import sys
import os
import logging
import csv
import time
import tempfile
import shutil
from pathlib import Path
import math
import mne
import numpy as np
from PyQt6 import QtWidgets, QtCore
from PyQt6.QtGui import QPixmap
from PyQt6.QtCore import QThread, pyqtSignal
import pyqtgraph as pg

LOGGER = logging.getLogger("janela_neuro")


def configurar_logging():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%H:%M:%S",
    )


def adicionar_no_path_se_necessario(novo_caminho: Path):
    caminho_str = str(novo_caminho)
    path_atual = os.environ.get("PATH", "")
    separador = os.pathsep
    entradas = path_atual.split(separador) if path_atual else []
    if caminho_str not in entradas:
        os.environ["PATH"] = caminho_str + separador + path_atual


def configurar_r_environment() -> bool:
    """Configura R_HOME de forma portátil sem hardcode obrigatório no código."""
    r_home_env = os.environ.get("R_HOME") or os.environ.get("NEURO_R_HOME")
    candidatos = [
        Path(r_home_env) if r_home_env else None,
        Path(r"C:\Program Files\R\R-4.5.2"),
        Path(r"C:\Program Files\R\R-4.4.0"),
        Path(r"C:\Program Files\R\R-4.3.3"),
    ]

    for candidato in candidatos:
        if not candidato:
            continue
        if candidato.exists():
            os.environ["R_HOME"] = str(candidato)
            bin_dir = candidato / "bin" / "x64"
            if bin_dir.exists():
                adicionar_no_path_se_necessario(bin_dir)
            LOGGER.info("R_HOME configurado para: %s", candidato)
            return True

    LOGGER.warning(
        "R_HOME nao encontrado. A analise R sera desativada nesta execucao."
    )
    return False


def caminho_saida_dir() -> Path:
    """Retorna diretório estável de saída sem depender do diretório de execução."""
    base_dir = Path(__file__).resolve().parent
    output_dir = Path(os.environ.get("NEURO_OUTPUT_DIR", base_dir / "output"))
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def calcular_parametros_sinal(raw, n_segmentos=20) -> tuple[list[dict], dict]:
    """
    Modificado para segmentar o sinal em N partes.
    Isso garante que o R tenha dados suficientes para gerar estatísticas.
    """
    dados = raw.get_data()
    sfreq = float(raw.info.get("sfreq", 1.0))
    ch_names = raw.ch_names
    
    # Tamanho de cada segmento
    n_amostras_total = dados.shape[1]
    tamanho_seg = n_amostras_total / n_segmentos
    
    metricas_acumuladas = []

    for i, canal_completo in enumerate(dados):
        for s in range(n_segmentos):
            start = int(s * tamanho_seg)
            end = int(start + tamanho_seg)
            sinal = canal_completo[start:end]
            
            if sinal.size < 2: continue
            
            # Cálculo de métricas por segmento
            rms = float(np.sqrt(np.mean(np.square(sinal))))
            wl = float(np.sum(np.abs(np.diff(sinal))))
            cruzamentos = int(np.sum((sinal[:-1] * sinal[1:]) < 0))
            zcr = float(cruzamentos / (sinal.size - 1))
            
            # Espectro para freq mediana
            psd = np.abs(np.fft.rfft(sinal - np.mean(sinal))) ** 2
            freqs = np.fft.rfftfreq(sinal.size, d=1.0 / sfreq)
            if np.sum(psd) > 0:
                idx_mediana = np.searchsorted(np.cumsum(psd), np.sum(psd) / 2.0)
                freq_mediana = float(freqs[min(idx_mediana, len(freqs)-1)])
            else:
                freq_mediana = 0.0

            metricas_acumuladas.append({
                "canal": ch_names[i],
                "rms": rms,
                "freq_mediana": freq_mediana,
                "zcr": zcr,
                "waveform_length": wl,
            })

    # Médias globais
    medias = {k: float(np.mean([m[k] for m in metricas_acumuladas])) for k in ["rms", "freq_mediana", "zcr", "waveform_length"]}
    return metricas_acumuladas, medias

# UI STYLE SHEET (Inspirado no estilo limpo do HTML5 UP)
ESTILO_PREMIUM = """
    QMainWindow { background-color: #f4f4f4; }
    
    /* Painel Lateral Estilo 'Sidebar' */
    QWidget#Sidebar { 
        background-color: #2c3e50; 
        border-right: 3px solid #1a252f;
    }
    
    QLabel { font-family: 'Helvetica Neue', Helvetica, Arial; color: #333; }
    QLabel#Logo { 
        color: #ffffff; 
        font-size: 22px; 
        font-weight: 300; 
        letter-spacing: 2px;
        padding: 20px;
    }
    
    /* Botões Estilo Flat Design */
    QPushButton {
        background-color: transparent;
        color: #bdc3c7;
        border: 1px solid #7f8c8d;
        border-radius: 2px;
        padding: 12px;
        text-align: left;
        font-size: 13px;
        margin: 5px 15px;
    }
    QPushButton:hover { 
        color: #ffffff; 
        border-color: #ffffff; 
        background-color: #34495e; 
    }
    
    /* Containers de Gráficos */
    QGroupBox {
        background-color: #ffffff;
        border: 1px solid #dcdde1;
        border-top: 4px solid #3498db;
        border-radius: 4px;
        margin-top: 20px;
        font-weight: bold;
        color: #2c3e50;
    }
"""


class LSLRealtimeWorker(QThread):
    """Worker que captura dados do LSL continuamente em tempo real."""
    
    # Sinais emitidos quando novos dados chegam
    dados_atualizados = pyqtSignal(np.ndarray, np.ndarray)  # (raw_bruto, raw_filt)
    status_atualizado = pyqtSignal(str)
    erro_captura = pyqtSignal(str)
    
    def __init__(self, sfreq: float = 250.0, buffer_size: int = 2500, intervalo_atualizar: float = 0.1):
        """
        Args:
            sfreq: Frequência de amostragem (Hz)
            buffer_size: Tamanho do buffer (amostras) - default 10 segundos a 250 Hz
            intervalo_atualizar: Intervalo mínimo entre emissões de sinais (segundos)
        """
        super().__init__()
        self.sfreq = sfreq
        self.buffer_size = buffer_size
        self.intervalo_atualizar = intervalo_atualizar
        self._running = False
        self._pause = False
        self.n_canais = 4  # Padrão: 4 canais EMG
        
    def run(self):
        """Loop principal que captura dados continuamente."""
        import time as _time

        self._running = True
        last_update = _time.time()
        
        try:
            import pylsl
        except ImportError:
            self.erro_captura.emit("pylsl não disponível. LSL em tempo real desativado.")
            self._running = False
            return
        
        # Inicializa streams
        emg_streams = []
        emg_proc_streams = []
        emg_inlet = None
        emg_proc_inlet = None
        
        # Busca por streams LSL
        timeout_busca = 5.0
        start_time = _time.time()
        
        while (_time.time() - start_time) < timeout_busca and self._running:
            available_streams = pylsl.resolve_streams(wait_time=0.5)
            
            for stream_info in available_streams:
                stream_name = stream_info.name()
                if stream_name == "EMG" and emg_inlet is None:
                    emg_inlet = pylsl.StreamInlet(stream_info, max_buflen=360)
                    self.n_canais = stream_info.channel_count()
                    self.status_atualizado.emit(f"Conectado ao stream EMG ({self.n_canais} canais)")
                elif stream_name == "EMG_Processado" and emg_proc_inlet is None:
                    emg_proc_inlet = pylsl.StreamInlet(stream_info, max_buflen=360)
                    self.status_atualizado.emit("Conectado ao stream EMG_Processado")
            
            if emg_inlet is not None and emg_proc_inlet is not None:
                break
        
        if emg_inlet is None:
            self.erro_captura.emit("Nenhum stream LSL 'EMG' encontrado.")
            self._running = False
            return
        
        # Buffers circulares para manter histórico
        buffer_bruto = np.zeros((self.n_canais, self.buffer_size))
        buffer_filt = np.zeros((self.n_canais, self.buffer_size))
        write_idx = 0
        
        self.status_atualizado.emit("Capturando dados em tempo real...")
        
        # Loop de captura
        while self._running:
            try:
                if self._pause:
                    _time.sleep(0.05)
                    continue
                
                # Puxa dados brutos
                sample, timestamp = emg_inlet.pull_sample(timeout=0.1)
                if sample is not None:
                    # Adiciona ao buffer
                    buffer_bruto[:, write_idx] = np.asarray(sample, dtype=float)
                    
                    # Se temos dados processados, puxa também
                    if emg_proc_inlet is not None:
                        sample_proc, _ = emg_proc_inlet.pull_sample(timeout=0.01)
                        if sample_proc is not None:
                            buffer_filt[:, write_idx] = np.asarray(sample_proc, dtype=float)
                        else:
                            # Filtro simples local como fallback
                            buffer_filt[:, write_idx] = np.asarray(sample, dtype=float) * 0.9
                    else:
                        buffer_filt[:, write_idx] = np.asarray(sample, dtype=float)
                    
                    write_idx = (write_idx + 1) % self.buffer_size
                    
                    # Emite sinal a cada intervalo_atualizar segundos
                    if (_time.time() - last_update) >= self.intervalo_atualizar:
                        # Reorganiza buffer em ordem cronológica
                        bruto_sorted = np.concatenate([
                            buffer_bruto[:, write_idx:],
                            buffer_bruto[:, :write_idx]
                        ], axis=1)
                        
                        filt_sorted = np.concatenate([
                            buffer_filt[:, write_idx:],
                            buffer_filt[:, :write_idx]
                        ], axis=1)
                        
                        self.dados_atualizados.emit(bruto_sorted, filt_sorted)
                        last_update = _time.time()
                
            except Exception as e:
                LOGGER.debug(f"Erro na captura LSL realtime: {e}")
                _time.sleep(0.01)
        
        self.status_atualizado.emit("Captura em tempo real parada.")
    
    def parar(self):
        """Para a captura de dados."""
        self._running = False
    
    def pausar(self, pausado: bool = True):
        """Pausa/retoma a captura sem encerrar a thread."""
        self._pause = pausado


class JanelaNeuro(QtWidgets.QMainWindow):
    def __init__(self, raw_bruto, raw_filt):
        super().__init__()
        self.setWindowTitle("Análise Neurofisiológica")
        self.setStyleSheet(ESTILO_PREMIUM)
        self.showMaximized()

        central = QtWidgets.QWidget()
        self.setCentralWidget(central)
        layout_main = QtWidgets.QHBoxLayout(central)
        layout_main.setContentsMargins(0, 0, 0, 0)
        layout_main.setSpacing(0)

        # --- SIDEBAR (HTML5 UP STYLE) ---
        sidebar = QtWidgets.QWidget()
        sidebar.setObjectName("Sidebar")
        sidebar.setFixedWidth(280)
        layout_side = QtWidgets.QVBoxLayout(sidebar)
        
        lbl_logo = QtWidgets.QLabel("Análise Miográfrica")
        lbl_logo.setObjectName("Logo")
        lbl_logo.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        
        btn1 = QtWidgets.QPushButton("📊 DASHBOARD PRINCIPAL")
        btn2 = QtWidgets.QPushButton("🔍 INSPEÇÃO DE CANAIS")
        btn3 = QtWidgets.QPushButton("💾 EXPORTAR RELATÓRIO")
        btn_toggle_metricas = QtWidgets.QPushButton("🧮 ALTERNAR VISUALIZAÇÃO")
        btn_reset = QtWidgets.QPushButton("🔄 RESETAR ZOOM")
        btn_reload_from_lsl = QtWidgets.QPushButton("⬇️ RECARREGAR DE LSL")
        btn1.clicked.connect(lambda: self.mostrar_em_desenvolvimento("Dashboard principal"))
        btn2.clicked.connect(lambda: self.mostrar_em_desenvolvimento("Inspeção de canais"))
        btn3.clicked.connect(lambda: self.mostrar_em_desenvolvimento("Exportar relatório"))
        btn_toggle_metricas.clicked.connect(self.alternar_visualizacao_analise)
        btn_reset.clicked.connect(self.reset_views)
        btn_reload_from_lsl.clicked.connect(self.recarregar_de_lsl)

        layout_side.addWidget(lbl_logo)
        layout_side.addSpacing(40)
        layout_side.addWidget(btn1)
        layout_side.addWidget(btn2)
        layout_side.addWidget(btn3)
        layout_side.addWidget(btn_toggle_metricas)
        layout_side.addWidget(btn_reset)
        layout_side.addWidget(btn_reload_from_lsl)
        layout_side.addStretch()
        
        lbl_footer = QtWidgets.QLabel("ELEX20 - 2026/1")
        lbl_footer.setStyleSheet("color: #7f8c8d; font-size: 10px; margin: 20px;")
        layout_side.addWidget(lbl_footer)

        # --- ÁREA DE CONTEÚDO (Grid de Gráficos) ---
        area_conteudo = QtWidgets.QWidget()
        layout_grid = QtWidgets.QGridLayout(area_conteudo)
        layout_grid.setContentsMargins(30, 30, 30, 30)
        layout_grid.setSpacing(20)

        # Boxes de Sinais (com referências aos layouts para atualização posterior)
        box_bruto = QtWidgets.QGroupBox("Sinal Bruto")
        self.layout_bruto = QtWidgets.QVBoxLayout(box_bruto)
        box_bruto.setSizePolicy(QtWidgets.QSizePolicy.Policy.Expanding, QtWidgets.QSizePolicy.Policy.Expanding)
        self.plot_bruto = self.criar_pyqtgraph(raw_bruto, "#2ecc71") # Verde Esmeralda
        self.layout_bruto.addWidget(self.plot_bruto)

        box_filt = QtWidgets.QGroupBox("Sinal Filtrado")
        self.layout_filt = QtWidgets.QVBoxLayout(box_filt)
        box_filt.setSizePolicy(QtWidgets.QSizePolicy.Policy.Expanding, QtWidgets.QSizePolicy.Policy.Expanding)
        self.plot_filt = self.criar_pyqtgraph(raw_filt, "#e74c3c") # Alizarin Red
        self.layout_filt.addWidget(self.plot_filt)

        # Box do R
        box_r = QtWidgets.QGroupBox("Análise Estatística")
        layout_r = QtWidgets.QVBoxLayout(box_r)
        box_r.setSizePolicy(QtWidgets.QSizePolicy.Policy.Expanding, QtWidgets.QSizePolicy.Policy.Expanding)
        self.lbl_modo_analise = QtWidgets.QLabel("Modo atual: Gráfico (R + ggplot2)")
        self.lbl_modo_analise.setStyleSheet("font-weight: normal; color: #34495e;")
        layout_r.addWidget(self.lbl_modo_analise)

        # Estado interno precisa existir antes de chamar carregar_r().
        self._tmp_dir_r_atual = None
        self._ultimo_bruto = None
        self._ultimo_filt = None
        self._graficos_ja_plotados = False
        self.window_seconds = 0.2

        self.metricas_por_canal, self.metricas_medias = calcular_parametros_sinal(raw_filt)

        self.painel_analise = QtWidgets.QStackedWidget()
        self.widget_graficos = self.criar_widget_graficos()
        self.widget_metricas = self.criar_widget_metricas()
        self.painel_analise.addWidget(self.widget_graficos)
        self.painel_analise.addWidget(self.widget_metricas)
        layout_r.addWidget(self.painel_analise)

        # Organizando no Grid
        layout_grid.addWidget(box_bruto, 0, 0)
        layout_grid.addWidget(box_filt, 1, 0)
        layout_grid.addWidget(box_r, 0, 1, 2, 1) # Ocupa duas linhas na coluna 1

        # Dá mais largura à coluna de análise (R) para evitar cortes/scroll horizontal
        layout_grid.setColumnStretch(0, 1)
        layout_grid.setColumnStretch(1, 2)

        layout_main.addWidget(sidebar)
        layout_main.addWidget(area_conteudo)

        # --- Inicializa thread de captura LSL em tempo real ---
        # buffer_size=1250 = 5 segundos de histórico a 250 Hz (escala menor no eixo X)
        self.lsl_worker = LSLRealtimeWorker(sfreq=250.0, buffer_size=1250, intervalo_atualizar=0.15)
        self.lsl_worker.dados_atualizados.connect(self.on_dados_lsl_recebidos)
        self.lsl_worker.status_atualizado.connect(self.on_status_lsl_atualizado)
        self.lsl_worker.erro_captura.connect(self.on_erro_lsl)
        self.lsl_worker.start()
        
        # Armazena referência aos plots items para atualização suave
        self.plot_items_bruto = []
        self.plot_items_filt = []
        self.raw_browser = raw_filt.copy()
        self.mne_browser = None

        # Abre o MNE browser depois da janela principal iniciar.
        QtCore.QTimer.singleShot(400, self.abrir_mne_browser)

    def criar_pyqtgraph(self, raw, cor_sinal):
        pw = pg.PlotWidget()
        pw.setBackground('#ffffff')
        pw.showGrid(x=True, y=True, alpha=0.1)
        
        # Configurar eixo X com escala pequena mas SEM mostrar números
        axis_x = pw.getAxis('bottom')
        axis_x.setTickSpacing(major=0.05, minor=0.01)
        axis_x.setLabel(text='Tempo', units='s')
        axis_x.setStyle(showValues=False)  # Esconde os números (reduz poluição visual)
        
        times = raw.times
        data = raw.get_data()
        
        # Plotando 2 canais com espessura premium
        plot_items = []
        for i in range(min(2, data.shape[0])):
            pen = pg.mkPen(color=cor_sinal, width=2, style=QtCore.Qt.PenStyle.SolidLine if i==0 else QtCore.Qt.PenStyle.DashLine)
            item = pw.plot(times, data[i, :], pen=pen)
            plot_items.append((item, times))
        
        # Remove bordas padrão para visual mais limpo
        pw.hideAxis('left')
        
        # Armazena referência aos items para atualização posterior
        setattr(pw, '_plot_items', plot_items)
        return pw
    
    def atualizar_plot_dados(self, plot_widget, dados_novos: np.ndarray) -> None:
        """Atualiza os dados dos plots existentes sem remover/recriar.
        
        Args:
            plot_widget: O PlotWidget já existente
            dados_novos: Array numpy com dados (n_canais x n_amostras)
        """
        try:
            plot_items = getattr(plot_widget, '_plot_items', None)
            if plot_items is None or len(plot_items) == 0:
                return
            
            # Mostra apenas a janela mais recente para ampliar a visualizacao
            n_samples = dados_novos.shape[1]
            sfreq = 250.0
            window_samples = max(1, int(self.window_seconds * sfreq))
            start_idx = max(0, n_samples - window_samples)
            dados_window = dados_novos[:, start_idx:]
            times = np.arange(dados_window.shape[1]) / sfreq
            
            # Atualiza cada plot item com os novos dados
            for idx, (plot_item, _) in enumerate(plot_items):
                if idx < dados_window.shape[0]:
                    plot_item.setData(times, dados_window[idx, :], connect="finite")

            plot_widget.setXRange(0.0, self.window_seconds, padding=0.0)
                    
        except Exception as e:
            LOGGER.debug(f"Erro ao atualizar dados do plot: {e}")

    def on_dados_lsl_recebidos(self, dados_bruto: np.ndarray, dados_filt: np.ndarray) -> None:
        """Slot chamado quando novos dados LSL chegam.
        
        Args:
            dados_bruto: Array com dados brutos (n_canais x n_amostras)
            dados_filt: Array com dados filtrados (n_canais x n_amostras)
        """
        try:
            # Guarda ultimo buffer para recarregar o R sem travar a UI
            self._ultimo_bruto = dados_bruto
            self._ultimo_filt = dados_filt

            # Atualiza os plots pyqtgraph em tempo real
            self.atualizar_plot_dados(self.plot_bruto, dados_bruto)
            self.atualizar_plot_dados(self.plot_filt, dados_filt)
            self.atualizar_mne_browser(dados_filt)
        except Exception as e:
            LOGGER.debug(f"Erro ao processar dados LSL: {e}")

    def on_status_lsl_atualizado(self, mensagem: str) -> None:
        """Slot para atualizar a barra de status com mensagens do LSL worker."""
        status_bar = self.statusBar()
        if status_bar:
            status_bar.showMessage(mensagem)
        LOGGER.info(f"LSL Status: {mensagem}")

    def on_erro_lsl(self, mensagem_erro: str) -> None:
        """Slot para tratar erros de captura LSL."""
        LOGGER.warning(f"Erro LSL: {mensagem_erro}")
        status_bar = self.statusBar()
        if status_bar:
            status_bar.showMessage(f"Erro: {mensagem_erro}")

    def atualizar_plots_pyqtgraph(self, raw_bruto, raw_filt) -> None:
        """Atualiza os gráficos pyqtgraph com novos dados.
        
        Args:
            raw_bruto: Dados brutos em formato MNE Raw
            raw_filt: Dados filtrados em formato MNE Raw
        """
        try:
            # Remove widgets antigos
            if self.plot_bruto is not None:
                self.layout_bruto.removeWidget(self.plot_bruto)
                self.plot_bruto.deleteLater()
            
            if self.plot_filt is not None:
                self.layout_filt.removeWidget(self.plot_filt)
                self.plot_filt.deleteLater()
            
            # Cria novos widgets
            self.plot_bruto = self.criar_pyqtgraph(raw_bruto, "#2ecc71")
            self.plot_filt = self.criar_pyqtgraph(raw_filt, "#e74c3c")
            
            # Adiciona novos widgets aos layouts
            self.layout_bruto.addWidget(self.plot_bruto)
            self.layout_filt.addWidget(self.plot_filt)
            
            LOGGER.info("Gráficos pyqtgraph atualizados com sucesso")
        except Exception as e:
            LOGGER.exception(f"Erro ao atualizar gráficos pyqtgraph: {e}")

    def criar_widget_graficos(self) -> QtWidgets.QWidget:
        container = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(container)
        self.lbl_status_graficos = QtWidgets.QLabel("Atualize para plotar os gráficos.")
        self.lbl_status_graficos.setStyleSheet("color: #7f8c8d; font-size: 12px;")
        layout.addWidget(self.lbl_status_graficos)

        scroll = QtWidgets.QScrollArea()
        scroll.setWidgetResizable(True)
        # Evita barra de rolagem horizontal indesejada
        scroll.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarPolicy.ScrollBarAlwaysOff)

        scroll_content = QtWidgets.QWidget()
        # Layout interno sem margens para ocupar todo o espaço do scroll
        self.layout_graficos = QtWidgets.QVBoxLayout(scroll_content)
        self.layout_graficos.setContentsMargins(0, 0, 0, 0)
        self.layout_graficos.setSpacing(12)

        self.lbl_r_box = QtWidgets.QLabel("Boxplot")
        self.lbl_r_pairs = QtWidgets.QLabel("Matriz de Dispersão")
        self.lbl_r_radar = QtWidgets.QLabel("Biplot")

        for lbl in (self.lbl_r_box, self.lbl_r_pairs, self.lbl_r_radar):
            lbl.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
            # Permite que o label expanda horizontalmente dentro do layout
            lbl.setSizePolicy(QtWidgets.QSizePolicy.Policy.Expanding, QtWidgets.QSizePolicy.Policy.Preferred)
            lbl.setMinimumHeight(200)
            lbl.setStyleSheet("background: #ffffff; border: 1px solid #dcdde1; padding: 20px;")
            lbl.setText("Atualize para plotar os gráficos")
            # Evita zoom/estiramento do pixmap a cada atualização
            lbl.setScaledContents(False)
            self.layout_graficos.addWidget(lbl)

        scroll.setWidget(scroll_content)
        # Garante que o scroll ocupe todo o espaço disponível no container
        layout.addWidget(scroll)
        return container

    def _mostrar_pixmap_em_label(self, label: QtWidgets.QLabel, caminho: Path) -> None:
        pix = QPixmap(str(caminho))
        if pix.isNull():
            label.setText("Falha ao carregar imagem")
            return

        # Se o label ainda não tem tamanho definido (a janela pode estar em construção), adiamos
        largura_label = max(1, label.width())
        if largura_label < 20:
            # aguarda o layout ser aplicado e tenta novamente
            QtCore.QTimer.singleShot(80, lambda: self._mostrar_pixmap_em_label(label, caminho))
            return

        # Escalona sem distorcer e sem zoom cumulativo
        alvo_largura = max(100, largura_label - 24)
        scaled = pix.scaled(
            alvo_largura,
            5000,
            QtCore.Qt.AspectRatioMode.KeepAspectRatio,
            QtCore.Qt.TransformationMode.SmoothTransformation,
        )
        label.setPixmap(scaled)
        # Ajusta altura para exibir a imagem completa (sem corte)
        label.setMinimumHeight(max(200, scaled.height() + 12))
        label.setText("")

    def _limpar_tmp_r_anterior(self) -> None:
        tmp_dir = getattr(self, "_tmp_dir_r_atual", None)
        if tmp_dir and tmp_dir.exists():
            shutil.rmtree(tmp_dir, ignore_errors=True)
        self._tmp_dir_r_atual = None

    def abrir_mne_browser(self) -> None:
        try:
            mne.viz.set_browser_backend("qt")
            self.mne_browser = mne.viz.plot_raw(
                self.raw_browser,
                title="MNE Inspector - Tempo Real",
                block=False,
                n_channels=min(4, len(self.raw_browser.ch_names)),
                show_options=True,
            )
            LOGGER.info("MNE Qt Browser aberto.")
        except Exception as e:
            LOGGER.warning(f"Nao foi possivel abrir MNE Browser: {e}")

    def atualizar_mne_browser(self, dados_filt: np.ndarray) -> None:
        if self.raw_browser is None or dados_filt is None:
            return

        try:
            n_canais = min(self.raw_browser._data.shape[0], dados_filt.shape[0])
            n_amostras = min(self.raw_browser._data.shape[1], dados_filt.shape[1])
            self.raw_browser._data[:, :] = 0.0
            self.raw_browser._data[:n_canais, -n_amostras:] = dados_filt[:n_canais, -n_amostras:]

            if self.mne_browser is not None:
                if hasattr(self.mne_browser, "_redraw"):
                    self.mne_browser._redraw(update_data=True)
                elif hasattr(self.mne_browser, "_update_data"):
                    self.mne_browser._update_data()
        except Exception as e:
            LOGGER.debug(f"Falha ao atualizar MNE Browser: {e}")


    def carregar_r(self):
        # R alimenta a janela; versão científica (Python) vai somente para output/.
        paths_r = gerar_grafico_r(self.metricas_por_canal)

        if paths_r:
            self._limpar_tmp_r_anterior()
            self._tmp_dir_r_atual = paths_r.get("tmp_dir")
            self._mostrar_pixmap_em_label(self.lbl_r_box, paths_r["boxplot"])
            self._mostrar_pixmap_em_label(self.lbl_r_pairs, paths_r["pares"])
            self._mostrar_pixmap_em_label(self.lbl_r_radar, paths_r["radar"])
            self.lbl_status_graficos.setText("Gráficos R atualizados.")
            self._graficos_ja_plotados = True
            LOGGER.info("Gráficos R atualizados na janela.")
        else:
            self.lbl_status_graficos.setText("Falha ao gerar gráficos R.")
            LOGGER.warning("R não disponível para renderização na janela.")

        # Sempre gera a versão científica em arquivos PNG no output.
        paths_py = gerar_grafico_python(self.metricas_por_canal)
        if paths_py:
            LOGGER.info("Versões científicas salvas em output/: %s", paths_py)
        else:
            LOGGER.warning("Não foi possível gerar gráficos científicos no output.")

    def criar_widget_metricas(self):
        container = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(container)

        resumo = QtWidgets.QLabel(
            (
                "Métricas médias (todos os canais) | "
                f"RMS: {self.metricas_medias['rms']:.6f} | "
                f"Média da Frequência Mediana: {self.metricas_medias['freq_mediana']:.3f} Hz | "
                f"ZCR: {self.metricas_medias['zcr']:.6f} | "
                f"Waveform Length: {self.metricas_medias['waveform_length']:.6f}"
            )
        )
        resumo.setWordWrap(True)
        layout.addWidget(resumo)

        tabela = QtWidgets.QTableWidget()
        tabela.setColumnCount(5)
        tabela.setHorizontalHeaderLabels(
            ["Canal", "RMS", "Freq. Mediana (Hz)", "Zero Crossing Rate", "Waveform Length"]
        )
        tabela.setRowCount(len(self.metricas_por_canal) + 1)

        tabela.setItem(0, 0, QtWidgets.QTableWidgetItem("Média (todos)"))
        tabela.setItem(0, 1, QtWidgets.QTableWidgetItem(f"{self.metricas_medias['rms']:.6f}"))
        tabela.setItem(0, 2, QtWidgets.QTableWidgetItem(f"{self.metricas_medias['freq_mediana']:.3f}"))
        tabela.setItem(0, 3, QtWidgets.QTableWidgetItem(f"{self.metricas_medias['zcr']:.6f}"))
        tabela.setItem(0, 4, QtWidgets.QTableWidgetItem(f"{self.metricas_medias['waveform_length']:.6f}"))

        for i, metrica in enumerate(self.metricas_por_canal, start=1):
            tabela.setItem(i, 0, QtWidgets.QTableWidgetItem(metrica["canal"]))
            tabela.setItem(i, 1, QtWidgets.QTableWidgetItem(f"{metrica['rms']:.6f}"))
            tabela.setItem(i, 2, QtWidgets.QTableWidgetItem(f"{metrica['freq_mediana']:.3f}"))
            tabela.setItem(i, 3, QtWidgets.QTableWidgetItem(f"{metrica['zcr']:.6f}"))
            tabela.setItem(i, 4, QtWidgets.QTableWidgetItem(f"{metrica['waveform_length']:.6f}"))

        header = tabela.horizontalHeader()
        if header is not None:
            header.setStretchLastSection(True)
            header.setSectionResizeMode(QtWidgets.QHeaderView.ResizeMode.ResizeToContents)
        layout.addWidget(tabela)
        return container

    def alternar_visualizacao_analise(self):
        indice_atual = self.painel_analise.currentIndex()
        novo_indice = 1 if indice_atual == 0 else 0
        self.painel_analise.setCurrentIndex(novo_indice)
        if novo_indice == 0:
            self.lbl_modo_analise.setText("Modo atual: Gráfico (R + ggplot2)")
        else:
            self.lbl_modo_analise.setText("Modo atual: Tabela de parâmetros")

    def reset_views(self):
        self.plot_bruto.autoRange()
        self.plot_filt.autoRange()

    def mostrar_em_desenvolvimento(self, recurso: str):
        QtWidgets.QMessageBox.information(
            self,
            "Recurso em desenvolvimento",
            f"{recurso} ainda não foi implementado.",
        )

    def closeEvent(self, a0) -> None:
        """Encerramento seguro da aplicação com limpeza de threads."""
        LOGGER.info("Encerrando aplicação...")
        
        # Para a thread LSL worker
        if hasattr(self, 'lsl_worker') and self.lsl_worker is not None:
            self.lsl_worker.parar()
            self.lsl_worker.wait(2000)  # Aguarda até 2 segundos
            if self.lsl_worker.isRunning():
                self.lsl_worker.terminate()
                self.lsl_worker.wait()

        self._limpar_tmp_r_anterior()

        if self.mne_browser is not None:
            try:
                self.mne_browser.close()
            except Exception:
                pass
        
        if a0 is not None:
            a0.accept()

    def capturar_dados_lsl(self) -> tuple[np.ndarray | None, np.ndarray | None]:
        """Captura dados de streams LSL: EMG e EMG_Processado.
        
        Returns:
            Tupla (emg_data, emg_proc_data) onde cada elemento pode ser ndarray ou None
        """
        try:
            import pylsl
        except ImportError:
            LOGGER.error("pylsl não disponível. Instale: pip install pylsl")
            QtWidgets.QMessageBox.critical(self, "Erro", "pylsl não instalado. Execute: pip install pylsl")
            return None, None

        try:
            LOGGER.info("Procurando por streams LSL...")
            streams = pylsl.resolve_streams(wait_time=3.0)
            if not streams:
                LOGGER.warning("Nenhum stream LSL encontrado.")
                QtWidgets.QMessageBox.warning(self, "Aviso", "Nenhum stream LSL encontrado.")
                return None, None

            emg_stream = None
            emg_proc_stream = None

            for stream_info in streams:
                if stream_info.name() == "EMG":
                    emg_stream = pylsl.StreamInlet(stream_info)
                    LOGGER.info("Stream EMG encontrado")
                elif stream_info.name() == "EMG_Processado":
                    emg_proc_stream = pylsl.StreamInlet(stream_info)
                    LOGGER.info("Stream EMG_Processado encontrado")

            if emg_stream is None:
                LOGGER.warning("Stream EMG não encontrado")
                QtWidgets.QMessageBox.warning(self, "Aviso", "Stream 'EMG' não encontrado.")
                return None, None

            # Captura dados por 5 segundos
            emg_data = []
            emg_proc_data = []
            timeout = 5.0
            import time
            start_time = time.time()

            while (time.time() - start_time) < timeout:
                if emg_stream:
                    sample, timestamp = emg_stream.pull_sample(timeout=0.1)
                    if sample is not None:
                        emg_data.append(sample)
                if emg_proc_stream:
                    sample, timestamp = emg_proc_stream.pull_sample(timeout=0.1)
                    if sample is not None:
                        emg_proc_data.append(sample)

            if not emg_data:
                LOGGER.warning("Nenhum dado capturado do EMG")
                QtWidgets.QMessageBox.warning(self, "Aviso", "Nenhum dado capturado.")
                return None, None

            LOGGER.info(f"Capturados {len(emg_data)} amostras de EMG")
            return np.array(emg_data), np.array(emg_proc_data) if emg_proc_data else None

        except Exception as e:
            LOGGER.exception(f"Erro ao capturar dados LSL: {e}")
            QtWidgets.QMessageBox.critical(self, "Erro", f"Erro ao capturar dados: {str(e)}")
            return None, None

    def recarregar_de_lsl(self):
        """Recarrega gráficos usando o ultimo buffer do LSL (sem travar UI)."""
        status_bar = self.statusBar()
        if status_bar:
            status_bar.showMessage("Atualizando gráficos com ultimo buffer...")

        if self._ultimo_bruto is None:
            if status_bar:
                status_bar.showMessage("Sem dados recentes do LSL")
            return

        try:
            sfreq = 250.0
            bruto = self._ultimo_bruto
            filt = self._ultimo_filt if self._ultimo_filt is not None else bruto

            raw_bruto_new = self._criar_raw_mne(bruto, sfreq, "EMG (LSL)")
            if raw_bruto_new is None:
                raise ValueError("Falha ao criar Raw MNE")

            raw_filt_new = self._criar_raw_mne(filt, sfreq, "EMG Proc (LSL)")
            if raw_filt_new is None:
                raw_filt_new = raw_bruto_new.copy().filter(1, 40)

            self.metricas_por_canal, self.metricas_medias = calcular_parametros_sinal(raw_filt_new)
            self.carregar_r()

            old_widget = self.painel_analise.widget(1)
            if old_widget is not None:
                self.painel_analise.removeWidget(old_widget)
                old_widget.deleteLater()
            self.widget_metricas = self.criar_widget_metricas()
            self.painel_analise.addWidget(self.widget_metricas)

            if status_bar:
                status_bar.showMessage("Recarregado!")
            LOGGER.info("Gráficos recarregados com buffer LSL")
        except Exception as e:
            LOGGER.exception(f"Erro: {e}")
            if status_bar:
                status_bar.showMessage(f"Erro: {str(e)}")

    def _criar_raw_mne(self, data: np.ndarray, sfreq: float, desc: str = "Dados LSL") -> mne.io.RawArray | None:
        """Cria objeto MNE Raw a partir de dados numpy.
        
        Args:
            data: Array numpy com dados (n_canais x n_amostras) ou (n_amostras,)
            sfreq: Frequência de amostragem em Hz
            desc: Descrição dos dados
            
        Returns:
            Objeto mne.io.RawArray ou None se houver erro
        """
        try:
            if data is None or len(data) == 0:
                LOGGER.error("Dados vazios ou None")
                return None
                
            # Garantir formato correto
            if len(data.shape) == 1:
                data = data.reshape(1, -1)
            
            # Converter para float se necessário
            data = np.asarray(data, dtype=float)
            
            # Cria info do MNE
            n_canais = int(data.shape[0])
            ch_names = [f"EMG_{i+1}" for i in range(n_canais)]
            info = mne.create_info(ch_names, int(sfreq), ch_types="emg")

            # Cria objeto Raw
            raw = mne.io.RawArray(data, info)
            LOGGER.info(f"Raw MNE criado: {n_canais} canais, {raw.n_times} amostras")
            return raw
            
        except Exception as e:
            LOGGER.exception(f"Erro ao criar Raw MNE: {e}")
            return None

def inferir_modo_sinal(metricas_por_canal: list[dict]) -> list[dict]:
    """Classifica canais em modos de energia com base no RMS.
    
    Args:
        metricas_por_canal: Lista de dicts com métricas por canal
        
    Returns:
        Lista de dicts com métricas classificadas em modos
    """
    if not metricas_por_canal or len(metricas_por_canal) == 0:
        return []

    try:
        rms_vals = np.array([m.get("rms", 0.0) for m in metricas_por_canal], dtype=float)
        if rms_vals.size == 0:
            return []
            
        q1, q2 = np.quantile(rms_vals, [0.33, 0.66])

        linhas = []
        for metrica in metricas_por_canal:
            rms = float(metrica.get("rms", 0.0))
            if rms <= q1:
                modo = "Baixa energia"
            elif rms <= q2:
                modo = "Media energia"
            else:
                modo = "Alta energia"

            linhas.append(
                {
                    "canal": str(metrica.get("canal", "Canal")),
                    "modo": modo,
                    "rms": float(metrica.get("rms", 0.0)),
                    "freq_mediana": float(metrica.get("freq_mediana", 0.0)),
                    "zcr": float(metrica.get("zcr", 0.0)),
                    "waveform_length": float(metrica.get("waveform_length", 0.0)),
                }
            )
        return linhas
    except Exception as e:
        LOGGER.exception(f"Erro ao inferir modo de sinal: {e}")
        return []


def salvar_metricas_csv(metricas_por_canal: list[dict], csv_path: Path) -> bool:
    """Salva métricas em arquivo CSV.
    
    Args:
        metricas_por_canal: Lista de dicts com métricas
        csv_path: Caminho do arquivo CSV de saída
        
    Returns:
        True se bem-sucedido, False caso contrário
    """
    try:
        linhas = inferir_modo_sinal(metricas_por_canal)
        if not linhas:
            LOGGER.warning("Nenhuma métrica para salvar")
            return False
            
        campos = ["canal", "modo", "rms", "freq_mediana", "zcr", "waveform_length"]
        with csv_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=campos)
            writer.writeheader()
            writer.writerows(linhas)
        LOGGER.info(f"Métricas salvas em: {csv_path}")
        return True
    except Exception as e:
        LOGGER.exception(f"Erro ao salvar métricas CSV: {e}")
        return False


def gerar_grafico_r(metricas_por_canal: list[dict] | None = None) -> dict[str, Path] | None:
    """Gera gráficos estatísticos usando R e ggplot2 para exibição nativa em Qt.
    
    Args:
        metricas_por_canal: Lista de dicts com métricas por canal
        
    Returns:
        Dicionário com caminhos dos PNGs temporários ou None se falhar
    """
    import tempfile as _tempfile

    temp_dir = Path(_tempfile.mkdtemp(prefix="neuro_r_"))
    output_csv = temp_dir / "metricas_canais.csv"
    
    # Usar diretório temporário para gráficos (não salva em output/)
    temp_png_box = temp_dir / "grafico_boxplot_facet_temp.png"
    temp_png_pairs = temp_dir / "grafico_matriz_dispersao_temp.png"
    temp_png_radar = temp_dir / "grafico_radar_temp.png"

    if not metricas_por_canal:
        LOGGER.warning("Sem metricas para gerar analise R.")
        return None

    salvar_metricas_csv(metricas_por_canal, output_csv)

    try:
        import rpy2.robjects as robjects
        from rpy2.robjects import packages as rpackages
        rpackages.importr('base')
        rpackages.importr('utils')
        rpackages.importr('stats')
        rpackages.importr('ggplot2')
        rpackages.importr('tidyr')
        rpackages.importr('dplyr')
        rpackages.importr('scales')

        output_r_csv = str(output_csv).replace("\\", "/")
        output_r_png_box = str(temp_png_box).replace("\\", "/")
        output_r_png_pairs = str(temp_png_pairs).replace("\\", "/")
        output_r_png_radar = str(temp_png_radar).replace("\\", "/")

        robjects.r(f'''
            library(ggplot2)
            library(tidyr)
            library(dplyr)
            library(scales)

            df <- read.csv("{output_r_csv}", stringsAsFactors = FALSE)
            df$modo <- factor(df$modo, levels = c("Baixa energia", "Media energia", "Alta energia"))

            nomes_features <- c(
                rms = "RMS",
                freq_mediana = "Freq. Mediana",
                zcr = "Zero Crossing Rate",
                waveform_length = "Waveform Length"
            )

            # 1) Boxplot combinado por feature (facets) + jitter + média
            long_df <- df %>%
                pivot_longer(
                    cols = c(rms, freq_mediana, zcr, waveform_length),
                    names_to = "feature",
                    values_to = "valor"
                ) %>%
                mutate(feature = factor(feature, levels = names(nomes_features), labels = unname(nomes_features)))

            p_box <- ggplot(long_df, aes(x = modo, y = valor, color = modo, fill = modo)) +
                geom_boxplot(outlier.shape = NA, alpha = 0.25, width = 0.58, linewidth = 0.5) +
                geom_jitter(width = 0.11, alpha = 0.75, size = 1.8) +
                stat_summary(fun = mean, geom = "point", shape = 23, size = 3, fill = "white", color = "black") +
                facet_wrap(~feature, scales = "free_y", ncol = 2) +
                scale_color_manual(values = c("Baixa energia" = "#1f78b4", "Media energia" = "#33a02c", "Alta energia" = "#e31a1c")) +
                scale_fill_manual(values = c("Baixa energia" = "#a6cee3", "Media energia" = "#b2df8a", "Alta energia" = "#fb9a99")) +
                labs(
                    x = "Modo inferido a partir do RMS",
                    y = "Valor da feature"
                ) + 
                theme_minimal(base_size = 12) +
                theme(
                    legend.position = "bottom",
                    legend.box.spacing = unit(0.5, "cm"),
                    legend.margin = margin(t = 10, b = 5, l = 5, r = 5),
                    plot.margin = margin(b = 60, t = 20, l = 20, r = 20),
                    panel.grid.minor = element_blank(),
                    axis.title.x = element_text(margin = margin(t = 20)),
                    axis.title.y = element_text(margin = margin(r = 20))
                )

            ggsave("{output_r_png_box}", plot = p_box, width = 12, height = 9, dpi = 120)

            # 2) Matriz de dispersão com correlação
            pair_df <- df %>% select(rms, freq_mediana, zcr, waveform_length)
            colnames(pair_df) <- unname(nomes_features[colnames(pair_df)])

            # Jitter robusto para evitar alinhamentos verticais/horizontais em sinais quase constantes
            pair_df_jittered <- pair_df %>%
                mutate(across(everything(), ~ {{
                    rng <- diff(range(.x, na.rm = TRUE))
                    amt <- if (!is.finite(rng) || rng == 0) 0.002 else max(rng * 0.03, 0.001)
                    jitter(.x, amount = amt)
                }}))

            if (requireNamespace("GGally", quietly = TRUE)) {{
                p_pairs <- suppressWarnings(GGally::ggpairs(
                    pair_df_jittered, # <--- Agora com dados garantidos
                    upper = list(continuous = GGally::wrap("cor", size = 4, color = "#2c3e50")),
                    lower = list(continuous = GGally::wrap("points", alpha = 0.35, size = 0.9, color = "#2c7fb8")),
                    diag = list(continuous = GGally::wrap("densityDiag", alpha = 0.6, fill = "#74a9cf"))
                )) +
                theme_minimal(base_size = 11) +
                theme(panel.grid = element_blank())

                ggsave("{output_r_png_pairs}", plot = p_pairs, width = 12, height = 10, dpi = 120)
            }} else {{
                panel_cor <- function(x, y, digits = 2, cex.cor = 1.1, ...) {{
                    old_usr <- par("usr")
                    on.exit(par(usr = old_usr))
                    par(usr = c(0, 1, 0, 1))
                    sx <- sd(x, na.rm = TRUE)
                    sy <- sd(y, na.rm = TRUE)
                    if (is.na(sx) || is.na(sy) || sx == 0 || sy == 0) {{
                        r <- 0
                    }} else {{
                        r <- suppressWarnings(cor(x, y, use = "pairwise.complete.obs"))
                        if (!is.finite(r)) r <- 0
                    }}
                    txt <- formatC(r, format = "f", digits = digits)
                    text(0.5, 0.5, txt, cex = cex.cor, col = "#2c3e50")
                }}

                png(filename = "{output_r_png_pairs}", width = 1300, height = 1100, res = 120)
                pairs(
                    pair_df,
                    pch = 19,
                    col = rgb(0.18, 0.48, 0.72, 0.55),
                    upper.panel = panel_cor
                )
                dev.off()
            }}

            # 3) Biplot PCA: pontos por modo + vetores das features
            matriz_features <- df %>%
                select(rms, freq_mediana, zcr, waveform_length) %>%
                mutate(across(everything(), ~ as.numeric(scale(.x))))

            matriz_features[!is.finite(as.matrix(matriz_features))] <- 0

            pca <- prcomp(matriz_features, center = FALSE, scale. = FALSE)
            scores <- as.data.frame(pca$x[, 1:2])
            scores$modo <- df$modo

            loadings <- as.data.frame(pca$rotation[, 1:2])
            loadings$feature <- c("RMS", "Freq. Mediana", "Zero Crossing Rate", "Waveform Length")

            names(scores)[1:2] <- c("PC1", "PC2")
            names(loadings)[1:2] <- c("PC1", "PC2")

            var_exp <- (pca$sdev^2) / sum(pca$sdev^2)
            pct1 <- round(var_exp[1] * 100, 1)
            pct2 <- round(var_exp[2] * 100, 1)

            escala_setas <- max(
                max(abs(scores$PC1), na.rm = TRUE),
                max(abs(scores$PC2), na.rm = TRUE)
            ) * 0.72 / max(
                max(abs(loadings$PC1), na.rm = TRUE),
                max(abs(loadings$PC2), na.rm = TRUE),
                1e-9
            )

            loadings <- loadings %>%
                mutate(
                    x_end = PC1 * escala_setas,
                    y_end = PC2 * escala_setas
                )

            p_biplot <- ggplot(scores, aes(x = PC1, y = PC2, color = modo)) +
                geom_hline(yintercept = 0, linewidth = 0.4, color = "gray80") +
                geom_vline(xintercept = 0, linewidth = 0.4, color = "gray80") +
                geom_point(size = 2.6, alpha = 0.85) +
                geom_segment(
                    data = loadings,
                    aes(x = 0, y = 0, xend = x_end, yend = y_end),
                    inherit.aes = FALSE,
                    color = "#111827",
                    linewidth = 0.9,
                    arrow = arrow(length = unit(0.22, "cm"))
                ) +
                scale_color_manual(values = c("Baixa energia" = "#1f78b4", "Media energia" = "#33a02c", "Alta energia" = "#e31a1c")) +
                labs(
                    x = paste0("PC1 (", pct1, "% da variância)"),
                    y = paste0("PC2 (", pct2, "% da variância)")
                    # Título removido conforme solicitado
                ) +
                coord_equal() +
                theme_minimal(base_size = 12) +
                theme(
                    legend.position = "bottom",
                    panel.grid.minor = element_blank(),
                    plot.margin = margin(18, 18, 18, 18),
                    plot.title = element_blank() # Garante a remoção de qualquer resquício de título
                )

            ggsave("{output_r_png_radar}", plot = p_biplot, width = 10, height = 8, dpi = 130)
        ''')

        LOGGER.info("Gráficos R gerados em PNG temporário.")
        return {
            "boxplot": temp_png_box,
            "pares": temp_png_pairs,
            "radar": temp_png_radar,
            "tmp_dir": temp_dir,
        }
        
    except ImportError as e:
        LOGGER.error("Dependencia Python ausente para R: %s", e)
        return None  # Falha
    except Exception as e:
        LOGGER.exception("Erro ao gerar grafico no R: %s", e)
        shutil.rmtree(temp_dir, ignore_errors=True)
        return None  # Falha


def gerar_grafico_python(metricas_por_canal: list[dict] | None = None) -> dict[str, Path] | None:
    """Gera gráficos estatísticos usando matplotlib (fallback).
    
    Args:
        metricas_por_canal: Lista de dicts com métricas por canal
        
    Returns:
        Caminhos dos PNGs científicos no output ou None se falhar
    """
    if not metricas_por_canal:
        return None

    try:
        import matplotlib.pyplot as plt
        from matplotlib.lines import Line2D
    except ImportError as e:
        LOGGER.error("Matplotlib nao disponivel para fallback Python: %s", e)
        return None

    linhas = inferir_modo_sinal(metricas_por_canal)
    if not linhas:
        return None

    np.random.seed(42)

    output_dir = caminho_saida_dir()
    output_png_box = output_dir / "grafico_boxplot_facet_py.png"
    output_png_pairs = output_dir / "grafico_matriz_dispersao_py.png"
    output_png_radar = output_dir / "grafico_radar_py.png"

    features = ["rms", "freq_mediana", "zcr", "waveform_length"]
    nomes_features = {
        "rms": "RMS",
        "freq_mediana": "Freq. Mediana",
        "zcr": "Zero Crossing Rate",
        "waveform_length": "Waveform Length",
    }
    modos = ["Baixa energia", "Media energia", "Alta energia"]
    # Paleta com alto contraste 
    cores = {
        "Baixa energia": "#0072B2",
        "Media energia": "#D55E00",
        "Alta energia": "#009E73",
    }
    marcadores = {
        "Baixa energia": "o",
        "Media energia": "s",
        "Alta energia": "^",
    }

    plt.style.use("seaborn-v0_8-whitegrid")
    plt.rcParams["font.family"] = "serif"
    plt.rcParams["font.serif"] = ["Cambria", "Georgia", "DejaVu Serif", "serif"]

    # 1) Boxplot facetado + jitter + media
    if not features: return 

    fig, axes = plt.subplots(2, 2, figsize=(13.2, 9.8), constrained_layout=False, dpi=150)
    fig.patch.set_facecolor("#FFFFFF")
    
    ax = None
    axes_list = axes.flatten() 

    for idx, feature in enumerate(features):
        ax = axes_list[idx]
        dados_modo = [
            [l[feature] for l in linhas if l["modo"] == modo]
            for modo in modos
        ]

        bp = ax.boxplot(
            dados_modo,
            patch_artist=True,
            tick_labels=modos,
            showfliers=False,
            widths=0.58,
            medianprops={"color": "#1F2937", "linewidth": 2.0},
            whiskerprops={"color": "#4B5563", "linewidth": 1.5},
            capprops={"color": "#4B5563", "linewidth": 1.5},
        )

        for patch, modo in zip(bp["boxes"], modos):
            patch.set_facecolor(cores[modo])
            patch.set_alpha(0.25)
            patch.set_edgecolor(cores[modo])
            patch.set_linewidth(1.5)

        for pos, (modo, valores) in enumerate(zip(modos, dados_modo), start=1):
            if not valores:
                continue
            jitter_x = np.random.normal(loc=pos, scale=0.045, size=len(valores))
            ax.scatter(
                jitter_x,
                valores,
                color=cores[modo],
                alpha=0.85,
                s=28,
                linewidths=0.8,
                edgecolors="#111111",
                marker=marcadores[modo],
            )
            media = float(np.mean(valores))
            ax.scatter([pos], [media], marker="D", s=64, color="#1F2937", zorder=3, edgecolors="#000000", linewidth=0.8)

        ax.set_title(nomes_features[feature], fontsize=12, fontweight="bold", color="#1F2937", family="serif")
        ax.grid(True, axis="y", alpha=0.3, linestyle="-", linewidth=0.6, color="#D1D5DB")
        ax.grid(False, axis="x")
        ax.tick_params(axis="x", rotation=35, labelsize=9, pad=6)
        ax.tick_params(axis="y", labelsize=9, pad=5)
        ax.set_facecolor("#FFFFFF")

    fig.suptitle("Distribuição das Features de EMG por Nível de Energia", 
                 fontsize=15, fontweight="bold", family="serif", y=0.98)
    
    handles_modo = [
        Line2D([0], [0], marker=marcadores[m], color='w', markerfacecolor=cores[m], 
               markeredgecolor="#111111", markersize=8, label=m)
        for m in modos
    ]
    
    fig.legend(
        handles=handles_modo, 
        title="Níveis de Energia", 
        loc="center left", 
        bbox_to_anchor=(1.0, 0.5), 
        frameon=False, 
        prop={"family": "serif", "size": 9}
    )
    
    fig.subplots_adjust(right=0.85, top=0.90)

    fig.savefig(output_png_box, dpi=300, bbox_inches="tight", facecolor="#FFFFFF", edgecolor="none")
    plt.close(fig)

    # 2) Matriz de dispersao e correlacao
    dados_feature = {feature: np.array([l[feature] for l in linhas], dtype=float) for feature in features}

    n = len(features)
    fig, axes = plt.subplots(n, n, figsize=(12.8, 12.8), constrained_layout=False, dpi=150)
    fig.patch.set_facecolor("#FFFFFF")

    for i in range(n):
        for j in range(n):
            ax = axes[i, j]
            fi = features[i]
            fj = features[j]

            if i == j:
                ax.text(
                    0.5,
                    0.5,
                    nomes_features[fi],
                    ha="center",
                    va="center",
                    fontsize=12,
                    fontweight="bold",
                    color="#111827",
                    family="serif",
                )
                ax.set_xticks([])
                ax.set_yticks([])
                ax.grid(False)
            elif i > j:
                for modo in modos:
                    xs = [l[fj] for l in linhas if l["modo"] == modo]
                    ys = [l[fi] for l in linhas if l["modo"] == modo]
                    if xs and ys:
                        ax.scatter(
                            xs,
                            ys,
                            s=28,
                            alpha=0.85,
                            color=cores[modo],
                            marker=marcadores[modo],
                            edgecolors="#111111",
                            linewidths=0.45,
                        )
            else:
                x = dados_feature[fj]
                y = dados_feature[fi]
                if x.size > 1 and y.size > 1 and float(np.std(x)) > 1e-12 and float(np.std(y)) > 1e-12:
                    with np.errstate(invalid="ignore", divide="ignore"):
                        corr = float(np.corrcoef(x, y)[0, 1])
                    if not np.isfinite(corr):
                        corr = 0.0
                else:
                    corr = 0.0
                cor_corr = "#374151" if abs(corr) < 0.4 else "#1F2937" if abs(corr) >= 0.4 else "#4B5563"
                ax.text(0.5, 0.56, f"r = {corr:.2f}", ha="center", va="center", fontsize=12, color=cor_corr, fontweight="bold", family="serif")
                ax.text(0.5, 0.38, "correlação", ha="center", va="center", fontsize=9, color="#6B7280", family="serif")
                ax.set_xlim(0, 1)
                ax.set_ylim(0, 1)

            if i == n - 1:
                ax.set_xlabel(nomes_features[fj], fontsize=10, family="serif", labelpad=8)
                ax.tick_params(axis="x", labelsize=8, pad=4)
            else:
                ax.set_xticklabels([])
            if j == 0:
                ax.set_ylabel(nomes_features[fi], fontsize=10, family="serif", labelpad=8)
                ax.tick_params(axis="y", labelsize=8, pad=4)
            else:
                ax.set_yticklabels([])

            ax.grid(True, alpha=0.2, linestyle=":", color="#D1D5DB", linewidth=0.6)
            ax.set_facecolor("#FFFFFF")
            for spine in ax.spines.values():
                spine.set_edgecolor("#D1D5DB")
                spine.set_linewidth(0.8)

    handles = [
        Line2D(
            [0],
            [0],
            marker=marcadores[m],
            linestyle="",
            color=cores[m],
            markerfacecolor=cores[m],
            markeredgecolor="#111111",
            markeredgewidth=0.8,
            label=m,
            markersize=8,
        )
        for m in modos
    ]
    fig.subplots_adjust(top=0.86, bottom=0.07, left=0.07, right=0.98, wspace=0.07, hspace=0.07)
    fig.legend(
        handles=handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.93),
        ncol=3,
        frameon=False,
        title="Modo de sinal",
        fontsize=10,
        title_fontsize=11,
        prop={"family": "serif"},
    )
    fig.suptitle(
        "Matriz de Dispersão e Correlação entre Features",
        fontsize=16,
        fontweight="bold",
        color="#0F172A",
        family="serif",
        y=0.985,
    )
    fig.savefig(output_png_pairs, dpi=300, bbox_inches="tight", pad_inches=0.28, facecolor="#FFFFFF", edgecolor="none")
    plt.close(fig)

    # 3) Biplot PCA: padroniza features, projeta em PC1/PC2, mostra pontos por modo e vetores das features
    X = np.array([[float(l[f]) for f in features] for l in linhas], dtype=float)
    if X.size == 0 or X.shape[0] < 2:
        return None

    means = np.nanmean(X, axis=0)
    stds = np.nanstd(X, axis=0)
    stds[~np.isfinite(stds) | (stds == 0.0)] = 1.0
    Xz = (X - means) / stds
    Xz[~np.isfinite(Xz)] = 0.0

    U, S, Vt = np.linalg.svd(Xz, full_matrices=False)
    if Vt.shape[0] < 2:
        return None

    scores = U[:, :2] * S[:2]
    loadings = Vt[:2, :].T
    var_exp = (S ** 2) / max(Xz.shape[0] - 1, 1)
    total_var = float(np.sum(var_exp)) if np.isfinite(np.sum(var_exp)) else 0.0
    pct1 = round(float(var_exp[0] / total_var * 100.0), 1) if total_var > 0 else 0.0
    pct2 = round(float(var_exp[1] / total_var * 100.0), 1) if total_var > 0 and len(var_exp) > 1 else 0.0

    max_score = max(float(np.max(np.abs(scores[:, 0]))), float(np.max(np.abs(scores[:, 1]))), 1e-9)
    max_loading = max(float(np.max(np.abs(loadings[:, 0]))), float(np.max(np.abs(loadings[:, 1]))), 1e-9)
    arrow_scale = max_score * 0.72 / max_loading
    loadings_scaled = loadings * arrow_scale

    fig = plt.figure(figsize=(12.6, 8.8), constrained_layout=False, dpi=150)
    fig.patch.set_facecolor("#FFFFFF")
    ax = fig.add_subplot(111)

    cores_vetores = {
        "rms": "#8E44AD",
        "freq_mediana": "#2980B9",
        "zcr": "#E67E22",
        "waveform_length": "#16A085",
    }

    for modo in modos:
        idxs = [i for i, l in enumerate(linhas) if l["modo"] == modo]
        if not idxs:
            continue
        ax.scatter(
            scores[idxs, 0],
            scores[idxs, 1],
            s=34,
            alpha=0.85,
            color=cores[modo],
            edgecolors="#111111",
            linewidths=0.6,
            label=modo,
            marker=marcadores[modo],
        )

    ax.axhline(0, color="#D1D5DB", linewidth=0.8)
    ax.axvline(0, color="#D1D5DB", linewidth=0.8)

    for i, feature in enumerate(features):
        x_end = float(loadings_scaled[i, 0])
        y_end = float(loadings_scaled[i, 1])
        ax.arrow(
            0.0,
            0.0,
            x_end,
            y_end,
            color=cores_vetores[feature],
            width=0.0,
            head_width=max_score * 0.035,
            length_includes_head=True,
            linewidth=1.1,
        )

    ax.set_xlabel(f"PC1 ({pct1}% da variância)", fontsize=11, family="serif")
    ax.set_ylabel(f"PC2 ({pct2}% da variância)", fontsize=11, family="serif")
    ax.set_title("Biplot PCA das Features", fontsize=15, fontweight="bold", color="#0F172A", family="serif")
    ax.grid(True, alpha=0.22, linestyle=":", color="#D1D5DB", linewidth=0.7)
    ax.set_aspect("equal", adjustable="datalim")
    handles_modo = [
        Line2D(
            [0],
            [0],
            marker=marcadores[m],
            linestyle="",
            markerfacecolor=cores[m],
            markeredgecolor="#111111",
            markeredgewidth=0.8,
            color=cores[m],
            label=m,
            markersize=8,
        )
        for m in modos
    ]
    handles_vetores = [
        Line2D([0], [0], color=cores_vetores[f], lw=2.2, label=nomes_features[f])
        for f in features
    ]

    fig.subplots_adjust(top=0.90, bottom=0.12, left=0.08, right=0.75)
    leg_modos = ax.legend(
        handles=handles_modo,
        title="Modos",
        loc="upper left",
        bbox_to_anchor=(1.01, 1.0),
        frameon=False,
        prop={"family": "serif", "size": 9},
        fontsize=10,
        title_fontsize=11,
    )
    ax.add_artist(leg_modos)
    ax.legend(
        handles=handles_vetores,
        title="Vetores das Features",
        loc="lower left",
        bbox_to_anchor=(1.01, 0.0),
        frameon=False,
        prop={"family": "serif", "size": 9},
        fontsize=9,
        title_fontsize=10,
    )

    fig.savefig(output_png_radar, dpi=300, bbox_inches="tight", pad_inches=0.25, facecolor="#FFFFFF", edgecolor="none")
    plt.close(fig)

    LOGGER.info("Gráficos científicos salvos em output/")
    return {
        "boxplot": output_png_box,
        "pares": output_png_pairs,
        "radar": output_png_radar,
    }


def criar_raw_vazio(n_canais: int = 4, duracao: float = 2.0, sfreq: float = 250.0) -> mne.io.RawArray:
    """Cria um RawArray vazio para inicialização.
    
    Args:
        n_canais: Número de canais
        duracao: Duração em segundos
        sfreq: Frequência de amostragem
        
    Returns:
        Objeto mne.io.RawArray vazio (será preenchido com dados do LSL)
    """
    n_amostras = int(duracao * sfreq)
    data = np.zeros((n_canais, n_amostras))
    ch_names = [f"EMG_{i+1}" for i in range(n_canais)]
    info = mne.create_info(ch_names, int(sfreq), ch_types="emg")
    raw = mne.io.RawArray(data, info)
    LOGGER.info(f"RawArray vazio criado: {n_canais} canais, {duracao}s")
    return raw

if __name__ == "__main__":
    configurar_logging()
    configurar_r_environment()

    app = QtWidgets.QApplication(sys.argv)
    app.setStyle("Fusion")

    try:
        sfreq = 250.0
        n_canais = 4
        raw = criar_raw_vazio(n_canais=n_canais, duracao=5.0, sfreq=sfreq)
        raw_filt = raw.copy()

        win = JanelaNeuro(raw, raw_filt)
        win.show()
        sys.exit(app.exec())
        
    except Exception as e:
        LOGGER.exception("Erro: %s", e)