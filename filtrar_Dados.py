from pylsl import StreamInlet, StreamOutlet, StreamInfo, resolve_streams, cf_float32
import time
import numpy as np
from collections import deque
from typing import Optional, List, Tuple


class EMGAggregator:
    def __init__(
        self,
        input_stream_name: str = "EMG",
        output_stream_name: str = "EMG_Processado",
        processing_mode: str = "raw",  # "raw", "rms", "normalized"
        rms_window_ms: float = 50.0,   # Janela RMS em milissegundos
        connection_timeout: float = 30.0,
    ):
        """
        Agregador de dados EMG com processamento RMS.
        
        Args:
            input_stream_name: Nome do stream LSL de entrada
            output_stream_name: Nome do stream LSL de saída
            processing_mode: Tipo de processamento
                - "raw": Repassa dados sem modificação
                - "rms": Calcula RMS em janela deslizante
                - "normalized": RMS + normalização para [0, 1]
            rms_window_ms: Tamanho da janela RMS em milissegundos (padrão: 50ms)
            connection_timeout: Tempo máximo para encontrar stream
        """
        self.input_stream_name = input_stream_name
        self.output_stream_name = output_stream_name
        self.connection_timeout = connection_timeout
        self.processing_mode = processing_mode
        self.rms_window_ms = rms_window_ms
        
        # Streams LSL
        self.inlet: Optional[StreamInlet] = None
        self.outlet: Optional[StreamOutlet] = None
        
        # Metadados
        self.channel_count: int = 0
        self.sample_rate: float = 0.0
        self.rms_window_samples: int = 0
        
        # Buffer circular para RMS (uma deque por canal)
        self.rms_buffers: List[deque] = []
        
        # Buffer para estatísticas de normalização (últimos 5 segundos)
        self.normalization_buffer: Optional[deque] = None
        
        # Contadores
        self.total_samples = 0
        self.error_count = 0
        
    def connect_input(self) -> None:
        """Conecta ao stream de entrada."""
        print(f"🔍 Procurando stream '{self.input_stream_name}'...")
        print(f"   Timeout: {self.connection_timeout}s")
        
        start_time = time.time()
        
        while time.time() - start_time < self.connection_timeout:
            streams = self._resolve_target_streams()
            
            if streams:
                self.inlet = StreamInlet(streams[0], max_buflen=360)
                self._initialize_input_metadata()
                self._initialize_processing_buffers()
                
                print(f"Conectado ao stream de entrada")
                print(f"Canais: {self.channel_count}")
                print(f"Taxa: {self.sample_rate} Hz")
                print(f"Janela RMS: {self.rms_window_ms}ms ({self.rms_window_samples} amostras)")
                return
            
            print(f"   Aguardando... ({int(time.time() - start_time)}s)", end='\r')
            time.sleep(0.5)
        
        raise RuntimeError(
            f"Stream '{self.input_stream_name}' não encontrado!\n"
            f"Execute o EMGEngine primeiro."
        )

    def _resolve_target_streams(self):
        """Busca streams disponíveis e filtra pelo nome configurado."""
        streams = resolve_streams(2.0)
        return [s for s in streams if s.name() == self.input_stream_name]

    def _initialize_input_metadata(self) -> None:
        """Lê metadados do stream de entrada já conectado."""
        if self.inlet is None:
            raise RuntimeError("Stream de entrada não inicializado")

        self.channel_count = self.inlet.channel_count
        self.sample_rate = self.inlet.info().nominal_srate()
        self.rms_window_samples = int((self.rms_window_ms / 1000.0) * self.sample_rate)

    def _initialize_processing_buffers(self) -> None:
        """Prepara buffers de RMS e normalização conforme o modo."""
        self.rms_buffers = [
            deque(maxlen=self.rms_window_samples)
            for _ in range(self.channel_count)
        ]

        self.normalization_buffer = None
        if self.processing_mode == "normalized":
            norm_buffer_size = int(self.sample_rate * 5.0)
            self.normalization_buffer = deque(maxlen=norm_buffer_size)
        
    def create_output(self) -> None:
        """Cria stream de saída."""
        if self.inlet is None:
            raise RuntimeError("Stream de entrada não inicializado")
        
        info = StreamInfo(
            name=self.output_stream_name,
            type="EMG",
            channel_count=self.channel_count,
            nominal_srate=self.sample_rate,
            channel_format=cf_float32,
            source_id=f"aggregator-rms-{int(time.time())}",
        )
        
        # Metadados sobre processamento
        desc = info.desc()
        desc.append_child_value("processing_mode", self.processing_mode)
        desc.append_child_value("rms_window_ms", str(self.rms_window_ms))
        desc.append_child_value("source_stream", self.input_stream_name)
        
        self.outlet = StreamOutlet(info, chunk_size=32)
        
        print(f"Stream de saída '{self.output_stream_name}' criado")
        print(f"Modo de processamento: {self.processing_mode.upper()}\n")
        
    def calculate_rms(self, channel_idx: int) -> float:
        """
        Calcula RMS (Root Mean Square) para um canal específico.
        
        RMS = sqrt(mean(x^2))
        
        Em EMG, RMS representa a "energia" ou "amplitude" do sinal muscular.
        Valores mais altos = contração muscular mais forte.
        
        Args:
            channel_idx: Índice do canal (0, 1, ...)
            
        Returns:
            Valor RMS
        """
        buffer = self.rms_buffers[channel_idx]
        
        if len(buffer) == 0:
            return 0.0
        
        # Converte para array NumPy
        data = np.array(buffer)
        
        # RMS = raiz quadrada da média dos quadrados
        rms = np.sqrt(np.mean(data ** 2))
        
        return float(rms)
    
    def process_sample(self, sample: List[float]) -> List[float]:
        """
        Processa amostra conforme modo selecionado.
        
        Args:
            sample: Lista de valores brutos (um por canal)
            
        Returns:
            Lista processada
        """
        # 1. Adiciona valores aos buffers RMS
        for i, value in enumerate(sample):
            self.rms_buffers[i].append(value)
        
        # 2. Processa conforme modo
        if self.processing_mode == "raw":
            # Modo RAW: retorna dados originais sem modificação
            return sample

        if self.processing_mode == "rms":
            # Modo RMS: calcula RMS para cada canal
            return self._compute_rms_values()

        if self.processing_mode == "normalized":
            # Modo NORMALIZADO: RMS + normalização para [0, 1]
            return self._normalize_rms_values(self._compute_rms_values())

        raise ValueError(f"Modo de processamento inválido: {self.processing_mode}")

    def _compute_rms_values(self) -> List[float]:
        """Calcula o RMS atual para todos os canais."""
        return [self.calculate_rms(i) for i in range(self.channel_count)]

    def _normalize_rms_values(self, rms_values: List[float]) -> List[float]:
        """Normaliza valores RMS para [0, 1] usando janela histórica."""
        if self.normalization_buffer is not None:
            self.normalization_buffer.append(rms_values.copy())

        # Enquanto não há dados suficientes, retorna RMS puro
        if not self.normalization_buffer or len(self.normalization_buffer) < 100:
            return rms_values

        buffer_array = np.array(list(self.normalization_buffer))

        # Calcula min/max por canal
        min_vals = np.min(buffer_array, axis=0)
        max_vals = np.max(buffer_array, axis=0)

        # Normaliza para [0, 1]
        rms_array = np.array(rms_values)
        range_vals = max_vals - min_vals
        range_vals[range_vals == 0] = 1.0  # Evita divisão por zero

        normalized = (rms_array - min_vals) / range_vals
        return normalized.tolist()
    
    def get_statistics(self) -> dict:
        """Retorna estatísticas em tempo real."""
        stats = {
            'total_samples': self.total_samples,
            'error_count': self.error_count,
            'processing_mode': self.processing_mode,
        }

        self._append_mode_statistics(stats)
        
        return stats

    def _append_mode_statistics(self, stats: dict) -> None:
        """Acrescenta estatísticas específicas do modo de processamento."""
        if self.processing_mode == "rms":
            stats['current_rms'] = self._compute_rms_values()
            return

        if self.processing_mode != "normalized" or not self.normalization_buffer:
            return

        if len(self.normalization_buffer) == 0:
            return

        buffer_array = np.array(list(self.normalization_buffer))
        stats['rms_mean'] = np.mean(buffer_array, axis=0).tolist()
        stats['rms_std'] = np.std(buffer_array, axis=0).tolist()
    
    def run(self) -> None:
        """Loop principal."""
        self.connect_input()
        self.create_output()
        
        if self.inlet is None or self.outlet is None:
            raise RuntimeError("Erro na inicialização")
        
        print("Agregador em execução (Ctrl+C para parar)...")
        print("Aguardando dados...\n")
        
        start_time = time.time()
        last_stats_time = start_time
        reconnect_attempts = 0
        
        try:
            while True:
                sample, timestamp = self.inlet.pull_sample(timeout=1.0)
                
                if sample is None or timestamp is None:
                    should_stop, reconnect_attempts = self._handle_missing_data(reconnect_attempts)
                    if should_stop:
                        break
                    continue
                
                reconnect_attempts = 0
                self._process_and_publish(sample, timestamp)
                
                # Estatísticas a cada 5 segundos
                current_time = time.time()
                if current_time - last_stats_time >= 5.0:
                    self._print_statistics(start_time, current_time)
                    last_stats_time = current_time
                    
        except KeyboardInterrupt:
            elapsed = time.time() - start_time
            print(f"\n✅ Encerrando...")
            print(f"   Total: {self.total_samples} amostras em {elapsed:.1f}s")
            print(f"   Taxa média: {self.total_samples / elapsed:.1f} Hz")

    def _handle_missing_data(self, reconnect_attempts: int) -> Tuple[bool, int]:
        """Controla política de reconexão quando não há amostra disponível."""
        if reconnect_attempts < 3:
            print(f"⚠️ Sem dados... (tentativa {reconnect_attempts + 1}/3)")
            return False, reconnect_attempts + 1

        print("Conexão perdida. Encerrando...")
        return True, reconnect_attempts

    def _process_and_publish(self, sample: List[float], timestamp: float) -> None:
        """Processa a amostra e publica no stream de saída."""
        if self.outlet is None:
            raise RuntimeError("Stream de saída não inicializado")

        try:
            processed_sample = self.process_sample(sample)
            self.outlet.push_sample(processed_sample, timestamp=timestamp)
            self.total_samples += 1
        except Exception as e:
            self.error_count += 1
            if self.error_count % 100 == 1:
                print(f"Erro ao processar: {e}")

    def _print_statistics(self, start_time: float, current_time: float) -> None:
        """Exibe estatísticas periódicas de execução."""
        elapsed = current_time - start_time
        rate = self.total_samples / elapsed if elapsed > 0 else 0.0
        stats = self.get_statistics()

        print(f"\nEstatísticas")
        print(f"Amostras: {self.total_samples}")
        print(f"Taxa: {rate:.1f} Hz")
        print(f"Modo: {self.processing_mode.upper()}")

        if 'current_rms' in stats:
            print(f"RMS atual: {[f'{v:.2f}' for v in stats['current_rms']]}")

        if 'rms_mean' in stats:
            print(f"RMS médio: {[f'{v:.2f}' for v in stats['rms_mean']]}")

        if self.error_count > 0:
            print(f"Erros: {self.error_count}")

        print()


def main():
    """Ponto de entrada principal."""
    
    # ========================================
    # CONFIGURAÇÃO: Escolha o modo aqui!
    # ========================================
    
    aggregator = EMGAggregator(
        input_stream_name="EMG",
        output_stream_name="EMG_Processado",
        
        # ESCOLHA UM MODO:
        processing_mode="rms",       # "raw", "rms", ou "normalized"
        
        # Tamanho da janela RMS (quanto maior, mais suave o sinal)
        rms_window_ms=50.0,          # 50ms é padrão para EMG
        
        connection_timeout=30.0,
    )
    
    try:
        aggregator.run()
    except Exception as e:
        print(f"\nErro crítico: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())