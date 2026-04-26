"""
Simulador de streams LSL para EMG e EMG Processado.
Envia dados contínuos de dois streams: "EMG" (sinal bruto) e "EMG_Processado" (filtrado).
"""

import sys
import time
import numpy as np
import logging
from pathlib import Path

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("simulador_lsl")

try:
    from pylsl import StreamInfo, StreamOutlet, cf_float32
except ImportError:
    logger.error("pylsl não instalado. Execute: pip install pylsl")
    sys.exit(1)


class SimuladorEMG:
    """Simula streams LSL de EMG bruto e processado."""

    def __init__(self, sfreq: int = 250, n_canais: int = 4) -> None:
        """
        Inicializa simulador de EMG com streams LSL.
        
        Args:
            sfreq: Frequência de amostragem em Hz (padrão 250)
            n_canais: Número de canais EMG (padrão 4)
        """
        self.sfreq = sfreq
        self.n_canais = n_canais
        self.amostra = 0
        self.tempo_inicio = time.time()

        # Criar streams LSL
        self._criar_streams()

    def _criar_streams(self) -> None:
        """Cria os dois streams LSL: EMG e EMG_Processado."""
        
        # Stream EMG (sinal bruto)
        info_emg = StreamInfo(
            name="EMG",
            type="EMG",
            channel_count=self.n_canais,
            nominal_srate=float(self.sfreq),
            channel_format=cf_float32,
            source_id="emg_simulator_bruto",
        )
        info_emg.desc().append_child_value("manufacturer", "Simulador Local")
        self.outlet_emg = StreamOutlet(info_emg)
        logger.info(f"Stream LSL 'EMG' criado ({self.n_canais} canais, {self.sfreq} Hz)")

        # Stream EMG_Processado (sinal filtrado)
        info_proc = StreamInfo(
            name="EMG_Processado",
            type="EMG",
            channel_count=self.n_canais,
            nominal_srate=float(self.sfreq),
            channel_format=cf_float32,
            source_id="emg_simulator_processado",
        )
        info_proc.desc().append_child_value("manufacturer", "Simulador Local")
        self.outlet_proc = StreamOutlet(info_proc)
        logger.info(f"Stream LSL 'EMG_Processado' criado ({self.n_canais} canais, {self.sfreq} Hz)")

    def gerar_emg_sintetico(self, tempo_s: float) -> np.ndarray:
        """
        Gera sinal EMG sintético com múltiplas frequências.
        
        Args:
            tempo_s: Tempo em segundos
            
        Returns:
            Array numpy de shape (n_canais, n_amostras) com tipo float64
        """
        t = np.arange(0, tempo_s, 1/self.sfreq)
        emg = np.zeros((self.n_canais, len(t)))

        # Cada canal com características diferentes
        for ch in range(self.n_canais):
            # Componentes de frequência (EMG típico 20-200 Hz)
            freq1 = 50 + ch * 10  # 50, 60, 70, 80 Hz
            freq2 = 150 + ch * 15  # 150, 165, 180, 195 Hz
            
            # Amplitude aleatória simulando contração muscular
            amplitude = np.random.uniform(0.8, 1.2)
            
            # Sinal composto
            sinal = (
                amplitude * 0.3 * np.sin(2 * np.pi * freq1 * t)
                + amplitude * 0.2 * np.sin(2 * np.pi * freq2 * t)
                + amplitude * 0.1 * np.random.normal(0, 1, len(t))  # Ruído
            )
            emg[ch, :] = sinal

        return emg

    def filtrar_emg(self, emg: np.ndarray, order: int = 4, lowcut: float = 20, highcut: float = 150) -> np.ndarray:
        """
        Aplica filtro passa-banda ao sinal EMG.
        
        Args:
            emg: Array numpy com EMG bruto (n_canais x n_amostras)
            order: Ordem do filtro Butterworth
            lowcut: Frequência de corte inferior (Hz)
            highcut: Frequência de corte superior (Hz)
            
        Returns:
            Array numpy filtrado com mesmo shape de entrada
        """
        from scipy import signal
        
        nyquist = self.sfreq / 2
        low = lowcut / nyquist
        high = highcut / nyquist
        
        # Garantir que os valores estejam entre 0 e 1
        low = np.clip(low, 0.001, 0.999)
        high = np.clip(high, 0.001, 0.999)
        
        if low >= high:
            low, high = 0.01, 0.99
        
        # Retorna tupla (b, a) do filtro
        try:
            result = signal.butter(order, [low, high], btype='band', output='ba')
            if result is not None and isinstance(result, (list, tuple)) and len(result) == 2:
                b, a = result[0], result[1]
            else:
                raise ValueError("Butter filter returned invalid type")
        except (ValueError, TypeError) as e:
            logger.warning(f"Erro ao criar filtro butter: {e}. Usando sem filtro.")
            b, a = np.array([1.0]), np.array([1.0])
        
        emg_filt = np.zeros_like(emg, dtype=float)
        for ch in range(emg.shape[0]):
            try:
                emg_filt[ch, :] = signal.filtfilt(b, a, emg[ch, :], padtype='even')
            except Exception as e:
                logger.warning(f"Erro ao filtrar canal {ch}: {e}. Copiando sinal original.")
                emg_filt[ch, :] = emg[ch, :]
        
        return emg_filt

    def enviar_stream(self, duracao_s: float = 10, intervalo_buffer: float = 0.1) -> None:
        """
        Envia dados continuamente para os streams LSL por tempo determinado.
        
        Args:
            duracao_s: Duração total da transmissão em segundos
            intervalo_buffer: Intervalo entre buffers em segundos
        """
        logger.info(f"Iniciando transmissão por {duracao_s}s (intervalo: {intervalo_buffer}s)")
        
        n_amostras_buffer = int(self.sfreq * intervalo_buffer)
        tempo_total = 0

        try:
            while tempo_total < duracao_s:
                # Gera dados para este buffer
                emg_buffer = self.gerar_emg_sintetico(intervalo_buffer)
                emg_proc_buffer = self.filtrar_emg(emg_buffer)

                # Envia cada amostra
                for i in range(emg_buffer.shape[1]):
                    self.outlet_emg.push_sample(emg_buffer[:, i].tolist())
                    self.outlet_proc.push_sample(emg_proc_buffer[:, i].tolist())

                # Log de progresso
                tempo_total += intervalo_buffer
                logger.info(f"Enviado: {tempo_total:.1f}s / {duracao_s}s")
                
                # Aguarda antes do próximo buffer
                time.sleep(intervalo_buffer)

            logger.info("Transmissão concluída.")

        except KeyboardInterrupt:
            logger.info("Transmissão interrompida pelo usuário.")
        except Exception as e:
            logger.exception(f"Erro durante transmissão: {e}")

    def enviar_continuo(self, intervalo_s: float = 0.1) -> None:
        """
        Envia dados continuamente até interrupção (Ctrl+C).
        
        Args:
            intervalo_s: Intervalo entre buffers em segundos
        """
        logger.info(f"Iniciando transmissão contínua (intervalo: {intervalo_s}s)")
        
        try:
            while True:
                emg_buffer = self.gerar_emg_sintetico(intervalo_s)
                emg_proc_buffer = self.filtrar_emg(emg_buffer)

                for i in range(emg_buffer.shape[1]):
                    self.outlet_emg.push_sample(emg_buffer[:, i].tolist())
                    self.outlet_proc.push_sample(emg_proc_buffer[:, i].tolist())

                logger.info(f"Buffer enviado ({self.amostra} amostras acumuladas)")
                self.amostra += emg_buffer.shape[1]
                
                time.sleep(intervalo_s)

        except KeyboardInterrupt:
            logger.info("Transmissão interrompida pelo usuário.")
        except Exception as e:
            logger.exception(f"Erro durante transmissão: {e}")


def main():
    """Ponto de entrada principal."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Simulador de streams LSL para EMG")
    parser.add_argument("--duracao", type=float, default=None, help="Duração em segundos (None para contínuo)")
    parser.add_argument("--sfreq", type=int, default=250, help="Frequência de amostragem (Hz)")
    parser.add_argument("--canais", type=int, default=4, help="Número de canais EMG")
    parser.add_argument("--intervalo", type=float, default=0.1, help="Intervalo entre buffers (s)")
    
    args = parser.parse_args()
    
    logger.info("="*60)
    logger.info("SIMULADOR DE STREAMS LSL - EMG")
    logger.info("="*60)
    logger.info(f"Configurações:")
    logger.info(f"  - Frequência: {args.sfreq} Hz")
    logger.info(f"  - Canais: {args.canais}")
    logger.info(f"  - Intervalo: {args.intervalo}s")
    logger.info(f"  - Duração: {'Contínua' if args.duracao is None else f'{args.duracao}s'}")
    logger.info("="*60)
    logger.info("Use Ctrl+C para parar")
    logger.info("="*60)
    
    simulador = SimuladorEMG(sfreq=args.sfreq, n_canais=args.canais)
    
    if args.duracao:
        simulador.enviar_stream(duracao_s=args.duracao, intervalo_buffer=args.intervalo)
    else:
        simulador.enviar_continuo(intervalo_s=args.intervalo)


if __name__ == "__main__":
    main()
