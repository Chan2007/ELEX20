"""
Script de teste e demonstração do sistema de análise LSL.
Simula o fluxo completo de captura de dados → processamento → análise.
"""

import sys
import time
import logging
import subprocess
import threading
from pathlib import Path

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | [%(name)s] | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("teste_lsl")


def iniciar_simulador_lsl(duracao=30, sfreq=250, canais=4, intervalo=0.1):
    """Inicia o simulador LSL em thread separada."""
    logger.info(f"Iniciando simulador LSL por {duracao}s...")
    
    cmd = [
        sys.executable,
        "simulador_lsl_emg.py",
        "--duracao", str(duracao),
        "--sfreq", str(sfreq),
        "--canais", str(canais),
        "--intervalo", str(intervalo),
    ]
    
    try:
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        return proc
    except Exception as e:
        logger.error(f"Erro ao iniciar simulador: {e}")
        return None


def testar_captura_lsl():
    """Testa captura de dados de streams LSL."""
    try:
        import pylsl
    except ImportError:
        logger.error("pylsl não instalado. Execute: pip install pylsl")
        return False

    logger.info("Aguardando streams LSL...")
    
    try:
        streams = pylsl.resolve_streams(wait_time=5.0)
        if not streams:
            logger.error("Nenhum stream encontrado após 5 segundos")
            return False

        logger.info(f"Encontrados {len(streams)} streams:")
        for stream in streams:
            logger.info(f"  - Nome: {stream.name()} | Tipo: {stream.type()} | Canais: {stream.channel_count()}")

        # Tenta conectar aos streams
        emg_stream = None
        emg_proc_stream = None

        for stream_info in streams:
            if stream_info.name() == "EMG":
                emg_stream = pylsl.StreamInlet(stream_info)
                logger.info("Conectado ao stream EMG")
            elif stream_info.name() == "EMG_Processado":
                emg_proc_stream = pylsl.StreamInlet(stream_info)
                logger.info("Conectado ao stream EMG_Processado")

        if emg_stream is None:
            logger.error("Stream EMG não encontrado")
            return False

        # Captura dados por 3 segundos
        logger.info("Capturando dados por 3 segundos...")
        emg_data = []
        start_time = time.time()

        while (time.time() - start_time) < 3:
            sample, _ = emg_stream.pull_sample(timeout=0.1)
            if sample:
                emg_data.append(sample)

        logger.info(f"✓ Capturadas {len(emg_data)} amostras")
        return True

    except Exception as e:
        logger.exception(f"Erro ao testar captura: {e}")
        return False


def main():
    """Função principal de teste."""
    logger.info("="*70)
    logger.info("TESTE DE INTEGRAÇÃO - ANÁLISE LSL COM EMG")
    logger.info("="*70)

    # Etapa 1: Iniciar simulador
    logger.info("\n[ETAPA 1] Iniciando simulador LSL...")
    simulador_proc = iniciar_simulador_lsl(duracao=30, intervalo=0.1)
    
    if simulador_proc is None:
        logger.error("Falha ao iniciar simulador")
        return False

    # Aguardar simulador começar
    time.sleep(2)

    # Etapa 2: Testar captura
    logger.info("\n[ETAPA 2] Testando captura de streams LSL...")
    if not testar_captura_lsl():
        logger.error("Falha no teste de captura")
        simulador_proc.terminate()
        return False

    logger.info("\n" + "="*70)
    logger.info("✓ TESTE CONCLUÍDO COM SUCESSO")
    logger.info("="*70)
    logger.info("\nPróximos passos:")
    logger.info("1. Execute 'python simulador_lsl_emg.py' em um terminal")
    logger.info("2. Execute 'python Grafico.py' em outro terminal")
    logger.info("3. Clique no botão '⬇️ RECARREGAR DE LSL' para capturar dados")
    logger.info("\n" + "="*70)

    # Aguardar simulador terminar
    simulador_proc.wait()
    logger.info("Simulador finalizado")
    return True


if __name__ == "__main__":
    try:
        success = main()
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        logger.info("Teste interrompido pelo usuário")
        sys.exit(1)
    except Exception as e:
        logger.exception(f"Erro geral: {e}")
        sys.exit(1)
