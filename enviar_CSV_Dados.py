import pylsl
import csv
from datetime import datetime


def main():
    print("Procurando stream 'EMG_Processado' na rede...")
    
    # Busca o stream transmitido pelo EMGEngine
    streams = pylsl.resolve_streams(1.0)

    stream_info = None
    for s in streams:
        if s.name() == "EMG_Processado":
            stream_info = s
            break

    if stream_info is None:
        raise RuntimeError("Stream 'EMG_Processado' not found")

    print(f"Stream encontrado: {stream_info}")

    # Conecta no stream
    inlet = pylsl.StreamInlet(stream_info)
    print(f"Conectado ao stream EMG_Processado")
    print(f"Canais: {inlet.channel_count}")
    print(f"Taxa de Amostragem: {inlet.info().nominal_srate()} Hz")
    print("\nRecebendo dados (Ctrl+C para parar)...\n")
    
    # Recebe e processa dados continuamente
    sample_count = 0
    flush_every = 100
    filename = f"emg_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"

    try:
        with open(filename, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['timestamp', 'canal1'])  # ou canal1, canal2

            print(f"Salvando em {filename}...")

            while True:
                # Puxa uma amostra (bloqueia até receber)
                sample, timestamp = inlet.pull_sample()
                if sample is None:
                    continue

                # Processa os dados aqui
                sample_count += 1

                # Exemplo: mostra a cada 500 amostras (~0.25s a 2000Hz)
                
                """
                if sample_count % 500 == 0:
                    print(f"Sample #{sample_count}: {sample} @ {timestamp:.3f}s")
                    
                """
                writer.writerow([timestamp, *sample])

                if sample_count % flush_every == 0:
                    f.flush()
    except KeyboardInterrupt:
        print(f"\nRecebidas {sample_count} amostras. Dados salvos em {filename}")

if __name__ == "__main__":
    main()


