import asyncio
import signal
import struct
import sys
from typing import Callable, Optional
from bleak import BleakScanner, BleakClient
from bleak.backends.characteristic import BleakGATTCharacteristic
from pylsl import StreamInfo, StreamOutlet, cf_float32


class EMGEngine:
    def __init__(
        self,
        device_name: str = "EMG-ESP32", # Nome do dispositivo BLE a ser conectado
        char_uuid: str = "12345678-1234-1234-1234-1234567890ac", # UUID da característica que transmite os dados EMG
        queue_maxsize: int = 2048,
        scan_timeout: float = 10.0,
    ):
        self.device_name = device_name
        self.char_uuid = char_uuid
        self.scan_timeout = scan_timeout
        self.is_running = False
        self._stop_event: Optional[asyncio.Event] = None

        self._queue: asyncio.Queue[float] = asyncio.Queue(maxsize=queue_maxsize)
        self._subscribers: list[Callable] = []

    def subscribe(self, callback: Callable) -> None:
        """Adiciona uma função (sync ou async) para receber os dados."""
        self._subscribers.append(callback)

    def unsubscribe(self, callback: Callable) -> None:
        """Remove uma função previamente registrada."""
        try:
            self._subscribers.remove(callback)
        except ValueError:
            pass

    def stop(self) -> None:
        """Solicita parada do engine de forma segura."""
        if self._stop_event is not None:
            self._stop_event.set()

    def _notification_handler(
        self, _: BleakGATTCharacteristic, data: bytearray
    ) -> None:
        try:
            value = struct.unpack('<f', data)[0]
        except struct.error as e:
            print(f"Erro de decodificação: {e} | Raw: {data.hex()}")
            return

        try:
            self._queue.put_nowait(value)
        except asyncio.QueueFull:
            pass

    async def _dispatch_worker(
        self,
        stop_event: asyncio.Event,
        max_samples: Optional[int] = None,
    ) -> None:
        """
        Worker separado: drena a fila e despacha para os subscribers.
        Roda em paralelo ao recebimento BLE, sem bloquear as notificações.
        """
        processed_count = 0

        while not stop_event.is_set() or not self._queue.empty():
            try:
                value = await asyncio.wait_for(self._queue.get(), timeout=1.0)
            except asyncio.TimeoutError:
                continue

            for callback in list(self._subscribers):
                try:
                    if asyncio.iscoroutinefunction(callback):
                        await callback(value)
                    else:
                        callback(value)
                except Exception as e:
                    print(f"Erro no subscriber '{callback.__name__}': {e}")

            processed_count += 1
            if max_samples is not None and processed_count >= max_samples:
                print(f"Limite de amostras atingido ({max_samples}). Encerrando...")
                stop_event.set()

            self._queue.task_done()

    async def run(
        self,
        stop_event: Optional[asyncio.Event] = None,
        max_run_seconds: Optional[float] = None,
        max_samples: Optional[int] = None,
    ) -> None:
        if stop_event is None:
            stop_event = asyncio.Event()

        self._stop_event = stop_event
        self.is_running = True
        timer_task: Optional[asyncio.Task] = None

        print(f"Buscando '{self.device_name}'...")
        device = await BleakScanner.find_device_by_filter(
            lambda d, ad: d.name == self.device_name,
            timeout=self.scan_timeout,
        )

        if not device:
            print(f"Dispositivo '{self.device_name}' não encontrado.")
            self.is_running = False
            return

        def on_disconnect(client: BleakClient) -> None:
            print(f"\nDispositivo {client.address} desconectado.")
            stop_event.set()

        async with BleakClient(device, disconnected_callback=on_disconnect) as client:
            print(f"Conectado em {device.address}. Transmitindo dados...")

            worker_task = asyncio.create_task(
                self._dispatch_worker(stop_event, max_samples=max_samples)
            )

            if max_run_seconds is not None:
                async def _stop_after_timeout() -> None:
                    await asyncio.sleep(max_run_seconds)
                    if not stop_event.is_set():
                        print(f"Tempo limite ({max_run_seconds}s) atingido. Encerrando...")
                        stop_event.set()

                timer_task = asyncio.create_task(_stop_after_timeout())

            await client.start_notify(self.char_uuid, self._notification_handler)

            try:
                await stop_event.wait()
            finally:
                print("Encerrando Transmissão...")
                await client.stop_notify(self.char_uuid)

                # Drena o que restou na fila antes de encerrar
                try:
                    await asyncio.wait_for(worker_task, timeout=5.0)
                except asyncio.TimeoutError:
                    worker_task.cancel()
                    await asyncio.gather(worker_task, return_exceptions=True)

                if timer_task is not None:
                    timer_task.cancel()
                    await asyncio.gather(timer_task, return_exceptions=True)

                self.is_running = False
                self._stop_event = None


# --- Exemplo de uso ---
async def _example():
    loop = asyncio.get_running_loop()
    stop_event = asyncio.Event()

    if sys.platform != "win32":
        for sig in (signal.SIGINT, signal.SIGTERM):
            loop.add_signal_handler(sig, stop_event.set)

    engine = EMGEngine()

    """     
    lsl_info = StreamInfo(
        name="EMG",
        type="EMG",
        channel_count=1,
        nominal_srate=0.0,
        channel_format=cf_float32,
        source_id="emg-esp32-ble",
    )
    lsl_outlet = StreamOutlet(lsl_info, chunk_size=1)
    print("LSL pronto: stream 'EMG' (1 canal).") 

    """

    # Impressão (Marco 1)
    def my_processor(value: float) -> None:
        print(f"EMG: {value:.4f}")

    # Envio para LSL (Marco 1-2)
    """
    def lsl_publisher(value: float) -> None:
        lsl_outlet.push_sample([value])
    """
    engine.subscribe(my_processor)

    """ 
    engine.subscribe(lsl_publisher)

    """

    # Exemplos de parada:
    # - Ctrl+C (SIGINT) em sistemas suportados
    # - engine.stop() em qualquer parte do código
    # - max_run_seconds para parar por tempo
    # - max_samples para parar por quantidade de amostras

    await engine.run(stop_event, max_run_seconds=1000)


if __name__ == "__main__":
    try:
        asyncio.run(_example())
    except KeyboardInterrupt:
        pass