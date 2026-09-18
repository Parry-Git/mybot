from __future__ import annotations

import queue
import struct
import threading
import time

from core.types import AudioPacket


class BrowserAudioPlayer:
    """Bound PCM in flight until the browser acknowledges rendered samples."""

    sample_rate = 24000

    def __init__(self, queue_seconds: float = 2.0) -> None:
        self.outgoing: queue.Queue = queue.Queue(maxsize=256)
        self._condition = threading.Condition()
        self._capacity = int(queue_seconds * 48000)
        self._pending = 0
        self._generation = 0
        self._connected = False
        self._closed = False
        self._callback = None
        self._last_progress = time.monotonic()

    def start(self) -> None:
        pass

    def attach(self) -> bool:
        with self._condition:
            if self._connected or self._closed:
                return False
            self._connected = True
            self.clear()
            return True

    def detach(self) -> None:
        with self._condition:
            self._connected = False
            self.clear()

    def emit(self, kind: str, value) -> None:
        with self._condition:
            if self._connected:
                self.outgoing.put_nowait({"type": kind, "value": value})

    def enqueue(self, packet: AudioPacket, cancel: threading.Event, on_first_playback=None) -> bool:
        if packet.sample_rate != self.sample_rate or len(packet.pcm) % 2:
            raise ValueError("browser output requires 24 kHz PCM16")
        for offset in range(0, len(packet.pcm), 960):
            pcm = packet.pcm[offset : offset + 960]
            with self._condition:
                while self._pending + len(pcm) > self._capacity:
                    if cancel.is_set() or not self._connected or self._closed:
                        return False
                    self._check_progress()
                    self._condition.wait(0.02)
                if cancel.is_set() or not self._connected or self._closed:
                    return False
                if self._pending == 0:
                    self._last_progress = time.monotonic()
                if on_first_playback is not None:
                    self._callback = on_first_playback
                    on_first_playback = None
                self._pending += len(pcm)
                self.outgoing.put_nowait(struct.pack("<I", self._generation) + pcm)
        return True

    def acknowledge(self, generation: int, samples: int) -> None:
        callback = None
        with self._condition:
            if generation != self._generation or samples <= 0:
                return
            if samples * 2 > self._pending:
                raise ValueError("browser acknowledged more audio than was sent")
            self._pending -= samples * 2
            self._last_progress = time.monotonic()
            callback, self._callback = self._callback, None
            self._condition.notify_all()
        if callback is not None:
            callback()

    def _check_progress(self) -> None:
        if self._pending and time.monotonic() - self._last_progress > 10:
            raise TimeoutError("browser stopped rendering audio; reconnect the microphone")

    def is_playing(self) -> bool:
        with self._condition:
            self._check_progress()
            return self._pending > 0

    def clear(self) -> None:
        with self._condition:
            self._generation += 1
            self._pending = 0
            self._callback = None
            while True:
                try:
                    self.outgoing.get_nowait()
                except queue.Empty:
                    break
            if self._connected:
                self.outgoing.put_nowait({"type": "clear", "value": self._generation})
            self._condition.notify_all()

    def close(self) -> None:
        with self._condition:
            self._closed = True
            self.detach()
