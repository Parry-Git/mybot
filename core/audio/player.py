from __future__ import annotations

import logging
import threading
import time
from collections import deque
from typing import Callable, Optional

import sounddevice as sd

from core.config import AudioSettings
from core.types import AudioPacket

logger = logging.getLogger(__name__)


class AudioPlayer:
    """Persistent callback-driven PCM player with bounded buffering."""

    def __init__(
        self,
        sample_rate: int = 24_000,
        settings: Optional[AudioSettings] = None,
    ) -> None:
        self.settings = settings or AudioSettings()
        self.sample_rate = sample_rate
        self._max_queue_bytes = max(
            2, int(self.settings.playback_queue_seconds * self.sample_rate) * 2
        )
        self._condition = threading.Condition()
        self._chunks: deque[tuple[memoryview, Optional[Callable[[], None]]]] = deque()
        self._current: Optional[memoryview] = None
        self._current_offset = 0
        self._current_callback: Optional[Callable[[], None]] = None
        self._queued_bytes = 0
        self._stream: Optional[sd.RawOutputStream] = None
        self._closed = False
        self._audible_until = 0.0
        self.last_status = ""
        self._idle = threading.Event()
        self._idle.set()

    def start(self) -> None:
        if not self.settings.playback_enabled or self._stream is not None:
            return
        self._closed = False
        if self.settings.resolved_backend == "pulse":
            from core.audio.pulse import PulseStream

            self._stream = PulseStream(
                self.sample_rate,
                self.sample_rate // 100,
                self._callback,
                device=self.settings.pulse_output,
            )
        else:
            self._stream = sd.RawOutputStream(
                samplerate=self.sample_rate,
                blocksize=max(1, self.sample_rate // 100),
                device=self.settings.output_device,
                channels=1,
                dtype="int16",
                latency="low",
                callback=self._callback,
            )
        self._stream.start()
        logger.info(
            "audio output started: %s Hz (%s)", self.sample_rate, self.settings.resolved_backend
        )

    def _check_stream(self) -> None:
        if self._stream is not None:
            check = getattr(self._stream, "check", None)
            if check is not None:
                check()
            elif not self._stream.active:
                raise RuntimeError("PortAudio output stream stopped unexpectedly")

    def _callback(self, outdata, _frames, _time_info, status) -> None:
        if status:
            # Logging from PortAudio's real-time callback can itself cause underruns.
            self.last_status = str(status)

        output = memoryview(outdata).cast("B")
        output[:] = b"\0" * len(output)
        position = 0
        callbacks: list[Callable[[], None]] = []

        with self._condition:
            while position < len(output):
                if self._current is None:
                    if not self._chunks:
                        break
                    self._current, self._current_callback = self._chunks.popleft()
                    self._current_offset = 0

                available = len(self._current) - self._current_offset
                count = min(available, len(output) - position)
                output[position : position + count] = self._current[
                    self._current_offset : self._current_offset + count
                ]
                if self._current_callback is not None:
                    callbacks.append(self._current_callback)
                    self._current_callback = None
                position += count
                self._current_offset += count
                self._queued_bytes -= count

                if self._current_offset == len(self._current):
                    self._current = None
                    self._current_offset = 0

            if self._queued_bytes == 0:
                self._idle.set()
            if position:
                latency = self._stream.latency if self._stream is not None else 0.0
                self._audible_until = (
                    time.perf_counter() + latency + position / (self.sample_rate * 2)
                )
            self._condition.notify_all()

        for callback in callbacks:
            callback()

    def enqueue(
        self,
        packet: AudioPacket,
        cancel: threading.Event,
        on_first_playback: Optional[Callable[[], None]] = None,
    ) -> bool:
        if packet.sample_rate != self.sample_rate:
            raise ValueError(
                f"audio sample rate mismatch: {packet.sample_rate} != {self.sample_rate}"
            )
        if len(packet.pcm) % 2:
            raise ValueError("PCM16 packets must contain complete samples")
        if not packet.pcm or cancel.is_set() or self._closed:
            return False
        if not self.settings.playback_enabled:
            return True

        self.start()
        self._check_stream()
        pcm = memoryview(packet.pcm)
        for offset in range(0, len(pcm), self._max_queue_bytes):
            chunk = pcm[offset : offset + self._max_queue_bytes]
            with self._condition:
                while (
                    self._queued_bytes + len(chunk) > self._max_queue_bytes
                    and not cancel.is_set()
                    and not self._closed
                ):
                    self._condition.wait(timeout=0.05)
                    self._check_stream()
                if cancel.is_set() or self._closed:
                    return False
                self._chunks.append((chunk, on_first_playback))
                on_first_playback = None
                self._queued_bytes += len(chunk)
                self._idle.clear()
                self._condition.notify_all()
        return True

    def clear(self) -> None:
        with self._condition:
            had_audio = self._queued_bytes > 0 or time.perf_counter() < self._audible_until
            self._chunks.clear()
            self._current = None
            self._current_offset = 0
            self._current_callback = None
            self._queued_bytes = 0
            self._idle.set()
            self._condition.notify_all()
        if had_audio and self._stream is not None and self.settings.resolved_backend == "pulse":
            self._stream.flush()

    def is_playing(self) -> bool:
        self._check_stream()
        with self._condition:
            return self._queued_bytes > 0 or time.perf_counter() < self._audible_until

    def wait_until_idle(self, timeout: Optional[float] = None) -> bool:
        deadline = None if timeout is None else time.monotonic() + timeout
        with self._condition:
            while self.is_playing():
                remaining = None if deadline is None else deadline - time.monotonic()
                if remaining is not None and remaining <= 0:
                    return False
                self._condition.wait(
                    timeout=min(0.02, remaining) if remaining is not None else 0.02
                )
        return True

    def stop_playback(self) -> None:
        self.clear()

    def stop(self) -> None:
        self._closed = True
        self.clear()
        if self._stream is not None:
            self._stream.stop()
            self._stream.close()
            self._stream = None

    close = stop
