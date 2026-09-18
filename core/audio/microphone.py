from __future__ import annotations

import logging
import queue
import time
from typing import Iterator, Optional

import sounddevice as sd

from core.config import AudioSettings

logger = logging.getLogger(__name__)


class Microphone:
    """Callback-driven microphone that never blocks the inference pipeline."""

    def __init__(self, settings: Optional[AudioSettings] = None, queue_frames: int = 200) -> None:
        self.settings = settings or AudioSettings()
        self._queue: queue.Queue[bytes] = queue.Queue(maxsize=queue_frames)
        self._stream: Optional[sd.RawInputStream] = None
        self._closed = False
        self.dropped_frames = 0
        self.unexpected_frame_count = 0
        self.last_status = ""
        self._last_frame_at = time.monotonic()

    def __enter__(self) -> "Microphone":
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.stop()

    def _callback(self, indata, frames, _time_info, status) -> None:
        self._last_frame_at = time.monotonic()
        if status:
            self.last_status = str(status)
        if frames != self.settings.frame_samples:
            self.unexpected_frame_count += 1
        try:
            self._queue.put_nowait(bytes(indata))
        except queue.Full:
            self.dropped_frames += 1
            try:
                self._queue.get_nowait()
                self._queue.put_nowait(bytes(indata))
            except (queue.Empty, queue.Full):
                pass

    def start(self) -> None:
        if self._stream is not None:
            return
        self._closed = False
        self._last_frame_at = time.monotonic()
        if self.settings.resolved_backend == "pulse":
            from core.audio.pulse import PulseStream

            self._stream = PulseStream(
                self.settings.sample_rate,
                self.settings.frame_samples,
                self._callback,
                recording=True,
                device=self.settings.pulse_input,
            )
        else:
            self._stream = sd.RawInputStream(
                samplerate=self.settings.sample_rate,
                blocksize=self.settings.frame_samples,
                device=self.settings.input_device,
                channels=self.settings.channels,
                dtype="int16",
                latency="low",
                callback=self._callback,
            )
        self._stream.start()
        logger.info(
            "microphone started: %s Hz, %s ms frames",
            self.settings.sample_rate,
            self.settings.frame_ms,
        )

    def stop(self) -> None:
        self._closed = True
        if self._stream is not None:
            self._stream.stop()
            self._stream.close()
            self._stream = None

    def stream(self) -> Iterator[bytes]:
        if self._stream is None:
            raise RuntimeError("microphone is not started")
        while not self._closed:
            check = getattr(self._stream, "check", None)
            if check is not None:
                check()
            elif not self._stream.active:
                raise RuntimeError("PortAudio input stream stopped unexpectedly")
            try:
                yield self._queue.get(timeout=0.2)
            except queue.Empty:
                if time.monotonic() - self._last_frame_at > 3:
                    raise TimeoutError("microphone has not delivered audio for 3 seconds")
                continue
