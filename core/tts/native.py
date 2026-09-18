from __future__ import annotations

import logging
import queue
import threading
import time
from pathlib import Path
from typing import Iterator, Optional

from core.config import TTSSettings
from core.types import AudioPacket

logger = logging.getLogger(__name__)


class _PCMQueue(queue.Queue):
    """Bound PCM buffering and pause native generation when playback blocks."""

    def __init__(self, pause=lambda: None, resume=lambda: None) -> None:
        super().__init__(maxsize=12)
        self.cancel = threading.Event()
        self._pause = pause
        self._resume = resume

    def put(self, pcm, block=True, timeout=None) -> None:
        paused = False
        try:
            for offset in range(0, len(pcm), 960):
                while not self.cancel.is_set():
                    try:
                        super().put(pcm[offset : offset + 960], timeout=0.02)
                        break
                    except queue.Full:
                        if not paused:
                            self._pause()
                            paused = True
                if self.cancel.is_set():
                    return
        finally:
            if paused:
                self._resume()


class NativeQwenTTSClient:
    """Qwen3-TTS backed by qwentts.cpp with real PCM streaming and cancellation."""

    sample_rate = 24_000

    def __init__(self, settings: Optional[TTSSettings] = None) -> None:
        self.settings = settings or TTSSettings()
        if not self.settings.ref_audio.is_file():
            raise FileNotFoundError(f"TTS reference audio not found: {self.settings.ref_audio}")
        try:
            from RealtimeTTS import QwenEngine, QwenVoice
        except ImportError as exc:
            raise ImportError(
                "native Qwen TTS is missing; run `python -m pip install -e .` and "
                "`python -m qwentts_cpp prefetch --model Qwen/Qwen3-TTS-12Hz-0.6B-Base --quant Q8_0`"
            ) from exc

        voice = QwenVoice(
            name=self.settings.ref_audio.stem,
            ref_audio=Path(self.settings.ref_audio),
            ref_text=self.settings.ref_text,
            language=self.settings.language,
        )
        logger.info("loading native Qwen TTS (%s, %s)", self.settings.quant, voice.clone_mode)
        self._engine = QwenEngine(
            model_id=self.settings.model,
            quant=self.settings.quant,
            voice=voice,
            local_files_only=self.settings.local_files_only,
            seed=self.settings.seed,
            startup_buffer_ms=self.settings.startup_buffer_ms,
        )
        self._lock = threading.Lock()
        self._state_lock = threading.Lock()
        self._engine.queue = _PCMQueue(
            pause=lambda: self._engine.pause(timeout=0), resume=self._engine.resume
        )
        self._active_stop: Optional[threading.Event] = None
        self._closed = False
        self._failed = False

    def _discard_queued_audio(self) -> None:
        while True:
            try:
                self._engine.queue.get_nowait()
            except queue.Empty:
                return

    def stream(self, text: str, cancel: threading.Event) -> Iterator[AudioPacket]:
        if not text.strip() or cancel.is_set():
            return
        with self._lock:
            with self._state_lock:
                if cancel.is_set():
                    return
                if self._closed or self._failed:
                    raise RuntimeError("native TTS is closed or its previous worker did not stop")
                self._discard_queued_audio()
                stopped = threading.Event()
                self._active_stop = stopped
                self._engine.queue.cancel = stopped
            result: list[bool] = []
            errors: list[BaseException] = []

            def synthesize() -> None:
                try:
                    if not cancel.is_set() and not stopped.is_set():
                        result.append(self._engine.synthesize(text.strip()))
                except BaseException as exc:
                    errors.append(exc)

            worker = threading.Thread(
                target=synthesize,
                name="qwen-tts-native",
                daemon=True,
            )
            worker.start()
            try:
                while worker.is_alive() or not self._engine.queue.empty():
                    if cancel.is_set() or stopped.is_set():
                        break
                    try:
                        pcm = self._engine.queue.get(timeout=0.02)
                    except queue.Empty:
                        continue
                    if pcm and not cancel.is_set() and not stopped.is_set():
                        yield AudioPacket(pcm=pcm, sample_rate=self.sample_rate)
                if errors:
                    raise errors[0]
                if result and not result[0] and not cancel.is_set() and not stopped.is_set():
                    raise RuntimeError(f"native TTS failed: {self._engine.last_error}")
            finally:
                stopped.set()
                deadline = time.monotonic() + 10
                while worker.is_alive() and time.monotonic() < deadline:
                    self._engine.stop()
                    worker.join(timeout=0.02)
                with self._state_lock:
                    self._active_stop = None
                    self._discard_queued_audio()
                    self._failed = worker.is_alive()
                if self._failed:
                    raise TimeoutError("native TTS did not stop within 10 seconds")

    def stop(self) -> None:
        with self._state_lock:
            if self._active_stop is not None:
                self._active_stop.set()
                self._engine.stop()
            self._discard_queued_audio()

    def close(self) -> None:
        self.stop()
        with self._lock:
            if not self._closed:
                self._closed = True
                if self._failed:
                    raise RuntimeError("cannot shut down native TTS while synthesis is stuck")
                self._engine.shutdown()
