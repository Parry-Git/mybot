from __future__ import annotations

import logging
from concurrent.futures import Future
from typing import TYPE_CHECKING, Optional

from core.asr.vad import VadDetector
from core.config import Settings

if TYPE_CHECKING:
    from core.asr.asr_client import ASRClient, ASRStreamingSession
    from core.orchestrator import ConversationPipeline

logger = logging.getLogger(__name__)


class VoiceInput:
    """Shared microphone/file framing, endpointing and streaming-ASR lifecycle."""

    def __init__(self, settings: Settings, asr: ASRClient, pipeline: ConversationPipeline) -> None:
        self.settings = settings
        self.asr = asr
        self.pipeline = pipeline
        self.vad = VadDetector(settings.audio, settings.vad)
        self._stream: Optional[ASRStreamingSession] = None

    def process(self, frame: bytes, now: Optional[float] = None) -> Optional[Future[str]]:
        if not self.settings.vad.barge_in and (
            self.pipeline.busy or self.pipeline.player.is_playing()
        ):
            self.reset()
            return None
        was_triggered = self.vad.triggered
        event = self.vad.process_chunk(frame, now=now)
        if event.speech_started:
            if self.settings.vad.barge_in:
                self.pipeline.interrupt()
            if self.asr.supports_streaming:
                self._stream = self.asr.start_stream()
                self._stream.push(event.started_audio)
        elif was_triggered and self._stream is not None:
            self._stream.push(frame)

        if was_triggered and not self.vad.triggered:
            stream, self._stream = self._stream, None
            if event.utterance is None:
                if stream is not None:
                    stream.cancel()
                return None
            future = stream.finish() if stream is not None else None
            logger.info("speech endpoint: %.2fs audio", event.utterance.duration_seconds)
            return self.pipeline.submit_utterance(event.utterance, future)
        return None

    def reset(self) -> None:
        if self._stream is not None:
            self._stream.cancel()
            self._stream = None
        self.vad.reset()

    close = reset
