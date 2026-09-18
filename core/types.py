from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from typing import Optional


@dataclass(frozen=True)
class Utterance:
    pcm: bytes
    sample_rate: int
    started_at: float
    ended_at: float

    @property
    def duration_seconds(self) -> float:
        return len(self.pcm) / (2 * self.sample_rate)


@dataclass(frozen=True)
class AudioPacket:
    pcm: bytes
    sample_rate: int

    @property
    def duration_seconds(self) -> float:
        return len(self.pcm) / (2 * self.sample_rate)


@dataclass
class TurnMetrics:
    speech_end: float = field(default_factory=time.perf_counter)
    asr_done: Optional[float] = None
    llm_first_token: Optional[float] = None
    first_phrase: Optional[float] = None
    tts_first_audio: Optional[float] = None
    playback_first_audio: Optional[float] = None
    completed: Optional[float] = None
    _lock: threading.Lock = field(default_factory=threading.Lock, init=False, repr=False)

    def mark_once(self, field_name: str) -> None:
        with self._lock:
            if getattr(self, field_name) is None:
                setattr(self, field_name, time.perf_counter())

    def milliseconds_from_speech_end(self, timestamp: Optional[float]) -> Optional[float]:
        if timestamp is None:
            return None
        return (timestamp - self.speech_end) * 1000

    def summary(self) -> dict[str, Optional[float]]:
        def elapsed(start: Optional[float], end: Optional[float]) -> Optional[float]:
            return None if start is None or end is None else (end - start) * 1000

        return {
            "asr_ms": self.milliseconds_from_speech_end(self.asr_done),
            "llm_first_token_ms": self.milliseconds_from_speech_end(self.llm_first_token),
            "first_phrase_ms": self.milliseconds_from_speech_end(self.first_phrase),
            "tts_first_audio_ms": self.milliseconds_from_speech_end(self.tts_first_audio),
            "playback_first_audio_ms": self.milliseconds_from_speech_end(self.playback_first_audio),
            "turn_ms": self.milliseconds_from_speech_end(self.completed),
            "llm_ttft_ms": elapsed(self.asr_done, self.llm_first_token),
            "tts_ttfa_ms": elapsed(self.first_phrase, self.tts_first_audio),
            "output_queue_ms": elapsed(self.tts_first_audio, self.playback_first_audio),
        }
