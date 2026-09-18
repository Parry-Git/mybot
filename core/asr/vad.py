from __future__ import annotations

import collections
import logging
import math
import time
from dataclasses import dataclass
from typing import Callable, Optional

import webrtcvad

from core.config import AudioSettings, VADSettings
from core.types import Utterance

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class VADEvent:
    speech_started: bool = False
    started_audio: bytes = b""
    utterance: Optional[Utterance] = None


class VadDetector:
    """Low-latency endpoint detector with pre-roll and bounded utterances."""

    def __init__(
        self,
        audio: Optional[AudioSettings] = None,
        settings: Optional[VADSettings] = None,
        speech_detector: Optional[Callable[[bytes], bool]] = None,
    ) -> None:
        self.audio = audio or AudioSettings()
        self.settings = settings or VADSettings()
        if self.audio.frame_ms not in (10, 20, 30):
            raise ValueError("WebRTC VAD frame size must be 10, 20, or 30 ms")
        self._vad = webrtcvad.Vad(self.settings.mode)
        self._speech_detector = speech_detector
        self._pre_roll_frames = max(1, math.ceil(self.settings.pre_roll_ms / self.audio.frame_ms))
        self._start_window_frames = max(
            1, math.ceil(self.settings.start_window_ms / self.audio.frame_ms)
        )
        self._end_silence_frames = max(
            1, math.ceil(self.settings.end_silence_ms / self.audio.frame_ms)
        )
        self._min_speech_frames = max(
            1, math.ceil(self.settings.min_speech_ms / self.audio.frame_ms)
        )
        self._max_frames = max(
            1, math.ceil(self.settings.max_utterance_seconds * 1000 / self.audio.frame_ms)
        )
        self._pre_roll: collections.deque[tuple[bytes, bool]] = collections.deque(
            maxlen=self._pre_roll_frames
        )
        self.reset()

    @property
    def triggered(self) -> bool:
        return self._triggered

    def is_speech(self, frame: bytes) -> bool:
        if len(frame) != self.audio.frame_bytes:
            raise ValueError(
                f"expected {self.audio.frame_bytes} PCM bytes per VAD frame, got {len(frame)}"
            )
        if self._speech_detector is not None:
            return bool(self._speech_detector(frame))
        return bool(self._vad.is_speech(frame, self.audio.sample_rate))

    def process_chunk(self, frame: bytes, now: Optional[float] = None) -> VADEvent:
        now = time.perf_counter() if now is None else now
        speech = self.is_speech(frame)

        if not self._triggered:
            self._pre_roll.append((frame, speech))
            recent = list(self._pre_roll)[-self._start_window_frames :]
            needed = math.ceil(len(recent) * self.settings.start_ratio)
            if (
                len(recent) >= self._start_window_frames
                and sum(flag for _, flag in recent) >= needed
            ):
                self._triggered = True
                self._started_at = now - (len(self._pre_roll) * self.audio.frame_ms / 1000)
                self._frames = [item for item, _ in self._pre_roll]
                self._speech_frames = sum(flag for _, flag in self._pre_roll)
                self._silence_frames = 0
                started_audio = b"".join(self._frames)
                self._pre_roll.clear()
                logger.debug("speech started")
                return VADEvent(speech_started=True, started_audio=started_audio)
            return VADEvent()

        self._frames.append(frame)
        if speech:
            self._speech_frames += 1
            self._silence_frames = 0
        else:
            self._silence_frames += 1

        reached_endpoint = self._silence_frames >= self._end_silence_frames
        reached_limit = len(self._frames) >= self._max_frames
        if not (reached_endpoint or reached_limit):
            return VADEvent()

        keep_frames = len(self._frames)
        if reached_endpoint:
            # Keep a short trailing pad but do not make ASR process the full endpoint delay.
            trailing_frames = max(1, math.ceil(80 / self.audio.frame_ms))
            keep_frames -= max(0, self._silence_frames - trailing_frames)
        pcm = b"".join(self._frames[:keep_frames])
        valid = self._speech_frames >= self._min_speech_frames
        started_at = self._started_at
        self.reset()
        if not valid:
            logger.debug("discarded short VAD segment")
            return VADEvent()
        logger.debug("speech ended after %.2f seconds", len(pcm) / (2 * self.audio.sample_rate))
        return VADEvent(
            utterance=Utterance(
                pcm=pcm,
                sample_rate=self.audio.sample_rate,
                started_at=started_at,
                ended_at=now,
            )
        )

    def reset(self) -> None:
        self._triggered = False
        self._frames: list[bytes] = []
        self._speech_frames = 0
        self._silence_frames = 0
        self._started_at = 0.0
        self._pre_roll.clear()
