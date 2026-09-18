from __future__ import annotations

import logging
import threading
from collections.abc import Callable
from contextlib import ExitStack

from core.asr.asr_client import ASRClient
from core.asr.voice_input import VoiceInput
from core.audio.base import Playback
from core.audio.microphone import Microphone
from core.audio.player import AudioPlayer
from core.config import Settings
from core.llm.llm_client import OpenAILLMClient
from core.orchestrator import ConversationPipeline
from core.tts.native import NativeQwenTTSClient
from core.types import TurnMetrics

logger = logging.getLogger("mybot")


class MyBot:
    """Assemble, warm up and close models and audio transports as one unit."""

    def __init__(
        self,
        settings: Settings,
        mode: str,
        warmup_asr: bool = True,
        player_factory: Callable[[], Playback] | None = None,
        on_event: Callable[[str, object], None] | None = None,
    ) -> None:
        self.settings = settings
        self.mode = mode
        self.running = True
        self._resources = ExitStack()
        self._stopped = False
        self._printed_phrase = False
        self._on_event = on_event or (lambda kind, value: None)

        try:
            self.llm = OpenAILLMClient(settings.llm)
            self._resources.callback(self.llm.close)
            self.asr = ASRClient(settings.asr) if mode != "text" else None
            if self.asr is not None:
                self._resources.callback(self.asr.close)
                if warmup_asr:
                    logger.info("warming up ASR with speech")
                    self.asr.warmup(str(settings.tts.ref_audio))
            self.tts = NativeQwenTTSClient(settings.tts)
            self._resources.callback(self.tts.close)
            self.player = (
                player_factory()
                if player_factory
                else AudioPlayer(sample_rate=self.tts.sample_rate, settings=settings.audio)
            )
            self._resources.callback(self.player.close)
            self.pipeline = ConversationPipeline(
                llm=self.llm,
                tts=self.tts,
                player=self.player,
                settings=settings,
                asr=self.asr,
                on_user_text=self._on_user_text,
                on_phrase=self._print_phrase if mode == "text" else None,
                on_response=self._on_response,
                on_metrics=self._on_metrics,
            )
            self._resources.callback(self.pipeline.close)
            self.voice_input = VoiceInput(settings, self.asr, self.pipeline) if self.asr else None
            if self.voice_input is not None:
                self._resources.callback(self.voice_input.close)
            self.microphone = Microphone(settings.audio) if mode == "voice" else None
            if self.microphone is not None:
                self._resources.callback(self.microphone.stop)
            logger.info("warming up TTS")
            for _packet in self.tts.stream("你好。", threading.Event()):
                pass
            logger.info("warming up LLM connection")
            probe = "".join(self.llm.stream_chat([{"role": "user", "content": "请只回复：就绪。"}]))
            if not probe.strip():
                raise RuntimeError("LLM warmup returned no visible content")
            self.player.start()
        except BaseException:
            self._resources.close()
            raise

    def _on_user_text(self, text: str) -> None:
        logger.info("User: %s", text)
        self._on_event("user", text)

    def _print_phrase(self, phrase: str) -> None:
        if not self._printed_phrase:
            print("AI: ", end="", flush=True)
            self._printed_phrase = True
        print(phrase, end="", flush=True)

    def _on_response(self, response: str) -> None:
        if self.mode == "text":
            if self._printed_phrase:
                print()
            self._printed_phrase = False
        else:
            logger.info("AI: %s", response)
        self._on_event("assistant", response)

    def _on_metrics(self, metrics: TurnMetrics) -> None:
        summary = {
            key: round(value, 1) if value is not None else None
            for key, value in metrics.summary().items()
        }
        logger.info("latency: %s", summary)
        self._on_event("metrics", summary)

    def stop(self) -> None:
        if self._stopped:
            return
        self._stopped = True
        self.running = False
        self._resources.close()
        logger.info("all workers and audio devices stopped")

    def run_text(self) -> None:
        logger.info("text mode ready; type quit to exit")
        try:
            while self.running:
                text = input("You: ").strip()
                if text.lower() in {"quit", "exit"}:
                    break
                if not text:
                    continue
                self._printed_phrase = False
                self.pipeline.submit_text(text)
                try:
                    self.pipeline.wait()
                except Exception:
                    logger.error("turn failed; check the error above and retry")
        except (EOFError, KeyboardInterrupt):
            pass
        finally:
            self.stop()

    def run_voice(self) -> None:
        assert self.microphone is not None
        assert self.voice_input is not None

        try:
            with self.microphone as microphone:
                ready = False
                dropped = microphone.dropped_frames
                for frame in microphone.stream():
                    if not self.running:
                        break
                    if not ready:
                        logger.info("voice mode ready; barge_in=%s", self.settings.vad.barge_in)
                        ready = True
                    if microphone.dropped_frames != dropped:
                        logger.warning(
                            "microphone dropped %s frames; resetting utterance",
                            microphone.dropped_frames - dropped,
                        )
                        dropped = microphone.dropped_frames
                        self.voice_input.reset()
                    try:
                        self.voice_input.process(frame)
                    except Exception:
                        logger.exception("voice input failed; resetting utterance")
                        self.voice_input.reset()
        except KeyboardInterrupt:
            pass
        except Exception:
            logger.exception("voice loop failed")
            raise
        finally:
            self.stop()
