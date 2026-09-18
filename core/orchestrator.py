from __future__ import annotations

import logging
import queue
import threading
import time
from collections.abc import Callable
from concurrent.futures import Future
from concurrent.futures import TimeoutError as FutureTimeout
from contextlib import closing
from typing import TYPE_CHECKING, Optional

from core.audio.base import Playback
from core.config import Settings
from core.llm.llm_client import LLMInterface, Message
from core.llm.streamer import TokenAggregator
from core.tts.base import StreamingTTS
from core.types import TurnMetrics, Utterance

if TYPE_CHECKING:
    from core.asr.asr_client import ASRClient

ASRResult = tuple[str, Optional[str]]

logger = logging.getLogger(__name__)
_SENTINEL = object()


class ConversationPipeline:
    """Cancelable ASR -> LLM -> TTS pipeline with bounded streaming queues."""

    def __init__(
        self,
        llm: LLMInterface,
        tts: StreamingTTS,
        player: Playback,
        settings: Settings,
        asr: Optional[ASRClient] = None,
        on_user_text: Optional[Callable[[str], None]] = None,
        on_phrase: Optional[Callable[[str], None]] = None,
        on_response: Optional[Callable[[str], None]] = None,
        on_metrics: Optional[Callable[[TurnMetrics], None]] = None,
    ) -> None:
        self.llm = llm
        self.tts = tts
        self.player = player
        self.settings = settings
        self.asr = asr
        self.on_user_text = on_user_text
        self.on_phrase = on_phrase
        self.on_response = on_response
        self.on_metrics = on_metrics
        self.history: list[Message] = []
        self._history_lock = threading.Lock()
        self._state_lock = threading.RLock()
        self._cancel: Optional[threading.Event] = None
        self._thread: Optional[threading.Thread] = None
        self._future: Optional[Future[str]] = None
        self._workers: set[threading.Thread] = set()
        self._closed = False

    def _cancel_current(self) -> None:
        with self._state_lock:
            cancel = self._cancel
            if cancel is not None:
                cancel.set()
            self.tts.stop()
            self.player.clear()

    @property
    def busy(self) -> bool:
        with self._state_lock:
            return bool(self._future is not None and not self._future.done())

    def interrupt(self) -> None:
        """Stop generated speech immediately; safe to call from the VAD loop."""
        self._cancel_current()

    def reset_conversation(self) -> None:
        with self._state_lock:
            self._cancel_current()
            with self._history_lock:
                self.history.clear()

    def _launch(self, target: Callable[[threading.Event], str], name: str) -> Future[str]:
        cancel = threading.Event()
        future: Future[str] = Future()

        def runner() -> None:
            try:
                future.set_result(target(cancel))
            except Exception as exc:
                cancel.set()
                with self._state_lock:
                    if self._cancel is cancel:
                        self.tts.stop()
                        self.player.clear()
                logger.exception("conversation turn failed")
                future.set_exception(exc)
            finally:
                with self._state_lock:
                    self._workers.discard(threading.current_thread())

        thread = threading.Thread(target=runner, name=name, daemon=True)
        with self._state_lock:
            if self._closed:
                raise RuntimeError("pipeline is closed")
            self._cancel_current()
            self._cancel = cancel
            self._thread = thread
            self._future = future
            self._workers.add(thread)
            thread.start()
        return future

    def submit_text(self, text: str) -> Future[str]:
        text = text.strip()
        if not text:
            raise ValueError("text must not be empty")
        metrics = TurnMetrics(speech_end=time.perf_counter())
        metrics.mark_once("asr_done")
        return self._launch(
            lambda cancel: self._respond_to_text(text, metrics, cancel),
            "text-turn",
        )

    def submit_utterance(
        self,
        utterance: Utterance,
        asr_future: Optional[Future[ASRResult]] = None,
    ) -> Future[str]:
        if self.asr is None and asr_future is None:
            raise RuntimeError("voice turns require an ASR client or streaming ASR result")
        metrics = TurnMetrics(speech_end=utterance.ended_at)

        def run(cancel: threading.Event) -> str:
            if asr_future is not None:
                while not cancel.is_set():
                    try:
                        text, _language = asr_future.result(timeout=0.05)
                        break
                    except FutureTimeout:
                        if asr_future.done():
                            raise
                        continue
                else:
                    return ""
            else:
                assert self.asr is not None
                text, _language = self.asr.transcribe(utterance.pcm, utterance.sample_rate)
            metrics.mark_once("asr_done")
            if cancel.is_set() or not text.strip():
                return ""
            return self._respond_to_text(text.strip(), metrics, cancel)

        return self._launch(run, "voice-turn")

    def _respond_to_text(
        self,
        text: str,
        metrics: TurnMetrics,
        cancel: threading.Event,
    ) -> str:
        with self._state_lock:
            if cancel.is_set():
                return ""
            if self.on_user_text is not None:
                self.on_user_text(text)
        with self._history_lock:
            history = list(self.history) + [{"role": "user", "content": text}]

        phrase_queue: queue.Queue[object] = queue.Queue(
            maxsize=self.settings.pipeline.text_queue_size
        )
        producer_done = threading.Event()
        aggregator = TokenAggregator(
            min_chars=self.settings.pipeline.min_phrase_chars,
            max_chars=self.settings.pipeline.max_phrase_chars,
        )
        producer_error: list[BaseException] = []

        def put_phrase(phrase: str) -> bool:
            while not cancel.is_set():
                try:
                    phrase_queue.put(phrase, timeout=0.05)
                    return True
                except queue.Full:
                    continue
            return False

        def produce() -> None:
            try:
                with closing(
                    self.llm.stream_chat(
                        history,
                        system_prompt=self.settings.llm.system_prompt,
                        cancel=cancel,
                    )
                ) as stream:
                    for token in stream:
                        if cancel.is_set():
                            return
                        metrics.mark_once("llm_first_token")
                        for phrase in aggregator.feed(token):
                            metrics.mark_once("first_phrase")
                            if not put_phrase(phrase):
                                return
                    if cancel.is_set():
                        return
                    for phrase in aggregator.finish():
                        metrics.mark_once("first_phrase")
                        if not put_phrase(phrase):
                            return
            except Exception as exc:
                producer_error.append(exc)
                cancel.set()
            finally:
                producer_done.set()
                try:
                    phrase_queue.put_nowait(_SENTINEL)
                except queue.Full:
                    pass

        producer = threading.Thread(target=produce, name="llm-stream", daemon=True)
        producer.start()
        first_packet = True

        def mark_first_playback() -> None:
            metrics.mark_once("playback_first_audio")

        try:
            while not cancel.is_set():
                try:
                    item = phrase_queue.get(timeout=0.05)
                except queue.Empty:
                    if producer_done.is_set():
                        break
                    continue
                if item is _SENTINEL:
                    break
                phrase = str(item)
                if self.on_phrase is not None:
                    self.on_phrase(phrase)
                with closing(self.tts.stream(phrase, cancel)) as packets:
                    for packet in packets:
                        if cancel.is_set():
                            break
                        metrics.mark_once("tts_first_audio")
                        callback = None
                        if first_packet:
                            callback = mark_first_playback
                            first_packet = False
                        if not self.player.enqueue(packet, cancel, callback):
                            cancel.set()
                            break
        except Exception:
            cancel.set()
            raise
        finally:
            producer.join(timeout=2)
        if producer_error:
            raise producer_error[0]
        response = aggregator.visible_text
        if cancel.is_set() or not response:
            return ""

        with self._state_lock:
            if cancel.is_set():
                return ""
            if self.on_response is not None:
                self.on_response(response)

        while self.player.is_playing() and not cancel.wait(0.05):
            pass
        with self._state_lock:
            if cancel.is_set():
                return ""
            with self._history_lock:
                self.history.extend(
                    [
                        {"role": "user", "content": text},
                        {"role": "assistant", "content": response},
                    ]
                )
                limit = self.settings.llm.max_history_messages
                self.history = self.history[-limit:]
            metrics.mark_once("completed")
            if self.on_metrics is not None:
                self.on_metrics(metrics)
        return response

    def wait(self, timeout: Optional[float] = None) -> bool:
        with self._state_lock:
            future = self._future
        if future is None:
            return True
        try:
            future.result(timeout=timeout)
        except FutureTimeout:
            if future.done():
                raise
            return False
        return True

    def close(self) -> None:
        with self._state_lock:
            if self._closed:
                return
            self._closed = True
            self._cancel_current()
            workers = list(self._workers)
        close_llm = getattr(self.llm, "close", None)
        if close_llm is not None:
            close_llm()
        for thread in workers:
            thread.join(timeout=12)
        try:
            if any(thread.is_alive() for thread in workers):
                raise TimeoutError("conversation workers did not stop")
            self.tts.close()
        finally:
            self.player.close()
