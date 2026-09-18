from __future__ import annotations

import threading
import time
from collections.abc import Generator, Sequence
from concurrent.futures import Future

import pytest

from core.audio.player import AudioPlayer
from core.config import AudioSettings, PipelineSettings, Settings
from core.llm.llm_client import LLMInterface, Message
from core.orchestrator import ConversationPipeline
from core.types import AudioPacket, TurnMetrics, Utterance


class FakeLLM(LLMInterface):
    def stream_chat(
        self,
        history: Sequence[Message],
        system_prompt: str | None = None,
        cancel: threading.Event | None = None,
    ) -> Generator[str, None, None]:
        answer = f"收到{history[-1]['content']}。"
        for char in answer:
            if cancel is not None and cancel.is_set():
                return
            time.sleep(0.002)
            yield char


class FakeTTS:
    sample_rate = 24_000

    def __init__(self) -> None:
        self.stop_count = 0

    def stream(self, text: str, cancel: threading.Event):
        if not cancel.is_set():
            yield AudioPacket(pcm=b"\x01\x00" * 240, sample_rate=self.sample_rate)

    def stop(self) -> None:
        self.stop_count += 1

    def close(self) -> None:
        return


def make_pipeline(responses: list[str], metrics: list[TurnMetrics]):
    settings = Settings(
        audio=AudioSettings(playback_enabled=False),
        pipeline=PipelineSettings(min_phrase_chars=2, max_phrase_chars=20),
    )
    tts = FakeTTS()
    player = AudioPlayer(sample_rate=tts.sample_rate, settings=settings.audio)
    pipeline = ConversationPipeline(
        llm=FakeLLM(),
        tts=tts,
        player=player,
        settings=settings,
        on_response=responses.append,
        on_metrics=metrics.append,
    )
    return pipeline, tts


def test_pipeline_streams_and_records_metrics() -> None:
    responses: list[str] = []
    metrics: list[TurnMetrics] = []
    pipeline, _tts = make_pipeline(responses, metrics)

    pipeline.submit_text("测试")
    assert pipeline.wait(timeout=2)

    assert responses == ["收到测试。"]
    assert pipeline.history[-1] == {"role": "assistant", "content": "收到测试。"}
    assert metrics[0].tts_first_audio is not None
    assert metrics[0].playback_first_audio is None
    pipeline.close()


def test_worker_error_reaches_caller_and_pipeline_recovers() -> None:
    pipeline, _ = make_pipeline([], [])
    original = pipeline.tts.stream

    def broken_tts(text, cancel):
        raise RuntimeError("synthesis failed")
        yield

    pipeline.tts.stream = broken_tts
    future = pipeline.submit_text("失败测试")
    with pytest.raises(RuntimeError, match="synthesis failed"):
        future.result(timeout=2)
    with pytest.raises(RuntimeError, match="synthesis failed"):
        pipeline.wait(timeout=2)
    assert pipeline.history == []
    pipeline.tts.stream = original
    assert pipeline.submit_text("恢复").result(timeout=2) == "收到恢复。"
    pipeline.close()


def test_interrupt_does_not_wait_for_unfinished_asr() -> None:
    pipeline, _ = make_pipeline([], [])
    pending: Future = Future()
    utterance = Utterance(b"\0\0" * 320, 16000, time.perf_counter(), time.perf_counter())
    old = pipeline.submit_utterance(utterance, pending)
    new = pipeline.submit_text("新问题")
    assert old.result(timeout=1) == ""
    assert new.result(timeout=2) == "收到新问题。"
    pipeline.close()


def test_failed_asr_timeout_is_not_retried_forever() -> None:
    pipeline, _ = make_pipeline([], [])
    failed: Future = Future()
    failed.set_exception(TimeoutError("ASR failed"))
    utterance = Utterance(b"\0\0" * 320, 16000, time.perf_counter(), time.perf_counter())
    future = pipeline.submit_utterance(utterance, failed)
    with pytest.raises(TimeoutError, match="ASR failed"):
        future.result(timeout=1)
    pipeline.close()


def test_llm_failure_is_propagated_and_producer_exits() -> None:
    pipeline, _ = make_pipeline([], [])
    finished = threading.Event()

    def broken_llm(*args, **kwargs):
        try:
            yield "开头。"
            raise ConnectionError("upstream offline")
        finally:
            finished.set()

    pipeline.llm.stream_chat = broken_llm
    with pytest.raises(ConnectionError, match="upstream offline"):
        pipeline.submit_text("测试").result(timeout=2)
    assert finished.is_set()
    assert pipeline.history == []
    pipeline.close()


def test_new_turn_cancels_previous_turn() -> None:
    responses: list[str] = []
    metrics: list[TurnMetrics] = []
    pipeline, tts = make_pipeline(responses, metrics)

    pipeline.submit_text("第一个很长的问题")
    time.sleep(0.004)
    pipeline.submit_text("第二个")
    assert pipeline.wait(timeout=2)

    assert responses == ["收到第二个。"]
    assert tts.stop_count >= 2
    pipeline.close()


def test_session_reset_cancels_work_and_discards_history() -> None:
    pipeline, _ = make_pipeline([], [])
    pipeline.submit_text("上一段对话").result(timeout=2)
    assert pipeline.history
    pending = pipeline.submit_utterance(
        Utterance(bytes(640), 16000, time.perf_counter(), time.perf_counter()), Future()
    )
    pipeline.reset_conversation()
    assert pending.result(timeout=1) == ""
    assert pipeline.history == []
    pipeline.submit_text("新会话").result(timeout=2)
    assert len(pipeline.history) == 2
    pipeline.close()
