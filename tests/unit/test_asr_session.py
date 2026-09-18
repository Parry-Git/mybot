import threading
from types import SimpleNamespace

import numpy as np
import pytest

from core.asr.asr_client import ASRStreamingSession
from core.config import ASRSettings


class Model:
    def __init__(self):
        self.samples = 0

    def init_streaming_state(self, **kwargs):
        return SimpleNamespace(text="", language="Chinese")

    def streaming_transcribe(self, samples, state):
        assert samples.dtype == np.float32
        self.samples += len(samples)

    def finish_streaming_transcribe(self, state):
        state.text = "测试"


def client(model):
    return SimpleNamespace(
        _inference_lock=threading.Lock(),
        _sessions_lock=threading.Lock(),
        _sessions=set(),
        _closed=False,
        model=model,
        settings=ASRSettings(),
    )


def test_finish_flushes_all_pcm_and_releases_session():
    model = Model()
    owner = client(model)
    session = ASRStreamingSession(owner)
    for _ in range(15):
        session.push(bytes(640))
    assert session.finish().result(timeout=2) == ("测试", "Chinese")
    assert session.join()
    assert model.samples == 4800
    assert not owner._sessions


def test_cancel_unblocks_session_waiting_for_frames():
    owner = client(Model())
    session = ASRStreamingSession(owner)
    session.cancel()
    assert session.join(timeout=1)
    assert session.finish().cancelled()
    assert not owner._sessions


def test_asr_worker_error_is_visible_without_blocking_finish():
    model = Model()

    def fail(**kwargs):
        raise RuntimeError("engine failed")

    model.init_streaming_state = fail
    session = ASRStreamingSession(client(model))
    with pytest.raises(RuntimeError, match="engine failed"):
        session.finish().result(timeout=1)
    assert session.join()
