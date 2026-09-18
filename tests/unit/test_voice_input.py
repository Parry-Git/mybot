from concurrent.futures import Future
from types import SimpleNamespace

from core.asr.voice_input import VoiceInput
from core.config import Settings, VADSettings


class Stream:
    def __init__(self):
        self.frames = []
        self.cancelled = False
        self.finished = False

    def push(self, frame):
        self.frames.append(frame)

    def finish(self):
        self.finished = True
        result = Future()
        result.set_result(("测试", "Chinese"))
        return result

    def cancel(self):
        self.cancelled = True


def make_voice(barge_in=True):
    stream = Stream()
    submissions = []
    interruptions = []
    asr = SimpleNamespace(supports_streaming=True, start_stream=lambda: stream)
    pipeline = SimpleNamespace(
        busy=False,
        player=SimpleNamespace(is_playing=lambda: False),
        interrupt=lambda: interruptions.append(True),
        submit_utterance=lambda utterance, future: submissions.append((utterance, future)),
    )
    settings = Settings(vad=VADSettings(barge_in=barge_in))
    voice = VoiceInput(settings, asr, pipeline)
    voice.vad._speech_detector = lambda frame: frame[0] == 1
    return voice, stream, submissions, interruptions


def test_streaming_asr_receives_pre_roll_and_each_frame_once():
    voice, stream, submissions, interruptions = make_voice()
    frames = [bytes(640)] * 8 + [b"\1" + bytes(639)] * 20 + [bytes(640)] * 14
    for frame in frames:
        voice.process(frame)
    assert len(submissions) == 1
    assert stream.finished
    # Trigger on the third voiced frame in a four-frame window: five silent
    # pre-roll frames plus all 20 voiced frames and 14 endpoint-silence frames.
    assert b"".join(stream.frames) == b"".join(frames[3:])
    assert len(interruptions) == 1


def test_short_speech_cancels_unused_asr_session():
    voice, stream, submissions, _ = make_voice()
    for frame in [bytes(640)] * 8 + [b"\1" + bytes(639)] * 4 + [bytes(640)] * 14:
        voice.process(frame)
    assert stream.cancelled
    assert not stream.finished
    assert not submissions


def test_half_duplex_discards_speaker_echo():
    voice, stream, submissions, _ = make_voice(barge_in=False)
    voice.pipeline.busy = True
    for _ in range(20):
        voice.process(b"\1" + bytes(639))
    assert not stream.frames
    assert not submissions
    assert not voice.vad.triggered
