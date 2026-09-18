from core.asr.vad import VadDetector
from core.config import AudioSettings, VADSettings


def test_vad_emits_prerolled_bounded_utterance() -> None:
    audio = AudioSettings(sample_rate=16_000, frame_ms=20)
    settings = VADSettings(
        pre_roll_ms=60,
        start_window_ms=40,
        start_ratio=1.0,
        end_silence_ms=60,
        min_speech_ms=40,
        max_utterance_seconds=10,
    )
    detector = VadDetector(
        audio,
        settings,
        speech_detector=lambda frame: frame[0] == 1,
    )
    silence = bytes(audio.frame_bytes)
    speech = bytes([1]) * audio.frame_bytes

    assert detector.process_chunk(silence, now=0.00).utterance is None
    assert not detector.process_chunk(speech, now=0.02).speech_started
    started = detector.process_chunk(speech, now=0.04)
    assert started.speech_started
    assert started.started_audio == silence + speech + speech

    detector.process_chunk(speech, now=0.06)
    detector.process_chunk(silence, now=0.08)
    detector.process_chunk(silence, now=0.10)
    ended = detector.process_chunk(silence, now=0.12)

    assert ended.utterance is not None
    assert ended.utterance.pcm.startswith(silence + speech + speech)
    assert ended.utterance.sample_rate == 16_000
    assert not detector.triggered
