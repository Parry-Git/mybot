from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[1]
load_dotenv(PROJECT_ROOT / ".env")


def _env_bool(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _env_int(name: str, default: int) -> int:
    value = os.getenv(name)
    return default if value is None else int(value)


def _env_float(name: str, default: float) -> float:
    value = os.getenv(name)
    return default if value is None else float(value)


def _env_path(name: str, default: Path) -> Path:
    value = os.getenv(name)
    path = Path(value).expanduser() if value else default
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return path.resolve()


def _env_optional_int(name: str) -> Optional[int]:
    value = os.getenv(name)
    return None if value in (None, "") else int(value)


@dataclass(frozen=True)
class AudioSettings:
    backend: str = field(default_factory=lambda: os.getenv("AUDIO_BACKEND", "auto"))
    pulse_input: Optional[str] = field(
        default_factory=lambda: os.getenv("PULSE_INPUT_DEVICE") or None
    )
    pulse_output: Optional[str] = field(
        default_factory=lambda: os.getenv("PULSE_OUTPUT_DEVICE") or None
    )
    sample_rate: int = field(default_factory=lambda: _env_int("AUDIO_SAMPLE_RATE", 16_000))
    channels: int = 1
    frame_ms: int = field(default_factory=lambda: _env_int("VAD_FRAME_MS", 20))
    input_device: Optional[int] = field(
        default_factory=lambda: _env_optional_int("AUDIO_INPUT_DEVICE")
    )
    output_device: Optional[int] = field(
        default_factory=lambda: _env_optional_int("AUDIO_OUTPUT_DEVICE")
    )
    playback_enabled: bool = field(default_factory=lambda: _env_bool("AUDIO_PLAYBACK", True))
    playback_queue_seconds: float = field(
        default_factory=lambda: _env_float("AUDIO_PLAYBACK_QUEUE_SECONDS", 2.0)
    )

    def __post_init__(self) -> None:
        if self.backend not in {"auto", "pulse", "portaudio"}:
            raise ValueError("AUDIO_BACKEND must be auto, pulse or portaudio")
        if self.sample_rate != 16_000 or self.channels != 1:
            raise ValueError("voice input must be 16000 Hz mono for streaming ASR")
        if self.frame_ms not in (10, 20, 30):
            raise ValueError("VAD_FRAME_MS must be 10, 20 or 30")
        if self.playback_queue_seconds < 0.02:
            raise ValueError("AUDIO_PLAYBACK_QUEUE_SECONDS must be at least 0.02")

    @property
    def resolved_backend(self) -> str:
        if self.backend != "auto":
            return self.backend
        if self.input_device is not None or self.output_device is not None:
            return "portaudio"
        return "pulse" if Path("/mnt/wslg/PulseServer").exists() else "portaudio"

    @property
    def frame_samples(self) -> int:
        return self.sample_rate * self.frame_ms // 1000

    @property
    def frame_bytes(self) -> int:
        return self.frame_samples * 2


@dataclass(frozen=True)
class VADSettings:
    mode: int = field(default_factory=lambda: _env_int("VAD_MODE", 2))
    pre_roll_ms: int = field(default_factory=lambda: _env_int("VAD_PRE_ROLL_MS", 160))
    start_window_ms: int = field(default_factory=lambda: _env_int("VAD_START_WINDOW_MS", 80))
    start_ratio: float = field(default_factory=lambda: _env_float("VAD_START_RATIO", 0.75))
    end_silence_ms: int = field(default_factory=lambda: _env_int("VAD_END_SILENCE_MS", 280))
    min_speech_ms: int = field(default_factory=lambda: _env_int("VAD_MIN_SPEECH_MS", 180))
    max_utterance_seconds: float = field(
        default_factory=lambda: _env_float("VAD_MAX_UTTERANCE_SECONDS", 30.0)
    )
    barge_in: bool = field(default_factory=lambda: _env_bool("BARGE_IN", False))

    def __post_init__(self) -> None:
        if not 0 < self.start_ratio <= 1:
            raise ValueError("VAD_START_RATIO must be in (0, 1]")
        if self.pre_roll_ms < self.start_window_ms:
            raise ValueError("VAD_PRE_ROLL_MS must cover VAD_START_WINDOW_MS")
        if min(self.start_window_ms, self.end_silence_ms, self.min_speech_ms) <= 0:
            raise ValueError("VAD timing settings must be positive")
        if not 0 < self.max_utterance_seconds <= 30:
            raise ValueError("VAD_MAX_UTTERANCE_SECONDS must be in (0, 30]")


@dataclass(frozen=True)
class ASRSettings:
    model: str = field(default_factory=lambda: os.getenv("ASR_MODEL", "Qwen/Qwen3-ASR-0.6B"))
    backend: str = field(default_factory=lambda: os.getenv("ASR_BACKEND", "vllm").lower())
    language: Optional[str] = field(default_factory=lambda: os.getenv("ASR_LANGUAGE") or None)
    context: str = field(default_factory=lambda: os.getenv("ASR_CONTEXT", ""))
    device: str = field(default_factory=lambda: os.getenv("ASR_DEVICE", "cuda:0"))
    attention: str = field(default_factory=lambda: os.getenv("ASR_ATTENTION", "sdpa"))
    compile: bool = field(default_factory=lambda: _env_bool("ASR_COMPILE", False))
    max_new_tokens: int = field(default_factory=lambda: _env_int("ASR_MAX_NEW_TOKENS", 128))
    vllm_gpu_memory_utilization: float = field(
        default_factory=lambda: _env_float("ASR_VLLM_GPU_MEMORY_UTILIZATION", 0.46)
    )
    vllm_max_model_len: int = field(
        default_factory=lambda: _env_int("ASR_VLLM_MAX_MODEL_LEN", 2048)
    )
    vllm_enforce_eager: bool = field(
        default_factory=lambda: _env_bool("ASR_VLLM_ENFORCE_EAGER", True)
    )
    stream_chunk_seconds: float = field(
        default_factory=lambda: _env_float("ASR_STREAM_CHUNK_SECONDS", 1.0)
    )

    def __post_init__(self) -> None:
        if self.stream_chunk_seconds <= 0:
            raise ValueError("ASR_STREAM_CHUNK_SECONDS must be positive")
        if not 0 < self.vllm_gpu_memory_utilization <= 1:
            raise ValueError("ASR_VLLM_GPU_MEMORY_UTILIZATION must be in (0, 1]")


@dataclass(frozen=True)
class LLMSettings:
    api_key: Optional[str] = field(
        default_factory=lambda: os.getenv("DEEPSEEK_API_KEY"), repr=False
    )
    base_url: str = field(
        default_factory=lambda: os.getenv("LLM_BASE_URL", "https://api.deepseek.com")
    )
    model: str = field(default_factory=lambda: os.getenv("LLM_MODEL", "deepseek-flash"))
    system_prompt: str = field(
        default_factory=lambda: os.getenv(
            "LLM_SYSTEM_PROMPT",
            "你是 MyBot，一名简洁、自然的语音助手。直接回答，不要使用 Markdown。",
        )
    )
    timeout_seconds: float = field(default_factory=lambda: _env_float("LLM_TIMEOUT_SECONDS", 45.0))
    max_output_tokens: int = field(default_factory=lambda: _env_int("LLM_MAX_OUTPUT_TOKENS", 512))
    thinking_enabled: bool = field(default_factory=lambda: _env_bool("LLM_THINKING", False))
    max_history_messages: int = field(
        default_factory=lambda: _env_int("LLM_MAX_HISTORY_MESSAGES", 20)
    )

    def __post_init__(self) -> None:
        if self.max_history_messages < 2 or self.max_history_messages % 2:
            raise ValueError(
                "LLM_MAX_HISTORY_MESSAGES must be a positive number of user/assistant pairs"
            )
        if self.timeout_seconds <= 0 or self.max_output_tokens < 1:
            raise ValueError("LLM timeout and token limit must be positive")


@dataclass(frozen=True)
class TTSSettings:
    model: str = field(
        default_factory=lambda: os.getenv("TTS_MODEL", "Qwen/Qwen3-TTS-12Hz-0.6B-Base")
    )
    quant: str = field(default_factory=lambda: os.getenv("TTS_QUANT", "Q8_0"))
    ref_audio: Path = field(
        default_factory=lambda: _env_path(
            "TTS_REF_AUDIO", PROJECT_ROOT / "assets" / "ref_audio.wav"
        )
    )
    ref_text: Optional[str] = field(default_factory=lambda: os.getenv("TTS_REF_TEXT") or None)
    language: str = field(default_factory=lambda: os.getenv("TTS_LANGUAGE", "chinese").lower())
    seed: int = field(default_factory=lambda: _env_int("TTS_SEED", -1))
    startup_buffer_ms: float = field(
        default_factory=lambda: _env_float("TTS_STARTUP_BUFFER_MS", 80.0)
    )
    local_files_only: bool = field(default_factory=lambda: _env_bool("TTS_LOCAL_FILES_ONLY", False))


@dataclass(frozen=True)
class PipelineSettings:
    min_phrase_chars: int = field(default_factory=lambda: _env_int("TTS_MIN_PHRASE_CHARS", 8))
    max_phrase_chars: int = field(default_factory=lambda: _env_int("TTS_MAX_PHRASE_CHARS", 42))
    text_queue_size: int = field(default_factory=lambda: _env_int("TTS_TEXT_QUEUE_SIZE", 2))

    def __post_init__(self) -> None:
        if self.text_queue_size < 1:
            raise ValueError("TTS_TEXT_QUEUE_SIZE must be positive")
        if not 1 <= self.min_phrase_chars <= self.max_phrase_chars:
            raise ValueError("phrase lengths must satisfy 1 <= min <= max")


@dataclass(frozen=True)
class Settings:
    audio: AudioSettings = field(default_factory=AudioSettings)
    vad: VADSettings = field(default_factory=VADSettings)
    asr: ASRSettings = field(default_factory=ASRSettings)
    llm: LLMSettings = field(default_factory=LLMSettings)
    tts: TTSSettings = field(default_factory=TTSSettings)
    pipeline: PipelineSettings = field(default_factory=PipelineSettings)


def load_settings() -> Settings:
    return Settings()
