from __future__ import annotations

import gc
import importlib
import logging
import queue
import threading
from concurrent.futures import Future
from typing import Optional, Union

import numpy as np

from core.config import ASRSettings

logger = logging.getLogger(__name__)

AudioInput = Union[bytes, np.ndarray, str]
ASRResult = tuple[str, Optional[str]]


class ASRStreamingSession:
    """Runs Qwen vLLM streaming ASR away from the microphone callback."""

    def __init__(self, client: "ASRClient") -> None:
        self._client = client
        # 4096 x 10 ms covers the 30-second utterance limit without
        # letting a temporarily slow model stall VAD endpoint detection.
        self._queue: queue.Queue[bytes] = queue.Queue(maxsize=4096)
        self._future: Future[ASRResult] = Future()
        self._closed = False
        self._cancel = threading.Event()
        self._finished = threading.Event()
        self._thread = threading.Thread(target=self._run, name="asr-stream", daemon=True)
        with client._sessions_lock:
            if client._closed:
                raise RuntimeError("ASR is closed")
            client._sessions.add(self)
        self._thread.start()

    def push(self, pcm: bytes) -> None:
        if self._future.done() and not self._future.cancelled():
            self._future.result()
        if not self._closed and not self._cancel.is_set() and pcm:
            try:
                self._queue.put_nowait(pcm)
            except queue.Full:
                self.cancel()
                raise BufferError(
                    "ASR input queue overflow; refusing to transcribe incomplete audio"
                )

    def finish(self) -> Future[ASRResult]:
        if not self._closed:
            self._closed = True
            self._finished.set()
        return self._future

    def cancel(self) -> None:
        self._closed = True
        self._cancel.set()
        self._finished.set()
        self._future.cancel()

    def join(self, timeout: float = 10) -> bool:
        self._thread.join(timeout)
        return not self._thread.is_alive()

    def _run(self) -> None:
        try:
            with self._client._inference_lock:
                if self._cancel.is_set():
                    return
                model = self._client.model
                settings = self._client.settings
                state = model.init_streaming_state(
                    context=settings.context,
                    language=settings.language,
                    chunk_size_sec=settings.stream_chunk_seconds,
                )
                pending = bytearray()
                flush_bytes = int(0.2 * 16_000 * 2)
                while True:
                    if self._cancel.is_set():
                        return
                    try:
                        item = self._queue.get(timeout=0.02)
                    except queue.Empty:
                        if self._finished.is_set():
                            break
                        continue
                    pending.extend(item)
                    if len(pending) >= flush_bytes:
                        samples = (
                            np.frombuffer(pending, dtype=np.int16).astype(np.float32) / 32768.0
                        )
                        pending.clear()
                        model.streaming_transcribe(samples, state)
                if pending:
                    samples = np.frombuffer(pending, dtype=np.int16).astype(np.float32) / 32768.0
                    model.streaming_transcribe(samples, state)
                model.finish_streaming_transcribe(state)
            if not self._future.done():
                self._future.set_result((state.text.strip(), state.language or None))
        except Exception as exc:
            if not self._future.done():
                self._future.set_exception(exc)
        finally:
            with self._client._sessions_lock:
                self._client._sessions.discard(self)


class ASRClient:
    def __init__(self, settings: Optional[ASRSettings] = None) -> None:
        import torch
        from qwen_asr import Qwen3ASRModel

        self.settings = settings or ASRSettings()
        self._inference_lock = threading.Lock()
        self._sessions_lock = threading.Lock()
        self._sessions: set[ASRStreamingSession] = set()
        self._closed = False
        logger.info("loading ASR %s with %s backend", self.settings.model, self.settings.backend)
        if self.settings.backend == "vllm":
            try:
                from vllm import ModelRegistry

                # qwen-asr only performs this registration in its server CLI.
                architecture = "Qwen3ASRForConditionalGeneration"
                if architecture not in ModelRegistry.get_supported_archs():
                    ModelRegistry.register_model(
                        architecture,
                        "qwen_asr.core.vllm_backend:Qwen3ASRForConditionalGeneration",
                    )
                self.model = Qwen3ASRModel.LLM(
                    model=self.settings.model,
                    worker_extension_cls="core.asr.vllm_worker.ASRWorkerLifecycle",
                    gpu_memory_utilization=self.settings.vllm_gpu_memory_utilization,
                    max_model_len=self.settings.vllm_max_model_len,
                    max_num_batched_tokens=self.settings.vllm_max_model_len,
                    max_num_seqs=1,
                    max_inference_batch_size=1,
                    max_new_tokens=self.settings.max_new_tokens,
                    enforce_eager=self.settings.vllm_enforce_eager,
                    disable_log_stats=True,
                )
            except ImportError as exc:
                raise RuntimeError("vLLM ASR requires: python -m pip install -e .") from exc
        elif self.settings.backend == "transformers":
            attention = self.settings.attention
            if attention == "auto":
                try:
                    importlib.import_module("flash_attn")
                except (ImportError, OSError):
                    attention = "sdpa"
                else:
                    attention = "flash_attention_2"
            logger.info("using %s attention", attention)
            self.model = Qwen3ASRModel.from_pretrained(
                self.settings.model,
                dtype=torch.bfloat16,
                device_map=self.settings.device,
                attn_implementation=attention,
                max_inference_batch_size=1,
                max_new_tokens=self.settings.max_new_tokens,
            )
            if self.settings.compile:
                self.model.model.forward = torch.compile(
                    self.model.model.forward,
                    mode="reduce-overhead",
                    dynamic=True,
                )
        else:
            raise ValueError(f"unsupported ASR backend: {self.settings.backend}")
        logger.info("ASR loaded")

    @property
    def supports_streaming(self) -> bool:
        return self.settings.backend == "vllm"

    def start_stream(self) -> ASRStreamingSession:
        if not self.supports_streaming:
            raise RuntimeError("streaming ASR requires ASR_BACKEND=vllm")
        return ASRStreamingSession(self)

    def transcribe(self, audio_data: AudioInput, sample_rate: int = 16_000) -> ASRResult:
        import torch

        if isinstance(audio_data, bytes):
            audio = np.frombuffer(audio_data, dtype=np.int16).astype(np.float32) / 32768.0
            model_input = (audio, sample_rate)
        elif isinstance(audio_data, np.ndarray):
            audio = np.asarray(audio_data, dtype=np.float32).reshape(-1)
            model_input = (audio, sample_rate)
        else:
            model_input = audio_data

        with self._inference_lock:
            if self._closed:
                raise RuntimeError("ASR is closed")
            try:
                results = self.model.transcribe(
                    audio=model_input,
                    context=self.settings.context,
                    language=self.settings.language,
                    return_time_stamps=False,
                )
            finally:
                if self.settings.backend == "transformers" and torch.cuda.is_available():
                    # Release transient ASR workspaces before the native TTS CUDA context runs.
                    torch.cuda.empty_cache()
        if not results:
            return "", None
        result = results[0]
        logger.debug("ASR: %s (%s)", result.text, result.language)
        return result.text.strip(), result.language

    def warmup(self, speech_audio: Optional[str] = None) -> None:
        if speech_audio:
            import librosa

            audio, _sample_rate = librosa.load(
                speech_audio,
                sr=16_000,
                mono=True,
                duration=3.0,
            )
        else:
            audio = np.zeros(4_000, dtype=np.float32)
        self.transcribe(audio, 16_000)

    def close(self) -> None:
        import torch

        with self._sessions_lock:
            if self._closed:
                return
            self._closed = True
            sessions = list(self._sessions)
        for session in sessions:
            session.cancel()
        for session in sessions:
            if not session.join():
                raise TimeoutError("streaming ASR worker did not stop")
        with self._inference_lock:
            if self.settings.backend == "vllm":
                engine = self.model.model.llm_engine
                try:
                    engine.collective_rpc("close_mybot_distributed", timeout=10)
                except Exception:
                    logger.warning("ASR distributed cleanup failed", exc_info=True)
                finally:
                    engine.engine_core.shutdown()
            self.model = None
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
