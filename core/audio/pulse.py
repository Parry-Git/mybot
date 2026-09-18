from __future__ import annotations

import ctypes as C
import threading
import time
from typing import Optional


class _SampleSpec(C.Structure):
    _fields_ = [("format", C.c_int), ("rate", C.c_uint32), ("channels", C.c_uint8)]


class _BufferAttr(C.Structure):
    _fields_ = [
        (name, C.c_uint32) for name in ("maxlength", "tlength", "prebuf", "minreq", "fragsize")
    ]


class PulseStream:
    """PCM16 mono stream using libpulse-simple, bypassing the WSL ALSA bridge.

    All blocking I/O and flushes belong to a single worker per connection.
    ctypes releases the GIL during native calls so CUDA inference can proceed.
    """

    def __init__(
        self,
        sample_rate: int,
        frame_samples: int,
        callback,
        *,
        recording: bool = False,
        device: Optional[str] = None,
    ) -> None:
        self._lib = C.CDLL("libpulse-simple.so.0")
        self._errors = C.CDLL("libpulse.so.0")
        self._errors.pa_strerror.argtypes = [C.c_int]
        self._errors.pa_strerror.restype = C.c_char_p
        self._lib.pa_simple_new.argtypes = [
            C.c_char_p,
            C.c_char_p,
            C.c_int,
            C.c_char_p,
            C.c_char_p,
            C.POINTER(_SampleSpec),
            C.c_void_p,
            C.POINTER(_BufferAttr),
            C.POINTER(C.c_int),
        ]
        self._lib.pa_simple_new.restype = C.c_void_p
        for name in ("pa_simple_read", "pa_simple_write"):
            function = getattr(self._lib, name)
            function.argtypes = [C.c_void_p, C.c_void_p, C.c_size_t, C.POINTER(C.c_int)]
            function.restype = C.c_int
        self._lib.pa_simple_flush.argtypes = [C.c_void_p, C.POINTER(C.c_int)]
        self._lib.pa_simple_flush.restype = C.c_int
        self._lib.pa_simple_get_latency.argtypes = [C.c_void_p, C.POINTER(C.c_int)]
        self._lib.pa_simple_get_latency.restype = C.c_uint64
        self._lib.pa_simple_free.argtypes = [C.c_void_p]
        self._lib.pa_simple_free.restype = None
        self.recording = recording
        self.sample_rate = sample_rate
        self.frame_samples = frame_samples
        self.callback = callback
        self.latency = 0.04
        spec = _SampleSpec(3, sample_rate, 1)  # PA_SAMPLE_S16LE
        frame_bytes = frame_samples * 2
        attrs = _BufferAttr(
            sample_rate * 2 // 10, sample_rate * 2 // 25, 0, frame_bytes, frame_bytes
        )
        error = C.c_int()
        self._handle = self._lib.pa_simple_new(
            None,
            b"MyBot",
            2 if recording else 1,
            device.encode() if device else None,
            b"microphone" if recording else b"speech",
            C.byref(spec),
            None,
            C.byref(attrs),
            C.byref(error),
        )
        if not self._handle:
            raise RuntimeError(self._error_text(error))
        self._stop = threading.Event()
        self._flush = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self.error: Optional[Exception] = None
        self.last_operation = "idle"

    def _error_text(self, error: C.c_int) -> str:
        return "PulseAudio: " + self._errors.pa_strerror(error.value).decode()

    def _check(self, result: int, error: C.c_int) -> None:
        if result < 0:
            raise RuntimeError(self._error_text(error))

    def start(self) -> None:
        if self._thread is not None:
            return
        self._thread = threading.Thread(
            target=self._run, name="pulse-input" if self.recording else "pulse-output", daemon=True
        )
        self._thread.start()

    def _run(self) -> None:
        frame_bytes = self.frame_samples * 2
        data = bytearray(frame_bytes)
        buffer = (C.c_char * frame_bytes).from_buffer(data)
        error = C.c_int()
        next_frame = time.monotonic()
        frame_seconds = self.frame_samples / self.sample_rate
        try:
            while not self._stop.is_set():
                if self.recording:
                    self.last_operation = "read"
                    self._check(
                        self._lib.pa_simple_read(self._handle, buffer, frame_bytes, C.byref(error)),
                        error,
                    )
                    if not self._stop.is_set():
                        self.callback(data, self.frame_samples, None, None)
                else:
                    # WSL's RDP sink can accept audio ahead of real time. Pacing
                    # prevents that hidden transport buffer from adding seconds.
                    if self._stop.wait(max(0, next_frame - time.monotonic())):
                        break
                    if self._flush.is_set():
                        self._flush.clear()
                        self.last_operation = "flush"
                        self._check(self._lib.pa_simple_flush(self._handle, C.byref(error)), error)
                    self.callback(data, self.frame_samples, None, None)
                    self.last_operation = "write"
                    self._check(
                        self._lib.pa_simple_write(
                            self._handle, buffer, frame_bytes, C.byref(error)
                        ),
                        error,
                    )
                    next_frame = max(next_frame + frame_seconds, time.monotonic())
        except Exception as exc:
            self.error = exc

    def check(self) -> None:
        if self.error is not None:
            raise RuntimeError("audio worker failed") from self.error

    def flush(self) -> None:
        self._flush.set()

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=3)
            if self._thread.is_alive():
                raise TimeoutError(
                    f"PulseAudio {'input' if self.recording else 'output'} did not stop during {self.last_operation}"
                )
            self._thread = None

    def close(self) -> None:
        self.stop()
        if self._handle:
            self._lib.pa_simple_free(self._handle)
            self._handle = None
