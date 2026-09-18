from __future__ import annotations

import threading
from typing import Iterator, Protocol

from core.types import AudioPacket


class StreamingTTS(Protocol):
    def stream(self, text: str, cancel: threading.Event) -> Iterator[AudioPacket]: ...

    def stop(self) -> None: ...

    def close(self) -> None: ...
