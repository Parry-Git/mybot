from __future__ import annotations

import threading
from collections.abc import Callable
from typing import Protocol

from core.types import AudioPacket


class Playback(Protocol):
    """Transport boundary used by the conversation pipeline."""

    def start(self) -> None: ...

    def enqueue(
        self,
        packet: AudioPacket,
        cancel: threading.Event,
        on_first_playback: Callable[[], None] | None = None,
    ) -> bool: ...

    def is_playing(self) -> bool: ...

    def clear(self) -> None: ...

    def close(self) -> None: ...
