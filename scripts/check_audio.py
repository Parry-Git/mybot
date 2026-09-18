#!/usr/bin/env python3
"""Short full-duplex device check. The launcher enforces an external timeout."""

from __future__ import annotations

import json
import sys
from contextlib import ExitStack
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from core.audio.microphone import Microphone
from core.audio.player import AudioPlayer
from core.config import load_settings


def main() -> None:
    settings = load_settings().audio
    with ExitStack() as resources:
        player = AudioPlayer(settings=settings)
        resources.callback(player.close)
        player.start()
        microphone = resources.enter_context(Microphone(settings))
        count = 0
        for _frame in microphone.stream():
            count += 1
            player.is_playing()
            if count * settings.frame_ms >= 500:
                break
        result = {
            "backend": settings.resolved_backend,
            "input_frames": count,
            "dropped_frames": microphone.dropped_frames,
        }
    print(json.dumps(result))


if __name__ == "__main__":
    main()
