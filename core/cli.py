"""Foreground text and direct-device diagnostics."""

from __future__ import annotations

import argparse
import logging
import signal
from dataclasses import replace
from typing import Optional

from core.app import MyBot
from core.config import load_settings


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Low-latency streaming voice assistant")
    parser.add_argument("--mode", choices=("voice", "text"), default="voice")
    parser.add_argument("--asr-backend", choices=("transformers", "vllm"))
    parser.add_argument("--no-playback", action="store_true")
    parser.add_argument("--skip-asr-warmup", action="store_true")
    parser.add_argument("--log-level", default="INFO")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper()),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    settings = load_settings()
    if args.asr_backend:
        settings = replace(settings, asr=replace(settings.asr, backend=args.asr_backend))
    if args.no_playback:
        settings = replace(
            settings,
            audio=replace(settings.audio, playback_enabled=False),
        )

    bot: Optional[MyBot] = None

    def handle_signal(_signum, _frame) -> None:
        # Resource teardown belongs in finally blocks, never inside a signal handler.
        raise KeyboardInterrupt

    signal.signal(signal.SIGINT, handle_signal)
    signal.signal(signal.SIGTERM, handle_signal)
    try:
        bot = MyBot(settings, args.mode, warmup_asr=not args.skip_asr_warmup)
        if args.mode == "voice":
            bot.run_voice()
        else:
            bot.run_text()
    except KeyboardInterrupt:
        pass
    finally:
        if bot is not None:
            bot.stop()
