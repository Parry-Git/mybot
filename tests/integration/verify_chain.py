#!/usr/bin/env python3
"""Exercise the real streaming pipeline without browser or microphone hardware."""

from __future__ import annotations

import argparse
import json
import logging
import sys
import threading
import time
import wave
from contextlib import ExitStack
from dataclasses import replace
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import numpy as np
from scipy.signal import resample_poly

from core.asr.asr_client import ASRClient
from core.asr.voice_input import VoiceInput
from core.audio.player import AudioPlayer
from core.config import load_settings
from core.llm.llm_client import OpenAILLMClient
from core.orchestrator import ConversationPipeline
from core.tts.native import NativeQwenTTSClient
from core.types import TurnMetrics

logger = logging.getLogger("verify")


class LoopbackHandler(BaseHTTPRequestHandler):
    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        if not body["messages"][-1]["content"].strip():
            self.send_error(400)
            return
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.end_headers()
        for token in ["链路测试已经通过，", "我已经收到你的语音。", "现在可以继续和我对话。"]:
            payload = {"choices": [{"index": 0, "delta": {"content": token}}]}
            self.wfile.write(("data: " + json.dumps(payload) + "\n\n").encode())
            self.wfile.flush()
        self.wfile.write(b"data: [DONE]\n\n")

    def log_message(self, *_args):
        pass


class RecordingPlayer(AudioPlayer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.recorded = bytearray()

    def enqueue(self, packet, cancel, on_first_playback=None):
        accepted = super().enqueue(packet, cancel, on_first_playback)
        if accepted:
            self.recorded.extend(packet.pcm)
        return accepted


def save_audio(path: Path, pcm: bytes, sample_rate: int) -> None:
    with wave.open(str(path), "wb") as out:
        out.setnchannels(1)
        out.setsampwidth(2)
        out.setframerate(sample_rate)
        out.writeframes(pcm)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--llm", choices=("live", "loopback"), default="live")
    parser.add_argument("--playback", action="store_true")
    parser.add_argument("--rounds", type=int, default=2)
    parser.add_argument("--output", type=Path, default=ROOT / "artifacts" / "verification")
    args = parser.parse_args()
    if args.rounds < 1:
        parser.error("--rounds must be positive")
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
    )
    settings = load_settings()
    settings = replace(
        settings,
        audio=replace(settings.audio, playback_enabled=args.playback),
        vad=replace(settings.vad, barge_in=False),
    )
    args.output.mkdir(parents=True, exist_ok=True)
    report = {"llm_mode": args.llm, "playback": args.playback, "rounds": [], "passed": False}

    with ExitStack() as resources:
        if args.llm == "loopback":
            logger.warning(
                "LOOPBACK LLM: real models and HTTP streaming, but no DeepSeek service validation"
            )
            server = ThreadingHTTPServer(("127.0.0.1", 0), LoopbackHandler)
            threading.Thread(target=server.serve_forever, daemon=True).start()
            resources.callback(server.server_close)
            resources.callback(server.shutdown)
            settings = replace(
                settings,
                llm=replace(
                    settings.llm,
                    api_key="local-test",
                    base_url=f"http://127.0.0.1:{server.server_port}/v1",
                ),
            )
        llm = OpenAILLMClient(settings.llm)
        resources.callback(llm.close)
        probe = "".join(llm.stream_chat([{"role": "user", "content": "请只回复：连接成功。"}]))
        if not probe.strip():
            raise RuntimeError("LLM returned no visible text")
        report["llm_probe"] = probe
        logger.info("LLM probe: %s", probe)

        asr = ASRClient(settings.asr)
        resources.callback(asr.close)
        asr.warmup(str(settings.tts.ref_audio))
        tts = NativeQwenTTSClient(settings.tts)
        resources.callback(tts.close)
        for _packet in tts.stream("你好。", threading.Event()):
            pass

        # Generate a known Chinese utterance, then feed it at microphone cadence.
        source = b"".join(
            packet.pcm for packet in tts.stream("你好，请用一句话介绍你自己。", threading.Event())
        )
        save_audio(args.output / "input.wav", source, tts.sample_rate)
        samples = np.frombuffer(source, dtype="<i2").astype(np.float32)
        resampled = np.clip(resample_poly(samples, 2, 3), -32768, 32767).astype("<i2").tobytes()
        pcm = bytes(16000) + resampled + bytes(32000)
        pcm += bytes((-len(pcm)) % settings.audio.frame_bytes)
        transcripts: list[str] = []
        metrics: list[TurnMetrics] = []
        player = RecordingPlayer(sample_rate=tts.sample_rate, settings=settings.audio)
        resources.callback(player.close)
        player.start()
        pipeline = ConversationPipeline(
            llm,
            tts,
            player,
            settings,
            asr,
            on_user_text=transcripts.append,
            on_metrics=metrics.append,
        )
        resources.callback(pipeline.close)
        voice = VoiceInput(settings, asr, pipeline)
        resources.callback(voice.close)

        for index in range(args.rounds):
            logger.info("round %s: feeding %.2fs audio through VAD", index + 1, len(pcm) / 32000)
            voice.reset()
            player.recorded.clear()
            before_transcripts, before_metrics = len(transcripts), len(metrics)
            futures = []
            start = time.monotonic()
            for offset in range(0, len(pcm), settings.audio.frame_bytes):
                due = start + offset / 32000
                time.sleep(max(0, due - time.monotonic()))
                future = voice.process(pcm[offset : offset + settings.audio.frame_bytes])
                if future is not None:
                    futures.append(future)
            if not futures:
                raise RuntimeError("VAD did not detect a complete utterance")
            answers = [future.result(timeout=90) for future in futures]
            if (
                not any(answers)
                or len(transcripts) == before_transcripts
                or len(metrics) == before_metrics
            ):
                raise RuntimeError("no completed ASR/LLM/TTS turn")
            if len(player.recorded) < 4800:
                raise RuntimeError("TTS produced less than 100 ms audio")
            recognized = "".join(transcripts[before_transcripts:])
            if "介绍" not in recognized:
                raise RuntimeError(f"Chinese ASR roundtrip lost the test intent: {recognized!r}")
            save_audio(args.output / f"response-{index + 1}.wav", player.recorded, tts.sample_rate)
            result = {
                "transcript": recognized,
                "answers": answers,
                "audio_seconds": len(player.recorded) / 48000,
                "metrics": [m.summary() for m in metrics[before_metrics:]],
            }
            report["rounds"].append(result)
            logger.info("round passed: %s", json.dumps(result, ensure_ascii=False))

        # Interrupt a live native generator, then prove its replacement still works.
        cancel = threading.Event()
        stream = tts.stream("这是一段会被打断的语音，用来检查取消之后能否继续正常说话。", cancel)
        next(stream)
        started = time.perf_counter()
        cancel.set()
        stream.close()
        report["tts_cancel_ms"] = (time.perf_counter() - started) * 1000
        recovery = sum(len(packet.pcm) for packet in tts.stream("恢复成功。", threading.Event()))
        if recovery == 0 or report["tts_cancel_ms"] > 2000:
            raise RuntimeError("native cancellation/recovery failed")

        report["passed"] = True

    report["resources_closed"] = True
    (args.output / "report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n"
    )
    logger.info("PASS (%s LLM); report: %s", args.llm, args.output / "report.json")


if __name__ == "__main__":
    main()
