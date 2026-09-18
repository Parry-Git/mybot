#!/usr/bin/env python3
"""Feed a known WAV through Chromium's microphone into the real running service."""

from __future__ import annotations

import argparse
import json
import sys
import time
import wave
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from playwright.sync_api import sync_playwright


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://127.0.0.1:8765")
    parser.add_argument("--input", type=Path, default=ROOT / "artifacts/verification/input.wav")
    parser.add_argument("--rounds", type=int, default=2)
    args = parser.parse_args()
    if args.rounds < 1:
        parser.error("--rounds must be positive")
    if not args.input.is_file():
        parser.error("input WAV missing; run tests/integration/verify_chain.py first")
    output = ROOT / "artifacts/verification"
    output.mkdir(parents=True, exist_ok=True)
    fixture = output / "browser-input.wav"
    with wave.open(str(args.input), "rb") as source, wave.open(str(fixture), "wb") as target:
        if source.getnchannels() != 1 or source.getsampwidth() != 2:
            parser.error("input must be mono PCM16 WAV")
        rate = source.getframerate()
        target.setnchannels(1)
        target.setsampwidth(2)
        target.setframerate(rate)
        target.writeframes(
            bytes(rate) + source.readframes(source.getnframes()) + bytes(rate * 2 * 16)
        )
    metrics = []
    errors = []
    pcm_packets = []
    clears = []
    with sync_playwright() as p:
        browser = p.chromium.launch(
            headless=True,
            args=[
                "--use-fake-ui-for-media-stream",
                "--use-fake-device-for-media-stream",
                f"--use-file-for-fake-audio-capture={fixture}",
                "--autoplay-policy=no-user-gesture-required",
                "--disable-audio-output",
                "--mute-audio",
            ],
        )
        context = browser.new_context(
            permissions=["microphone"], viewport={"width": 1280, "height": 900}
        )
        page = context.new_page()
        page.on("pageerror", lambda error: errors.append(str(error)))

        def on_frame(data):
            if isinstance(data, bytes):
                pcm_packets.append(data[4:])
            else:
                message = json.loads(data)
                if message.get("type") == "metrics":
                    metrics.append(message["value"])
                if message.get("type") == "clear":
                    clears.append(time.monotonic())
                if message.get("type") == "error":
                    errors.append(message["value"])
                    print(f"service error: {message['value']}", flush=True)

        page.on("websocket", lambda socket: socket.on("framereceived", on_frame))
        page.goto(args.url)
        page.get_by_role("button", name="开启麦克风").click()
        page.wait_for_function(
            "document.querySelector('#status').textContent.includes('正在聆听')", timeout=15000
        )
        for index in range(args.rounds):
            page.wait_for_function(
                "n => document.querySelectorAll('.entry.assistant').length >= n",
                arg=index + 1,
                timeout=90000,
            )
            deadline = time.monotonic() + 60
            while len(metrics) <= index and time.monotonic() < deadline:
                page.wait_for_timeout(100)
            if len(metrics) <= index:
                raise RuntimeError("browser did not acknowledge rendered audio")
            print(
                f"round {index + 1}: {json.dumps(metrics[index], ensure_ascii=False)}", flush=True
            )
        transcripts = page.locator(".entry.user p").all_text_contents()
        replies = page.locator(".entry.assistant p").all_text_contents()
        if not all("介绍" in text for text in transcripts[: args.rounds]):
            raise RuntimeError(f"ASR changed the test intent: {transcripts}")
        page.screenshot(path=str(output / "browser-desktop.png"), full_page=True)
        page.get_by_role("button", name="关闭麦克风").click()
        page.wait_for_function("document.querySelector('#session-state').textContent === '未连接'")
        page.get_by_role("button", name="开启麦克风").click()
        page.wait_for_function(
            "document.querySelector('#status').textContent.includes('正在聆听')", timeout=15000
        )
        count = len(pcm_packets)
        deadline = time.monotonic() + 60
        while len(pcm_packets) == count and time.monotonic() < deadline:
            page.wait_for_timeout(20)
        if len(pcm_packets) == count:
            raise RuntimeError("reconnected session produced no audio to interrupt")
        count = len(clears)
        interrupted_at = time.monotonic()
        page.get_by_role("button", name="打断当前回复").click()
        deadline = time.monotonic() + 2
        while len(clears) == count and time.monotonic() < deadline:
            page.wait_for_timeout(10)
        if len(clears) == count:
            raise RuntimeError("interrupt did not clear browser playback")
        interrupt_ms = (clears[-1] - interrupted_at) * 1000
        page.wait_for_timeout(500)
        count = len(metrics)
        page.get_by_role("button", name="关闭麦克风").click()
        page.wait_for_function("document.querySelector('#session-state').textContent === '未连接'")
        page.get_by_role("button", name="开启麦克风").click()
        page.wait_for_function(
            "document.querySelector('#status').textContent.includes('正在聆听')", timeout=15000
        )
        deadline = time.monotonic() + 90
        while len(metrics) == count and time.monotonic() < deadline:
            page.wait_for_timeout(100)
        if len(metrics) == count:
            raise RuntimeError("conversation did not recover after interruption")
        page.get_by_role("button", name="关闭麦克风").click()
        page.wait_for_function("document.querySelector('#session-state').textContent === '未连接'")
        page.set_viewport_size({"width": 390, "height": 844})
        page.screenshot(path=str(output / "browser-mobile.png"), full_page=True)
        if page.evaluate("document.documentElement.scrollWidth > innerWidth"):
            raise RuntimeError("mobile layout overflows horizontally")
        browser.close()
    if errors:
        raise RuntimeError(f"browser errors: {errors}")
    if not pcm_packets or any(m["playback_first_audio_ms"] is None for m in metrics):
        raise RuntimeError("no rendered TTS audio")
    with wave.open(str(output / "browser-response.wav"), "wb") as response:
        response.setnchannels(1)
        response.setsampwidth(2)
        response.setframerate(24000)
        response.writeframes(b"".join(pcm_packets))
    report = {
        "passed": True,
        "transcripts": transcripts,
        "replies": replies,
        "metrics": metrics,
        "reconnect_passed": True,
        "interrupt_ms": round(interrupt_ms, 1),
        "interrupt_recovery_passed": True,
        "browser_audio_rendered": True,
        "physical_speaker_verified": False,
    }
    (output / "browser-report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n"
    )
    print(f"PASS: {output / 'browser-report.json'}")


if __name__ == "__main__":
    main()
