from __future__ import annotations

import asyncio
import json
import logging
import os
import queue
from contextlib import asynccontextmanager, suppress
from pathlib import Path

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from core.app import MyBot
from core.audio.browser import BrowserAudioPlayer
from core.config import load_settings

logger = logging.getLogger("mybot")
STATIC = Path(__file__).resolve().parent / "static"


def create_app() -> FastAPI:
    settings = load_settings()
    if settings.audio.frame_ms != 20:
        raise ValueError("browser mode requires VAD_FRAME_MS=20")
    player = BrowserAudioPlayer(settings.audio.playback_queue_seconds)

    @asynccontextmanager
    async def lifespan(app):
        bot = await asyncio.to_thread(MyBot, settings, "web", True, lambda: player, player.emit)
        app.state.bot = bot
        logger.info("web mode ready")
        try:
            yield
        finally:
            await asyncio.to_thread(bot.stop)

    app = FastAPI(lifespan=lifespan, docs_url=None, redoc_url=None)
    app.mount("/static", StaticFiles(directory=STATIC), name="static")

    @app.get("/")
    async def index():
        return FileResponse(STATIC / "index.html", headers={"Cache-Control": "no-store"})

    @app.get("/health")
    async def health():
        return {"ready": True, "barge_in": settings.vad.barge_in, "pid": os.getpid()}

    @app.websocket("/audio")
    async def audio(websocket: WebSocket):
        # Prevent unrelated websites from opening a microphone session on localhost.
        origin = websocket.headers.get("origin")
        expected = f"http://{websocket.headers.get('host')}"
        if origin != expected:
            await websocket.close(code=1008)
            return
        await websocket.accept()
        if not player.attach():
            await websocket.close(code=1008, reason="another tab is already connected")
            return
        bot = app.state.bot
        bot.voice_input.reset()
        bot.pipeline.reset_conversation()
        player.emit("connected", {"barge_in": settings.vad.barge_in})

        async def send():
            while True:
                try:
                    item = player.outgoing.get_nowait()
                except queue.Empty:
                    await asyncio.sleep(0.005)
                    continue
                if isinstance(item, bytes):
                    await websocket.send_bytes(item)
                else:
                    await websocket.send_json(item)

        sender = asyncio.create_task(send())

        connected = True

        def check_turn(future):
            if connected and not future.cancelled() and future.exception() is not None:
                player.emit("error", "本轮处理失败，请查看服务日志后重试。")

        try:
            while True:
                message = await websocket.receive()
                if message["type"] == "websocket.disconnect":
                    break
                if sender.done():
                    sender.result()
                pcm = message.get("bytes")
                if pcm is not None:
                    if len(pcm) != settings.audio.frame_bytes:
                        raise ValueError("incorrect microphone frame size")
                    future = bot.voice_input.process(pcm)
                    if future is not None:
                        future.add_done_callback(check_turn)
                elif message.get("text"):
                    value = json.loads(message["text"])
                    if value.get("type") == "played":
                        player.acknowledge(int(value["generation"]), int(value["samples"]))
                    elif value.get("type") == "interrupt":
                        bot.pipeline.interrupt()
                        bot.voice_input.reset()
        except WebSocketDisconnect:
            pass
        except Exception:
            logger.exception("browser audio connection failed")
        finally:
            connected = False
            bot.pipeline.interrupt()
            bot.voice_input.reset()
            player.detach()
            sender.cancel()
            with suppress(asyncio.CancelledError, WebSocketDisconnect, RuntimeError):
                await sender
            with suppress(RuntimeError):
                await websocket.close()

    return app
