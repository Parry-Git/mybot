from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient
from starlette.websockets import WebSocketDisconnect

from core.config import AudioSettings, Settings
from core.web import server as web_app


@pytest.fixture
def app(monkeypatch):
    state = SimpleNamespace(frames=[], closed=False, resets=0)

    def reset():
        state.resets += 1

    def fake_bot(settings, mode, warmup, player_factory, emit):
        state.player = player_factory()
        return SimpleNamespace(
            pipeline=SimpleNamespace(reset_conversation=reset, interrupt=reset),
            voice_input=SimpleNamespace(reset=reset, process=state.frames.append),
            stop=lambda: setattr(state, "closed", True),
        )

    monkeypatch.setattr(web_app, "MyBot", fake_bot)
    monkeypatch.setattr(web_app, "load_settings", lambda: Settings())
    return web_app.create_app(), state


def test_web_audio_origin_single_client_and_reconnect(app):
    application, state = app
    with TestClient(application) as client:
        assert client.get("/").status_code == 200
        for asset in ("app.js", "audio-worklet.js", "style.css"):
            assert client.get(f"/static/{asset}").status_code == 200
        assert client.get("/health").json()["ready"]
        with pytest.raises(WebSocketDisconnect):
            with client.websocket_connect(
                "/audio", headers={"origin": "https://unrelated.example"}
            ):
                pass
        headers = {"origin": "http://testserver"}
        with client.websocket_connect("/audio", headers=headers) as socket:
            assert socket.receive_json()["type"] == "clear"
            assert socket.receive_json()["type"] == "connected"
            with pytest.raises(WebSocketDisconnect):
                with client.websocket_connect("/audio", headers=headers) as other:
                    other.receive_json()
            socket.send_bytes(bytes(640))
            socket.send_json({"type": "interrupt"})
        assert state.frames == [bytes(640)]
        with client.websocket_connect("/audio", headers=headers) as socket:
            assert socket.receive_json()["type"] == "clear"
    assert state.closed
    assert not state.player._connected


def test_web_rejects_wrong_microphone_frame(app):
    application, state = app
    with TestClient(application) as client:
        with client.websocket_connect("/audio", headers={"origin": "http://testserver"}) as socket:
            socket.receive_json()
            socket.receive_json()
            socket.send_bytes(b"wrong frame")
            with pytest.raises(WebSocketDisconnect):
                socket.receive_json()
    assert state.frames == []


def test_browser_frame_setting_is_validated_before_loading_models(monkeypatch):
    monkeypatch.setattr(
        web_app, "load_settings", lambda: Settings(audio=AudioSettings(frame_ms=10))
    )
    with pytest.raises(ValueError, match="VAD_FRAME_MS"):
        web_app.create_app()
