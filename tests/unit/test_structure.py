import ast
import tomllib
from pathlib import Path
from types import SimpleNamespace

import pytest

from core import app
from core.config import PROJECT_ROOT, Settings
from core.web.server import STATIC


def test_checkout_paths_and_packaged_browser_assets():
    root = Path(__file__).resolve().parents[2]
    assert PROJECT_ROOT == root
    assert STATIC == root / "core/web/static"
    for name in ("index.html", "app.js", "audio-worklet.js", "style.css"):
        assert (STATIC / name).is_file()
    metadata = tomllib.loads((root / "pyproject.toml").read_text())
    assert metadata["tool"]["setuptools"]["package-data"]["core.web"] == ["static/*"]
    assert metadata["tool"]["pytest"]["ini_options"]["testpaths"] == ["tests/unit"]


def test_core_never_imports_entry_scripts_or_tests():
    for path in (PROJECT_ROOT / "core").rglob("*.py"):
        tree = ast.parse(path.read_text())
        for node in ast.walk(tree):
            modules = []
            if isinstance(node, ast.Import):
                modules = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom) and node.module:
                modules = [node.module]
            for module in modules:
                assert module.split(".")[0] not in {"main", "scripts", "tests"}, path


@pytest.mark.parametrize("failed_stage", ["tts", "player"])
def test_application_closes_resources_on_partial_initialization(monkeypatch, failed_stage):
    closed = []
    monkeypatch.setattr(
        app, "OpenAILLMClient", lambda _: SimpleNamespace(close=lambda: closed.append("llm"))
    )
    monkeypatch.setattr(
        app,
        "ASRClient",
        lambda _: SimpleNamespace(warmup=lambda _: None, close=lambda: closed.append("asr")),
    )

    def create_tts(_):
        if failed_stage == "tts":
            raise RuntimeError("TTS initialization failed")
        return SimpleNamespace(sample_rate=24000, close=lambda: closed.append("tts"))

    def create_player():
        raise RuntimeError("player initialization failed")

    monkeypatch.setattr(app, "NativeQwenTTSClient", create_tts)
    with pytest.raises(RuntimeError, match="initialization failed"):
        app.MyBot(Settings(), "web", player_factory=create_player)
    assert closed == (["asr", "llm"] if failed_stage == "tts" else ["tts", "asr", "llm"])
