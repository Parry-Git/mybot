#!/usr/bin/env python3
"""Start, inspect and gracefully stop the local voice assistant."""

from __future__ import annotations

import argparse
import fcntl
import getpass
import json
import os
import signal
import socket
import subprocess
import sys
import time
from dataclasses import replace
from pathlib import Path
from urllib.request import urlopen

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
ARTIFACTS = ROOT / "artifacts"
LOCK = ARTIFACTS / "mybot.lock"
LOG = ARTIFACTS / "mybot.log"


def web_ready(port: int, pid: int) -> bool:
    try:
        with urlopen(f"http://127.0.0.1:{port}/health", timeout=0.5) as response:
            status = json.load(response)
        return status.get("ready") is True and status.get("pid") == pid
    except (OSError, ValueError):
        return False


def active_pid() -> int | None:
    if not LOCK.exists():
        return None
    with LOCK.open("r+") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            value = lock.read().strip()
            return int(value) if value else None
    return None


def validate_llm() -> None:
    from dotenv import set_key

    from core.config import load_settings
    from core.llm.llm_client import OpenAILLMClient

    settings = load_settings()
    key = settings.llm.api_key
    entered = False
    if not key:
        if not sys.stdin.isatty():
            raise RuntimeError("请先在 .env 配置 DEEPSEEK_API_KEY，或在交互终端运行 ./run.sh start")
        key = getpass.getpass("请输入新的 DeepSeek API Key（输入不回显）: ").strip()
        if not key:
            raise RuntimeError("未输入 API Key")
        entered = True
    client = OpenAILLMClient(replace(settings.llm, api_key=key, max_output_tokens=32))
    print("正在验证 LLM 连接...", flush=True)
    try:
        text = "".join(client.stream_chat([{"role": "user", "content": "请只回复：连接成功。"}]))
        if not text.strip():
            raise RuntimeError("LLM 没有返回可朗读的内容")
    finally:
        client.close()
    if entered:
        path = ROOT / ".env"
        if not path.exists():
            path.touch(mode=0o600)
        path.chmod(0o600)
        set_key(str(path), "DEEPSEEK_API_KEY", key)
        print("密钥已写入本地 .env（权限 600，Git 忽略）。", flush=True)
    os.environ["DEEPSEEK_API_KEY"] = key
    print("LLM 流式连接通过。", flush=True)


def start(wait_seconds: float, mode: str = "web", port: int = 8765) -> None:
    ARTIFACTS.mkdir(exist_ok=True)
    with LOCK.open("a+") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            print(f"MyBot 已在运行，PID={active_pid()}。日志：{LOG}")
            return
        if mode == "web":
            with socket.socket() as probe:
                try:
                    probe.bind(("127.0.0.1", port))
                except OSError as exc:
                    raise RuntimeError(
                        f"端口 {port} 已被占用，请关闭占用程序或使用 --port 指定其他端口"
                    ) from exc
        if mode == "voice":
            print("正在检查麦克风和播放设备...", flush=True)
            try:
                audio = subprocess.run(
                    [sys.executable, str(ROOT / "scripts" / "check_audio.py")],
                    cwd=ROOT,
                    capture_output=True,
                    text=True,
                    timeout=15,
                )
            except subprocess.TimeoutExpired as exc:
                raise RuntimeError("音频设备无响应；请改用默认浏览器模式 ./run.sh start") from exc
            if audio.returncode:
                raise RuntimeError("音频设备检查失败：\n" + audio.stderr[-2000:])
            print("音频设备通过：" + audio.stdout.strip(), flush=True)
        validate_llm()
        if LOG.exists():
            LOG.replace(LOG.with_suffix(".previous.log"))
        child_env = dict(os.environ, MYBOT_LOCK_FD=str(lock.fileno()), PYTHONUNBUFFERED="1")
        with LOG.open("w") as log:
            process = subprocess.Popen(
                [
                    sys.executable,
                    str(Path(__file__).resolve()),
                    "_run",
                    "--mode",
                    mode,
                    "--port",
                    str(port),
                ],
                cwd=ROOT,
                env=child_env,
                stdin=subprocess.DEVNULL,
                stdout=log,
                stderr=subprocess.STDOUT,
                start_new_session=True,
                pass_fds=(lock.fileno(),),
            )
        lock.seek(0)
        lock.truncate()
        lock.write(str(process.pid))
        lock.flush()
    print(f"MyBot 已在后台启动，PID={process.pid}。正在加载模型和预热...", flush=True)
    deadline = time.monotonic() + wait_seconds
    while time.monotonic() < deadline:
        if process.poll() is not None:
            tail = LOG.read_text(errors="replace").splitlines()[-16:]
            raise RuntimeError("启动失败：\n" + "\n".join(tail))
        if (
            web_ready(port, process.pid)
            if mode == "web"
            else "voice mode ready" in LOG.read_text(errors="replace")
        ):
            print(
                f"MyBot 已就绪，打开 http://127.0.0.1:{port} 开启麦克风。"
                if mode == "web"
                else f"MyBot 已就绪，可以直接说话。日志：{LOG}"
            )
            return
        time.sleep(1)
    print(f"模型仍在后台加载，请运行 ./run.sh status 或 ./run.sh logs。日志：{LOG}")


def stop() -> None:
    pid = active_pid()
    if pid is None:
        print("MyBot 未运行。")
        return
    try:
        os.kill(pid, signal.SIGTERM)
    except ProcessLookupError:
        return
    deadline = time.monotonic() + 30
    while active_pid() is not None and time.monotonic() < deadline:
        time.sleep(0.2)
    if active_pid() is not None:
        raise RuntimeError(f"PID={pid} 仍在清理资源，请查看 {LOG}")
    print("MyBot 已停止，GPU 和音频资源已释放。")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "action", nargs="?", default="start", choices=("start", "stop", "status", "logs", "_run")
    )
    parser.add_argument(
        "--wait", type=float, default=240, help="seconds to wait for background readiness"
    )
    parser.add_argument("--mode", choices=("web", "voice"), default="web")
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args()
    if args.action == "_run":
        # Keep the inherited advisory lock for the entire application lifetime.
        lock_fd = int(os.environ.pop("MYBOT_LOCK_FD"))
        os.set_inheritable(lock_fd, False)
        try:
            if args.mode == "web":
                import logging

                import uvicorn

                from core.web.server import create_app

                logging.basicConfig(
                    level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
                )
                uvicorn.run(
                    create_app(),
                    host="127.0.0.1",
                    port=args.port,
                    access_log=False,
                    ws_max_size=4096,
                    ws_max_queue=64,
                )
            else:
                from core.cli import main as run_bot

                sys.argv = [str(ROOT / "main.py"), "--mode", "voice"]
                run_bot()
        finally:
            os.close(lock_fd)
        return
    if args.action == "start":
        start(args.wait, args.mode, args.port)
    elif args.action == "stop":
        stop()
    elif args.action == "status":
        pid = active_pid()
        ready = LOG.exists() and "mode ready" in LOG.read_text(errors="replace")
        print(f"MyBot {'已就绪' if ready else '加载中'}，PID={pid}" if pid else "MyBot 未运行。")
    elif args.action == "logs":
        if not LOG.exists():
            raise RuntimeError("暂无日志，请先运行 ./run.sh start")
        os.execvp("tail", ["tail", "-n", "60", "-f", str(LOG)])


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n已退出当前命令；后台任务可用 ./run.sh status 查看。")
    except Exception as exc:
        print(f"{type(exc).__name__}: {exc}", file=sys.stderr)
        sys.exit(1)
