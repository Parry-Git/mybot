import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from core.config import LLMSettings
from core.llm.llm_client import OpenAILLMClient


def test_cancellation_closes_a_stalled_http_token_stream():
    release = threading.Event()
    first_token = threading.Event()

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            self.rfile.read(int(self.headers["Content-Length"]))
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.end_headers()
            chunk = {"choices": [{"index": 0, "delta": {"content": "hello"}}]}
            self.wfile.write(("data: " + json.dumps(chunk) + "\n\n").encode())
            self.wfile.flush()
            release.wait(timeout=5)

        def log_message(self, *_args):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    client = OpenAILLMClient(
        LLMSettings(
            api_key="local-test",
            timeout_seconds=10,
            base_url=f"http://127.0.0.1:{server.server_port}/v1",
        )
    )
    cancel = threading.Event()
    errors = []

    def consume():
        try:
            for _token in client.stream_chat([{"role": "user", "content": "test"}], cancel=cancel):
                first_token.set()
        except Exception as exc:
            errors.append(exc)

    thread = threading.Thread(target=consume, daemon=True)
    thread.start()
    try:
        assert first_token.wait(timeout=2)
        cancel.set()
        thread.join(timeout=1)
        assert not thread.is_alive(), "a cancelled turn must not wait for the server's next token"
        assert not errors
    finally:
        release.set()
        thread.join(timeout=3)
        client.close()
        server.shutdown()
        server.server_close()
