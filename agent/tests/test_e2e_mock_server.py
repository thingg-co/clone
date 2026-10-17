"""Drive the real Client and Agent against a mock OpenAI-compatible server that
streams SSE chunks the way llama.cpp does, including split tool-call deltas and
a final usage chunk."""

import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from frank_local.client import Client
from frank_local.config import Settings
from frank_local.loop import Agent


def sse(obj):
    return f"data: {json.dumps(obj)}\n\n".encode()


def chunk(delta, finish=None, usage=None):
    body = {"id": "x", "object": "chat.completion.chunk", "created": 0, "model": "mock",
            "choices": [{"index": 0, "delta": delta, "finish_reason": finish}]}
    if usage:
        body["usage"] = usage
    return body


class Handler(BaseHTTPRequestHandler):
    requests: list = []
    script: list = []

    def log_message(self, *a):  # silence
        pass

    def _send(self, code, payload, stream=False):
        self.send_response(code)
        self.send_header("Content-Type", "text/event-stream" if stream else "application/json")
        self.end_headers()

    def do_GET(self):
        if self.path.endswith("/models"):
            self._send(200, None)
            self.wfile.write(json.dumps({"data": [{"id": "mock-model"}]}).encode())
        elif self.path.endswith("/props"):
            self._send(200, None)
            self.wfile.write(json.dumps({"default_generation_settings": {"n_ctx": 4096}, "build_info": "mock"}).encode())
        else:
            self._send(404, None)

    def do_POST(self):
        n = int(self.headers.get("Content-Length", 0))
        req = json.loads(self.rfile.read(n))
        Handler.requests.append(req)
        kind, payload = Handler.script.pop(0)
        if kind == "error":
            self._send(400, None)
            self.wfile.write(json.dumps({"error": {"message": payload, "type": "BadRequestError", "code": 400}}).encode())
            return
        if not req.get("stream"):
            self._send(200, None)
            self.wfile.write(json.dumps({"choices": [{"message": {"role": "assistant", "content": payload}}],
                                         "usage": {"prompt_tokens": 1, "completion_tokens": 1}}).encode())
            return
        self._send(200, None, stream=True)
        if kind == "text":
            self.wfile.write(sse(chunk({"role": "assistant", "reasoning_content": "hmm "})))
            for word in payload.split(" "):
                self.wfile.write(sse(chunk({"content": word + " "})))
            self.wfile.write(sse(chunk({}, finish="stop")))
        else:  # tool call split over several chunks, like llama.cpp
            name, args = payload
            self.wfile.write(sse(chunk({"role": "assistant", "reasoning_content": "hmm "})))
            self.wfile.write(sse(chunk({"tool_calls": [{"index": 0, "id": "call_abc", "type": "function", "function": {"name": name, "arguments": ""}}]})))
            text = json.dumps(args)
            for i in range(0, len(text), 7):
                self.wfile.write(sse(chunk({"tool_calls": [{"index": 0, "function": {"arguments": text[i:i + 7]}}]})))
            self.wfile.write(sse(chunk({}, finish="tool_calls")))
        self.wfile.write(sse(chunk({}, usage={"prompt_tokens": 120, "completion_tokens": 30})))
        self.wfile.write(b"data: [DONE]\n\n")


@pytest.fixture
def server():
    srv = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    t = threading.Thread(target=srv.serve_forever, daemon=True)
    t.start()
    Handler.requests = []
    yield f"http://127.0.0.1:{srv.server_port}/v1"
    srv.shutdown()


def test_probe_and_tool_loop(server, tmp_path, ui):
    (tmp_path / "data.txt").write_text("answer=42\n")
    settings = Settings(base_url=server, permission_mode="auto", backend="llama.cpp", reasoning_effort="high", thinking="native")
    client = Client(settings)
    info = client.probe()
    assert info["models"] == ["mock-model"] and info["context"] == 4096 and client.model == "mock-model"

    Handler.script = [
        ("tool", ("read_file", {"path": "data.txt"})),
        ("text", "The answer is 42."),
    ]
    agent = Agent(settings, client, ui, tmp_path, context_window=info["context"])
    out = agent.run_turn("what is the answer?")
    assert out.strip() == "The answer is 42."
    assert agent.last_usage == {"prompt_tokens": 120, "completion_tokens": 30}

    first, second = Handler.requests
    assert first["stream"] is True and first["tools"] and first["chat_template_kwargs"] == {"reasoning_effort": "high"}
    assert [m["role"] for m in second["messages"]] == ["system", "user", "assistant", "tool"]
    assert second["messages"][2]["tool_calls"][0]["id"] == "call_abc"
    assert "answer=42" in second["messages"][3]["content"]
    assert second["messages"][2]["reasoning_content"] == "hmm "


def test_refused_reasoning_effort_is_dropped_and_the_request_retried(server, tmp_path, ui):
    Handler.requests.clear()
    settings = Settings(base_url=server, permission_mode="auto", backend="vllm", reasoning_effort="high", thinking="native")
    client = Client(settings)
    client.probe()
    Handler.script = [
        ("error", "Unexpected reasoning effort high. Supported types are xhigh (default), medium, and low."),
        ("text", "First answer."),
        ("text", "Second answer."),
    ]
    agent = Agent(settings, client, ui, tmp_path, context_window=4096)
    assert agent.run_turn("hello").strip() == "First answer."
    assert agent.run_turn("again").strip() == "Second answer."
    refused, retried, later = Handler.requests
    assert refused["reasoning_effort"] == "high"
    assert "reasoning_effort" not in retried and "reasoning_effort" not in later
