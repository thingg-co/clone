"""The chat server end to end: real worker processes against the mock model server."""

import http.client
import json
import shutil
import subprocess
import threading
import time
from pathlib import Path

from http.server import ThreadingHTTPServer

import pytest

from frank_local.client import Turn
from frank_local.config import Settings
from frank_local.loop import Agent
from frank_local.session import Session
from frank_local.web.server import make_server

from test_e2e_mock_server import Handler
from test_loop import FakeClient


class Slow(Handler):
    """Like the mock server, but a ("slow", seconds) item answers nothing for that long, the
    way a server reading a long prompt does before its first token."""

    def do_POST(self):
        if Handler.script and Handler.script[0][0] == "slow":
            _, seconds = Handler.script.pop(0)
            self.rfile.read(int(self.headers.get("Content-Length", 0)))
            Handler.requests.append({"slow": True})
            end = time.time() + seconds
            while time.time() < end:
                time.sleep(0.05)
            return
        super().do_POST()


@pytest.fixture
def model():
    srv = ThreadingHTTPServer(("127.0.0.1", 0), Slow)
    srv.daemon_threads = True
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{srv.server_port}/v1"
    srv.shutdown()
    srv.server_close()


@pytest.fixture
def web(model, tmp_path):
    server = model
    Handler.requests = []
    project = tmp_path / "project"
    project.mkdir()
    home = tmp_path / "home"
    overrides = {"base_url": server, "api_key": "k", "home": str(home), "backend": "llama.cpp", "thinking": "native"}
    settings = Settings(base_url=server, api_key="k", home=home, web_port=0, backend="llama.cpp")
    httpd, hub = make_server(settings, overrides)
    t = threading.Thread(target=httpd.serve_forever, daemon=True)
    t.start()
    yield {"port": httpd.server_port, "project": project, "home": home, "hub": hub}
    hub.close_all()
    httpd.shutdown()
    httpd.server_close()


def call(w, method, path, body=None, headers=None):
    conn = http.client.HTTPConnection("127.0.0.1", w["port"], timeout=20)
    h = {"X-Frank": "1", "Content-Type": "application/json", **(headers or {})}
    conn.request(method, path, body=None if body is None else json.dumps(body), headers=h)
    r = conn.getresponse()
    data = r.read()
    conn.close()
    return r.status, json.loads(data) if data else None


def events_until_done(w, sid, after=0, timeout=30, on_event=None):
    """Read the chat's event stream until the turn ends; return the events."""
    conn = http.client.HTTPConnection("127.0.0.1", w["port"], timeout=timeout)
    conn.request("GET", f"/api/chats/{sid}/events?after={after}")
    r = conn.getresponse()
    assert r.status == 200
    out, deadline = [], time.time() + timeout
    while time.time() < deadline:
        line = r.fp.readline().decode()
        if line.startswith("data: ") and line.startswith("data: ["):
            for ev in json.loads(line[6:]):
                out.append(ev)
                if on_event:
                    on_event(ev)
                if ev["type"] == "done":
                    conn.close()
                    return out
    conn.close()
    raise AssertionError(f"no done event; got {[e['type'] for e in out]}")


def new_chat(w, text, mode="ask"):
    status, data = call(w, "POST", "/api/chats", {"cwd": str(w["project"]), "text": text, "effort": "high", "mode": mode})
    assert status == 200, data
    return data["id"]


def test_a_chat_from_the_page(web):
    Handler.script = [("text", "Hello from the model.")]
    sid = new_chat(web, "say hello")
    evs = events_until_done(web, sid)
    kinds = [e["type"] for e in evs]
    assert "reasoning" in kinds and "text" in kinds and kinds[-1] == "done"
    recs = [e["record"]["message"] for e in evs if e["type"] == "record"]
    assert recs[0] == {"role": "user", "content": "say hello"}
    assert recs[-1]["content"].strip() == "Hello from the model."
    assert evs[-1]["state"]["window"] == 4096 and evs[-1]["state"]["effort"] == "high"

    _, view = call(web, "GET", f"/api/chats/{sid}")
    assert not view["running"] and [r["type"] for r in view["records"]] == ["meta", "message", "message"]
    _, listing = call(web, "GET", "/api/chats")
    assert [(c["id"], c["title"]) for c in listing["chats"]] == [(sid, "say hello")]
    _, prompts = call(web, "GET", "/api/prompts")
    assert prompts["prompts"][0] == "say hello"  # the terminal's Up-arrow history has it too
    _, state = call(web, "GET", "/api/state")
    assert state["model"] == "mock-model" and str(web["project"]) in state["projects"]


def test_permission_question_and_answer(web):
    Handler.script = [("tool", ("write_file", {"path": "out.txt", "content": "made on the page\n"})), ("text", "Written.")]
    sid = new_chat(web, "write the file")

    def answer(ev):
        if ev["type"] == "permission":
            assert ev["name"] == "write_file"
            assert call(web, "POST", f"/api/chats/{sid}/answer", {"id": ev["id"], "answer": "y"})[0] == 200

    evs = events_until_done(web, sid, on_event=answer)
    assert "answered" in [e["type"] for e in evs]
    assert (web["project"] / "out.txt").read_text() == "made on the page\n"
    calls = [e["record"]["message"] for e in evs if e["type"] == "record" and e["record"]["message"].get("tool_calls")]
    assert calls[0]["tool_calls"][0]["summary"] == "out.txt (17 chars)"  # what the terminal shows


def test_stop_while_the_model_is_still_reading(web):
    Handler.script = [("slow", 20)]
    srv = web["hub"]
    sid = new_chat(web, "a long question")
    time.sleep(1.5)  # the worker is up and waiting on the model
    t0 = time.time()
    assert call(web, "POST", f"/api/chats/{sid}/stop", {})[0] == 200
    events_until_done(web, sid, timeout=15)
    assert time.time() - t0 < 10
    msgs = Session(srv.home / "sessions" / f"{sid}.jsonl").messages()
    assert [m["content"] for m in msgs] == ["a long question", "[interrupted]"]


def test_the_terminal_and_the_page_share_a_session(web, ui):
    Handler.script = [("text", "First answer.")]
    sid = new_chat(web, "from the page")
    events_until_done(web, sid)

    # The terminal resumes it (frank -r ID) and adds a turn.
    session = Session.find(web["home"], sid)
    terminal = Agent(Settings(permission_mode="auto"), FakeClient([Turn(text="Terminal answer.")]), ui, web["project"], context_window=100_000)
    terminal.load_history(session.messages())
    terminal.session = session
    terminal.run_turn("from the terminal")

    # The page's worker sees that turn before its next request.
    Handler.script = [("text", "Second answer.")]
    _, view = call(web, "GET", f"/api/chats/{sid}")
    assert call(web, "POST", f"/api/chats/{sid}/send", {"text": "back on the page"})[0] == 200
    events_until_done(web, sid, after=view["seq"])
    sent = [m["content"] for m in Handler.requests[-1]["messages"][1:]]
    assert sent[2:] == ["from the terminal", "Terminal answer.", "back on the page"]


def test_busy_in_the_terminal(web):
    Handler.script = [("text", "ok")]
    sid = new_chat(web, "hello")
    events_until_done(web, sid)
    with Session(web["home"] / "sessions" / f"{sid}.jsonl").turn("the terminal"):
        status, data = call(web, "POST", f"/api/chats/{sid}/send", {"text": "again"})
        assert status == 409 and "the terminal" in data["error"]
        _, listing = call(web, "GET", "/api/chats")
        assert listing["chats"][0]["holder"].startswith("the terminal")


def test_rename_delete_and_commands(web):
    Handler.script = [("text", "ok")]
    sid = new_chat(web, "hello")
    events_until_done(web, sid)
    _, view = call(web, "GET", f"/api/chats/{sid}")
    assert call(web, "POST", f"/api/chats/{sid}/send", {"text": "/tools"})[0] == 200
    evs = events_until_done(web, sid, after=view["seq"])
    assert any(e["type"] == "note" and "spawn_subagent" in e["text"] for e in evs)

    assert call(web, "POST", f"/api/chats/{sid}/rename", {"title": "Greeting"})[0] == 200
    assert call(web, "GET", "/api/chats")[1]["chats"][0]["title"] == "Greeting"
    assert call(web, "POST", f"/api/chats/{sid}/delete", {})[0] == 200
    assert call(web, "GET", "/api/chats")[1]["chats"] == []
    assert (web["home"] / "sessions" / "trash" / f"{sid}.jsonl").exists()


def test_other_sites_are_turned_away(web):
    assert call(web, "POST", "/api/chats", {"cwd": "/", "text": "x"}, headers={"X-Frank": ""})[0] == 403
    assert call(web, "GET", "/api/chats", headers={"Host": "evil.example:8089"})[0] == 403
    conn = http.client.HTTPConnection("127.0.0.1", web["port"], timeout=5)
    conn.request("GET", "/")
    r = conn.getresponse()
    assert "script-src 'self'" in r.getheader("Content-Security-Policy")
    assert b"chat.js" in r.read()
    conn.close()


@pytest.mark.skipif(shutil.which("node") is None, reason="node is not installed")
def test_markdown_escapes_and_renders():
    js = Path(__file__).resolve().parent.parent / "frank_local" / "web" / "markdown.js"
    script = """
    const M = require(process.argv[1]);
    const cases = process.argv.slice(2).map((s) => M.render(s));
    console.log(JSON.stringify(cases));
    """
    inputs = [
        "<img src=x onerror=alert(1)> **bold** `<code>`",
        "[a](javascript:alert(1)) [b](https://example.com/x?a=1&b=2)",
        "```js\nconst a = '<b>';\n```",
        "| x | y |\n|---|---|\n| <i> | 2 |",
        "- [x] done\n- [ ] todo\n\n1. one\n2. two",
    ]
    out = json.loads(subprocess.run(["node", "-e", script, str(js), *inputs], capture_output=True, text=True, check=True).stdout)
    assert "<img" not in out[0] and "&lt;img" in out[0] and "<strong>bold</strong>" in out[0] and "<code>&lt;code&gt;</code>" in out[0]
    assert "javascript:" not in out[1].replace("[a](javascript:alert(1))", "") and 'href="https://example.com/x?a=1&amp;b=2"' in out[1]
    assert "&lt;b&gt;" in out[2] and '<span>js</span>' in out[2]
    assert "<td>&lt;i&gt;</td>" in out[3]
    assert "☑" in out[4] and "<ol><li>one</li><li>two</li></ol>" in out[4]
