"""frank --web: the chat page, on the same sessions, settings and prompt history as the terminal.

This process serves the page, reads the session files for it, and relays between the
browser and one worker process per chat in use (worker.py), which runs the agent.

There is no login. Anyone who can reach the port can have the agent run commands as
this user, so it listens on 127.0.0.1 unless --host or web_host says otherwise. The Host
check and the X-Frank header only stop other web sites from using the page through the
browser of someone who has it open.
"""

from __future__ import annotations

import json
import os
import re
import secrets
import signal
import subprocess
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlparse

from prompt_toolkit.history import FileHistory

from .. import __version__, thinking
from ..client import Client
from ..config import Settings
from ..session import Session, load_titles, save_title
from .worker import for_page

HERE = Path(__file__).resolve().parent
STATIC = {
    "/": ("page.html", "text/html"),
    "/chat.js": ("chat.js", "text/javascript"),
    "/markdown.js": ("markdown.js", "text/javascript"),
}
CSP = ("default-src 'none'; script-src 'self'; style-src 'self' 'unsafe-inline'; connect-src 'self'; "
       "img-src 'self' data:; base-uri 'none'; form-action 'none'; frame-ancestors 'none'")
SID = re.compile(r"^[0-9A-Za-z][0-9A-Za-z._-]{0,80}$")
IDLE_SECONDS = 1800   # a worker idle this long exits; the next message starts another from the file
MAX_BODY = 4 << 20
BOOT = secrets.token_hex(4)  # tells the page the server restarted and its view is stale


class Conflict(Exception):
    """Not now: a turn is running, or the session is busy elsewhere."""


class Chat:
    """A session open on the page: its worker process and the events of the turn in progress."""

    def __init__(self, hub: "Hub", session: Session):
        self.hub = hub
        self.session = session
        self.cond = threading.Condition()
        self.proc: subprocess.Popen | None = None
        self.events: list[dict[str, Any]] = []
        self.seq = 0
        self.running = False
        self.turn_start = 0  # records in the file when the turn began; the page replays the turn after them
        self.pending: dict[str, Any] | None = None
        self.state: dict[str, Any] = {}
        self.touched = time.time()
        self.started = 0.0  # when the turn in progress began

    def _publish(self, event: dict[str, Any]) -> None:  # with self.cond held
        self.seq += 1
        event["seq"] = self.seq
        self.events.append(event)
        self.cond.notify_all()

    def _start_worker(self, overrides: dict[str, Any]) -> None:
        cwd = self.session.meta().get("cwd") or ""
        if not cwd or not Path(cwd).is_dir():
            raise Conflict(f"the project folder {cwd or '(none)'} is not there any more")
        spec = {"session": str(self.session.path), "cwd": cwd, "overrides": overrides}
        # Its own process group, so Stop reaches the shell commands it runs, as Ctrl-C does.
        self.proc = subprocess.Popen(
            [sys.executable, "-m", "frank_local.web.worker", json.dumps(spec)],
            cwd=cwd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, start_new_session=True,
        )
        threading.Thread(target=self._pump, args=(self.proc,), daemon=True).start()

    def _pump(self, proc: subprocess.Popen) -> None:
        assert proc.stdout is not None
        for line in proc.stdout:
            try:
                ev = json.loads(line)
            except ValueError:
                continue
            with self.cond:
                kind = ev.get("type")
                if kind == "permission":
                    self.pending = ev
                elif kind == "answered":
                    self.pending = None
                elif kind == "done":
                    self.running = False
                    self.pending = None
                    self.state = ev.get("state") or self.state
                    self.touched = time.time()
                self._publish(ev)
        code = proc.wait()
        with self.cond:
            if self.proc is proc:
                self.proc = None
            if self.running:
                self.running = False
                self.pending = None
                self._publish({"type": "note", "level": "error", "text": f"the agent's process exited (code {code})"})
                self._publish({"type": "done", "state": self.state})

    def _write(self, obj: dict[str, Any]) -> None:
        assert self.proc is not None and self.proc.stdin is not None
        self.proc.stdin.write((json.dumps(obj, ensure_ascii=False) + "\n").encode("utf-8"))
        self.proc.stdin.flush()

    def submit(self, op: dict[str, Any], overrides: dict[str, Any]) -> None:
        with self.cond:
            if self.running:
                raise Conflict("a turn is already running in this chat")
            if holder := self.session.holder():
                raise Conflict(f"a turn is running in this session in {holder}")
            if self.proc is None or self.proc.poll() is not None:
                self._start_worker(overrides)
            self.events = []
            self.turn_start = len(self.session.records())
            self.running = True
            self.pending = None
            self.touched = self.started = time.time()
            try:
                self._write(op)
            except OSError:
                self.running = False
                raise Conflict("the agent's process went away; send again") from None

    def answer(self, qid: Any, answer: str) -> None:
        with self.cond:
            if not self.pending or self.pending.get("id") != qid or self.proc is None:
                raise Conflict("that question is no longer open")
            try:
                self._write({"op": "answer", "id": qid, "answer": answer})
            except OSError:
                raise Conflict("the agent's process went away") from None

    def stop(self) -> None:
        with self.cond:
            if self.running and self.proc is not None:
                try:
                    os.killpg(self.proc.pid, signal.SIGINT)
                except (ProcessLookupError, PermissionError):
                    pass

    def close(self) -> None:
        with self.cond:
            proc, self.proc = self.proc, None
        if proc is None:
            return
        try:
            if proc.stdin:
                proc.stdin.close()  # the worker exits when its input ends
            proc.wait(timeout=5)
        except (OSError, subprocess.TimeoutExpired):
            proc.kill()

    def view(self) -> dict[str, Any]:
        """The records to show and, during a turn, the events so far: the page draws the
        records from before the turn and replays the turn on top."""
        with self.cond:
            records = self.session.records()
            if self.running:
                records = records[: self.turn_start]
            return {
                "id": self.session.id,
                "records": [for_page(r) for r in records],
                "events": list(self.events) if self.running else [],
                "seq": self.seq,
                "running": self.running,
                "started": self.started if self.running else None,
                "state": self.state,
                "holder": None if self.running else self.session.holder(),
                "boot": BOOT,
            }


class Hub:
    """All chats, the session list and what the page needs to start a new one."""

    def __init__(self, settings: Settings, overrides: dict[str, Any]):
        self.settings = settings
        self.overrides = overrides
        self.home = settings.home
        self.chats: dict[str, Chat] = {}
        self.lock = threading.Lock()
        self._probed: tuple[float, dict[str, Any]] = (0.0, {})
        self.home.mkdir(parents=True, exist_ok=True)
        self.history = FileHistory(str(self.home / "history"))  # the terminal's Up-arrow history

    def chat(self, sid: str) -> Chat:
        if not SID.match(sid):
            raise KeyError(sid)
        with self.lock:
            if sid in self.chats:
                return self.chats[sid]
            path = self.home / "sessions" / f"{sid}.jsonl"
            if not path.is_file():
                raise KeyError(sid)
            chat = self.chats[sid] = Chat(self, Session(path))
            return chat

    def create(self, cwd: str) -> Chat:
        folder = Path(cwd).expanduser()
        if not folder.is_absolute() or not folder.is_dir():
            raise ValueError(f"not a folder: {cwd}")
        session = Session.new(self.home, folder.resolve(), self.probe().get("model") or "")
        with self.lock:
            chat = self.chats[session.id] = Chat(self, session)
        return chat

    def send(self, chat: Chat, text: str, effort: Any, mode: Any) -> None:
        text = text.strip()
        if not text:
            raise ValueError("nothing to send")
        overrides = dict(self.overrides)
        if effort in thinking.LEVELS:
            overrides["reasoning_effort"] = effort
        if mode in ("ask", "auto", "readonly"):
            overrides["permission_mode"] = mode
        op = {"op": "command", "line": text} if text.startswith("/") else {"op": "send", "text": text}
        chat.submit({**op, "effort": effort, "mode": mode}, overrides)
        try:
            self.history.store_string(text)
        except OSError:
            pass

    def delete(self, chat: Chat) -> None:
        if chat.running:
            raise Conflict("stop the turn first")
        if holder := chat.session.holder():
            raise Conflict(f"a turn is running in this session in {holder}")
        chat.close()
        chat.session.trash()
        save_title(self.home, chat.session.id, "")
        with self.lock:
            self.chats.pop(chat.session.id, None)

    def probe(self) -> dict[str, Any]:
        at, info = self._probed
        if time.time() - at < 5:
            return info
        client = Client(self.settings)
        info = client.probe()
        info["model"] = client.model
        self._probed = (time.time(), info)
        return info

    def listing(self) -> list[dict[str, Any]]:
        titles = load_titles(self.home)
        out = []
        for s in Session.list_all(self.home):
            try:
                info = s.summary(titles)
            except OSError:
                continue
            if not info["title"]:
                continue  # started and left without a word
            chat = self.chats.get(s.id)
            info["running"] = bool(chat and chat.running)
            info["holder"] = None if info["running"] else s.holder()
            out.append(info)
            if len(out) >= 300:
                break
        return out

    def projects(self, listing: list[dict[str, Any]]) -> list[str]:
        """Folders for a new chat: the ones used lately, then the rest of the projects folder."""
        seen: list[str] = []
        for info in listing:
            cwd = info.get("cwd") or ""
            if cwd and cwd not in seen and Path(cwd).is_dir():
                seen.append(cwd)
            if len(seen) >= 12:
                break
        root = Path(self.settings.projects_dir).expanduser()
        if root.is_dir():
            dirs = [d for d in root.iterdir() if d.is_dir() and not d.name.startswith(".")]
            for d in sorted(dirs, key=lambda d: d.stat().st_mtime, reverse=True):
                if str(d) not in seen:
                    seen.append(str(d))
        return seen[:80]

    def state(self) -> dict[str, Any]:
        info = self.probe()
        listing = self.listing()
        return {
            "version": __version__,
            "base_url": self.settings.base_url,
            "model": info.get("model") or "",
            "context": info.get("context"),
            "server": info.get("server"),
            "error": info.get("error"),
            "effort": self.settings.reasoning_effort,
            "mode": self.settings.permission_mode,
            "levels": list(thinking.LEVELS),
            "projects": self.projects(listing),
            "home": str(Path.home()),
            "boot": BOOT,
        }

    def prompts(self, limit: int = 200) -> list[str]:
        out: list[str] = []
        try:
            for s in self.history.load_history_strings():  # newest first
                if s not in out:
                    out.append(s)
                if len(out) >= limit:
                    break
        except OSError:
            pass
        return out

    def reap(self) -> None:
        """Let workers that have sat idle go; their sessions are all on disk."""
        while True:
            time.sleep(60)
            with self.lock:
                chats = list(self.chats.values())
            for chat in chats:
                if chat.proc is not None and not chat.running and time.time() - chat.touched > IDLE_SECONDS:
                    chat.close()

    def close_all(self) -> None:
        with self.lock:
            chats = list(self.chats.values())
        for chat in chats:
            chat.close()


class Handler(BaseHTTPRequestHandler):
    hub: Hub
    protocol_version = "HTTP/1.1"
    server_version = "frank"

    def log_message(self, fmt: str, *args: Any) -> None:
        pass

    def send(self, code: int, payload: Any, ctype: str = "application/json", headers: dict[str, str] | None = None) -> None:
        data = payload if isinstance(payload, bytes) else json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", ctype + "; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        for k, v in {"Cache-Control": "no-store", "X-Content-Type-Options": "nosniff", **(headers or {})}.items():
            self.send_header(k, v)
        self.end_headers()
        self.wfile.write(data)

    def allowed(self) -> bool:
        """The Host check stops a hostile site from rebinding its own name to this address."""
        host = (self.headers.get("Host") or "").strip().lower()
        host = (host.rsplit(":", 1)[0] if re.search(r":\d+$", host) else host).strip("[]")
        me = os.uname().nodename.lower().split(".")[0]
        if host in ("localhost", me) or host.startswith(me + ".") or re.fullmatch(r"[0-9a-f:.]+", host):
            return True
        self.send(403, {"error": f"open the page by the machine's own name or address, not {host!r}"})
        return False

    def body(self) -> dict[str, Any]:
        n = int(self.headers.get("Content-Length") or 0)
        if n > MAX_BODY:
            raise ValueError("request too large")
        data = json.loads(self.rfile.read(n) or b"{}")
        if not isinstance(data, dict):
            raise ValueError("body must be a JSON object")
        return data

    def do_GET(self) -> None:
        if not self.allowed():
            return
        url = urlparse(self.path)
        hub = self.hub
        try:
            if url.path in STATIC:
                name, ctype = STATIC[url.path]
                self.send(200, (HERE / name).read_bytes(), ctype, {"Content-Security-Policy": CSP, "X-Frame-Options": "DENY"})
            elif url.path == "/api/state":
                self.send(200, hub.state())
            elif url.path == "/api/chats":
                self.send(200, {"chats": hub.listing()})
            elif url.path == "/api/prompts":
                self.send(200, {"prompts": hub.prompts()})
            elif m := re.fullmatch(r"/api/chats/([^/]+)", url.path):
                self.send(200, hub.chat(m.group(1)).view())
            elif m := re.fullmatch(r"/api/chats/([^/]+)/events", url.path):
                after = self.headers.get("Last-Event-ID") or (parse_qs(url.query).get("after") or ["0"])[0]
                self.events(hub.chat(m.group(1)), int(after) if str(after).isdigit() else 0)
            else:
                self.send(404, {"error": "not found"})
        except KeyError:
            self.send(404, {"error": "no such chat"})
        except (OSError, ValueError) as e:
            self.send(500, {"error": f"{type(e).__name__}: {e}"})

    def events(self, chat: Chat, after: int) -> None:
        """Server-sent events: each message is a batch of the turn's events, with its last seq as the id."""
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream; charset=utf-8")
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Accel-Buffering", "no")
        self.end_headers()
        self.close_connection = True
        last = after
        try:
            self.wfile.write(f"event: hello\ndata: {json.dumps({'boot': BOOT, 'seq': chat.seq})}\n\n".encode())
            self.wfile.flush()
            while True:
                with chat.cond:
                    if last > chat.seq:  # from before a restart
                        last = 0
                    batch = [e for e in chat.events if e["seq"] > last]
                    if not batch:
                        chat.cond.wait(timeout=20)
                        batch = [e for e in chat.events if e["seq"] > last]
                if batch:
                    last = batch[-1]["seq"]
                    data = json.dumps(batch, ensure_ascii=False)
                    self.wfile.write(f"id: {last}\ndata: {data}\n\n".encode("utf-8"))
                else:
                    self.wfile.write(b": still here\n\n")
                self.wfile.flush()
        except OSError:  # the page went away
            pass

    def do_POST(self) -> None:
        if not self.allowed():
            return
        # Another site cannot set this header on a request to here, so it cannot press the buttons.
        if self.headers.get("X-Frank") != "1":
            return self.send(403, {"error": "missing X-Frank header"})
        hub = self.hub
        path = urlparse(self.path).path
        try:
            body = self.body()
            if path == "/api/chats":
                chat = hub.create(str(body.get("cwd") or ""))
                hub.send(chat, str(body.get("text") or ""), body.get("effort"), body.get("mode"))
                return self.send(200, {"id": chat.session.id})
            m = re.fullmatch(r"/api/chats/([^/]+)/(send|stop|answer|rename|delete)", path)
            if not m:
                return self.send(404, {"error": "not found"})
            chat, action = hub.chat(m.group(1)), m.group(2)
            if action == "send":
                hub.send(chat, str(body.get("text") or ""), body.get("effort"), body.get("mode"))
            elif action == "stop":
                chat.stop()
            elif action == "answer":
                chat.answer(body.get("id"), str(body.get("answer") or "n"))
            elif action == "rename":
                save_title(hub.home, chat.session.id, str(body.get("title") or ""))
            elif action == "delete":
                hub.delete(chat)
            self.send(200, {"ok": True})
        except KeyError:
            self.send(404, {"error": "no such chat"})
        except Conflict as e:
            self.send(409, {"error": str(e)})
        except ValueError as e:
            self.send(400, {"error": str(e)})
        except OSError as e:
            self.send(500, {"error": f"{type(e).__name__}: {e}"})


def _interrupt(*_: Any) -> None:
    raise KeyboardInterrupt


def make_server(settings: Settings, overrides: dict[str, Any] | None = None) -> tuple[ThreadingHTTPServer, Hub]:
    hub = Hub(settings, dict(overrides or {}))
    handler = type("ChatHandler", (Handler,), {"hub": hub})
    httpd = ThreadingHTTPServer((settings.web_host or "127.0.0.1", settings.web_port), handler)
    httpd.daemon_threads = True
    return httpd, hub


def serve(settings: Settings, overrides: dict[str, Any] | None = None) -> int:
    httpd, hub = make_server(settings, overrides)
    host = settings.web_host or "127.0.0.1"
    shown = os.uname().nodename if host in ("0.0.0.0", "::") else host
    print(f"frank {__version__}: chat page on http://{shown}:{httpd.server_port}, sessions in {settings.home / 'sessions'}", flush=True)
    if host not in ("127.0.0.1", "localhost", "::1"):
        print("no login: anyone who can reach this port can run commands here through the agent", flush=True)
    threading.Thread(target=hub.reap, daemon=True).start()
    signal.signal(signal.SIGTERM, _interrupt)  # systemctl stop: close the workers on the way out
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        hub.close_all()
        httpd.server_close()
    return 0
