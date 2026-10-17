"""Append-only JSONL transcripts so a session can be resumed, in the terminal or on the web page.

Both write the same files. A turn holds a lock on its session, so two processes never
add to one at the same time, and each picks up what the other added before it goes on.
"""

from __future__ import annotations

import datetime as dt
import json
import os
import secrets
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator

try:
    import fcntl
except ImportError:  # Windows: no locking, the terminal and the page should not share a session there
    fcntl = None  # type: ignore[assignment]

TITLE_CHARS = 80


class SessionBusy(Exception):
    """Another process is in the middle of a turn in this session."""


def text_of(content: Any) -> str:
    """Message content as plain text, whether a string or a list of parts."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "\n".join(str(p.get("text", "")) for p in content if isinstance(p, dict))
    return ""


def title_from(text: str) -> str:
    line = " ".join(text.split())
    return line if len(line) <= TITLE_CHARS else line[: TITLE_CHARS - 1].rstrip() + "…"


class Session:
    def __init__(self, path: Path):
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.synced = 0  # file size after our last read or write; more than that was written elsewhere

    @classmethod
    def new(cls, home: Path, cwd: Path, model: str) -> "Session":
        sid = dt.datetime.now().strftime("%Y%m%d-%H%M%S") + "-" + secrets.token_hex(2)
        s = cls(home / "sessions" / f"{sid}.jsonl")
        s.append({"type": "meta", "cwd": str(cwd), "model": model, "created": dt.datetime.now().isoformat(timespec="seconds")})
        return s

    @property
    def id(self) -> str:
        return self.path.stem

    def append(self, record: dict[str, Any]) -> None:
        with self.path.open("ab") as fh:
            fh.write((json.dumps(record, ensure_ascii=False) + "\n").encode("utf-8"))
            self.synced = fh.tell()

    def add_message(self, message: dict[str, Any], **extra: Any) -> None:
        """extra goes into the record beside the message (the reasoning of a final answer, for
        the web page) and never back to the model."""
        self.append({"type": "message", "message": message, **{k: v for k, v in extra.items() if v}})

    def _read(self) -> tuple[list[dict[str, Any]], int]:
        data = self.path.read_bytes()
        out = []
        for line in data.decode("utf-8", "replace").splitlines():
            line = line.strip()
            if line:
                try:
                    out.append(json.loads(line))
                except json.JSONDecodeError:  # a line still being written
                    continue
        return out, len(data)

    def records(self) -> list[dict[str, Any]]:
        return self._read()[0]

    def messages(self) -> list[dict[str, Any]]:
        """Replay the transcript. A compaction record resets history to the summary that follows it."""
        records, self.synced = self._read()
        msgs: list[dict[str, Any]] = []
        for r in records:
            if r.get("type") == "message":
                msgs.append(r["message"])
            elif r.get("type") == "compacted":
                msgs = []
        return msgs

    def changed_elsewhere(self) -> bool:
        try:
            return self.path.stat().st_size != self.synced
        except FileNotFoundError:
            return False

    def meta(self) -> dict[str, Any]:
        return self.head()[0]

    def head(self) -> tuple[dict[str, Any], str]:
        """The meta record and the first user message, without reading the whole file."""
        meta: dict[str, Any] = {}
        with self.path.open(encoding="utf-8", errors="replace") as fh:
            for line in fh:
                try:
                    r = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if r.get("type") == "meta" and not meta:
                    meta = r
                elif r.get("type") == "message" and r["message"].get("role") == "user":
                    return meta, text_of(r["message"].get("content"))
        return meta, ""

    def summary(self, titles: dict[str, str] | None = None) -> dict[str, Any]:
        """What a list of sessions shows. No title means nothing was ever said in it."""
        meta, first = self.head()
        st = self.path.stat()
        return {
            "id": self.id,
            "title": (titles or {}).get(self.id) or title_from(first),
            "cwd": meta.get("cwd", ""),
            "model": meta.get("model", ""),
            "created": meta.get("created", ""),
            "updated": st.st_mtime,
            "size": st.st_size,
        }

    # ---- one turn at a time ------------------------------------------------
    @property
    def lock_path(self) -> Path:
        return self.path.with_suffix(".lock")

    @contextmanager
    def turn(self, owner: str) -> Iterator[None]:
        """Hold the session for a turn. Raises SessionBusy when another process holds it."""
        if fcntl is None:
            yield
            return
        fh = self.lock_path.open("a+", encoding="utf-8")
        try:
            for attempt in range(20):  # holder() takes the lock for an instant when it looks
                try:
                    fcntl.flock(fh, fcntl.LOCK_EX | fcntl.LOCK_NB)
                    break
                except BlockingIOError:
                    if attempt == 19:
                        fh.seek(0)
                        raise SessionBusy(fh.read().strip() or "another process") from None
                    time.sleep(0.05)
            fh.seek(0)
            fh.truncate()
            fh.write(f"{owner} (pid {os.getpid()})")
            fh.flush()
            try:
                yield
            finally:
                fh.seek(0)
                fh.truncate()
                fcntl.flock(fh, fcntl.LOCK_UN)
        finally:
            fh.close()

    def holder(self) -> str | None:
        """Who is in the middle of a turn in this session, or None."""
        if fcntl is None or not self.lock_path.exists():
            return None
        with self.lock_path.open("a+", encoding="utf-8") as fh:
            try:
                fcntl.flock(fh, fcntl.LOCK_SH | fcntl.LOCK_NB)
            except BlockingIOError:
                fh.seek(0)
                return fh.read().strip() or "another process"
            fcntl.flock(fh, fcntl.LOCK_UN)
        return None

    def trash(self) -> None:
        """Move the transcript out of the list. Kept under sessions/trash/ in case it was a mistake."""
        bin_ = self.path.parent / "trash"
        bin_.mkdir(exist_ok=True)
        self.path.rename(bin_ / self.path.name)
        self.lock_path.unlink(missing_ok=True)

    # ---- finding sessions --------------------------------------------------
    @staticmethod
    def list_all(home: Path) -> list["Session"]:
        """Most recently changed first, so a session continued on the web page counts as the latest."""
        d = home / "sessions"
        if not d.is_dir():
            return []
        paths = []
        for p in d.glob("*.jsonl"):
            try:
                paths.append((p.stat().st_mtime, p.name, p))
            except FileNotFoundError:  # moved to the trash meanwhile
                continue
        return [Session(p) for _, _, p in sorted(paths, reverse=True)]

    @staticmethod
    def latest(home: Path, cwd: Path | None = None) -> "Session | None":
        """The last session with anything in it, for this directory if one is given."""
        for s in Session.list_all(home):
            meta, first = s.head()
            if first and (cwd is None or meta.get("cwd") == str(cwd)):
                return s
        return None

    @staticmethod
    def find(home: Path, prefix: str) -> "Session | None":
        for s in Session.list_all(home):
            if s.id.startswith(prefix):
                return s
        return None


def titles_path(home: Path) -> Path:
    return home / "sessions" / "titles.json"


def load_titles(home: Path) -> dict[str, str]:
    """Names given on the web page, by session id."""
    try:
        data = json.loads(titles_path(home).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    return {str(k): str(v) for k, v in data.items()} if isinstance(data, dict) else {}


def save_title(home: Path, sid: str, title: str) -> None:
    titles = load_titles(home)
    if title.strip():
        titles[sid] = title_from(title)
    else:
        titles.pop(sid, None)
    path = titles_path(home)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(titles, ensure_ascii=False, indent=1), encoding="utf-8")
    tmp.replace(path)
