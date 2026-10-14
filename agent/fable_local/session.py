"""Append-only JSONL transcripts so a session can be resumed."""

from __future__ import annotations

import datetime as dt
import json
import secrets
from pathlib import Path
from typing import Any


class Session:
    def __init__(self, path: Path):
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)

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
        with self.path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(record, ensure_ascii=False) + "\n")

    def add_message(self, message: dict[str, Any]) -> None:
        self.append({"type": "message", "message": message})

    def records(self) -> list[dict[str, Any]]:
        out = []
        with self.path.open(encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if line:
                    try:
                        out.append(json.loads(line))
                    except json.JSONDecodeError:
                        continue
        return out

    def messages(self) -> list[dict[str, Any]]:
        """Replay the transcript. A compaction record resets history to the summary that follows it."""
        msgs: list[dict[str, Any]] = []
        for r in self.records():
            if r.get("type") == "message":
                msgs.append(r["message"])
            elif r.get("type") == "compacted":
                msgs = []
        return msgs

    def meta(self) -> dict[str, Any]:
        for r in self.records():
            if r.get("type") == "meta":
                return r
        return {}

    @staticmethod
    def list_all(home: Path) -> list["Session"]:
        d = home / "sessions"
        if not d.is_dir():
            return []
        return [Session(p) for p in sorted(d.glob("*.jsonl"), reverse=True)]

    @staticmethod
    def latest(home: Path, cwd: Path | None = None) -> "Session | None":
        for s in Session.list_all(home):
            if cwd is None or s.meta().get("cwd") == str(cwd):
                return s
        return None

    @staticmethod
    def find(home: Path, prefix: str) -> "Session | None":
        for s in Session.list_all(home):
            if s.id.startswith(prefix):
                return s
        return None
