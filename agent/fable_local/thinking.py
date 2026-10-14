"""Thinking: which kind a model gets, the prompt for it, and parsing <think> blocks out of a stream.

Two kinds. Native: the model reasons by itself and the server returns it as
reasoning_content (gpt-oss, Qwen3-Next Thinking). Prompted: the model has no
reasoning mode, so the system prompt asks for a <think> block at the start of
every response and the client lifts it out of the content.
"""

from __future__ import annotations

import re
from typing import Any

from . import prompts

LEVELS = ("off", "low", "medium", "high", "max")
NATIVE_NAMES = re.compile(r"gpt-oss|thinking|reasoner|qwq|glm|-r1|deepseek-v(?:3\.[1-9]|[4-9])|nemotron|qwen3\.[5-9]|flash-next|laguna|"
                          r"mimo|dots3|k2-horizon|seed-oss", re.I)
OPEN, CLOSE = "<think>", "</think>"

# What each family of natively reasoning models takes as reasoning_effort, by our level.
# A missing level, or None for the whole family, means the parameter is left out.
STANDARD_EFFORT = {"low": "low", "medium": "medium", "high": "high", "max": "high"}
EFFORTS: tuple[tuple[re.Pattern[str], dict[str, str] | None], ...] = (
    (re.compile(r"gpt-oss", re.I), {**STANDARD_EFFORT, "off": "low"}),  # cannot be switched off
    (re.compile(r"qwen3\.[89]|flash-next|qwen4", re.I), {"low": "low", "medium": "medium", "high": "xhigh", "max": "xhigh"}),
    (re.compile(r"glm-5", re.I), {"low": "low", "medium": "high", "high": "max", "max": "max"}),
    (re.compile(r"nemotron|laguna|qwen3\.[5-7]|mimo|dots3", re.I), None),  # on or off only, through enable_thinking
    (re.compile(r"seed-oss", re.I), None),         # a thinking budget instead, unlimited unless set
    # DeepSeek V4's template (the one in the GGUF builds) knows only high and max; anything
    # else leaves its effort instruction out.
    (re.compile(r"deepseek", re.I), {"high": "high", "max": "max"}),
)
_BLOCK = re.compile(r"^\s*<think>.*?(</think>\s*|$)", re.S)


def mode(settings: Any, model: str) -> str:
    """off | native | prompted for this model under these settings."""
    if settings.reasoning_effort == "off" or settings.thinking == "off":
        return "off"
    if settings.thinking in ("native", "prompted"):
        return settings.thinking
    return "native" if NATIVE_NAMES.search(model or "") else "prompted"


def server_effort(settings: Any, model: str) -> str | None:
    """The reasoning_effort to send for this model, or None to send nothing."""
    if mode(settings, model) == "prompted":
        return None  # no reasoning mode to steer; the depth is asked for in the system prompt
    table = next((t for pattern, t in EFFORTS if pattern.search(model or "")), STANDARD_EFFORT)
    return table.get(settings.reasoning_effort) if table else None


def prompt_block(settings: Any, model: str) -> str:
    """System prompt section for prompted thinking; empty for native and off."""
    if mode(settings, model) != "prompted":
        return ""
    parts = prompts.sections("thinking")
    return f"# Thinking\n\n{parts['protocol']}\n\n{parts[settings.reasoning_effort]}\n"


def strip_think(text: str) -> str:
    return _BLOCK.sub("", text or "", count=1)


class ThinkSplitter:
    """Splits streamed content into ('reasoning', s) and ('text', s) pieces.

    Only a block that opens the response counts, so a literal <think> later in
    an answer or in code passes through untouched. Tags may arrive split across
    chunks.
    """

    def __init__(self) -> None:
        self.state = "start"  # start | think | text
        self.buf = ""

    def feed(self, s: str) -> list[tuple[str, str]]:
        out: list[tuple[str, str]] = []
        self.buf += s
        if self.state == "start":
            lead = self.buf.lstrip()
            if lead.startswith(OPEN):
                self.state, self.buf = "think", lead[len(OPEN):]
            elif OPEN.startswith(lead):
                return out  # whitespace or a partial opening tag so far
            else:
                self.state = "text"
        if self.state == "think":
            end = self.buf.find(CLOSE)
            if end >= 0:
                if self.buf[:end]:
                    out.append(("reasoning", self.buf[:end]))
                self.state, self.buf = "text", self.buf[end + len(CLOSE):].lstrip("\n")
            else:
                keep = _partial_suffix(self.buf, CLOSE)
                if len(self.buf) > keep:
                    out.append(("reasoning", self.buf[: len(self.buf) - keep]))
                    self.buf = self.buf[len(self.buf) - keep:]
                return out
        if self.state == "text" and self.buf:
            out.append(("text", self.buf))
            self.buf = ""
        return out

    def flush(self) -> list[tuple[str, str]]:
        """Whatever is still held back when the stream ends."""
        kind = "reasoning" if self.state == "think" else "text"
        rest, self.buf = self.buf, ""
        return [(kind, rest)] if rest.strip() else []


def _partial_suffix(buf: str, tag: str) -> int:
    """Length of the longest suffix of buf that is a proper prefix of tag."""
    for n in range(min(len(tag) - 1, len(buf)), 0, -1):
        if buf.endswith(tag[:n]):
            return n
    return 0
