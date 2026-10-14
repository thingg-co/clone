"""Prompt files live next to this module as Markdown so they can be edited without touching code."""

from __future__ import annotations

import re
from functools import lru_cache
from pathlib import Path

PROMPTS_DIR = Path(__file__).resolve().parent


@lru_cache
def load(name: str) -> str:
    """A prompt file. `<name>.local.md`, which git ignores, replaces `<name>.md` when it exists."""
    local = PROMPTS_DIR / f"{name}.local.md"
    return (local if local.is_file() else PROMPTS_DIR / f"{name}.md").read_text(encoding="utf-8")


@lru_cache
def sections(name: str) -> dict[str, str]:
    """Parse a prompt file into {heading: body} by its `## heading` lines."""
    text = load(name)
    out: dict[str, str] = {}
    parts = re.split(r"^## (\S+)\s*$", text, flags=re.M)
    # parts = [preamble, name1, body1, name2, body2, ...]
    for i in range(1, len(parts) - 1, 2):
        out[parts[i]] = parts[i + 1].strip()
    return out


def tool_descriptions() -> dict[str, str]:
    """tools.md as {tool_name: description}."""
    return sections("tools")
