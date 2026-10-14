"""Build the system prompt from prompts/system.md plus environment and instruction files."""

from __future__ import annotations

import datetime as dt
import os
import platform
import subprocess
from pathlib import Path

from . import prompts, thinking
from .config import Settings

INSTRUCTION_FILES = ("AGENTS.md", "CLAUDE.md", "FABLE.md")
MAX_INSTRUCTION_CHARS = 20_000


def _git(cwd: Path, *args: str) -> str | None:
    try:
        p = subprocess.run(["git", *args], cwd=cwd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, timeout=3)
    except (OSError, subprocess.TimeoutExpired):
        return None
    return p.stdout.decode("utf-8", "replace").strip() if p.returncode == 0 else None


def git_block(cwd: Path) -> str:
    top = _git(cwd, "rev-parse", "--show-toplevel")
    if not top:
        return "- Git: not a repository"
    branch = _git(cwd, "rev-parse", "--abbrev-ref", "HEAD") or "?"
    status = _git(cwd, "status", "--porcelain") or ""
    dirty = len(status.splitlines())
    last = _git(cwd, "log", "-1", "--format=%h %s") or ""
    return f"- Git: repository at {top}, branch {branch}, {dirty} changed file(s), last commit: {last}"


def instruction_files(cwd: Path) -> list[Path]:
    """User-level and project-level instruction files, outermost first."""
    found: list[Path] = []
    home = Path.home()
    for p in (home / ".claude" / "CLAUDE.md", home / ".config" / "fable" / "CLAUDE.md", home / ".config" / "fable" / "AGENTS.md"):
        if p.is_file():
            found.append(p)
    chain = [cwd, *cwd.parents]
    chain.reverse()
    for d in chain:
        if d == home:
            continue
        for name in INSTRUCTION_FILES:
            p = d / name
            if p.is_file() and p not in found:
                found.append(p)
    return found


def instructions_block(cwd: Path, extra: str = "") -> str:
    parts: list[str] = []
    baked = prompts.load("user")
    for p in instruction_files(cwd):
        text = p.read_text(encoding="utf-8", errors="replace").strip()
        if text and text in baked:  # already in the user prompt file
            continue
        if len(text) > MAX_INSTRUCTION_CHARS:
            text = text[:MAX_INSTRUCTION_CHARS] + "\n[truncated]"
        parts.append(f"Contents of {p}:\n\n{text}")
    if extra.strip():
        parts.append(extra.strip())
    if not parts:
        return ""
    return "\n# User and project instructions\n\nFollow these; they override the defaults above.\n\n" + "\n\n".join(parts) + "\n"


def build_system_prompt(settings: Settings, cwd: Path, model: str) -> str:
    template = prompts.load("system")
    return template.format(
        name=settings.name,
        cwd=cwd,
        platform=f"{platform.system()} {platform.release()} ({platform.machine()})",
        shell=os.environ.get("SHELL", "/bin/sh"),
        date=dt.date.today().isoformat(),
        model=model or "(unknown)",
        permission_mode=settings.permission_mode,
        git=git_block(cwd),
        thinking=thinking.prompt_block(settings, model),
        user_instructions=prompts.load("user").strip(),
        instructions=instructions_block(cwd, settings.system_prompt_extra),
    )


def subagent_system_prompt(parent_prompt: str) -> str:
    return parent_prompt.rstrip() + "\n\n# Subagent brief\n\n" + prompts.load("subagent")
