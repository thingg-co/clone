"""Decide whether a tool call may run: read-only tools always, safe shell commands
always, everything else by mode (auto, ask, readonly)."""

from __future__ import annotations

import re
import shlex
from typing import Any, Callable

from .tools.base import Tool, ToolContext

AskFn = Callable[[str, str, str], str]  # (tool name, summary, detail) -> "y" | "n" | "a"

SAFE_FIRST = {
    "ls", "cat", "head", "tail", "wc", "pwd", "echo", "printf", "which", "type", "file", "stat", "du", "df",
    "tree", "date", "whoami", "uname", "rg", "grep", "egrep", "fgrep", "find", "fd", "sort", "uniq", "cut", "tr",
    "diff", "true", "false", "test", "[", "jq", "column", "nl", "basename", "dirname", "realpath", "readlink",
    "md5sum", "sha256sum", "shasum", "less", "more", "xargs",
}
GIT_SAFE = {"status", "diff", "log", "show", "branch", "blame", "ls-files", "rev-parse", "remote", "describe",
            "shortlog", "grep", "cat-file", "name-rev", "tag", "worktree"}
UNSAFE_TOKENS = re.compile(r"(>|>>|\$\(|`|\brm\b|\bsudo\b|\bmv\b|\bdd\b|\bmkfs|\bchmod\b|\bchown\b|\bkill\b|\bpkill\b|\btee\b)")
SPLIT = re.compile(r"\|\||&&|;|\||\n")


def is_safe_command(command: str, extra_prefixes: list[str] | None = None) -> bool:
    cmd = command.strip()
    if not cmd:
        return False
    for prefix in extra_prefixes or []:
        if cmd == prefix or cmd.startswith(prefix.rstrip() + " "):
            return True
    if UNSAFE_TOKENS.search(cmd):
        return False
    for segment in SPLIT.split(cmd):
        segment = segment.strip()
        if not segment:
            continue
        try:
            tokens = shlex.split(segment)
        except ValueError:
            return False
        if not tokens:
            continue
        # skip leading VAR=value assignments
        while tokens and re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*=.*", tokens[0]):
            tokens.pop(0)
        if not tokens:
            return False
        head = tokens[0]
        if head == "git":
            if len(tokens) < 2 or tokens[1] not in GIT_SAFE:
                return False
            if tokens[1] in ("branch", "tag", "remote", "worktree") and any(t.startswith("-") and t not in ("-a", "-v", "-vv", "--list", "-l", "--show-current") for t in tokens[2:]):
                return False
            continue
        if head == "find" and any(t in ("-delete", "-exec", "-execdir", "-ok") for t in tokens):
            return False
        if head not in SAFE_FIRST:
            return False
    return True


class PermissionManager:
    def __init__(self, mode: str, ask: AskFn, allowed_commands: list[str] | None = None):
        if mode not in ("ask", "auto", "readonly"):
            raise ValueError(f"unknown permission mode {mode!r}")
        self.mode = mode
        self.ask = ask
        self.allowed_commands = list(allowed_commands or [])
        self.always: set[str] = set()

    def check(self, tool: Tool, args: dict[str, Any], ctx: ToolContext) -> tuple[bool, str]:
        """Return (allowed, reason)."""
        if tool.read_only:
            return True, "read-only"
        if self.mode == "readonly":
            return False, "read-only mode"
        if self.mode == "auto":
            return True, "auto"
        if tool.name == "bash" and is_safe_command(str(args.get("command", "")), self.allowed_commands):
            return True, "safe command"
        if tool.name in self.always:
            return True, "always allowed this session"
        answer = (self.ask(tool.name, tool.summary(args), tool.detail(args, ctx)) or "n").strip().lower()
        if answer.startswith("a"):
            self.always.add(tool.name)
            return True, "allowed for the session"
        if answer.startswith("y"):
            return True, "allowed"
        return False, "denied by user"
