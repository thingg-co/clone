from __future__ import annotations

import os
import signal
import subprocess
from typing import Any

from .base import Tool, ToolContext, ToolResult, truncate_middle

CWD_MARK = "\x1e__FRANK_CWD__="


def _stop(proc: subprocess.Popen) -> bytes:
    """End the command and everything it started, and return the output read so far.

    The command has a process group of its own, so this reaches what bash left running too:
    a job sent to the background, or the children of a command that timed out. It sends SIGINT
    as Ctrl-C would, then SIGTERM, then SIGKILL, a second apart.
    """
    out = b""
    for sig in (signal.SIGINT, signal.SIGTERM, signal.SIGKILL):
        try:
            os.killpg(proc.pid, sig)
        except ProcessLookupError:  # nothing left in the group
            pass
        try:
            return proc.communicate(timeout=1)[0] or b""
        except subprocess.TimeoutExpired as e:
            out = e.output or out
        except KeyboardInterrupt:  # Ctrl-C again: on to the next signal
            pass
    # What still holds the output open has left the group (setsid, a daemon); stop reading.
    if proc.stdout:
        proc.stdout.close()
    try:
        proc.wait(timeout=1)
    except subprocess.TimeoutExpired:
        pass
    return out


class Bash(Tool):
    name = "bash"
    read_only = False
    parameters = {
        "type": "object",
        "properties": {
            "command": {"type": "string", "description": "The command to run"},
            "timeout": {"type": "integer", "description": "Seconds before the command is killed (max 600)"},
        },
        "required": ["command"],
    }

    def summary(self, args: dict[str, Any]) -> str:
        cmd = str(args.get("command", "")).strip().replace("\n", " ⏎ ")
        return cmd if len(cmd) <= 160 else cmd[:157] + "..."

    def detail(self, args: dict[str, Any], ctx: ToolContext) -> str:
        return f"$ {args.get('command', '')}\n(in {ctx.cwd})"

    def run(self, args: dict[str, Any], ctx: ToolContext) -> ToolResult:
        command = str(args.get("command", ""))
        if not command.strip():
            return ToolResult("empty command", is_error=True)
        settings = ctx.agent.settings
        timeout = int(args.get("timeout") or settings.bash_timeout)
        timeout = max(1, min(timeout, 600))
        # Run the command, then report the final working directory so `cd` persists.
        script = f"{command}\n__rc=$?\nprintf '\\n{CWD_MARK}%s' \"$PWD\"\nexit $__rc"
        env = dict(os.environ, FRANK_AGENT="1", GIT_TERMINAL_PROMPT="0")
        # A session of its own, so that an interrupt or a timeout can end all of it (_stop).
        # It also has no terminal, so a password prompt fails at once instead of waiting.
        proc = subprocess.Popen(
            ["bash", "-c", script],
            cwd=ctx.cwd,
            env=env,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        try:
            raw = proc.communicate(timeout=timeout)[0] or b""
        except subprocess.TimeoutExpired:
            partial = _stop(proc).decode("utf-8", "replace")
            return ToolResult(f"command timed out after {timeout}s\n{partial}", is_error=True)
        except BaseException:  # Ctrl-C in the terminal, Stop on the chat page
            _stop(proc)
            raise
        out = raw.decode("utf-8", "replace")
        if CWD_MARK in out:
            out, _, new_cwd = out.rpartition(CWD_MARK)
            new_cwd = new_cwd.strip()
            if new_cwd and os.path.isdir(new_cwd):
                ctx.cwd = type(ctx.cwd)(new_cwd)
                ctx.agent.cwd = ctx.cwd
            out = out[:-1] if out.endswith("\n") else out  # the newline printf added
        out = out.rstrip("\n")
        out = truncate_middle(out, ctx.agent.tool_output_limit)
        if proc.returncode != 0:
            return ToolResult(f"{out}\n[exit code {proc.returncode}]".strip(), is_error=True)
        return ToolResult(out if out.strip() else "(no output)")
