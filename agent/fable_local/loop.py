"""The agent loop: user message, model turn, tool calls, repeat until the model answers."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

from . import prompts, thinking
from .client import Client, ContextOverflow, ToolCall, Turn
from .config import Settings
from .permissions import PermissionManager
from .prompt import build_system_prompt, subagent_system_prompt
from .render import UI
from .session import Session
from .tools import Registry, ToolContext, ToolResult, build_registry, truncate_middle

COMPACT_NOTE = "[context compacted] Summary of the session so far:\n\n"
CHARS_PER_TOKEN = 3.5


class Agent:
    def __init__(
        self,
        settings: Settings,
        client: Client,
        ui: UI,
        cwd: Path | str,
        *,
        registry: Registry | None = None,
        permissions: PermissionManager | None = None,
        session: Session | None = None,
        depth: int = 0,
        system_prompt: str | None = None,
        context_window: int | None = None,
    ):
        self.settings = settings
        self.client = client
        self.ui = ui
        self.cwd = Path(cwd)
        self.depth = depth
        self.registry = registry or build_registry(
            read_only=settings.permission_mode == "readonly",
            allow_subagents=depth < settings.max_subagent_depth,
        )
        self.permissions = permissions or PermissionManager(settings.permission_mode, ui.ask_permission, settings.allowed_commands)
        self.session = session
        self.context_window = int(context_window or settings.context_window or 32768)
        # one tool result may take about a tenth of the window (3.5 chars per token)
        self.tool_output_limit = settings.max_tool_output_chars or max(30_000, int(self.context_window * 0.35))
        self.system_prompt = system_prompt or build_system_prompt(settings, self.cwd, client.model)
        self.messages: list[dict[str, Any]] = [{"role": "system", "content": self.system_prompt}]
        self.ctx = ToolContext(cwd=self.cwd, agent=self)
        self.last_usage: dict[str, int] = {}
        self._floor = 1  # len(messages) right after the last compaction; blocks back-to-back compactions

    # ---- history ----------------------------------------------------------
    def load_history(self, messages: list[dict[str, Any]]) -> None:
        self.messages = [self.messages[0], *[m for m in messages if m.get("role") != "system"]]
        self.repair()

    def _record(self, msg: dict[str, Any]) -> None:
        self.messages.append(msg)
        if self.session:
            self.session.add_message(msg)

    def repair(self) -> None:
        """Make the message list valid again after an interruption."""
        if len(self.messages) < 2:
            return
        last = self.messages[-1]
        if last.get("role") == "assistant" and last.get("tool_calls"):
            for tc in last["tool_calls"]:
                self._record({"role": "tool", "tool_call_id": tc["id"], "name": tc["function"]["name"], "content": "error: interrupted by the user before this tool ran"})
            self._record({"role": "assistant", "content": "[interrupted]"})
        elif last.get("role") == "user":
            self._record({"role": "assistant", "content": "[interrupted]"})

    def set_effort(self, level: str) -> bool:
        """Change the reasoning effort. True if the system prompt had to change (the server's prefix cache is lost)."""
        self.settings.reasoning_effort = level
        if self.depth:
            return False
        new = build_system_prompt(self.settings, self.cwd, self.client.model)
        changed = new != self.messages[0]["content"]
        self.system_prompt = new
        self.messages[0] = {"role": "system", "content": new}
        return changed

    # ---- main loop ----------------------------------------------------------
    def run_turn(self, user_text: str) -> str:
        self._record({"role": "user", "content": user_text})
        final = ""
        overflows = 0
        for _ in range(self.settings.max_iterations):
            self.maybe_compact()
            try:
                turn = self._step()
            except ContextOverflow:
                overflows += 1
                if overflows > 2:
                    raise
                self.compact(hard=True)
                continue
            final = turn.text
            if not turn.tool_calls:
                return final
            for tc in turn.tool_calls:
                self._execute_tool(tc)
        self.ui.warn(f"stopped after {self.settings.max_iterations} tool rounds in one turn")
        return final

    def _step(self) -> Turn:
        tools = self.registry.schemas() or None
        turn: Turn | None = None
        # never ask for more than the window has left, or the server rejects the request
        room = self.context_window - max(self.tokens_used(), self.estimate_tokens()) - 2048
        for kind, payload in self.client.stream(self.messages, tools, max_tokens=max(1024, min(self.settings.max_tokens, room))):
            if kind == "text":
                self.ui.text(payload)
            elif kind == "reasoning":
                self.ui.reasoning(payload)
            elif kind == "done":
                turn = payload
        self.ui.turn_end()
        assert turn is not None
        if turn.usage:
            self.last_usage = turn.usage
        # Kept exactly as produced, <think> block included, so the server's prefix cache still matches.
        content = (turn.raw or turn.text) if self.settings.send_reasoning_back else turn.text
        msg: dict[str, Any] = {"role": "assistant", "content": content or None}
        if turn.tool_calls:
            msg["tool_calls"] = [tc.to_message_part() for tc in turn.tool_calls]
            inline = thinking.strip_think(turn.raw) != turn.raw  # the reasoning is already in the content
            if turn.reasoning and self.settings.send_reasoning_back and not inline:
                msg["reasoning_content"] = turn.reasoning
        self._record(msg)
        if turn.finish_reason == "length":
            self.ui.warn("response was cut off at max_tokens")
        return turn

    def _execute_tool(self, tc: ToolCall) -> None:
        tool = self.registry.get(tc.name)
        if tool is None:
            self.ui.tool_call(tc.name, "(unknown tool)")
            self._tool_message(tc, ToolResult(f"unknown tool {tc.name!r}; available: {', '.join(self.registry.names())}", is_error=True))
            return
        try:
            args = tc.parsed()
        except (ValueError, json.JSONDecodeError) as e:
            self.ui.tool_call(tc.name, "(bad arguments)")
            self._tool_message(tc, ToolResult(f"arguments are not a JSON object: {e}", is_error=True))
            return
        self.ui.tool_call(tc.name, tool.summary(args))
        allowed, reason = self.permissions.check(tool, args, self.ctx)
        if not allowed:
            result = ToolResult(f"permission denied ({reason}). Do not retry the same call; ask the user or take another approach.", is_error=True)
        else:
            t0 = time.time()
            try:
                result = tool.run(args, self.ctx)
            except Exception as e:  # a tool crash is information for the model, not a crash of the agent
                result = ToolResult(f"{tc.name} failed: {type(e).__name__}: {e}", is_error=True)
            result.content = truncate_middle(result.content, self.tool_output_limit)
            if time.time() - t0 > 30:
                self.ui.info(f"{tc.name} took {time.time() - t0:.0f}s")
        self.ui.tool_result(tc.name, result)
        self._tool_message(tc, result)

    def _tool_message(self, tc: ToolCall, result: ToolResult) -> None:
        content = f"error: {result.content}" if result.is_error and not result.content.startswith("error") else result.content
        self._record({"role": "tool", "tool_call_id": tc.id, "name": tc.name, "content": content or "(empty)"})

    # ---- context management -------------------------------------------------
    def estimate_tokens(self) -> int:
        return int(sum(len(json.dumps(m, ensure_ascii=False)) for m in self.messages) / CHARS_PER_TOKEN)

    def tokens_used(self) -> int:
        if self.last_usage:
            return self.last_usage.get("prompt_tokens", 0) + self.last_usage.get("completion_tokens", 0)
        return self.estimate_tokens()

    def maybe_compact(self) -> None:
        if len(self.messages) <= self._floor + 2:
            return  # nothing new since the last compaction (or the session just started)
        if self.tokens_used() > self.settings.compact_threshold * self.context_window:
            self.compact()

    def transcript(self, limit_chars: int) -> str:
        """Plain-text rendering of the conversation, most recent part kept within limit_chars."""
        parts: list[str] = []
        for m in self.messages[1:]:
            role = m.get("role")
            if role == "user":
                parts.append(f"[user]\n{m.get('content', '')}")
            elif role == "assistant":
                if text := thinking.strip_think(m.get("content") or "").strip():
                    parts.append(f"[assistant]\n{text}")
                for tc in m.get("tool_calls") or []:
                    parts.append(f"[tool call] {tc['function']['name']} {tc['function'].get('arguments', '')[:2000]}")
            elif role == "tool":
                parts.append(f"[tool result {m.get('name', '')}]\n{str(m.get('content', ''))[:4000]}")
        text = "\n\n".join(parts)
        if len(text) > limit_chars:
            text = "[... earlier part of the transcript dropped ...]\n\n" + text[-limit_chars:]
        return text

    def compact(self, hard: bool = False) -> None:
        self.ui.info("compacting context...")
        # a longer window earns a longer summary: 800 words at 128k, 6400 at 1M
        out_tokens = max(2048, min(16384, self.context_window // 64))
        brief = prompts.load("compact").format(words=out_tokens * 800 // 2048)
        summary = ""
        if not hard:
            # Ask at the end of the real conversation, tools and all: the server has that
            # prefix cached, so the summary costs seconds instead of a refill of the window.
            try:
                summary = self.client.complete([*self.messages, {"role": "user", "content": brief}], max_tokens=out_tokens + 4096, tools=self.registry.schemas() or None)
            except ContextOverflow:
                pass
        if not summary:
            budget = int(self.context_window * CHARS_PER_TOKEN * (0.35 if hard else 0.6))
            request = [
                {"role": "system", "content": "You write continuation summaries for a coding agent."},
                {"role": "user", "content": f"Transcript:\n\n{self.transcript(budget)}\n\n{brief}"},
            ]
            try:
                summary = self.client.complete(request, max_tokens=out_tokens + 4096)
            except ContextOverflow:
                pass
        if not summary:
            summary = self.transcript(int(self.context_window * CHARS_PER_TOKEN * 0.25))
        last_role = self.messages[-1].get("role")
        pending = [self.messages[-1]] if last_role == "user" else []
        self.messages = [
            self.messages[0],
            {"role": "user", "content": COMPACT_NOTE + summary},
            {"role": "assistant", "content": "Understood. Continuing from the summary."},
            *pending,
        ]
        if last_role == "tool":  # compacted in the middle of a turn
            self.messages.append({"role": "user", "content": "Continue the task described in the summary from its next step."})
        if self.session:
            self.session.append({"type": "compacted"})
            for m in self.messages[1:]:
                self.session.add_message(m)
        self.last_usage = {}
        self._floor = len(self.messages)
        self.ui.info(f"compacted to about {self.estimate_tokens()} tokens")

    def reset(self) -> None:
        self.messages = [self.messages[0]]
        self.last_usage = {}
        self._floor = 1
        self.ctx.state.clear()

    # ---- subagents ------------------------------------------------------------
    def spawn_child(self, read_only: bool = True) -> "Agent | None":
        s = self.settings
        if self.depth >= s.max_subagent_depth:
            return None
        client = self.client
        window = self.context_window
        if s.subagent_base_url or s.subagent_model:
            client = Client(s, base_url=s.subagent_base_url or None, api_key=s.subagent_api_key or None, model=s.subagent_model or None)
            info = client.probe()
            window = info.get("context") or window
        ro = read_only or s.permission_mode == "readonly"
        registry = build_registry(read_only=ro, allow_subagents=False)
        perms = PermissionManager("readonly", self.ui.ask_permission) if ro else self.permissions
        child = Agent(
            s, client, self.ui.child(), self.cwd,
            registry=registry, permissions=perms, session=None, depth=self.depth + 1,
            system_prompt=subagent_system_prompt(self.system_prompt), context_window=window,
        )
        child.ctx.state["read_files"] = set(self.ctx.state.get("read_files", set()))
        return child
