from __future__ import annotations

from typing import Any

from .base import Tool, ToolContext, ToolResult


class SpawnSubagent(Tool):
    name = "spawn_subagent"
    read_only = True  # the child enforces its own permissions
    parameters = {
        "type": "object",
        "properties": {
            "task": {"type": "string", "description": "Complete brief: what to find or do, where, and what to report"},
            "read_only": {"type": "boolean", "description": "Restrict the child to read-only tools (default true)"},
        },
        "required": ["task"],
    }

    def summary(self, args: dict[str, Any]) -> str:
        t = str(args.get("task", "")).strip().replace("\n", " ")
        return (t[:120] + "...") if len(t) > 120 else t

    def run(self, args: dict[str, Any], ctx: ToolContext) -> ToolResult:
        task = str(args.get("task", "")).strip()
        if not task:
            return ToolResult("task is empty", is_error=True)
        child = ctx.agent.spawn_child(read_only=bool(args.get("read_only", True)))
        if child is None:
            return ToolResult("subagent depth limit reached; do the work directly", is_error=True)
        try:
            report = child.run_turn(task)
        except Exception as e:  # the parent should hear about it, not crash
            return ToolResult(f"subagent failed: {e}", is_error=True)
        return ToolResult(report or "(subagent returned nothing)")
