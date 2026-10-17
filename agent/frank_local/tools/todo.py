from __future__ import annotations

from typing import Any

from .base import Tool, ToolContext, ToolResult

MARK = {"pending": "[ ]", "in_progress": "[>]", "done": "[x]", "completed": "[x]"}


def render_todos(todos: list[dict[str, Any]]) -> str:
    return "\n".join(f"{MARK.get(str(t.get('status', 'pending')), '[ ]')} {t.get('content', '')}" for t in todos)


class TodoWrite(Tool):
    name = "todo_write"
    read_only = True  # touches nothing outside the session
    parameters = {
        "type": "object",
        "properties": {
            "todos": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "content": {"type": "string"},
                        "status": {"type": "string", "enum": ["pending", "in_progress", "done"]},
                    },
                    "required": ["content", "status"],
                },
            }
        },
        "required": ["todos"],
    }

    def summary(self, args: dict[str, Any]) -> str:
        todos = args.get("todos") or []
        done = sum(1 for t in todos if t.get("status") in ("done", "completed"))
        return f"{done}/{len(todos)} done"

    def run(self, args: dict[str, Any], ctx: ToolContext) -> ToolResult:
        todos = args.get("todos")
        if not isinstance(todos, list):
            return ToolResult("todos must be a list", is_error=True)
        ctx.state["todos"] = todos
        text = render_todos(todos)
        return ToolResult(f"todo list updated ({len(todos)} items)", display=text)
