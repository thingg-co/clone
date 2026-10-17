from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

from ..prompts import tool_descriptions

if TYPE_CHECKING:
    from ..loop import Agent


@dataclass
class ToolResult:
    content: str
    is_error: bool = False
    display: str | None = None  # what to show the user instead of content, if different


@dataclass
class ToolContext:
    """What a tool gets to see when it runs."""

    cwd: Path
    agent: "Agent"
    state: dict[str, Any] = field(default_factory=dict)  # per-agent scratch, e.g. the todo list

    def resolve(self, path: str) -> Path:
        p = Path(path).expanduser()
        return p if p.is_absolute() else (self.cwd / p).resolve()


class Tool:
    name: str = ""
    read_only: bool = False
    parameters: dict[str, Any] = {"type": "object", "properties": {}}

    @property
    def description(self) -> str:
        return tool_descriptions().get(self.name, self.__doc__ or self.name)

    def schema(self) -> dict[str, Any]:
        return {
            "type": "function",
            "function": {"name": self.name, "description": self.description, "parameters": self.parameters},
        }

    def summary(self, args: dict[str, Any]) -> str:
        """One line shown to the user when the call starts."""
        return ", ".join(f"{k}={str(v)[:60]!r}" for k, v in args.items())

    def detail(self, args: dict[str, Any], ctx: ToolContext) -> str:
        """Shown in the permission prompt; defaults to the summary."""
        return self.summary(args)

    def run(self, args: dict[str, Any], ctx: ToolContext) -> ToolResult:  # pragma: no cover - abstract
        raise NotImplementedError


class Registry:
    def __init__(self) -> None:
        self._tools: dict[str, Tool] = {}

    def add(self, tool: Tool) -> None:
        self._tools[tool.name] = tool

    def get(self, name: str) -> Tool | None:
        return self._tools.get(name)

    def names(self) -> list[str]:
        return list(self._tools)

    def schemas(self) -> list[dict[str, Any]]:
        return [t.schema() for t in self._tools.values()]

    def __contains__(self, name: str) -> bool:
        return name in self._tools


def truncate_middle(text: str, limit: int) -> str:
    if len(text) <= limit:
        return text
    head = limit * 2 // 3
    tail = limit - head
    cut = len(text) - head - tail
    return text[:head] + f"\n\n[... {cut} characters truncated ...]\n\n" + text[-tail:]
