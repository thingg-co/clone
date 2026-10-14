"""Tool registry. Each tool is a Tool subclass; build_registry picks the set for a mode."""

from __future__ import annotations

from .base import Registry, Tool, ToolContext, ToolResult, truncate_middle
from .fs import EditFile, GlobFiles, GrepFiles, ListDir, ReadFile, WriteFile
from .shell import Bash
from .subagent import SpawnSubagent
from .todo import TodoWrite
from .web import WebFetch

ALL_TOOLS: list[type[Tool]] = [
    Bash, ReadFile, WriteFile, EditFile, GlobFiles, GrepFiles, ListDir, WebFetch, TodoWrite, SpawnSubagent,
]


def build_registry(*, read_only: bool = False, allow_subagents: bool = True) -> Registry:
    reg = Registry()
    for cls in ALL_TOOLS:
        if read_only and not cls.read_only:
            continue
        if cls is SpawnSubagent and not allow_subagents:
            continue
        reg.add(cls())
    return reg


__all__ = ["Registry", "Tool", "ToolContext", "ToolResult", "build_registry", "truncate_middle"]
