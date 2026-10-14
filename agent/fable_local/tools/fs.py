from __future__ import annotations

import difflib
import os
import re
import shutil
import subprocess
from pathlib import Path
from typing import Any

from .base import Tool, ToolContext, ToolResult, truncate_middle

SKIP_DIRS = {".git", "node_modules", ".venv", "venv", "__pycache__", ".tox", ".mypy_cache", ".pytest_cache",
             "build", "dist", "target", ".next", ".cache", ".tools", "models", "llama.cpp"}
MAX_READ_BYTES = 20 * 1024 * 1024


def _is_binary(sample: bytes) -> bool:
    return b"\x00" in sample


class ReadFile(Tool):
    name = "read_file"
    read_only = True
    parameters = {
        "type": "object",
        "properties": {
            "path": {"type": "string", "description": "File path, absolute or relative to the working directory"},
            "offset": {"type": "integer", "description": "First line to read, 1-based (default 1)"},
            "limit": {"type": "integer", "description": "Number of lines to read (default 2000)"},
        },
        "required": ["path"],
    }

    def summary(self, args: dict[str, Any]) -> str:
        extra = ""
        if args.get("offset") or args.get("limit"):
            extra = f" [{args.get('offset', 1)}:+{args.get('limit', 2000)}]"
        return f"{args.get('path', '')}{extra}"

    def run(self, args: dict[str, Any], ctx: ToolContext) -> ToolResult:
        p = ctx.resolve(str(args["path"]))
        if p.is_dir():
            return ToolResult(f"{p} is a directory; use list_dir", is_error=True)
        if not p.is_file():
            return ToolResult(f"no such file: {p}", is_error=True)
        if p.stat().st_size > MAX_READ_BYTES:
            return ToolResult(f"{p} is larger than 20 MB; read it in parts with offset/limit or search it", is_error=True)
        raw = p.read_bytes()
        if _is_binary(raw[:8192]):
            return ToolResult(f"{p} looks binary ({len(raw)} bytes); not shown", is_error=True)
        lines = raw.decode("utf-8", "replace").splitlines()
        offset = max(1, int(args.get("offset") or 1))
        limit = max(1, int(args.get("limit") or 2000))
        chunk = lines[offset - 1: offset - 1 + limit]
        if not chunk and lines:
            return ToolResult(f"offset {offset} is past the end ({len(lines)} lines)", is_error=True)
        width = len(str(offset + len(chunk)))
        body = "\n".join(f"{i:>{width}}\t{line}" for i, line in enumerate(chunk, start=offset))
        if offset - 1 + limit < len(lines):
            body += f"\n[... {len(lines) - (offset - 1 + limit)} more lines; total {len(lines)}]"
        ctx.state.setdefault("read_files", set()).add(str(p))
        return ToolResult(truncate_middle(body, ctx.agent.tool_output_limit) if body else "(empty file)")


class WriteFile(Tool):
    name = "write_file"
    read_only = False
    parameters = {
        "type": "object",
        "properties": {
            "path": {"type": "string", "description": "File path to write"},
            "content": {"type": "string", "description": "Full file content"},
        },
        "required": ["path", "content"],
    }

    def summary(self, args: dict[str, Any]) -> str:
        return f"{args.get('path', '')} ({len(str(args.get('content', '')))} chars)"

    def detail(self, args: dict[str, Any], ctx: ToolContext) -> str:
        p = ctx.resolve(str(args.get("path", "")))
        content = str(args.get("content", ""))
        head = "\n".join(content.splitlines()[:30])
        verb = "overwrite" if p.exists() else "create"
        return f"{verb} {p} ({len(content)} chars)\n---\n{head}" + ("\n..." if content.count("\n") > 30 else "")

    def run(self, args: dict[str, Any], ctx: ToolContext) -> ToolResult:
        p = ctx.resolve(str(args["path"]))
        content = str(args.get("content", ""))
        existed = p.exists()
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(content, encoding="utf-8")
        ctx.state.setdefault("read_files", set()).add(str(p))
        return ToolResult(f"{'overwrote' if existed else 'created'} {p} ({len(content)} chars, {content.count(chr(10)) + 1} lines)")


class EditFile(Tool):
    name = "edit_file"
    read_only = False
    parameters = {
        "type": "object",
        "properties": {
            "path": {"type": "string", "description": "File to edit"},
            "old_string": {"type": "string", "description": "Exact text to replace"},
            "new_string": {"type": "string", "description": "Replacement text"},
            "replace_all": {"type": "boolean", "description": "Replace every occurrence (default false)"},
        },
        "required": ["path", "old_string", "new_string"],
    }

    def summary(self, args: dict[str, Any]) -> str:
        return str(args.get("path", ""))

    def detail(self, args: dict[str, Any], ctx: ToolContext) -> str:
        p = ctx.resolve(str(args.get("path", "")))
        try:
            text = p.read_text(encoding="utf-8")
            new_text = self._apply(text, args)
            diff = _unified(text, new_text, str(p))
            return diff or "(no change)"
        except Exception as e:  # show the problem instead of failing the prompt
            return f"edit {p}: {e}"

    @staticmethod
    def _apply(text: str, args: dict[str, Any]) -> str:
        old = str(args["old_string"])
        new = str(args["new_string"])
        if old == "":
            raise ValueError("old_string is empty")
        if old == new:
            raise ValueError("old_string and new_string are identical")
        n = text.count(old)
        if n == 0:
            raise ValueError("old_string not found in file; read the file and copy the text exactly")
        if n > 1 and not args.get("replace_all"):
            raise ValueError(f"old_string occurs {n} times; include more context to make it unique, or set replace_all")
        return text.replace(old, new) if args.get("replace_all") else text.replace(old, new, 1)

    def run(self, args: dict[str, Any], ctx: ToolContext) -> ToolResult:
        p = ctx.resolve(str(args["path"]))
        if not p.is_file():
            return ToolResult(f"no such file: {p}", is_error=True)
        read = ctx.state.get("read_files", set())
        if str(p) not in read and ctx.agent.settings.permission_mode != "auto":
            return ToolResult(f"read {p} with read_file before editing it", is_error=True)
        text = p.read_text(encoding="utf-8")
        try:
            new_text = self._apply(text, args)
        except ValueError as e:
            return ToolResult(str(e), is_error=True)
        p.write_text(new_text, encoding="utf-8")
        diff = _unified(text, new_text, str(p))
        return ToolResult(f"edited {p}\n{diff}")


def _unified(old: str, new: str, name: str, max_lines: int = 80) -> str:
    lines = list(difflib.unified_diff(old.splitlines(), new.splitlines(), fromfile=name, tofile=name, lineterm="", n=2))
    if len(lines) > max_lines:
        lines = lines[:max_lines] + [f"... ({len(lines) - max_lines} more diff lines)"]
    return "\n".join(lines)


class ListDir(Tool):
    name = "list_dir"
    read_only = True
    parameters = {
        "type": "object",
        "properties": {"path": {"type": "string", "description": "Directory (default: working directory)"}},
    }

    def summary(self, args: dict[str, Any]) -> str:
        return str(args.get("path", "."))

    def run(self, args: dict[str, Any], ctx: ToolContext) -> ToolResult:
        p = ctx.resolve(str(args.get("path") or "."))
        if not p.is_dir():
            return ToolResult(f"not a directory: {p}", is_error=True)
        entries = sorted(p.iterdir(), key=lambda e: (not e.is_dir(), e.name.lower()))
        rows = [f"{e.name}/" if e.is_dir() else e.name for e in entries[:500]]
        more = f"\n[... {len(entries) - 500} more]" if len(entries) > 500 else ""
        return ToolResult(f"{p}:\n" + "\n".join(rows) + more if rows else f"{p}: (empty)")


class GlobFiles(Tool):
    name = "glob_files"
    read_only = True
    parameters = {
        "type": "object",
        "properties": {
            "pattern": {"type": "string", "description": "Glob such as **/*.py"},
            "path": {"type": "string", "description": "Directory to search (default: working directory)"},
        },
        "required": ["pattern"],
    }

    def summary(self, args: dict[str, Any]) -> str:
        return f"{args.get('pattern', '')} in {args.get('path', '.')}"

    def run(self, args: dict[str, Any], ctx: ToolContext) -> ToolResult:
        root = ctx.resolve(str(args.get("path") or "."))
        pattern = str(args["pattern"])
        if not root.is_dir():
            return ToolResult(f"not a directory: {root}", is_error=True)
        hits: list[Path] = []
        for p in root.glob(pattern):
            rel = p.relative_to(root)
            if any(part in SKIP_DIRS for part in rel.parts[:-1]):
                continue
            if p.is_file():
                hits.append(p)
            if len(hits) >= 2000:
                break
        hits.sort(key=lambda p: p.stat().st_mtime, reverse=True)
        shown = hits[:500]
        body = "\n".join(str(p.relative_to(root)) for p in shown)
        if len(hits) > 500:
            body += f"\n[... {len(hits) - 500} more]"
        return ToolResult(body if body else "no matches")


class GrepFiles(Tool):
    name = "grep_files"
    read_only = True
    parameters = {
        "type": "object",
        "properties": {
            "pattern": {"type": "string", "description": "Regular expression"},
            "path": {"type": "string", "description": "Directory or file to search (default: working directory)"},
            "glob": {"type": "string", "description": "Only files matching this glob, like *.py"},
            "ignore_case": {"type": "boolean"},
            "context": {"type": "integer", "description": "Lines of context around each match"},
            "max_results": {"type": "integer", "description": "Cap on matching lines (default 200)"},
        },
        "required": ["pattern"],
    }

    def summary(self, args: dict[str, Any]) -> str:
        where = args.get("path", ".")
        g = f" ({args['glob']})" if args.get("glob") else ""
        return f"/{args.get('pattern', '')}/ in {where}{g}"

    def run(self, args: dict[str, Any], ctx: ToolContext) -> ToolResult:
        target = ctx.resolve(str(args.get("path") or "."))
        if not target.exists():
            return ToolResult(f"no such path: {target}", is_error=True)
        pattern = str(args["pattern"])
        max_results = int(args.get("max_results") or 200)
        context = int(args.get("context") or 0)
        rg = shutil.which("rg")
        if rg:
            cmd = [rg, "-n", "--no-heading", "--color", "never", "--max-count", str(max_results)]
            if args.get("ignore_case"):
                cmd.append("-i")
            if context:
                cmd += ["-C", str(context)]
            if args.get("glob"):
                cmd += ["-g", str(args["glob"])]
            cmd += ["-e", pattern, str(target)]
            try:
                proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=60)
            except subprocess.TimeoutExpired:
                return ToolResult("grep timed out", is_error=True)
            if proc.returncode == 2:
                return ToolResult(proc.stderr.decode("utf-8", "replace").strip(), is_error=True)
            lines = proc.stdout.decode("utf-8", "replace").splitlines()
        else:
            lines = self._python_grep(target, pattern, args, context)
        if not lines:
            return ToolResult("no matches")
        if len(lines) > max_results:
            lines = lines[:max_results] + [f"[... capped at {max_results} lines]"]
        body = "\n".join(_relativize(line, ctx.cwd) for line in lines)
        return ToolResult(truncate_middle(body, ctx.agent.tool_output_limit))

    @staticmethod
    def _python_grep(target: Path, pattern: str, args: dict[str, Any], context: int) -> list[str]:
        flags = re.IGNORECASE if args.get("ignore_case") else 0
        try:
            rx = re.compile(pattern, flags)
        except re.error as e:
            return [f"bad regex: {e}"]
        glob = str(args.get("glob") or "*")
        files = [target] if target.is_file() else [
            p for p in target.rglob(glob)
            if p.is_file() and not any(part in SKIP_DIRS for part in p.relative_to(target).parts[:-1])
        ]
        out: list[str] = []
        for f in files:
            try:
                raw = f.read_bytes()
            except OSError:
                continue
            if _is_binary(raw[:4096]):
                continue
            lines = raw.decode("utf-8", "replace").splitlines()
            for i, line in enumerate(lines):
                if rx.search(line):
                    lo, hi = max(0, i - context), min(len(lines), i + context + 1)
                    for j in range(lo, hi):
                        sep = ":" if j == i else "-"
                        out.append(f"{f}{sep}{j + 1}{sep}{lines[j]}")
                    if len(out) > 5000:
                        return out
        return out


def _relativize(line: str, cwd: Path) -> str:
    prefix = str(cwd) + os.sep
    return line[len(prefix):] if line.startswith(prefix) else line
