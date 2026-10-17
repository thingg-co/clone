from __future__ import annotations

import re
from html.parser import HTMLParser
from typing import Any

import httpx

from .base import Tool, ToolContext, ToolResult


class _Text(HTMLParser):
    SKIP = {"script", "style", "noscript", "svg", "head"}
    BLOCK = {"p", "div", "br", "li", "h1", "h2", "h3", "h4", "h5", "h6", "tr", "pre", "section", "article", "blockquote"}

    def __init__(self) -> None:
        super().__init__()
        self.parts: list[str] = []
        self._skip = 0

    def handle_starttag(self, tag: str, attrs: Any) -> None:
        if tag in self.SKIP:
            self._skip += 1
        elif tag in self.BLOCK:
            self.parts.append("\n")

    def handle_endtag(self, tag: str) -> None:
        if tag in self.SKIP and self._skip:
            self._skip -= 1
        elif tag in self.BLOCK:
            self.parts.append("\n")

    def handle_data(self, data: str) -> None:
        if not self._skip:
            self.parts.append(data)


def html_to_text(html: str) -> str:
    p = _Text()
    p.feed(html)
    text = "".join(p.parts)
    text = re.sub(r"[ \t\r\f\v]+", " ", text)
    text = re.sub(r"\n\s*\n+", "\n\n", text)
    return text.strip()


class WebFetch(Tool):
    name = "web_fetch"
    read_only = True
    parameters = {
        "type": "object",
        "properties": {
            "url": {"type": "string", "description": "http(s) URL"},
            "max_chars": {"type": "integer", "description": "Truncate the text to this many characters (default 20000)"},
        },
        "required": ["url"],
    }

    def summary(self, args: dict[str, Any]) -> str:
        return str(args.get("url", ""))

    def run(self, args: dict[str, Any], ctx: ToolContext) -> ToolResult:
        url = str(args["url"])
        if not url.startswith(("http://", "https://")):
            return ToolResult("only http(s) URLs are supported", is_error=True)
        max_chars = int(args.get("max_chars") or 20000)
        try:
            r = httpx.get(url, follow_redirects=True, timeout=30, headers={"User-Agent": "frank-local/0.1 (+local agent)"})
        except httpx.HTTPError as e:
            return ToolResult(f"fetch failed: {e}", is_error=True)
        if r.status_code >= 400:
            return ToolResult(f"HTTP {r.status_code} for {url}", is_error=True)
        ctype = r.headers.get("content-type", "")
        body = html_to_text(r.text) if "html" in ctype else r.text
        if len(body) > max_chars:
            body = body[:max_chars] + f"\n[... truncated at {max_chars} of {len(body)} characters]"
        return ToolResult(f"{url} ({ctype.split(';')[0] or 'unknown type'}, {len(r.content)} bytes)\n\n{body}")
