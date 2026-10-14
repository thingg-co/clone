#!/usr/bin/env python3
"""Measure the model a server is serving and append the result to a file. Standard library only.

    python3 scripts/bench.py --base-url http://127.0.0.1:8080/v1 --api-key KEY --label qwen3.8-27b

Four things, in the order that matters here: recall across the window
(scripts/context-test.py does that part), whether the model thinks, whether
it calls a tool and uses the result, and how fast it decodes. One JSON line
per run goes to --out, which is what the control page ranks models from.
"""

import argparse
import json
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent
TOOLS = [{
    "type": "function",
    "function": {
        "name": "get_weather",
        "description": "Get the current weather for a city",
        "parameters": {"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]},
    },
}]


class Server:
    def __init__(self, base: str, key: str):
        self.base, self.headers = base.rstrip("/"), {"Content-Type": "application/json", "Authorization": f"Bearer {key}"}

    def get(self, path: str, root: bool = False):
        url = (self.base.removesuffix("/v1") if root else self.base) + path
        with urllib.request.urlopen(urllib.request.Request(url, headers=self.headers), timeout=30) as r:
            return json.load(r)

    def post(self, body: dict, timeout: int = 900):
        req = urllib.request.Request(self.base + "/chat/completions", data=json.dumps(body).encode(), headers=self.headers)
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return json.load(r)

    def stream(self, body: dict, timeout: int = 900):
        """Yield the JSON chunks of a streamed completion."""
        body = {**body, "stream": True, "stream_options": {"include_usage": True}}
        req = urllib.request.Request(self.base + "/chat/completions", data=json.dumps(body).encode(), headers=self.headers)
        with urllib.request.urlopen(req, timeout=timeout) as r:
            for raw in r:
                line = raw.decode("utf-8", "replace").strip()
                if line.startswith("data:") and line[5:].strip() != "[DONE]":
                    try:
                        yield json.loads(line[5:])
                    except ValueError:
                        pass


def window_of(server: Server, model: str):
    for m in server.get("/models")["data"]:
        if m["id"] == model and m.get("max_model_len"):  # vLLM
            return int(m["max_model_len"])
    try:  # llama.cpp
        return int(server.get("/props", root=True)["default_generation_settings"]["n_ctx"])
    except (urllib.error.URLError, KeyError, ValueError, OSError):
        return None


def reasoning_of(message: dict) -> str:
    text = message.get("reasoning_content") or message.get("reasoning") or ""
    content = (message.get("content") or "").lstrip()
    if not text and content.startswith("<think>"):  # a server without a reasoning parser
        text = content[7:].split("</think>")[0]
    return text


def speed(server: Server, model: str) -> dict:
    """Decode speed on a short prompt: tokens after the first one, over the time they took."""
    t0, first, last, pieces, usage = time.time(), None, None, 0, {}
    for chunk in server.stream({"model": model, "max_tokens": 400, "messages": [{"role": "user", "content": "Write the numbers from 1 to 150, one per line, and nothing else."}]}):
        if chunk.get("usage"):
            usage = chunk["usage"]
        for choice in chunk.get("choices") or []:
            delta = choice.get("delta") or {}
            if delta.get("content") or delta.get("reasoning_content") or delta.get("reasoning"):
                first = first or time.time()
                last = time.time()
                pieces += 1
    tokens = usage.get("completion_tokens") or pieces
    if not first or last <= first or tokens < 2:
        return {"decode_tps": None, "ttft_ms": None}
    return {"decode_tps": round((tokens - 1) / (last - first), 1), "ttft_ms": round((first - t0) * 1000)}


def tools_and_thinking(server: Server, model: str) -> dict:
    """One request that should think and call a tool, then a second that should use the tool's answer."""
    out = {"thinking": False, "tool_call": False, "tool_loop": False}
    msgs = [{"role": "system", "content": "You are a helpful assistant. Use the tools when they apply."},
            {"role": "user", "content": "What is the weather in Lisbon right now? Use the tool."}]
    msg = server.post({"model": model, "messages": msgs, "tools": TOOLS, "max_tokens": 8192})["choices"][0]["message"]
    out["thinking"] = bool(reasoning_of(msg).strip())
    calls = msg.get("tool_calls") or []
    if not calls or calls[0]["function"]["name"] != "get_weather":
        return out
    out["tool_call"] = True
    assistant = {"role": "assistant", "content": msg.get("content"), "tool_calls": calls}
    if msg.get("reasoning_content"):
        assistant["reasoning_content"] = msg["reasoning_content"]
    msgs += [assistant, {"role": "tool", "tool_call_id": calls[0].get("id", "call_0"), "name": "get_weather",
                         "content": json.dumps({"city": "Lisbon", "temp_c": 21, "sky": "clear"})}]
    final = server.post({"model": model, "messages": msgs, "tools": TOOLS, "max_tokens": 8192})["choices"][0]["message"]
    out["tool_loop"] = "21" in (final.get("content") or "")
    out["thinking"] = out["thinking"] or bool(reasoning_of(final).strip())
    return out


def ladder(window) -> list:
    """Prompt sizes, in thousands of tokens: 100, 250, 500 and 1000 where they fit, then the top of the window."""
    if not window:
        return [100]
    top = (window - 8000) // 1000
    sizes = [s for s in (100, 250, 500, 1000) if s <= top]
    if not sizes or top > sizes[-1] * 1.1:
        sizes.append(top)
    return sizes


def recall(args, model: str, sizes: list) -> list:
    with tempfile.NamedTemporaryFile("r", suffix=".jsonl") as tmp:
        cmd = [sys.executable, "-I", str(HERE / "context-test.py"), "--base-url", args.base_url, "--api-key", args.api_key, "--model", model,
               "--sizes", ",".join(map(str, sizes)), "--timeout", str(args.timeout), "--record", tmp.name, "--label", args.label, "--verbose"]
        subprocess.run(cmd, check=False)
        rows = [json.loads(line) for line in Path(tmp.name).read_text().splitlines() if line.strip()]
    return [step(r) for r in rows]


def step(r: dict) -> dict:
    """One recall step as results.jsonl keeps it, from a context-test.py --record line."""
    return {"tokens": r["prompt_tokens"], "right": r["right"], "of": r["of"], "tool_call": r["tool_call"], "seconds": r["seconds"],
            "finish": r.get("finish"), "written": r.get("written"), **({"samples": r["samples"]} if r.get("samples") else {})}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", default="http://127.0.0.1:8080/v1")
    ap.add_argument("--api-key", default="local")
    ap.add_argument("--model", help="default: the first model the server lists")
    ap.add_argument("--label", required=True, help="what to file the result under, normally the profile key")
    ap.add_argument("--title", help="display name (default: the label)")
    ap.add_argument("--detail", default="", help="one line about the build: size, quantisation")
    ap.add_argument("--note", default="", help="one line about the configuration: stretched or native, cache precision")
    ap.add_argument("--sizes", help="prompt sizes in thousands of tokens (default: a ladder up to the window)")
    ap.add_argument("--timeout", type=int, default=14400, help="seconds allowed for one recall request")
    ap.add_argument("--recall-from", metavar="FILE", help="take the recall steps recorded under --label in FILE "
                    "(context-test.py --record) instead of running them again")
    ap.add_argument("--out", default=str(HERE.parent / "run" / "results.jsonl"))
    args = ap.parse_args()
    server = Server(args.base_url, args.api_key)

    try:
        model = args.model or server.get("/models")["data"][0]["id"]
    except (urllib.error.URLError, KeyError, IndexError, OSError) as e:
        print(f"cannot list models at {args.base_url}: {e}")
        return 2
    window = window_of(server, model)
    record = {"kind": "bench", "date": time.strftime("%Y-%m-%d"), "label": args.label, "title": args.title or args.label,
              "detail": args.detail, "note": args.note, "model": model, "window": window}
    print(f"{args.label}: {model}, window {window}", flush=True)

    for name, step in (("speed", speed), ("tools and thinking", tools_and_thinking)):
        try:
            record.update(step(server, model))
        except (urllib.error.URLError, OSError, KeyError, ValueError) as e:  # a failed check is a result too
            record.setdefault("errors", []).append(f"{name}: {e}")
        print(f"  {name}: " + ", ".join(f"{k} {record.get(k)}" for k in ("decode_tps", "ttft_ms", "thinking", "tool_call", "tool_loop") if k in record), flush=True)

    if args.recall_from:
        rows = [json.loads(line) for line in Path(args.recall_from).read_text().splitlines() if line.strip()]
        record["recall"] = [step(r) for r in rows if r.get("label") == args.label]
    else:
        sizes = [int(s) for s in args.sizes.split(",")] if args.sizes else ladder(window)
        record["recall"] = recall(args, model, sizes)
    if record["recall"]:
        first = record["recall"][0]
        record["prefill_tps"] = round(first["tokens"] / max(1, first["seconds"]))
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(record) + "\n")
    print(f"recorded in {out}: " + ", ".join(f"{r['tokens'] // 1000}k {r['right']}/{r['of']}" for r in record["recall"]), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
