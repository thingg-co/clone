#!/usr/bin/env python3
"""Tool-calling smoke test for an OpenAI-compatible server. Standard library only.

    python3 scripts/smoke-test.py --base-url http://omega.local:8080/v1 --api-key KEY [--model NAME]

Exit 0 when the model calls the tool and then uses its result.
"""

import argparse
import json
import sys
import time
import urllib.error
import urllib.request

TOOLS = [{
    "type": "function",
    "function": {
        "name": "get_weather",
        "description": "Get the current weather for a city",
        "parameters": {"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]},
    },
}]


def post(url, key, payload, timeout=600):
    req = urllib.request.Request(url, data=json.dumps(payload).encode(), method="POST",
                                 headers={"Content-Type": "application/json", "Authorization": f"Bearer {key}"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.load(r)


def get(url, key):
    req = urllib.request.Request(url, headers={"Authorization": f"Bearer {key}"})
    with urllib.request.urlopen(req, timeout=30) as r:
        return json.load(r)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", default="http://127.0.0.1:8080/v1")
    ap.add_argument("--api-key", default="local")
    ap.add_argument("--model", default=None)
    a = ap.parse_args()
    base = a.base_url.rstrip("/")

    try:
        models = [m["id"] for m in get(f"{base}/models", a.api_key)["data"]]
    except (urllib.error.URLError, KeyError, OSError) as e:
        print(f"FAIL cannot list models at {base}: {e}")
        return 2
    model = a.model or models[0]
    print(f"server {base}\nmodels {', '.join(models)}\nusing  {model}")

    msgs = [{"role": "system", "content": "You are a helpful assistant. Use the tools when they apply."},
            {"role": "user", "content": "What is the weather in Lisbon right now? Use the tool."}]
    t0 = time.time()
    r = post(f"{base}/chat/completions", a.api_key, {"model": model, "messages": msgs, "tools": TOOLS, "max_tokens": 512})
    dt = time.time() - t0
    msg = r["choices"][0]["message"]
    calls = msg.get("tool_calls") or []
    if not calls or calls[0]["function"]["name"] != "get_weather":
        print(f"FAIL no tool call. content={msg.get('content')!r}")
        return 1
    call = calls[0]
    args = json.loads(call["function"]["arguments"] or "{}")
    print(f"step 1 ok: get_weather({args}) in {dt:.1f}s")

    asst = {"role": "assistant", "content": msg.get("content"), "tool_calls": calls}
    if msg.get("reasoning_content"):
        asst["reasoning_content"] = msg["reasoning_content"]
    msgs += [asst, {"role": "tool", "tool_call_id": call.get("id", "call_0"), "name": "get_weather",
                    "content": json.dumps({"city": "Lisbon", "temp_c": 21, "sky": "clear"})}]
    t0 = time.time()
    r = post(f"{base}/chat/completions", a.api_key, {"model": model, "messages": msgs, "tools": TOOLS, "max_tokens": 256})
    dt = time.time() - t0
    text = r["choices"][0]["message"].get("content") or ""
    u = r.get("usage", {})
    tps = u.get("completion_tokens", 0) / dt if dt else 0
    if "21" not in text:
        print(f"WARN tool result not used: {text[:120]!r}")
        return 1
    print(f"step 2 ok: {text.strip()[:100]!r}")
    print(f"usage  prompt {u.get('prompt_tokens')} completion {u.get('completion_tokens')} tokens, ~{tps:.0f} tok/s")
    print("PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
