#!/usr/bin/env python3
"""Long-context acceptance test for an OpenAI-compatible server. Standard library only.

    python3 scripts/context-test.py --base-url http://omega.local:8080/v1 --api-key KEY [--sizes 100,250,500,1000]

Fills the prompt with numbered records, each holding a random six-digit code,
then asks for three of them (near the start, the middle and the end) through a
tool call. It checks three things at once at each size: the server accepts the
prompt, the model still calls the tool, and it can find facts anywhere in the
window. Sizes are thousands of tokens, approximate. Every size is a prefix of
the next, so with prefix caching only the first pass over new text is slow.

Exit 0 when every size returns all three codes.
"""

import argparse
import json
import random
import sys
import time
import urllib.error
import urllib.request

TOKENS_PER_RECORD = 27.4  # a first guess; corrected from the server's own count after each request
TOOLS = [{
    "type": "function",
    "function": {
        "name": "report_codes",
        "description": "Report the access codes that were asked for",
        "parameters": {"type": "object", "properties": {"codes": {"type": "array", "items": {"type": "integer"}, "description": "the codes, in the order asked"}}, "required": ["codes"]},
    },
}]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", default="http://localhost:8080/v1")
    ap.add_argument("--api-key", default="local")
    ap.add_argument("--model", help="default: the first model the server lists")
    ap.add_argument("--sizes", default="100,250,500,1000", help="prompt sizes in thousands of tokens")
    ap.add_argument("--timeout", type=int, default=3600)
    ap.add_argument("--verbose", action="store_true", help="also print what was asked, what came back and the start of the reasoning")
    ap.add_argument("--record", metavar="FILE", help="append each result to FILE as a JSON line (the control page reads run/recall.jsonl)")
    ap.add_argument("--label", help="name to record the results under, normally the profile key (default: the model id)")
    args = ap.parse_args()
    headers = {"Content-Type": "application/json", "Authorization": f"Bearer {args.api_key}"}
    base = args.base_url.rstrip("/")

    with urllib.request.urlopen(urllib.request.Request(base + "/models", headers=headers), timeout=20) as r:
        listed = json.load(r)["data"]
    model = args.model or listed[0]["id"]
    window = next((m.get("max_model_len") for m in listed if m["id"] == model), None)
    if window is None:  # llama.cpp reports its window under /props
        try:
            with urllib.request.urlopen(urllib.request.Request(base.removesuffix("/v1") + "/props", headers=headers), timeout=20) as r:
                window = json.load(r)["default_generation_settings"]["n_ctx"]
        except (urllib.error.URLError, OSError, KeyError, ValueError):
            pass
    print(f"model {model}, window {window or 'not reported'}")

    sizes = [int(s) * 1000 for s in args.sizes.split(",")]
    random.seed(7)
    codes = [random.randint(100000, 999999) for _ in range(int(max(sizes) / 8) + 1)]  # more than any tokenizer needs
    per_record = TOKENS_PER_RECORD
    failed = False
    for size in sizes:
        if window and size > window - 8000:
            size = window - 8000  # leave room for the question, the answer and its reasoning
        n = int(size / per_record)
        if per_record == TOKENS_PER_RECORD:
            n = int(n * 0.96)  # tokenizers differ by a few percent; stay under until this one is measured
        elif window and size > window * 0.9:
            n = int(n * 0.985)  # record numbers get longer as they count up, so later records cost more
        d = None
        for attempt in range(4):
            asks = [min(137, n // 8), n // 2 + 11, n - min(205, n // 8)]  # start, middle and end, also for small tests
            records = "\n".join(f"Record {i}: the access code for locker {i} is {codes[i]}." for i in range(n))
            body = {
                "model": model, "max_tokens": 4000, "temperature": 0, "tools": TOOLS,
                "messages": [
                    {"role": "system", "content": "You answer from the records the user provides. Report results with the report_codes tool."},
                    {"role": "user", "content": f"{records}\n\nFrom the records above, find the access codes for lockers {asks[0]}, {asks[1]} and {asks[2]}, in that order, and report them with the tool."},
                ],
            }
            t0 = time.time()
            try:
                req = urllib.request.Request(base + "/chat/completions", data=json.dumps(body).encode(), headers=headers)
                with urllib.request.urlopen(req, timeout=args.timeout) as r:
                    d = json.load(r)
                break
            except urllib.error.HTTPError as e:
                text = e.read()[:300]
                if e.code == 400 and b"context length" in text and attempt < 3:
                    n = int(n * 0.98)  # over the window after all: fewer records and try again
                    continue
                print(f"{size:>9,} tokens asked: HTTP {e.code} {text[:200]!r}")
                break
        if d is None:
            failed = True
            continue
        per_record = d["usage"]["prompt_tokens"] / n
        calls = d["choices"][0]["message"].get("tool_calls") or []
        # "length" means the answer ran into max_tokens, often while still thinking
        finish, written = d["choices"][0].get("finish_reason"), d["usage"].get("completion_tokens")
        try:
            got = json.loads(calls[0]["function"]["arguments"]).get("codes") if calls else None
        except (ValueError, AttributeError):
            got = None
        want = [codes[a] for a in asks]
        right = sum(1 for g, w in zip(got or [], want) if g == w)
        where = ", ".join(f"{name} {'ok' if (got or [None] * 3)[i:i + 1] == [want[i]] else 'WRONG'}" for i, name in enumerate(("start", "middle", "end")))
        seconds = time.time() - t0
        print(f"{d['usage']['prompt_tokens']:>9,} prompt tokens  {seconds:6.0f}s  tool call {'yes' if calls else 'NO'}  recall {right}/3  ({where})", flush=True)
        if args.verbose:
            msg = d["choices"][0]["message"]
            print(f"            asked {asks}  want {want}  got {got}  (stopped: {finish}, {written} tokens written)")
            print(f"            raw arguments: {[c['function'].get('arguments', '')[:200] for c in calls] or (msg.get('content') or '')[:200]!r}")
            print(f"            reasoning: {(msg.get('reasoning_content') or msg.get('reasoning') or '')[:400]!r}", flush=True)
        if args.record:  # one JSON line per size, for the control page
            with open(args.record, "a", encoding="utf-8") as fh:
                fh.write(json.dumps({
                    "date": time.strftime("%Y-%m-%d"), "label": args.label or model, "model": model, "window": window,
                    "prompt_tokens": d["usage"]["prompt_tokens"], "right": right, "of": 3, "tool_call": bool(calls), "seconds": round(seconds),
                    "finish": finish, "written": written,
                }) + "\n")
        failed = failed or right != 3
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
