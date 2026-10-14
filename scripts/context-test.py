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
    args = ap.parse_args()
    headers = {"Content-Type": "application/json", "Authorization": f"Bearer {args.api_key}"}
    base = args.base_url.rstrip("/")

    with urllib.request.urlopen(urllib.request.Request(base + "/models", headers=headers), timeout=20) as r:
        listed = json.load(r)["data"]
    model = args.model or listed[0]["id"]
    window = next((m.get("max_model_len") for m in listed if m["id"] == model), None)
    print(f"model {model}, window {window or 'not reported'}")

    sizes = [int(s) * 1000 for s in args.sizes.split(",")]
    random.seed(7)
    codes = [random.randint(100000, 999999) for _ in range(int(max(sizes) / 20) + 1)]  # more than any tokenizer needs
    per_record = TOKENS_PER_RECORD
    failed = False
    for size in sizes:
        if window and size > window - 8000:
            size = window - 8000  # leave room for the question, the answer and its reasoning
        n = int(size / per_record)
        if per_record == TOKENS_PER_RECORD:
            n = int(n * 0.96)  # tokenizers differ by a few percent; stay under until this one is measured
        asks = [137, n // 2 + 11, n - 205]
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
        except urllib.error.HTTPError as e:
            print(f"{size:>9,} tokens asked: HTTP {e.code} {e.read()[:200]!r}")
            failed = True
            continue
        per_record = d["usage"]["prompt_tokens"] / n
        calls = d["choices"][0]["message"].get("tool_calls") or []
        try:
            got = json.loads(calls[0]["function"]["arguments"]).get("codes") if calls else None
        except (ValueError, AttributeError):
            got = None
        want = [codes[a] for a in asks]
        right = sum(1 for g, w in zip(got or [], want) if g == w)
        where = ", ".join(f"{name} {'ok' if (got or [None] * 3)[i:i + 1] == [want[i]] else 'WRONG'}" for i, name in enumerate(("start", "middle", "end")))
        print(f"{d['usage']['prompt_tokens']:>9,} prompt tokens  {time.time() - t0:6.0f}s  tool call {'yes' if calls else 'NO'}  recall {right}/3  ({where})", flush=True)
        if args.verbose:
            msg = d["choices"][0]["message"]
            print(f"            asked {asks}  want {want}  got {got}")
            print(f"            raw arguments: {[c['function'].get('arguments', '')[:200] for c in calls] or (msg.get('content') or '')[:200]!r}")
            print(f"            reasoning: {(msg.get('reasoning_content') or msg.get('reasoning') or '')[:400]!r}", flush=True)
        failed = failed or right != 3
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
