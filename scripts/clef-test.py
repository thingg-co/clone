#!/usr/bin/env python3
"""The recall test, adapted for a decision model (Clef) served at /v1/systemone. Standard library only.

    python3 scripts/clef-test.py --base-url http://127.0.0.1:8080 --api-key KEY --label clef

A decision model writes no text, so it cannot return codes through a tool call. It gets
the same numbered records as scripts/context-test.py and two kinds of question:

  pick    which of eight codes belongs to locker N. The wrong options are the codes of
          the records next to it, ten away, and with two digits of the locker number
          swapped: the mistakes the generative models made.
  claims  "the code for locker N is X", true for some lockers and a neighbour's code for
          others. This is the job of a checker next to the main model.

Three lockers per size, near the start, the middle and the end. One JSON line per size
goes to --out, with the probability the model gave each answer.
"""

import argparse
import json
import random
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent
TOKENS_PER_RECORD = 27.4  # first guess, corrected from the server's count


def post(base: str, key: str, body: dict, timeout: int) -> dict:
    req = urllib.request.Request(base + "/v1/systemone", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json", "Authorization": f"Bearer {key}"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.load(r)


def swapped(n: int, limit: int):
    """n with two neighbouring digits swapped, if that is another locker that exists."""
    s = str(n)
    for i in range(len(s) - 1):
        t = s[:i] + s[i + 1] + s[i] + s[i + 2:]
        if t != s and not t.startswith("0") and int(t) < limit:
            return int(t)
    return None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", default="http://127.0.0.1:8080", help="server root (no /v1)")
    ap.add_argument("--api-key", default="local")
    ap.add_argument("--model", default="clef")
    ap.add_argument("--label", default="clef")
    ap.add_argument("--title", default="")
    ap.add_argument("--detail", default="")
    ap.add_argument("--sizes", default="4,16,56", help="record text sizes in thousands of tokens; the profiles take 64k")
    ap.add_argument("--timeout", type=int, default=3600)
    ap.add_argument("--out", default=str(HERE.parent / "run" / "results.jsonl"))
    args = ap.parse_args()
    base = args.base_url.rstrip("/").removesuffix("/v1")

    random.seed(7)
    sizes = [int(s) * 1000 for s in args.sizes.split(",")]
    codes = [random.randint(100000, 999999) for _ in range(int(max(sizes) / 8) + 1)]
    rng = random.Random(11)
    per_record, steps = TOKENS_PER_RECORD, []
    for size in sizes:
        n = int(size / per_record * (0.96 if per_record == TOKENS_PER_RECORD else 1))
        lockers = [min(137, n // 8), n // 2 + 11, n - min(205, n // 8)]
        state = "\n".join(f"Record {i}: the access code for locker {i} is {codes[i]}." for i in range(n))
        picks, claims, truth = {}, {}, {}
        for k, locker in enumerate(lockers):
            wrong = [codes[j] for j in (locker - 1, locker + 1, locker - 10, locker + 10) if 0 <= j < n]
            sw = swapped(locker, n)
            if sw is not None:
                wrong.append(codes[sw])
            wrong += rng.sample(codes[:n], 8)
            options = [codes[locker]] + [c for c in dict.fromkeys(wrong) if c != codes[locker]][:7]
            rng.shuffle(options)
            picks[f"pick{k}"] = {"type": "choice", "instructions": f"Which is the access code for locker {locker}?",
                                 "criteria": {str(c): None for c in options}}
            # one true and one false claim per locker, in separate requests so neither gives the other away
            near = codes[sw] if sw is not None else codes[locker + 1]
            for r, claim in enumerate((codes[locker], near) if k % 2 == 0 else (near, codes[locker])):
                claims.setdefault(r, {})[f"claim{k}"] = {"type": "noul", "instructions": f"Is the access code for locker {locker} {claim}?"}
                truth[(r, k)] = claim == codes[locker]
        t0, tokens, result = time.time(), 0, {"picks": [], "claims": []}
        try:
            d = post(base, args.api_key, {"model": args.model, "state": state, "questions": picks}, args.timeout)
            tokens = d.get("usage", {}).get("input_tokens", 0)
            for k, locker in enumerate(lockers):
                a = d["answers"][f"pick{k}"]
                result["picks"].append({"locker": locker, "right": a["choice"] == str(codes[locker]),
                                        "p_right": round(a["probabilities"].get(str(codes[locker]), 0.0), 3), "chose": a["choice"]})
            for r in sorted(claims):
                d = post(base, args.api_key, {"model": args.model, "state": state, "questions": claims[r]}, args.timeout)
                for k, locker in enumerate(lockers):
                    p = d["answers"][f"claim{k}"]["noul"]
                    result["claims"].append({"locker": locker, "true": truth[(r, k)], "p_true": round(p, 3), "right": (p >= 0.5) == truth[(r, k)]})
        except (urllib.error.URLError, OSError, KeyError, ValueError) as e:
            result["error"] = str(e)[:300]
        seconds = round(time.time() - t0)
        if tokens:
            per_record = tokens / n
        step = {"tokens": tokens or size, "picks_right": sum(p["right"] for p in result["picks"]), "picks_of": len(lockers),
                "claims_right": sum(c["right"] for c in result["claims"]), "claims_of": 2 * len(lockers), "seconds": seconds, **result}
        steps.append(step)
        print(f"{step['tokens']:>9,} tokens  {seconds:5d}s  picked {step['picks_right']}/{step['picks_of']}  "
              f"claims judged {step['claims_right']}/{step['claims_of']}" + (f"  error: {result['error']}" if "error" in result else ""), flush=True)
        for c in result["claims"]:
            print(f"            locker {c['locker']:>6}  {'true ' if c['true'] else 'false'} claim  p(true) {c['p_true']:.3f}", flush=True)
        if "error" in result:
            break
    record = {"kind": "checker", "date": time.strftime("%Y-%m-%d"), "label": args.label, "title": args.title or args.label,
              "detail": args.detail, "model": args.model, "steps": steps}
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(record) + "\n")
    print(f"recorded in {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
