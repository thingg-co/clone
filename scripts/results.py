#!/usr/bin/env python3
"""Publish the measured results: merge new bench runs into docs/results.json and rewrite
the results tables in docs/index.html and README.md from it. Standard library only.

    python3 scripts/results.py                         # print the ranking
    scp omega.local:frank/run/results.jsonl run/       # fetch the Spark's runs first
    python3 scripts/results.py --publish               # write docs/results.json and the table

The ranking is the control page's (spark/panel.py results()): the largest prompt answered
exactly with every smaller one exact too, then the largest with at most one code wrong at each
size, then the share of codes right, then decode speed, with sizes to the nearest 50k.
"""

import argparse
import html
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "spark"))
from panel import results  # noqa: E402  (the one ranking, shared with the control page)

FIELDS = ("label", "title", "detail", "lab", "note", "model", "window", "decode_tps", "prefill_tps",
          "thinking", "tool_call", "tool_loop", "recall", "date")
START, END = "<!-- results:start -->", "<!-- results:end -->"


def tokens(n) -> str:
    if not n:
        return "?"
    if n >= 1_000_000:
        return f"{n / 1e6:.2f}".rstrip("0").rstrip(".") + "M"
    return f"{round(n / 1000)}k"


def rate(n) -> str:
    return "?" if n is None else f"{round(n):,} tok/s"


def rows(models: list) -> str:
    out = []
    for m in models:
        steps = " ".join(
            f'<span class="{"pass" if r["right"] == r["of"] else "fail"}">{tokens(r["tokens"])}&nbsp;{r["right"]}/{r["of"]}</span>'
            for r in m["recall"])
        checks = "yes" if m.get("thinking") and m.get("tool_loop") else ", ".join(
            [("thinks" if m.get("thinking") else "no thinking"), ("tool loop" if m.get("tool_loop") else "no tool loop")])
        out.append(
            "          <tr>\n"
            f'            <td class="model"><b>{html.escape(m.get("title") or m["label"])}</b><span>{html.escape(m.get("detail") or "")}</span></td>\n'
            f'            <td>{tokens(m.get("window"))}<br><small>{html.escape(m.get("note") or "")}</small></td>\n'
            f'            <td class="num {"pass" if m["reach"] else "fail"}">{tokens(m["reach"]) if m["reach"] else "none"}</td>\n'
            f'            <td class="steps">{steps}</td>\n'
            f'            <td>{checks}</td>\n'
            f'            <td class="num">{rate(m.get("decode_tps"))}</td>\n'
            f'            <td class="num">{rate(m.get("prefill_tps"))}</td>\n'
            "          </tr>")
    return "\n".join(out)


def markdown(models: list) -> str:
    out = ["| | model | window served | exact up to | codes right, by prompt size | thinks, tools | decode | read-in |",
           "|---|---|---|---|---|---|---|---|"]
    for i, m in enumerate(models, 1):
        steps = ", ".join(f"{tokens(r['tokens'])} {r['right']}/{r['of']}" for r in m["recall"])
        checks = "yes" if m.get("thinking") and m.get("tool_loop") else ("tool loop only" if m.get("tool_loop") else "no")
        name = (m.get("title") or m["label"]) + (f" ({m['detail']})" if m.get("detail") else "")
        window = tokens(m.get("window")) + (f", {m['note']}" if m.get("note") else "")
        out.append(f"| {i} | {name} | {window} | {tokens(m['reach']) if m['reach'] else 'none'} | {steps} | {checks} | "
                   f"{rate(m.get('decode_tps'))} | {rate(m.get('prefill_tps'))} |")
    return "\n".join(out)


def splice(path: Path, block: str, indent: str = "") -> bool:
    """Replace what sits between the markers in a file. False when the markers are missing."""
    text = path.read_text()
    if text.count(START) != 1 or text.count(END) != 1:
        print(f"{path} has no {START} ... {END} block; left alone")
        return False
    head, rest = text.split(START)
    _, tail = rest.split(END)
    path.write_text(head + START + "\n" + block + "\n" + indent + END + tail)
    return True


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default=str(ROOT / "run" / "results.jsonl"), help="bench runs to merge (one JSON line each)")
    ap.add_argument("--publish", action="store_true", help="write docs/results.json and the table in docs/index.html")
    args = ap.parse_args()
    published = ROOT / "docs" / "results.json"
    merged = results(published, Path(args.runs))
    for i, m in enumerate(merged["models"], 1):
        steps = ", ".join(f"{tokens(r['tokens'])} {r['right']}/{r['of']}" for r in m["recall"])
        print(f"{i:2}. {m.get('title') or m['label']:<26} {m['label']:<26} exact to {tokens(m['reach']) if m['reach'] else 'none':<6} {steps}")
    if not args.publish:
        return 0

    base = json.loads(published.read_text()) if published.is_file() else {}
    base["updated"] = time.strftime("%Y-%m-%d")
    base["models"] = [{k: m[k] for k in FIELDS if k in m} for m in merged["models"]]
    if merged.get("checkers"):
        base["checkers"] = merged["checkers"]
    published.write_text(json.dumps(base, indent=1) + "\n")

    ok = splice(ROOT / "docs" / "index.html", rows(merged["models"]), " " * 10)
    ok = splice(ROOT / "README.md", markdown(merged["models"])) and ok
    print(f"wrote {published.relative_to(ROOT)} and the tables in docs/index.html and README.md")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
