#!/usr/bin/env python3
"""Grade stored answers against each item's `expect` block with an LLM judge.

Every key under `expect` becomes one or more binary checks (list values are
split, one check per element). Keys starting with "should_" are soft: they
are reported but do not decide whether the item passes. An item passes when
all of its hard checks pass.

Two judges are supported, both read from the environment so no key is ever
passed on the command line:

    --judge mimo   CODEATLAS_LLM_BASE_URL / _API_KEY / _MODEL  (chat/completions)
    --judge luna   CODEATLAS_SHARED_LLM_LUNA_BASE_URL / _API_KEY / _MODEL  (/v1/responses)

Both are also contestants in the runs they grade, so run both judges and
compare: agreement is the signal, and a judge that favours its own model
shows up as disagreement rather than hiding inside one score.

Usage:
    python evals/judge.py --answers run/answers.jsonl --judge mimo --out run/judged-mimo.jsonl
    python evals/judge.py --report run/judged-mimo.jsonl run/judged-luna.jsonl
"""

import argparse
import json
import os
import re
import sys
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import requests
import yaml

ITEMS = Path(__file__).parent / "admin_config" / "items.yaml"

INSTRUCTIONS = """You are grading an answer written for a product manager about a Django admin.
Each check below is one thing a correct answer must satisfy. Grade every check independently as PASS or FAIL.

Rules:
- Paraphrase is fine. Product answers deliberately avoid code identifiers, so a plain-language equivalent of a page, field or key counts as naming it.
- "must_not" checks PASS when the answer avoids the described behaviour.
- "answer" and "verdict" checks are about the answer's actual conclusion, not whether the words appear somewhere.
- Do not reward hedging. If a check states a definite fact, an answer that only says it is unsure FAILS that check.
- If a check expects "not determinable" or "not configurable", a confident answer that invents a mechanism FAILS it.
- An answer that only asks a clarifying question, without answering, FAILS every content check.
- "should_" checks are graded the same way; the caller treats them as optional.

Reply with JSON only, no prose, in exactly this shape:
{"checks": [{"id": "<check id>", "pass": true, "why": "<one short sentence>"}]}"""


def load_items():
    return {i["id"]: i for i in yaml.safe_load(ITEMS.read_text())["items"]}


def checks_for(item):
    """Flatten expect into (id, text, soft) checks."""
    out = []
    for key, value in (item.get("expect") or {}).items():
        soft = key.startswith("should_")
        if isinstance(value, list):
            for n, v in enumerate(value, 1):
                out.append((f"{key}[{n}]", f"{key}: {v}", soft))
        elif isinstance(value, dict):
            for k, v in value.items():
                out.append((f"{key}.{k}", f"{key} {k}: {v}", soft))
        else:
            out.append((key, f"{key}: {' '.join(str(value).split())}", soft))
    return out


def build_prompt(item, answer, checks):
    lines = [
        f"QUESTION: {' '.join(item['question'].split())}",
        "",
        "ANSWER BEING GRADED:",
        answer.strip() or "(empty)",
        "",
        "CHECKS:",
        *[f"- id={cid} :: {text}" for cid, text, _ in checks],
    ]
    if item.get("notes"):
        lines += ["", f"GRADER CONTEXT (not part of the answer): {' '.join(str(item['notes']).split())}"]
    return "\n".join(lines)


def call_mimo(prompt):
    base = os.environ["CODEATLAS_LLM_BASE_URL"].rstrip("/")
    r = requests.post(
        f"{base}/chat/completions",
        headers={"Authorization": f"Bearer {os.environ['CODEATLAS_LLM_API_KEY']}"},
        json={
            "model": os.environ.get("CODEATLAS_LLM_MODEL", "mimo-v2.5"),
            "messages": [{"role": "system", "content": INSTRUCTIONS},
                         {"role": "user", "content": prompt}],
            "temperature": 0,
            "max_tokens": 2000,
        },
        timeout=180,
    )
    return r.status_code, (r.json()["choices"][0]["message"]["content"] if r.ok else r.text)


def call_luna(prompt):
    base = os.environ["CODEATLAS_SHARED_LLM_LUNA_BASE_URL"].rstrip("/")
    r = requests.post(
        f"{base}/responses",
        headers={"Authorization": f"Bearer {os.environ['CODEATLAS_SHARED_LLM_LUNA_API_KEY']}"},
        json={
            "model": os.environ.get("CODEATLAS_SHARED_LLM_LUNA_MODEL", "gpt-5.6-luna"),
            "instructions": INSTRUCTIONS,
            "input": prompt,
            "reasoning": {"effort": "low"},
        },
        timeout=180,
    )
    if not r.ok:
        return r.status_code, r.text
    text = "".join(
        part.get("text", "")
        for o in r.json().get("output", []) if o.get("type") == "message"
        for part in o.get("content", [])
    )
    return 200, text


JUDGES = {"mimo": call_mimo, "luna": call_luna}


def parse(text):
    m = re.search(r"\{.*\}", text or "", re.S)
    if not m:
        return None
    try:
        return {c["id"]: c for c in json.loads(m.group(0)).get("checks", [])}
    except (ValueError, KeyError, TypeError):
        return None


def judge_one(judge, item, rec):
    checks = checks_for(item)
    prompt = build_prompt(item, rec.get("answer") or "", checks)
    for attempt in range(4):
        status, text = JUDGES[judge](prompt)
        if status == 429 or status >= 500:
            time.sleep(15 * (attempt + 1))
            continue
        got = parse(text) if status == 200 else None
        if got is None:
            time.sleep(3)
            continue
        results = []
        for cid, ctext, soft in checks:
            g = got.get(cid) or {}
            results.append({"id": cid, "soft": soft, "pass": bool(g.get("pass")),
                            "missing": cid not in got, "why": g.get("why", "")})
        hard = [c for c in results if not c["soft"]]
        return {
            "item_id": rec["item_id"], "arm": rec["arm"], "judge": judge,
            "item_pass": all(c["pass"] for c in hard) if hard else None,
            "hard_passed": sum(c["pass"] for c in hard), "hard_total": len(hard),
            "checks": results,
        }
    return {"item_id": rec["item_id"], "arm": rec["arm"], "judge": judge, "error": f"{status} {str(text)[:200]}"}


def run(args):
    items = load_items()
    recs = [json.loads(l) for l in Path(args.answers).read_text().splitlines() if l.strip()]
    recs = [r for r in recs if r.get("item_id") in items and r.get("answer")]
    done = set()
    out = Path(args.out)
    if out.exists():
        for l in out.read_text().splitlines():
            r = json.loads(l)
            if "error" not in r:
                done.add((r["item_id"], r["arm"]))
    todo = [r for r in recs if (r["item_id"], r["arm"]) not in done]
    print(f"{len(todo)} answers to grade with {args.judge} ({len(done)} already done)")
    with ThreadPoolExecutor(args.workers) as pool, out.open("a") as fh:
        for res in pool.map(lambda r: judge_one(args.judge, items[r["item_id"]], r), todo):
            fh.write(json.dumps(res) + "\n"); fh.flush()
            tag = "ERR" if "error" in res else ("PASS" if res["item_pass"] else "fail")
            print(f"  {res['arm']:<8} {res['item_id']:<38} {tag}", flush=True)


def report(paths):
    items = load_items()
    rows = [json.loads(l) for p in paths for l in Path(p).read_text().splitlines() if l.strip()]
    rows = [r for r in rows if "error" not in r]
    by = defaultdict(dict)                       # (item, arm) -> {judge: pass}
    for r in rows:
        by[(r["item_id"], r["arm"])][r["judge"]] = r["item_pass"]
    judges = sorted({r["judge"] for r in rows})
    arms = sorted({r["arm"] for r in rows})

    def rate(xs):
        xs = [x for x in xs if x is not None]
        return f"{sum(xs)/len(xs):.2f}" if xs else " n/a"

    print(f"\nItem pass rate (all hard checks pass). Judges: {', '.join(judges)}\n")
    print(f"{'arm':<10}" + "".join(f"{j:>10}" for j in judges) + f"{'both':>10}{'n':>6}")
    for arm in arms:
        keys = [k for k in by if k[1] == arm]
        cols = [rate(by[k].get(j) for k in keys) for j in judges]
        both = rate((all(by[k].get(j) for j in judges) if all(j in by[k] for j in judges) else None) for k in keys)
        print(f"{arm:<10}" + "".join(f"{c:>10}" for c in cols) + f"{both:>10}{len(keys):>6}")

    if len(judges) == 2:
        a, b = judges
        pairs = [(v[a], v[b]) for v in by.values() if a in v and b in v]
        agree = sum(x == y for x, y in pairs)
        print(f"\nJudge agreement: {agree}/{len(pairs)} = {agree/len(pairs):.2f}")
        for arm in arms:
            ps = [(v[a], v[b]) for k, v in by.items() if k[1] == arm and a in v and b in v]
            print(f"  on {arm:<8} {a} passes {sum(x for x,_ in ps):>2}, {b} passes {sum(y for _,y in ps):>2}  (n={len(ps)})")

    for dim, field in (("scope_class", "scope_class"), ("kind", "kind"), ("band", "band")):
        print(f"\nby {dim} — pass rate where both judges agree it passed")
        print(f"{'':<16}" + "".join(f"{arm:>10}" for arm in arms))
        groups = sorted({items[k[0]].get(field) or "(none)" for k in by})
        for g in groups:
            cells = []
            for arm in arms:
                keys = [k for k in by if k[1] == arm and (items[k[0]].get(field) or "(none)") == g]
                cells.append(rate((all(by[k].get(j) for j in judges) if all(j in by[k] for j in judges) else None) for k in keys))
            print(f"  {g:<14}" + "".join(f"{c:>10}" for c in cells))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--answers")
    ap.add_argument("--judge", choices=sorted(JUDGES))
    ap.add_argument("--out")
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--report", nargs="+")
    args = ap.parse_args()
    if args.report:
        report(args.report)
    elif args.answers and args.judge and args.out:
        run(args)
    else:
        sys.exit("pass --answers, --judge and --out, or --report FILES")


if __name__ == "__main__":
    main()
