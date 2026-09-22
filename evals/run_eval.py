#!/usr/bin/env python3
"""Run the PM admin-config eval set against a live CodeAtlas instance.

Deterministic grading (retrieval recall, audience compliance) happens inline.
Facet correctness needs a judge, so this writes judge_tasks.jsonl for a
separate step rather than calling an LLM itself — that keeps the runner
dependency-light and keeps the judge swappable.

Usage:
    python evals/run_eval.py --base-url http://localhost:8000 \
        --workspace myrepo --username ishaan --reps 3

Credentials come from the environment, never from argv:
    CODEATLAS_EVAL_PASSWORD   required unless --cookie is passed
    CODEATLAS_EVAL_COOKIE     an existing ca_session value, as an alternative
"""

import argparse
import json
import os
import re
import sys
import time
import urllib.error
import urllib.request
from collections import defaultdict
from datetime import datetime, timezone
from http.cookiejar import CookieJar
from pathlib import Path

try:
    import yaml
except ImportError:
    sys.exit("PyYAML is required: pip install -r evals/requirements.txt")

ITEMS = Path(__file__).parent / "admin_config" / "items.yaml"


# --------------------------------------------------------------- transport

def _opener():
    return urllib.request.build_opener(
        urllib.request.HTTPCookieProcessor(CookieJar())
    )


def _post(opener, url, payload, timeout):
    body = json.dumps(payload).encode()
    req = urllib.request.Request(
        url, data=body, headers={"Content-Type": "application/json"}
    )
    with opener.open(req, timeout=timeout) as resp:
        return json.loads(resp.read())


def login(base_url, username, password, timeout=30):
    """Authenticate and return an opener carrying the ca_session cookie."""
    opener = _opener()
    _post(opener, f"{base_url}/auth/login",
          {"username": username, "password": password}, timeout)
    return opener


def opener_from_cookie(base_url, cookie_value):
    """Alternative to login() when you already hold a session token."""
    opener = _opener()
    opener.addheaders = [("Cookie", f"ca_session={cookie_value}")]
    return opener


def ask(opener, base_url, workspace, question, user_type,
        llm_mode="auto", branch=None, timeout=600):
    url = f"{base_url}/repo/ask-llm?workspace={workspace}"
    if branch is not None:
        url += f"&branch={branch}"
    payload = {
        "question": question,
        "answer_user_type": user_type,
        "llm_mode": llm_mode,
        "follow_up": False,
        "deep_investigation": False,
    }
    started = time.time()
    try:
        data = _post(opener, url, payload, timeout)
        data["_elapsed_s"] = round(time.time() - started, 1)
        return data
    except urllib.error.HTTPError as e:
        return {"_error": f"HTTP {e.code}", "_body": e.read()[:400].decode("utf-8", "replace"),
                "_elapsed_s": round(time.time() - started, 1)}
    except Exception as e:                                  # noqa: BLE001
        return {"_error": type(e).__name__, "_body": str(e)[:400],
                "_elapsed_s": round(time.time() - started, 1)}


# --------------------------------------------------------------- grading

def trace_paths(response):
    """Every source path the agent actually opened, from agent_trace."""
    out = []
    for entry in response.get("agent_trace") or []:
        path = (entry.get("result") or {}).get("path")
        if path:
            out.append(path)
    return out


def _glob_regex(pattern):
    """Globstar-aware matcher. fnmatch has no ** semantics: it treats * as
    matching '/' too, so '**/RMS/**' fails against a top-level 'RMS/admin.py'.
    Here '**/' matches an optional leading path, '**' any span, '*' one segment.
    """
    out, i = [], 0
    while i < len(pattern):
        if pattern.startswith("**/", i):
            out.append("(?:.*/)?")
            i += 3
        elif pattern.startswith("**", i):
            out.append(".*")
            i += 2
        elif pattern[i] == "*":
            out.append("[^/]*")
            i += 1
        elif pattern[i] == "?":
            out.append("[^/]")
            i += 1
        else:
            out.append(re.escape(pattern[i]))
            i += 1
    return re.compile("".join(out) + r"\Z")


def grade_retrieval(response, evidence_paths):
    """None when the item declares no expected evidence (e.g. true negatives)."""
    if not evidence_paths:
        return None, []
    opened = trace_paths(response)
    matchers = [_glob_regex(p) for p in evidence_paths]
    hit = any(m.match(o) for m in matchers for o in opened)
    return hit, opened


# A product-team answer must not leak implementation detail. Mirrors the
# guardrail in app/llm/client.py::_clean_product_answer — we assert it holds.
LEAK_PATTERNS = [
    (r"\b\w+\.py\b", "python filename"),
    (r"\bline\s+\d+", "line reference"),
    (r"\b[a-z_]+/[a-z_]+\.py", "file path"),
]


def grade_audience(response, must_not_contain):
    answer = response.get("answer") or ""
    hits = []
    for needle in must_not_contain or []:
        if needle.lower() in answer.lower():
            hits.append(needle)
    for pattern, label in LEAK_PATTERNS:
        if re.search(pattern, answer):
            hits.append(label)
    return (len(hits) == 0), hits


def judge_task(item, response, arm, rep):
    """Payload for the separate facet-correctness judging step."""
    return {
        "item_id": item["id"],
        "arm": arm,
        "rep": rep,
        "question": item["question"],
        "expect": item.get("expect", {}),
        "notes": item.get("notes"),
        "gold_citations": item.get("gold_citations"),
        "answer": response.get("answer"),
        "instruction": (
            "Grade each key under `expect` as a separate PASS/FAIL against the "
            "answer. Do not reward hedging: an item whose expect block states a "
            "definite answer fails if the response only says it is uncertain. "
            "Conversely, where expect declares a verdict of not_determinable, a "
            "confident answer fails. Return {key: bool, ...} plus one line of "
            "reasoning per FAIL."
        ),
    }


# --------------------------------------------------------------- run

def run_arm(opener, args, items, arm, results_fh, judge_fh):
    """arm 'fresh'      - ask as product_team directly
       arm 'audience'   - ask as dev_team FIRST, so the product answer is
                          built by reusing cached dev evidence. This is the
                          precise way to exercise answer_from_cached_audience_
                          evidence (commit bc8eed1) rather than hoping to hit it.
    """
    for item in items:
        for rep in range(1, args.reps + 1):
            if arm == "audience":
                ask(opener, args.base_url, args.workspace,
                    item["question"], "dev_team", args.llm_mode, args.branch)

            resp = ask(opener, args.base_url, args.workspace,
                       item["question"], "product_team", args.llm_mode, args.branch)

            retrieval, opened = grade_retrieval(resp, item.get("evidence_paths"))
            audience, leaks = grade_audience(resp, item.get("must_not_contain"))

            row = {
                "item_id": item["id"],
                "arm": arm,
                "rep": rep,
                "scope_class": item.get("scope_class"),
                "kind": item.get("kind"),
                "band": item.get("band"),
                "priority": item.get("priority"),
                "error": resp.get("_error"),
                "retrieval_pass": retrieval,
                "audience_pass": audience,
                "leaks": leaks,
                "paths_opened": opened,
                "tool_calls": resp.get("agent_tool_calls"),
                "rounds": resp.get("agent_rounds"),
                "needs_clarification": resp.get("needs_clarification"),
                "provider_used": resp.get("provider_used"),
                "retrieval_mode": resp.get("retrieval_mode"),
                "elapsed_s": resp.get("_elapsed_s"),
                "answer": resp.get("answer"),
            }
            results_fh.write(json.dumps(row) + "\n")
            results_fh.flush()

            if not resp.get("_error"):
                judge_fh.write(json.dumps(judge_task(item, resp, arm, rep)) + "\n")
                judge_fh.flush()

            flag = "ERR" if resp.get("_error") else (
                "ok" if (retrieval is not False and audience) else "MISS")
            print(f"  [{arm}] {item['id']:<36} rep{rep} {flag:>4} "
                  f"{row['elapsed_s']}s tools={row['tool_calls']}", flush=True)


def scorecard(rows):
    """Deterministic cells only. Facet columns come from the judging step."""
    def rate(subset, key):
        vals = [r[key] for r in subset if r.get(key) is not None]
        return f"{sum(vals)/len(vals):.2f}" if vals else "  - "

    print("\n" + "=" * 68)
    print("DETERMINISTIC SCORECARD  (facets require the judging step)")
    print("=" * 68)

    for dimension, field in (("class", "scope_class"), ("kind", "kind"), ("band", "band")):
        groups = defaultdict(list)
        for r in rows:
            if r.get(field):
                groups[r[field]].append(r)
        if not groups:
            continue
        print(f"\n{dimension:<22} retrieval  audience   n   errors")
        for name, subset in sorted(groups.items()):
            errs = sum(1 for r in subset if r.get("error"))
            print(f"  {name:<20} {rate(subset,'retrieval_pass'):>6}   "
                  f"{rate(subset,'audience_pass'):>6}  {len(subset):>3}   {errs:>3}")

    arms = defaultdict(list)
    for r in rows:
        arms[r["arm"]].append(r)
    if len(arms) > 1:
        print("\nARM COMPARISON (the bc8eed1 audience-cache question)")
        for name, subset in sorted(arms.items()):
            print(f"  {name:<20} retrieval {rate(subset,'retrieval_pass')}  "
                  f"audience {rate(subset,'audience_pass')}  n={len(subset)}")
        print("  Compare paired by item_id, not in aggregate — see the README.")

    flaky = defaultdict(set)
    for r in rows:
        flaky[(r["item_id"], r["arm"])].add(bool(r.get("retrieval_pass")))
    unstable = [k for k, v in flaky.items() if len(v) > 1]
    if unstable:
        print(f"\nUNSTABLE across reps ({len(unstable)}): "
              f"{', '.join(sorted({i for i, _ in unstable}))}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", default="http://localhost:8000")
    ap.add_argument("--workspace", required=True)
    ap.add_argument("--username")
    ap.add_argument("--branch", type=int)
    ap.add_argument("--llm-mode", default="auto")
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--arms", default="fresh",
                    help="comma-separated: fresh,audience")
    ap.add_argument("--only", help="substring filter on item id")
    ap.add_argument("--priority", help="only items at this priority")
    ap.add_argument("--out-dir", default="evals/runs")
    args = ap.parse_args()

    spec = yaml.safe_load(ITEMS.read_text())
    items = spec["items"]
    if args.only:
        items = [i for i in items if args.only in i["id"]]
    if args.priority:
        items = [i for i in items if i.get("priority") == args.priority]
    if not items:
        sys.exit("no items matched")

    cookie = os.environ.get("CODEATLAS_EVAL_COOKIE")
    if cookie:
        opener = opener_from_cookie(args.base_url, cookie)
    else:
        password = os.environ.get("CODEATLAS_EVAL_PASSWORD")
        if not (args.username and password):
            sys.exit("set CODEATLAS_EVAL_COOKIE, or pass --username with "
                     "CODEATLAS_EVAL_PASSWORD in the environment")
        opener = login(args.base_url, args.username, password)

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out = Path(args.out_dir) / stamp
    out.mkdir(parents=True, exist_ok=True)

    arms = [a.strip() for a in args.arms.split(",") if a.strip()]
    total = len(items) * args.reps * len(arms)
    print(f"{len(items)} items x {args.reps} reps x {len(arms)} arm(s) "
          f"= {total} calls -> {out}")
    print("NOTE: answers are cached in-process. For a true cold run, restart "
          "the CodeAtlas server first; see the README.\n")

    (out / "meta.json").write_text(json.dumps({
        "started_utc": stamp, "workspace": args.workspace,
        "branch": args.branch, "llm_mode": args.llm_mode,
        "reps": args.reps, "arms": arms, "item_count": len(items),
        "source_commit": spec.get("meta", {}).get("source_commit"),
    }, indent=2))

    with (out / "results.jsonl").open("w") as rf, \
         (out / "judge_tasks.jsonl").open("w") as jf:
        for arm in arms:
            print(f"\n--- arm: {arm} ---")
            run_arm(opener, args, items, arm, rf, jf)

    rows = [json.loads(l) for l in (out / "results.jsonl").read_text().splitlines()]
    scorecard(rows)
    print(f"\nresults      {out/'results.jsonl'}")
    print(f"judge tasks  {out/'judge_tasks.jsonl'}  ({sum(1 for _ in open(out/'judge_tasks.jsonl'))} to grade)")


if __name__ == "__main__":
    main()
