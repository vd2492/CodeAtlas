#!/usr/bin/env python3
"""Update the README repo-traffic badge from GitHub traffic data."""

import argparse
import json
import os
import sys
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.parse import quote
from urllib.request import Request, urlopen


START_MARKER = "<!-- repo-traffic:start -->"
END_MARKER = "<!-- repo-traffic:end -->"


def fetch_unique_visitors(repo: str, token: str) -> int:
    request = Request(
        f"https://api.github.com/repos/{repo}/traffic/views",
        headers={
            "Accept": "application/vnd.github+json",
            "Authorization": f"Bearer {token}",
            "X-GitHub-Api-Version": "2022-11-28",
            "User-Agent": "codeatlas-traffic-badge",
        },
    )
    try:
        with urlopen(request, timeout=30) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except HTTPError as error:
        detail = error.read().decode("utf-8", errors="replace")
        if error.code == 403:
            raise RuntimeError(
                "GitHub denied access to repository traffic. Add a fine-grained "
                "TRAFFIC_TOKEN secret with read-only Administration permission "
                "for this repository."
            ) from error
        raise RuntimeError(f"GitHub traffic API failed ({error.code}): {detail}") from error
    except URLError as error:
        raise RuntimeError(f"Could not reach GitHub traffic API: {error}") from error
    return int(payload.get("uniques") or 0)


def badge_markdown(unique_visitors: int) -> str:
    label = quote("unique visitors (14d)")
    message = quote(str(unique_visitors))
    return (
        f"![Unique visitors in the last 14 days]"
        f"(https://img.shields.io/badge/{label}-{message}-blue)"
    )


def update_readme(path: Path, unique_visitors: int) -> None:
    original = path.read_text()
    block = f"{START_MARKER}\n{badge_markdown(unique_visitors)}\n{END_MARKER}"
    if START_MARKER in original and END_MARKER in original:
        before, rest = original.split(START_MARKER, 1)
        _, after = rest.split(END_MARKER, 1)
        updated = f"{before}{block}{after}"
    else:
        lines = original.splitlines()
        insert_at = 1 if lines and lines[0].startswith("# ") else 0
        lines[insert_at:insert_at] = ["", block]
        updated = "\n".join(lines) + ("\n" if original.endswith("\n") else "")
    path.write_text(updated)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", required=True, help="owner/repo, for example vd2492/CodeAtlas")
    parser.add_argument("--readme", default="README.md")
    args = parser.parse_args()

    token = os.environ.get("GH_TOKEN") or os.environ.get("GITHUB_TOKEN")
    if not token:
        print("GH_TOKEN or GITHUB_TOKEN is required.", file=sys.stderr)
        return 1

    try:
        unique_visitors = fetch_unique_visitors(args.repo, token)
        update_readme(Path(args.readme), unique_visitors)
    except Exception as error:
        print(error, file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
