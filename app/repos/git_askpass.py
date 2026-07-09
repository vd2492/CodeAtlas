#!/usr/bin/env python3
"""Minimal non-interactive Git credential helper for CodeAtlas.

Git invokes this program with a prompt such as "Username for ..." or
"Password for ...". Values are passed only through the subprocess environment
created by app.repos.git_auth; they are never written to command arguments,
repository URLs, SQLite, or logs.
"""

from __future__ import annotations

import os
import sys


def main() -> int:
    prompt = " ".join(sys.argv[1:]).lower()
    username = os.environ.get("CODEATLAS_GIT_ASKPASS_USERNAME", "")
    token = os.environ.get("CODEATLAS_GIT_ASKPASS_TOKEN", "")
    if "username" in prompt:
        print(username)
        return 0
    if "password" in prompt or "token" in prompt:
        print(token)
        return 0
    print(token or username)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
