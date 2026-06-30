"""Clone a repository into a workspace via HTTPS, SSH, or the GitHub CLI.

Used by the admin "add repository" flow (Phase 2). Private repos rely on the
host's existing git/SSH credentials or `gh auth login`.
"""

import shutil
import subprocess
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

from ..config import repo_clone_dir, workspace_dir

CLONE_TIMEOUT = 600
SENSITIVE_URL_QUERY_KEYS = {
    "access_token", "api_key", "apikey", "auth", "authorization", "oauth_token",
    "password", "passwd", "private_token", "token",
}


def sanitize_clone_url(source_url: str) -> str:
    """Remove URL credentials from the value persisted to the DB and audit log."""
    if "://" not in source_url:
        return source_url
    parsed = urlsplit(source_url)
    sanitized_netloc = parsed.netloc.rsplit("@", 1)[-1]
    query_pairs = parse_qsl(parsed.query, keep_blank_values=True)
    sanitized_query_pairs = [
        (key, "[redacted]" if key.lower() in SENSITIVE_URL_QUERY_KEYS else value)
        for key, value in query_pairs
    ]
    if (
        sanitized_netloc == parsed.netloc
        and sanitized_query_pairs == query_pairs
    ):
        return source_url
    return urlunsplit((
        parsed.scheme,
        sanitized_netloc,
        parsed.path,
        urlencode(sanitized_query_pairs),
        parsed.fragment,
    ))


def clone_repo(source_url: str, method: str, workspace: str):
    """Clone source_url into the workspace's repo dir. method: https|ssh|gh."""
    dest = repo_clone_dir(workspace)
    if dest.exists():
        raise RuntimeError(f"workspace repo already exists at {dest}")
    dest.parent.mkdir(parents=True, exist_ok=True)

    if method == "gh":
        cmd = ["gh", "repo", "clone", source_url, str(dest)]
    elif method in ("https", "ssh"):
        # git infers protocol from the URL form; --depth 1 keeps indexing fast.
        cmd = ["git", "clone", "--depth", "1", source_url, str(dest)]
    else:
        raise ValueError(f"unknown clone method: {method!r}")

    result = subprocess.run(cmd, capture_output=True, text=True, timeout=CLONE_TIMEOUT)
    if result.returncode != 0:
        raise RuntimeError(f"clone failed: {result.stderr.strip() or result.stdout.strip()}")
    return dest


def remove_workspace(workspace: str):
    """Delete a workspace's entire directory (clone + graph + config) from disk.
    Used by the admin "delete repository" flow; safe if the dir is missing."""
    target = workspace_dir(workspace)
    shutil.rmtree(target, ignore_errors=True)
    return target
