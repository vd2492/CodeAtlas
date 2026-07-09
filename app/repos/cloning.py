"""Clone a repository into a workspace via HTTPS, SSH, or the GitHub CLI.

Used by the admin "add repository" and reclone flows. Private GitHub and
Bitbucket HTTPS repos use centrally configured read-only credentials via
GIT_ASKPASS; credentials are never embedded in URLs or command arguments.
"""

import shutil
import subprocess

from ..config import repo_clone_dir, workspace_dir
from .git_auth import (
    git_env_for_gh_cli,
    git_env_for_url,
    sanitize_git_error,
    sanitize_url_for_storage,
    validate_clone_url,
)

CLONE_TIMEOUT = 600


def sanitize_clone_url(source_url: str) -> str:
    """Remove URL credentials from the value persisted to the DB and audit log."""
    return sanitize_url_for_storage(source_url)


def clone_repo(source_url: str, method: str, workspace: str):
    """Clone source_url into the workspace's repo dir. method: https|ssh|gh."""
    validate_clone_url(source_url, method)
    dest = repo_clone_dir(workspace)
    if dest.exists():
        raise RuntimeError(f"workspace repo already exists at {dest}")
    dest.parent.mkdir(parents=True, exist_ok=True)

    if method == "gh":
        cmd = ["gh", "repo", "clone", source_url, str(dest)]
        env = git_env_for_gh_cli()
    elif method in ("https", "ssh"):
        # git infers protocol from the URL form; --depth 1 keeps indexing fast.
        cmd = ["git", "clone", "--depth", "1", source_url, str(dest)]
        env = git_env_for_url(source_url)
    else:
        raise ValueError(f"unknown clone method: {method!r}")

    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        timeout=CLONE_TIMEOUT,
        env=env,
    )
    if result.returncode != 0:
        detail = sanitize_git_error(result.stderr or result.stdout)
        raise RuntimeError(f"clone failed: {detail}")
    return dest


def remove_repo_clone(workspace: str):
    """Remove only a repository working copy while preserving its graph/config."""
    target = repo_clone_dir(workspace)
    shutil.rmtree(target, ignore_errors=True)
    return target


def remove_workspace(workspace: str):
    """Delete a workspace's entire directory (clone + graph + config) from disk.
    Used by the admin "delete repository" flow; safe if the dir is missing."""
    target = workspace_dir(workspace)
    shutil.rmtree(target, ignore_errors=True)
    return target
