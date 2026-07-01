"""Branch discovery, freshness checks, and isolated atomic indexing.

Approved branches are indexed into immutable version workspaces. The active
workspace pointer changes only after Graphify completes successfully.
"""

from __future__ import annotations

import re
import shutil
import subprocess
import threading
from concurrent.futures import Future, ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

from .. import db
from ..config import (
    BRANCH_FRESHNESS_INTERVAL_SECONDS,
    BRANCH_SYNC_MAX_WORKERS,
    BRANCH_SYNC_POLL_SECONDS,
    BRANCH_VERSION_RETENTION_SECONDS,
    branch_version_workspace,
    graph_path,
    repo_clone_dir,
    retrieval_config_path,
    workspace_dir,
)
from .indexing import index_repo

GIT_TIMEOUT = 300
_CREDENTIAL_URL_RE = re.compile(r"(https?://)[^/@\s]+@", re.IGNORECASE)
_git_locks: dict[str, threading.Lock] = {}
_git_locks_guard = threading.Lock()


def _safe_git_error(value: str) -> str:
    return _CREDENTIAL_URL_RE.sub(r"\1[redacted]@", (value or "").strip())[:1000]


def _git_lock(repo: Path) -> threading.Lock:
    key = str(repo.resolve())
    with _git_locks_guard:
        if key not in _git_locks:
            _git_locks[key] = threading.Lock()
        return _git_locks[key]


def _git(repo: Path, *args: str, timeout: int = GIT_TIMEOUT, check: bool = True):
    with _git_lock(repo):
        result = subprocess.run(
            ["git", *args],
            cwd=str(repo),
            capture_output=True,
            text=True,
            timeout=timeout,
        )
    if check and result.returncode != 0:
        detail = _safe_git_error(result.stderr or result.stdout)
        raise RuntimeError(detail or f"git {' '.join(args)} failed")
    return result


def _current_branch_and_commit(repo: Path) -> tuple[str, str | None]:
    if not (repo / ".git").exists():
        return "default", None
    commit_result = _git(repo, "rev-parse", "HEAD", check=False)
    commit = commit_result.stdout.strip() if commit_result.returncode == 0 else None
    branch_result = _git(repo, "symbolic-ref", "--short", "HEAD", check=False)
    branch = branch_result.stdout.strip() if branch_result.returncode == 0 else ""
    return branch or (f"detached-{commit[:12]}" if commit else "default"), commit


def _indexed_timestamp(workspace: str) -> str | None:
    path = graph_path(workspace)
    try:
        timestamp = path.stat().st_mtime
    except OSError:
        return None
    return datetime.fromtimestamp(timestamp, timezone.utc).strftime("%Y-%m-%d %H:%M:%S")


def ensure_repo_branch(repo: dict) -> dict:
    existing = db.list_repo_branches(repo["id"])
    if existing:
        return existing[0]

    source = repo_clone_dir(repo["workspace"])
    branch_name, commit = _current_branch_and_commit(source)
    indexed_at = _indexed_timestamp(repo["workspace"])
    ready = indexed_at is not None and repo["status"] in {"indexed", "published"}
    branch = db.create_repo_branch(
        repo["id"],
        branch_name,
        workspace=repo["workspace"] if ready else None,
        indexed_commit_sha=commit if ready else None,
        index_status="ready" if ready else "never_indexed",
        freshness_status="unknown",
        indexed_at=indexed_at,
        is_legacy=True,
    )
    can_sync = bool(commit and not branch_name.startswith("detached-"))
    db.update_repo_branch_settings(
        branch["id"],
        allow_user_sync=can_sync,
        auto_sync=False,
        strict_freshness=False,
        freshness_interval_seconds=BRANCH_FRESHNESS_INTERVAL_SECONDS,
    )
    if ready:
        db.add_legacy_repo_branch_version(branch["id"], commit, repo["workspace"])
    return db.get_repo_branch(branch["id"])


def ensure_legacy_repo_branches() -> None:
    for repo in db.list_repos():
        ensure_repo_branch(repo)


def restore_branches_after_reclone(repo: dict) -> None:
    """Clear missing-clone errors and bind a graph-only placeholder branch."""
    source = repo_clone_dir(repo["workspace"])
    branch_name, commit = _current_branch_and_commit(source)
    legacy = db.get_legacy_repo_branch(repo["id"])
    if (
        legacy
        and legacy["name"] == "default"
        and not legacy.get("indexed_commit_sha")
        and commit
        and not db.get_repo_branch_by_name(repo["id"], branch_name)
    ):
        db.bind_repo_branch_to_clone(legacy["id"], branch_name, commit)

    for branch in db.list_repo_branches(repo["id"]):
        db.update_repo_branch_state(
            branch["id"],
            job_stage="idle",
            freshness_status="unknown",
        )
        db.clear_repo_branch_error(branch["id"])


def _remote_default_branch(source: Path) -> str | None:
    result = _git(source, "ls-remote", "--symref", "origin", "HEAD", check=False)
    for line in result.stdout.splitlines():
        parts = line.split()
        if len(parts) == 3 and parts[0] == "ref:" and parts[2] == "HEAD":
            ref = parts[1]
            if ref.startswith("refs/heads/"):
                return ref[len("refs/heads/"):]
    return None


def discover_remote_branches(repo: dict) -> list[dict]:
    source = repo_clone_dir(repo["workspace"])
    if not source.is_dir():
        raise RuntimeError("Repository clone is not available.")
    default_branch = _remote_default_branch(source)
    if default_branch:
        db.set_repo_default_branch(repo["id"], default_branch)
    result = _git(source, "ls-remote", "--heads", "origin")
    branches = []
    for line in result.stdout.splitlines():
        parts = line.split()
        if len(parts) != 2 or not parts[1].startswith("refs/heads/"):
            continue
        branches.append({
            "name": parts[1][len("refs/heads/"):],
            "commit_sha": parts[0],
            "is_default": parts[1] == f"refs/heads/{default_branch}",
        })
    branches.sort(key=lambda item: item["name"])
    return branches


def approve_repo_branch(repo: dict, branch_name: str) -> dict:
    branch_name = (branch_name or "").strip()
    source = repo_clone_dir(repo["workspace"])
    if not branch_name or _git(
        source, "check-ref-format", "--branch", branch_name, check=False
    ).returncode != 0:
        raise ValueError("Invalid Git branch name.")
    available = {item["name"]: item for item in discover_remote_branches(repo)}
    if branch_name not in available:
        raise ValueError("The branch does not exist on the remote.")
    existing = db.get_repo_branch_by_name(repo["id"], branch_name)
    if existing:
        return db.get_repo_branch(existing["id"])
    branch = db.create_repo_branch(
        repo["id"],
        branch_name,
        is_default=bool(available[branch_name].get("is_default")),
    )
    db.update_repo_branch_settings(
        branch["id"],
        allow_user_sync=True,
        auto_sync=False,
        strict_freshness=False,
        freshness_interval_seconds=BRANCH_FRESHNESS_INTERVAL_SECONDS,
    )
    db.update_repo_branch_state(
        branch["id"],
        remote_commit_sha=available[branch_name]["commit_sha"],
        checked=True,
    )
    return db.get_repo_branch(branch["id"])


def _fetch_branch(branch: dict) -> str:
    source = repo_clone_dir(branch["repo_workspace"])
    remote_ref = f"refs/remotes/origin/{branch['name']}"
    source_ref = f"+refs/heads/{branch['name']}:{remote_ref}"
    _git(source, "fetch", "--no-tags", "--depth", "50", "origin", source_ref)
    result = _git(source, "rev-parse", remote_ref)
    return result.stdout.strip()


def _freshness_after_fetch(branch: dict, remote_sha: str) -> tuple[str, int]:
    indexed_sha = branch.get("indexed_commit_sha")
    if not indexed_sha:
        return "unknown", 0
    if indexed_sha == remote_sha:
        return "up_to_date", 0

    source = repo_clone_dir(branch["repo_workspace"])
    ancestor = _git(
        source, "merge-base", "--is-ancestor", indexed_sha, remote_sha, check=False
    )
    if ancestor.returncode != 0:
        return "diverged", 0
    count = _git(source, "rev-list", "--count", f"{indexed_sha}..{remote_sha}")
    try:
        behind = max(1, int(count.stdout.strip()))
    except ValueError:
        behind = 1
    return "behind", behind


def check_branch_freshness(branch_id: int) -> dict:
    branch = db.get_repo_branch(branch_id)
    if not branch:
        raise RuntimeError("Branch is no longer available.")
    db.update_repo_branch_state(
        branch_id,
        job_stage="fetching",
        freshness_status="checking",
    )
    try:
        remote_sha = _fetch_branch(branch)
        freshness, behind = _freshness_after_fetch(branch, remote_sha)
        db.update_repo_branch_state(
            branch_id,
            freshness_status=freshness,
            remote_commit_sha=remote_sha,
            behind_count=behind,
            checked=True,
            job_stage="idle",
        )
        db.clear_repo_branch_error(branch_id)
    except Exception as exc:
        db.update_repo_branch_state(
            branch_id,
            freshness_status="remote_unavailable",
            last_error=str(exc),
            checked=True,
            job_stage="idle",
        )
    return db.get_repo_branch(branch_id)


def _copy_retrieval_config(repo_workspace: str, version_workspace: str) -> None:
    source = retrieval_config_path(repo_workspace)
    if not source.exists():
        return
    target = retrieval_config_path(version_workspace)
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, target)


def _remove_version_workspace(repo_workspace: str, version_workspace: str) -> None:
    if version_workspace == repo_workspace:
        return
    source = repo_clone_dir(repo_workspace)
    target = repo_clone_dir(version_workspace)
    if source.is_dir() and target.exists():
        _git(source, "worktree", "remove", "--force", str(target), check=False)
        _git(source, "worktree", "prune", check=False)
    shutil.rmtree(workspace_dir(version_workspace), ignore_errors=True)


def cleanup_expired_branch_versions() -> None:
    for version in db.list_expired_repo_branch_versions(
        BRANCH_VERSION_RETENTION_SECONDS
    ):
        _remove_version_workspace(version["repo_workspace"], version["workspace"])
        db.delete_repo_branch_version(version["id"])


def delete_approved_branch(branch_id: int) -> None:
    branch = db.get_repo_branch(branch_id)
    if not branch:
        return
    if branch["is_legacy"]:
        raise ValueError("The repository's original branch cannot be removed.")
    for version in db.list_repo_branch_versions(branch_id):
        _remove_version_workspace(branch["repo_workspace"], version["workspace"])
    db.delete_repo_branch(branch_id)


def record_legacy_index(repo: dict) -> dict:
    branch = ensure_repo_branch(repo)
    _, commit = _current_branch_and_commit(repo_clone_dir(repo["workspace"]))
    if not commit:
        db.update_repo_branch_state(
            branch["id"],
            index_status="ready",
            freshness_status="unknown",
        )
        return db.get_repo_branch(branch["id"])
    db.activate_repo_branch_version(
        branch["id"],
        commit,
        repo["workspace"],
        verified_remote=False,
    )
    return db.get_repo_branch(branch["id"])


def copy_config_to_active_branch_workspaces(repo: dict) -> None:
    for branch in db.list_repo_branches(repo["id"]):
        workspace = branch.get("workspace")
        if workspace and workspace != repo["workspace"]:
            _copy_retrieval_config(repo["workspace"], workspace)


def remove_repo_branch_workspaces(repo: dict) -> None:
    for branch in db.list_repo_branches(repo["id"]):
        for version in db.list_repo_branch_versions(branch["id"]):
            _remove_version_workspace(repo["workspace"], version["workspace"])


def _prepare_version_workspace(branch: dict, commit_sha: str) -> str:
    workspace = branch_version_workspace(
        branch["repo_workspace"], branch["id"], commit_sha
    )
    existing = db.find_repo_branch_version(branch["id"], commit_sha)
    if (
        existing
        and repo_clone_dir(existing["workspace"]).is_dir()
        and graph_path(existing["workspace"]).is_file()
    ):
        _copy_retrieval_config(branch["repo_workspace"], existing["workspace"])
        return existing["workspace"]

    _remove_version_workspace(branch["repo_workspace"], workspace)
    source = repo_clone_dir(branch["repo_workspace"])
    target = repo_clone_dir(workspace)
    target.parent.mkdir(parents=True, exist_ok=True)
    _git(source, "worktree", "add", "--detach", str(target), commit_sha)
    _copy_retrieval_config(branch["repo_workspace"], workspace)
    db.update_repo_branch_state(branch["id"], job_stage="indexing")
    try:
        index_repo(workspace)
    except Exception:
        _remove_version_workspace(branch["repo_workspace"], workspace)
        raise
    return workspace


def sync_and_index_branch(branch_id: int) -> dict:
    branch = db.get_repo_branch(branch_id)
    if not branch:
        raise RuntimeError("Branch is no longer available.")
    previous_workspace = branch.get("workspace")
    remote_reached = False
    db.update_repo_branch_state(
        branch_id,
        index_status="indexing",
        job_stage="fetching",
        freshness_status="checking",
    )
    try:
        remote_sha = _fetch_branch(branch)
        remote_reached = True
        current = db.get_repo_branch(branch_id)
        freshness, behind = _freshness_after_fetch(current, remote_sha)
        db.update_repo_branch_state(
            branch_id,
            job_stage="preparing",
            freshness_status=freshness,
            remote_commit_sha=remote_sha,
            behind_count=behind,
            checked=True,
        )
        if current.get("indexed_commit_sha") == remote_sha and previous_workspace:
            db.update_repo_branch_state(
                branch_id,
                index_status="ready",
                job_stage="idle",
            )
            db.clear_repo_branch_error(branch_id)
            return db.get_repo_branch(branch_id)

        workspace = _prepare_version_workspace(current, remote_sha)
        db.activate_repo_branch_version(branch_id, remote_sha, workspace)
        cleanup_expired_branch_versions()
    except Exception as exc:
        state = {
            "index_status": "ready" if previous_workspace else "failed",
            "job_stage": "idle",
            "last_error": str(exc),
            "checked": True,
        }
        if not remote_reached:
            state["freshness_status"] = "remote_unavailable"
        db.update_repo_branch_state(branch_id, **state)
    return db.get_repo_branch(branch_id)


_executor = ThreadPoolExecutor(
    max_workers=BRANCH_SYNC_MAX_WORKERS,
    thread_name_prefix="codeatlas-branch",
)
_jobs: dict[int, Future] = {}
_jobs_lock = threading.Lock()
_service_stop = threading.Event()
_scheduler_thread: threading.Thread | None = None


def submit_branch_job(branch_id: int, sync: bool = True) -> bool:
    with _jobs_lock:
        current = _jobs.get(branch_id)
        if current and not current.done():
            return False
        future = _executor.submit(
            sync_and_index_branch if sync else check_branch_freshness,
            branch_id,
        )
        _jobs[branch_id] = future

        def clear(completed: Future) -> None:
            with _jobs_lock:
                if _jobs.get(branch_id) is completed:
                    _jobs.pop(branch_id, None)

        future.add_done_callback(clear)
        return True


def branch_job_running(branch_id: int) -> bool:
    with _jobs_lock:
        future = _jobs.get(branch_id)
        return bool(future and not future.done())


def _branch_check_due(branch: dict) -> bool:
    last_checked = branch.get("last_checked_at")
    if not last_checked:
        return True
    try:
        checked = datetime.strptime(last_checked, "%Y-%m-%d %H:%M:%S")
    except ValueError:
        return True
    elapsed = datetime.utcnow() - checked
    return elapsed.total_seconds() >= max(
        60, int(branch.get("freshness_interval_seconds") or 300)
    )


def _scheduler_loop() -> None:
    while not _service_stop.wait(BRANCH_SYNC_POLL_SECONDS):
        for branch in db.list_all_repo_branches():
            source = repo_clone_dir(branch["repo_workspace"])
            if (source / ".git").exists() and _branch_check_due(branch):
                submit_branch_job(branch["id"], sync=bool(branch["auto_sync"]))
        cleanup_expired_branch_versions()


def start_branch_services() -> None:
    global _scheduler_thread
    db.recover_interrupted_repo_branch_jobs()
    if _scheduler_thread and _scheduler_thread.is_alive():
        return
    _service_stop.clear()
    _scheduler_thread = threading.Thread(
        target=_scheduler_loop,
        name="codeatlas-branch-scheduler",
        daemon=True,
    )
    _scheduler_thread.start()


def stop_branch_services() -> None:
    _service_stop.set()
