"""Admin and user HTTP APIs for approved repository branches."""

import threading
import time

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel

from .. import db
from ..auth.sessions import require_admin, require_user
from ..config import (
    BRANCH_FRESHNESS_INTERVAL_SECONDS,
    BRANCH_USER_SYNC_COOLDOWN_SECONDS,
)
from .branches import (
    approve_repo_branch,
    branch_job_running,
    delete_approved_branch,
    discover_remote_branches,
    submit_branch_job,
)

router = APIRouter(tags=["repo-branches"])
_user_sync_times: dict[tuple[int, int], float] = {}
_user_sync_lock = threading.Lock()


class ApproveBranchRequest(BaseModel):
    name: str


class BranchSettingsRequest(BaseModel):
    allow_user_sync: bool
    auto_sync: bool
    strict_freshness: bool
    freshness_interval_seconds: int = BRANCH_FRESHNESS_INTERVAL_SECONDS


def _require_repo(slug: str) -> dict:
    repo = db.get_repo_by_slug(slug)
    if not repo:
        raise HTTPException(status_code=404, detail=f"No repo with slug '{slug}'.")
    return repo


def _require_repo_branch(repo: dict, branch_id: int) -> dict:
    branch = db.get_repo_branch(branch_id)
    if not branch or branch["repo_id"] != repo["id"]:
        raise HTTPException(status_code=404, detail="Repository branch not found.")
    return branch


def _authorized_repo(workspace: str, user: dict) -> dict:
    repo = db.get_repo_by_workspace(workspace)
    if not repo:
        raise HTTPException(status_code=404, detail="Repository not found.")
    if user["role"] != "admin" and not db.user_has_repo(user["id"], repo["workspace"]):
        raise HTTPException(
            status_code=403,
            detail="You do not have access to this repository.",
        )
    return repo


def _branch_payload(branch: dict, admin: bool = False) -> dict:
    payload = {
        "id": branch["id"],
        "name": branch["name"],
        "indexed_commit_sha": branch.get("indexed_commit_sha"),
        "remote_commit_sha": branch.get("remote_commit_sha"),
        "indexed_at": branch.get("indexed_at"),
        "last_checked_at": branch.get("last_checked_at"),
        "index_status": branch["index_status"],
        "job_stage": branch.get("job_stage", "idle"),
        "freshness_status": branch["freshness_status"],
        "behind_count": branch.get("behind_count", 0),
        "last_error": branch.get("last_error"),
        "allow_user_sync": bool(branch["allow_user_sync"]),
        "auto_sync": bool(branch["auto_sync"]),
        "strict_freshness": bool(branch["strict_freshness"]),
        "freshness_interval_seconds": branch["freshness_interval_seconds"],
        "is_legacy": bool(branch["is_legacy"]),
        "available": bool(
            branch.get("workspace") and branch["index_status"] in {"ready", "indexing"}
        ),
        "job_running": branch_job_running(branch["id"]),
    }
    if admin:
        payload["workspace"] = branch.get("workspace")
    return payload


@router.get("/admin/repos/{slug}/branches")
def admin_list_branches(slug: str, admin: dict = Depends(require_admin)):
    repo = _require_repo(slug)
    return {
        "branches": [
            _branch_payload(branch, admin=True)
            for branch in db.list_repo_branches(repo["id"])
        ]
    }


@router.post("/admin/repos/{slug}/branches/discover")
def admin_discover_branches(slug: str, admin: dict = Depends(require_admin)):
    repo = _require_repo(slug)
    try:
        branches = discover_remote_branches(repo)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Branch discovery failed: {exc}")
    approved = {
        branch["name"] for branch in db.list_repo_branches(repo["id"])
    }
    return {
        "branches": [
            {**branch, "approved": branch["name"] in approved}
            for branch in branches
        ]
    }


@router.post("/admin/repos/{slug}/branches")
def admin_approve_branch(
    slug: str,
    request: ApproveBranchRequest,
    admin: dict = Depends(require_admin),
):
    repo = _require_repo(slug)
    try:
        branch = approve_repo_branch(repo, request.name)
    except (RuntimeError, ValueError) as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    accepted = submit_branch_job(branch["id"], sync=True)
    db.record_audit(
        admin["username"],
        "approve_branch",
        slug,
        request.name,
    )
    return {
        "branch": _branch_payload(db.get_repo_branch(branch["id"]), admin=True),
        "accepted": accepted,
    }


@router.patch("/admin/repos/{slug}/branches/{branch_id}")
def admin_update_branch(
    slug: str,
    branch_id: int,
    request: BranchSettingsRequest,
    admin: dict = Depends(require_admin),
):
    repo = _require_repo(slug)
    branch = _require_repo_branch(repo, branch_id)
    interval = max(60, min(86400, request.freshness_interval_seconds))
    db.update_repo_branch_settings(
        branch["id"],
        allow_user_sync=request.allow_user_sync,
        auto_sync=request.auto_sync,
        strict_freshness=request.strict_freshness,
        freshness_interval_seconds=interval,
    )
    db.record_audit(
        admin["username"],
        "update_branch",
        slug,
        f"{branch['name']} user_sync={request.allow_user_sync} "
        f"auto_sync={request.auto_sync} strict={request.strict_freshness}",
    )
    return {
        "branch": _branch_payload(db.get_repo_branch(branch_id), admin=True)
    }


@router.post("/admin/repos/{slug}/branches/{branch_id}/sync")
def admin_sync_branch(
    slug: str,
    branch_id: int,
    admin: dict = Depends(require_admin),
):
    repo = _require_repo(slug)
    branch = _require_repo_branch(repo, branch_id)
    accepted = submit_branch_job(branch["id"], sync=True)
    db.record_audit(
        admin["username"],
        "sync_branch",
        slug,
        branch["name"],
    )
    return {
        "branch": _branch_payload(db.get_repo_branch(branch_id), admin=True),
        "accepted": accepted,
    }


@router.delete("/admin/repos/{slug}/branches/{branch_id}")
def admin_delete_branch(
    slug: str,
    branch_id: int,
    admin: dict = Depends(require_admin),
):
    repo = _require_repo(slug)
    branch = _require_repo_branch(repo, branch_id)
    if branch_job_running(branch_id):
        raise HTTPException(status_code=409, detail="Branch indexing is in progress.")
    try:
        delete_approved_branch(branch_id)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    db.record_audit(admin["username"], "delete_branch", slug, branch["name"])
    return {"deleted": branch_id}


@router.get("/repo/branches")
def user_list_branches(
    workspace: str = Query(...),
    user: dict = Depends(require_user),
):
    repo = _authorized_repo(workspace, user)
    branches = []
    for branch in db.list_repo_branches(repo["id"]):
        payload = _branch_payload(branch)
        payload["can_sync"] = bool(
            user["role"] == "admin" or branch["allow_user_sync"]
        )
        branches.append(payload)
    return {
        "branches": branches
    }


@router.post("/repo/branches/{branch_id}/sync")
def user_sync_branch(
    branch_id: int,
    workspace: str = Query(...),
    user: dict = Depends(require_user),
):
    repo = _authorized_repo(workspace, user)
    branch = _require_repo_branch(repo, branch_id)
    if user["role"] != "admin" and not branch["allow_user_sync"]:
        raise HTTPException(
            status_code=403,
            detail="User-triggered synchronization is disabled for this branch.",
        )
    if branch_job_running(branch_id):
        return {"branch": _branch_payload(branch), "accepted": False}

    if user["role"] != "admin" and BRANCH_USER_SYNC_COOLDOWN_SECONDS:
        key = (user["id"], branch_id)
        now = time.monotonic()
        with _user_sync_lock:
            previous = _user_sync_times.get(key)
            if previous is not None and now - previous < BRANCH_USER_SYNC_COOLDOWN_SECONDS:
                remaining = int(BRANCH_USER_SYNC_COOLDOWN_SECONDS - (now - previous)) + 1
                raise HTTPException(
                    status_code=429,
                    detail=f"Please wait {remaining} seconds before syncing again.",
                )
            _user_sync_times[key] = now

    accepted = submit_branch_job(branch_id, sync=True)
    db.record_audit(user["username"], "sync_branch", repo["slug"], branch["name"])
    return {
        "branch": _branch_payload(db.get_repo_branch(branch_id)),
        "accepted": accepted,
    }
