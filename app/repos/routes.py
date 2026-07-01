"""Admin repository-management routes.

The full lifecycle: add → clone → index → tune retrieval config → publish →
grant/revoke access → delete, plus an admin "test" panel to check
retrieval/answer quality for a specific workspace before exposing it. All routes
require an admin session, and privileged actions are recorded to the audit log.
"""

import re
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel

from .. import db
from ..auth.sessions import require_admin
from ..config import DEFAULT_WORKSPACE, repo_clone_dir
from ..retrieval.config_schema import (
    RetrievalConfig,
    load_retrieval_config,
    save_retrieval_config,
)
from .cloning import clone_repo, remove_repo_clone, remove_workspace, sanitize_clone_url
from .branches import (
    copy_config_to_active_branch_workspaces,
    ensure_repo_branch,
    record_legacy_index,
    remove_repo_branch_workspaces,
    restore_branches_after_reclone,
)
from .indexing import index_repo

router = APIRouter(prefix="/admin/repos", tags=["admin-repos"])

SLUG_RE = re.compile(r"^[a-z0-9][a-z0-9-]{0,62}$")


class AddRepoRequest(BaseModel):
    slug: str
    name: str
    source_url: str
    clone_method: str  # https | ssh | gh


class GrantRequest(BaseModel):
    username: str


class PrivacyRequest(BaseModel):
    allow_shared_fallback: bool


class UpdateRepoRequest(BaseModel):
    name: Optional[str] = None
    source_url: Optional[str] = None


class RecloneRepoRequest(BaseModel):
    source_url: Optional[str] = None
    clone_method: Optional[str] = None


class TestRequest(BaseModel):
    question: str


def _require_repo(slug: str) -> dict:
    repo = db.get_repo_by_slug(slug)
    if not repo:
        raise HTTPException(status_code=404, detail=f"No repo with slug '{slug}'.")
    return repo


def _repo_with_clone_state(repo: dict) -> dict:
    clone = repo_clone_dir(repo["workspace"])
    return {
        **repo,
        "clone_available": clone.is_dir() and (clone / ".git").exists(),
    }


@router.get("")
def list_repos(admin: dict = Depends(require_admin)):
    return {"repos": [_repo_with_clone_state(repo) for repo in db.list_repos()]}


@router.post("")
def add_repo(req: AddRepoRequest, admin: dict = Depends(require_admin)):
    """Register + clone a repo (https/ssh/gh) into a new workspace."""
    if not SLUG_RE.match(req.slug):
        raise HTTPException(
            status_code=400,
            detail="slug must be lowercase letters, digits, and dashes.",
        )
    if req.clone_method not in ("https", "ssh", "gh"):
        raise HTTPException(status_code=400, detail="clone_method must be https, ssh, or gh.")
    if db.get_repo_by_slug(req.slug):
        raise HTTPException(status_code=409, detail=f"slug '{req.slug}' already exists.")

    workspace = req.slug
    stored_source_url = sanitize_clone_url(req.source_url)
    repo = db.create_repo(
        req.slug, req.name, stored_source_url, req.clone_method, workspace
    )
    try:
        clone_repo(req.source_url, req.clone_method, workspace)
    except Exception as exc:
        db.set_repo_status(req.slug, "new")  # leave row; surface the error
        raise HTTPException(status_code=400, detail=f"Clone failed: {exc}")
    db.set_repo_status(req.slug, "cloned")
    ensure_repo_branch(db.get_repo_by_slug(req.slug))
    db.record_audit(admin["username"], "add_repo", req.slug, stored_source_url)
    return {"repo": db.get_repo_by_slug(req.slug)}


@router.post("/{slug}/reclone")
def reclone_repo(
    slug: str,
    req: RecloneRepoRequest,
    admin: dict = Depends(require_admin),
):
    """Restore a missing working copy without replacing the current graph."""
    repo = _require_repo(slug)
    clone_path = repo_clone_dir(repo["workspace"])
    if clone_path.exists():
        raise HTTPException(
            status_code=409,
            detail="Repository clone path already exists; refusing to overwrite it.",
        )

    source_url = (
        (req.source_url or "").strip()
        or (repo.get("source_url") or "").strip()
    )
    clone_method = (
        (req.clone_method or "").strip()
        or (repo.get("clone_method") or "").strip()
    )
    if not source_url:
        raise HTTPException(status_code=400, detail="source_url is required.")
    if clone_method not in ("https", "ssh", "gh"):
        raise HTTPException(
            status_code=400,
            detail="clone_method must be https, ssh, or gh.",
        )

    try:
        clone_repo(source_url, clone_method, repo["workspace"])
        stored_source_url = sanitize_clone_url(source_url)
        db.update_repo(
            slug,
            source_url=stored_source_url,
            clone_method=clone_method,
        )
        if repo["status"] == "new":
            db.set_repo_status(slug, "cloned")
        restored = db.get_repo_by_slug(slug)
        ensure_repo_branch(restored)
        restore_branches_after_reclone(restored)
    except Exception as exc:
        remove_repo_clone(repo["workspace"])
        raise HTTPException(status_code=400, detail=f"Reclone failed: {exc}")

    db.record_audit(admin["username"], "reclone_repo", slug, stored_source_url)
    return {"repo": _repo_with_clone_state(db.get_repo_by_slug(slug))}


@router.post("/{slug}/index")
def index_repo_route(slug: str, admin: dict = Depends(require_admin)):
    """Run graphify indexing for the workspace."""
    repo = _require_repo(slug)
    if repo["status"] == "new":
        raise HTTPException(status_code=409, detail="Repo not cloned yet.")
    try:
        index_repo(repo["workspace"])
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Indexing failed: {exc}")
    db.set_repo_status(slug, "indexed")
    record_legacy_index(db.get_repo_by_slug(slug))
    db.record_audit(admin["username"], "index_repo", slug)
    return {"repo": db.get_repo_by_slug(slug)}


@router.get("/{slug}/config")
def get_retrieval_config(slug: str, admin: dict = Depends(require_admin)):
    """Read the safe, config-only retrieval settings for this repo."""
    repo = _require_repo(slug)
    return {"config": load_retrieval_config(repo["workspace"]).to_dict()}


@router.put("/{slug}/config")
def update_retrieval_config(slug: str, config: dict, admin: dict = Depends(require_admin)):
    """Update retrieval settings (stopwords, synonyms, boosts, limits, ...).

    Data-only: unknown keys are dropped by RetrievalConfig.from_dict. Effect on
    retrieval lands in Phase 3; this persists the config now.
    """
    repo = _require_repo(slug)
    parsed = RetrievalConfig.from_dict(config)
    save_retrieval_config(repo["workspace"], parsed)
    copy_config_to_active_branch_workspaces(repo)
    return {"config": parsed.to_dict()}


@router.post("/{slug}/publish")
def publish_repo(slug: str, admin: dict = Depends(require_admin)):
    """Mark the workspace published and exposable to granted users."""
    repo = _require_repo(slug)
    if repo["status"] not in ("indexed", "published"):
        raise HTTPException(status_code=409, detail="Index the repo before publishing.")
    db.set_repo_status(slug, "published")
    db.record_audit(admin["username"], "publish_repo", slug)
    return {"repo": db.get_repo_by_slug(slug)}


@router.post("/{slug}/grant")
def grant_access(slug: str, req: GrantRequest, admin: dict = Depends(require_admin)):
    """Grant a user access to this repo."""
    repo = _require_repo(slug)
    user = db.get_user_by_username(req.username)
    if not user:
        raise HTTPException(status_code=404, detail=f"No user '{req.username}'.")
    db.grant_access(user["id"], repo["id"])
    db.record_audit(admin["username"], "grant", slug, req.username)
    return {"granted": {"username": req.username, "slug": slug}}


@router.post("/{slug}/revoke")
def revoke_access(slug: str, req: GrantRequest, admin: dict = Depends(require_admin)):
    """Revoke a user's access to this repo. Takes effect on their next page load
    (the Ask UI re-reads authorized repos from /auth/me)."""
    repo = _require_repo(slug)
    user = db.get_user_by_username(req.username)
    if not user:
        raise HTTPException(status_code=404, detail=f"No user '{req.username}'.")
    db.revoke_access(user["id"], repo["id"])
    db.record_audit(admin["username"], "revoke", slug, req.username)
    return {"revoked": {"username": req.username, "slug": slug}}


@router.get("/{slug}/members")
def list_members(slug: str, admin: dict = Depends(require_admin)):
    """Users explicitly granted access to this repo (admins reach every repo
    implicitly and are not listed here)."""
    repo = _require_repo(slug)
    return {"members": db.list_repo_members(repo["id"])}


@router.patch("/{slug}")
def update_repo_details(slug: str, req: UpdateRepoRequest, admin: dict = Depends(require_admin)):
    """Edit a repo's display name (and optionally source URL). slug/workspace
    are immutable; status and shared-LLM have their own controls."""
    _require_repo(slug)
    name = req.name.strip() if req.name is not None else None
    if name == "":
        raise HTTPException(status_code=400, detail="name cannot be empty.")
    if name is None and req.source_url is None:
        raise HTTPException(status_code=400, detail="nothing to update.")
    stored_source_url = (
        sanitize_clone_url(req.source_url) if req.source_url is not None else None
    )
    db.update_repo(slug, name=name, source_url=stored_source_url)
    db.record_audit(admin["username"], "update_repo", slug, name or stored_source_url)
    return {"repo": db.get_repo_by_slug(slug)}


@router.patch("/{slug}/privacy")
def set_privacy(slug: str, req: PrivacyRequest, admin: dict = Depends(require_admin)):
    """Toggle whether the shared "Kimi"/Mimo LLM tier may be used for this repo.
    When off, private code is never sent to the shared endpoint (enforced in
    /repo/ask-llm)."""
    _require_repo(slug)
    db.set_repo_shared_fallback(slug, req.allow_shared_fallback)
    db.record_audit(
        admin["username"], "set_privacy", slug,
        f"allow_shared_fallback={req.allow_shared_fallback}",
    )
    return {"repo": db.get_repo_by_slug(slug)}


@router.delete("/{slug}")
def delete_repo(slug: str, admin: dict = Depends(require_admin)):
    """Delete a repo: remove its DB row (grants cascade) and its workspace
    directory (clone + graph + config). The seeded default repo is protected."""
    repo = _require_repo(slug)
    if repo["workspace"] == DEFAULT_WORKSPACE:
        raise HTTPException(status_code=400, detail="The default demo repo cannot be deleted.")
    remove_repo_branch_workspaces(repo)
    db.delete_repo(slug)
    remove_workspace(repo["workspace"])
    db.record_audit(admin["username"], "delete_repo", slug)
    return {"deleted": slug}


@router.post("/{slug}/test")
def test_repo(slug: str, req: TestRequest, admin: dict = Depends(require_admin)):
    """Run retrieval + LLM against this workspace so the admin can judge answer
    quality before granting users access."""
    repo = _require_repo(slug)
    # Lazy import avoids a circular import (main mounts this router at load time).
    from ..main import answer_question

    try:
        branch = db.get_legacy_repo_branch(repo["id"])
        workspace = branch["workspace"] if branch and branch.get("workspace") else repo["workspace"]
        return answer_question(
            req.question,
            workspace=workspace,
            allow_shared_fallback=bool(repo["allow_shared_fallback"]),
        )
    except RuntimeError as error:
        raise HTTPException(status_code=400, detail=str(error))
    except Exception as error:
        raise HTTPException(status_code=500, detail=f"Test failed: {error}")
