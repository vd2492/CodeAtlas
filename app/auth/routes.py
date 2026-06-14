"""Authentication & user-management routes (Phase 2).

Sessions are cookie-based and DB-backed. The first user created (when the store
is empty) becomes the admin; thereafter admins create users and grant repo
access.
"""

import json
from typing import List, Optional

from fastapi import APIRouter, Depends, HTTPException, Request, Response
from pydantic import BaseModel

from .. import db
from ..llm.client import sniff_provider
from . import crypto
from .security import hash_password, verify_password
from .sessions import (
    COOKIE_NAME,
    clear_session_cookie,
    get_current_user,
    require_admin,
    require_user,
    set_session_cookie,
)

router = APIRouter(prefix="/auth", tags=["auth"])


def load_user_llm(user_id: int) -> Optional[dict]:
    """Decrypt a user's stored BYOK creds into {provider, base_url, api_key,
    model}, or None if unset/undecryptable. Used as LLM tier 1 for that user."""
    blob = db.get_user_llm_creds(user_id)
    if not blob:
        return None
    try:
        return json.loads(crypto.decrypt(blob))
    except Exception:
        return None


def _public_user(user: dict) -> dict:
    return {"id": user["id"], "username": user["username"], "role": user["role"]}


def _public_repo(repo: dict) -> dict:
    return {
        "slug": repo["slug"],
        "name": repo["name"],
        "workspace": repo["workspace"],
        "status": repo["status"],
    }


class Credentials(BaseModel):
    username: str
    password: str


class CreateUserRequest(BaseModel):
    username: str
    password: str
    role: str = "user"
    grant_slugs: Optional[List[str]] = None


class LlmCredsRequest(BaseModel):
    api_key: str
    provider: Optional[str] = None   # auto-sniffed from the key if omitted
    base_url: Optional[str] = None
    model: Optional[str] = None


@router.get("/status")
def status():
    """Whether an admin has been bootstrapped yet (drives setup vs login UI)."""
    return {"bootstrapped": db.user_count() > 0}


@router.post("/bootstrap")
def bootstrap(creds: Credentials, response: Response):
    """Create the first admin. Only allowed while the user store is empty."""
    if db.user_count() > 0:
        raise HTTPException(status_code=403, detail="Already bootstrapped.")
    if not creds.username or not creds.password:
        raise HTTPException(status_code=400, detail="username and password required.")
    user = db.create_user(creds.username, hash_password(creds.password), role="admin")
    token = db.create_session(user["id"])
    set_session_cookie(response, token)
    db.record_audit(user["username"], "bootstrap_admin", user["username"])
    return {"user": _public_user(user)}


@router.post("/login")
def login(creds: Credentials, response: Response):
    user = db.get_user_by_username(creds.username)
    if not user or not verify_password(creds.password, user["password_hash"]):
        db.record_audit(creds.username, "login_failed")
        raise HTTPException(status_code=401, detail="Invalid username or password.")
    token = db.create_session(user["id"])
    set_session_cookie(response, token)
    db.record_audit(user["username"], "login")
    return {"user": _public_user(user)}


@router.post("/logout")
def logout(request: Request, response: Response):
    db.delete_session(request.cookies.get(COOKIE_NAME))
    clear_session_cookie(response)
    return {"ok": True}


@router.get("/me")
def me(user: Optional[dict] = Depends(get_current_user)):
    if not user:
        return {"user": None, "repos": []}
    if user["role"] == "admin":
        repos = db.list_repos()
    else:
        repos = db.list_repos_for_user(user["id"])
    return {"user": _public_user(user), "repos": [_public_repo(r) for r in repos]}


@router.get("/admin/users")
def list_users(admin: dict = Depends(require_admin)):
    return {"users": db.list_users()}


@router.get("/admin/audit")
def list_audit(admin: dict = Depends(require_admin), limit: int = 100):
    return {"audit": db.list_audit(limit)}


@router.post("/admin/users")
def create_user(req: CreateUserRequest, admin: dict = Depends(require_admin)):
    if req.role not in ("admin", "user"):
        raise HTTPException(status_code=400, detail="role must be 'admin' or 'user'.")
    if db.get_user_by_username(req.username):
        raise HTTPException(status_code=409, detail="username already exists.")
    user = db.create_user(req.username, hash_password(req.password), role=req.role)
    db.record_audit(admin["username"], "create_user", req.username, f"role={req.role}")

    granted = []
    for slug in req.grant_slugs or []:
        repo = db.get_repo_by_slug(slug)
        if repo:
            db.grant_access(user["id"], repo["id"])
            db.record_audit(admin["username"], "grant", slug, req.username)
            granted.append(slug)
    return {"user": _public_user(user), "granted": granted}


@router.delete("/admin/users/{username}")
def delete_user(username: str, admin: dict = Depends(require_admin)):
    """Delete a user (sessions + grants cascade). Cannot delete yourself or the
    last remaining admin."""
    target = db.get_user_by_username(username)
    if not target:
        raise HTTPException(status_code=404, detail=f"No user '{username}'.")
    if target["id"] == admin["id"]:
        raise HTTPException(status_code=400, detail="You cannot delete your own account.")
    if target["role"] == "admin" and db.admin_count() <= 1:
        raise HTTPException(status_code=400, detail="Cannot delete the last admin.")
    db.delete_user(target["id"])
    db.record_audit(admin["username"], "delete_user", username)
    return {"deleted": username}


# --- BYOK: the logged-in user's own LLM key (tier 1) -------------------------

@router.get("/me/llm")
def get_my_llm(user: dict = Depends(require_user)):
    """Non-secret view of the user's stored key (never returns the key itself)."""
    creds = load_user_llm(user["id"])
    if not creds:
        return {"configured": False}
    key = creds.get("api_key", "")
    return {
        "configured": True,
        "provider": creds.get("provider"),
        "base_url": creds.get("base_url"),
        "model": creds.get("model"),
        "key_hint": f"…{key[-4:]}" if len(key) >= 4 else "set",
    }


@router.put("/me/llm")
def set_my_llm(req: LlmCredsRequest, user: dict = Depends(require_user)):
    if not req.api_key.strip():
        raise HTTPException(status_code=400, detail="api_key is required.")
    defaults = sniff_provider(req.api_key)
    creds = {
        "provider": req.provider or defaults["provider"],
        "base_url": req.base_url or defaults["base_url"],
        "model": req.model or defaults["model"],
        "api_key": req.api_key.strip(),
    }
    if creds["provider"] == "openai_compatible" and not creds["base_url"]:
        raise HTTPException(
            status_code=400,
            detail="openai_compatible keys require a base_url (e.g. https://host/v1).",
        )
    db.set_user_llm_creds(user["id"], crypto.encrypt(json.dumps(creds)))
    db.record_audit(user["username"], "set_llm_key", None, creds["provider"])
    return {"configured": True, "provider": creds["provider"],
            "base_url": creds["base_url"], "model": creds["model"]}


@router.delete("/me/llm")
def clear_my_llm(user: dict = Depends(require_user)):
    db.set_user_llm_creds(user["id"], None)
    db.record_audit(user["username"], "clear_llm_key")
    return {"configured": False}
