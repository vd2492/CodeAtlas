"""Authentication & user-management routes (Phase 2).

Sessions are cookie-based and DB-backed. The first user created (when the store
is empty) becomes the admin; thereafter admins create users and grant repo
access.
"""

import json
import os
import re
import time
from collections import defaultdict
from typing import List, Optional
from urllib.parse import urlparse

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

try:
    from google.auth.transport import requests as google_requests
    from google.oauth2 import id_token as google_id_token
except ImportError:  # pragma: no cover - depends on deployment extras.
    google_requests = None
    google_id_token = None

router = APIRouter(prefix="/auth", tags=["auth"])

AUTH_MODE = os.environ.get("CODEATLAS_AUTH_MODE", "password").strip().lower()
if AUTH_MODE not in {"password", "mixed", "google"}:
    AUTH_MODE = "password"
GOOGLE_CLIENT_ID = os.environ.get("CODEATLAS_GOOGLE_CLIENT_ID", "").strip()
GOOGLE_ALLOWED_DOMAINS = {
    domain.strip().lower().lstrip("@")
    for domain in os.environ.get("CODEATLAS_GOOGLE_ALLOWED_DOMAINS", "").split(",")
    if domain.strip()
}
GOOGLE_AUTO_CREATE = os.environ.get(
    "CODEATLAS_GOOGLE_AUTO_CREATE", "false"
).lower() in {"1", "true", "yes"}
LOGIN_RATE_LIMIT = int(os.environ.get("CODEATLAS_LOGIN_RATE_LIMIT", "10"))
LOGIN_RATE_WINDOW_SECONDS = int(
    os.environ.get("CODEATLAS_LOGIN_RATE_WINDOW_SECONDS", "300")
)
_login_failures: "dict[str, list[float]]" = defaultdict(list)
EMAIL_RE = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")


def enforce_login_rate_limit(username: str) -> None:
    now = time.monotonic()
    key = (username or "").strip().lower()
    hits = [
        timestamp
        for timestamp in _login_failures[key]
        if now - timestamp < LOGIN_RATE_WINDOW_SECONDS
    ]
    _login_failures[key] = hits
    if len(hits) >= LOGIN_RATE_LIMIT:
        raise HTTPException(
            status_code=429,
            detail=(
                f"Too many failed login attempts. Please wait "
                f"{LOGIN_RATE_WINDOW_SECONDS} seconds and retry."
            ),
        )


def record_login_failure(username: str) -> None:
    key = (username or "").strip().lower()
    _login_failures[key].append(time.monotonic())


def clear_login_failures(username: str) -> None:
    key = (username or "").strip().lower()
    _login_failures.pop(key, None)


def password_login_enabled() -> bool:
    return AUTH_MODE in {"password", "mixed"}


def google_login_enabled() -> bool:
    return AUTH_MODE in {"mixed", "google"}


def normalize_email(email: str) -> str:
    return (email or "").strip().lower()


def validate_email(email: str) -> str:
    normalized = normalize_email(email)
    if not normalized or not EMAIL_RE.match(normalized):
        raise HTTPException(status_code=400, detail="A valid email address is required.")
    return normalized


def enforce_allowed_google_domain(email: str, hosted_domain: str = None) -> None:
    if not GOOGLE_ALLOWED_DOMAINS:
        return
    email_domain = email.rsplit("@", 1)[-1].lower()
    hd_domain = (hosted_domain or "").strip().lower()
    if email_domain in GOOGLE_ALLOWED_DOMAINS or hd_domain in GOOGLE_ALLOWED_DOMAINS:
        return
    raise HTTPException(
        status_code=403,
        detail="This Google account is not allowed for this CodeAtlas instance.",
    )


def user_auth_status(user: dict) -> str:
    credential_login = bool(user.get("credential_login")) or str(
        user.get("password_hash") or ""
    ).startswith("pbkdf2_sha256$")
    google_linked = bool(user.get("google_linked")) or bool(user.get("google_sub"))
    google_pending = bool(user.get("email")) and not google_linked
    if credential_login and google_linked:
        return "password_google"
    if credential_login and google_pending:
        return "password_google_pending"
    if google_linked:
        return "google_linked"
    if google_pending:
        return "google_pending"
    return "password_only"


def verify_google_credential(credential: str) -> dict:
    if not GOOGLE_CLIENT_ID:
        raise HTTPException(status_code=503, detail="Google login is not configured.")
    if google_id_token is None or google_requests is None:
        raise HTTPException(
            status_code=503,
            detail="Google login dependency is not installed on the server.",
        )
    try:
        return google_id_token.verify_oauth2_token(
            credential,
            google_requests.Request(),
            GOOGLE_CLIENT_ID,
        )
    except Exception as exc:
        raise HTTPException(status_code=401, detail="Invalid Google credential.") from exc


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
    return {
        "id": user["id"],
        "username": user["username"],
        "email": user.get("email"),
        "display_name": user.get("display_name"),
        "role": user["role"],
        "user_type": user.get("user_type") or "dev_team",
        "auth_status": user_auth_status(user),
    }


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


class GoogleCredentialRequest(BaseModel):
    credential: str


class CreateUserRequest(BaseModel):
    username: str
    password: str
    role: str = "user"
    user_type: str = "dev_team"
    grant_slugs: Optional[List[str]] = None


class GrantGoogleAccessRequest(BaseModel):
    email: str
    confirm_email: str
    grant_slugs: List[str]
    role: str = "user"
    user_type: str = "dev_team"


class UpdateUserRequest(BaseModel):
    username: Optional[str] = None
    password: Optional[str] = None
    user_type: Optional[str] = None


class LlmCredsRequest(BaseModel):
    api_key: str
    provider: Optional[str] = None   # auto-sniffed from the key if omitted
    base_url: Optional[str] = None
    model: Optional[str] = None


def validate_llm_key_endpoint(api_key: str, base_url: str) -> None:
    """Catch MiMo key/endpoint mismatches before storing unusable credentials."""
    key = (api_key or "").strip()
    hostname = (urlparse(base_url or "").hostname or "").lower()
    is_token_plan_endpoint = (
        hostname.startswith("token-plan-")
        and hostname.endswith(".xiaomimimo.com")
    )
    is_mimo_endpoint = hostname == "api.xiaomimimo.com" or is_token_plan_endpoint

    if key.startswith("tp-") and is_mimo_endpoint and not is_token_plan_endpoint:
        raise HTTPException(
            status_code=400,
            detail=(
                "MiMo Token Plan keys (tp-…) require the China, Singapore, or "
                "Europe Token Plan endpoint shown in your MiMo account."
            ),
        )
    if key.startswith("sk-") and is_token_plan_endpoint:
        raise HTTPException(
            status_code=400,
            detail=(
                "MiMo Token Plan endpoints require a tp-… key. For a pay-as-you-go "
                "MiMo sk-… key, select MiMo pay-as-you-go."
            ),
        )


@router.get("/status")
def status():
    """Whether an admin has been bootstrapped yet (drives setup vs login UI)."""
    return {"bootstrapped": db.user_count() > 0}


@router.get("/config")
def auth_config():
    """Public auth settings needed by the static login screens."""
    return {
        "auth_mode": AUTH_MODE,
        "password_login_enabled": password_login_enabled(),
        "google_login_enabled": google_login_enabled() and bool(GOOGLE_CLIENT_ID),
        "google_client_id": GOOGLE_CLIENT_ID,
    }


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
    if not password_login_enabled():
        raise HTTPException(status_code=403, detail="Credential login is disabled.")
    enforce_login_rate_limit(creds.username)
    user = db.get_user_by_username(creds.username)
    if not user or not verify_password(creds.password, user["password_hash"]):
        record_login_failure(creds.username)
        db.record_audit(creds.username, "login_failed")
        raise HTTPException(status_code=401, detail="Invalid username or password.")
    clear_login_failures(creds.username)
    token = db.create_session(user["id"])
    set_session_cookie(response, token)
    db.record_audit(user["username"], "login")
    return {"user": _public_user(user)}


@router.post("/google")
def google_login(req: GoogleCredentialRequest, response: Response):
    if not google_login_enabled():
        raise HTTPException(status_code=403, detail="Google login is disabled.")
    id_info = verify_google_credential(req.credential)
    email = validate_email(id_info.get("email") or "")
    if str(id_info.get("email_verified")).lower() not in {"true", "1"}:
        raise HTTPException(status_code=403, detail="Google email is not verified.")
    google_sub = (id_info.get("sub") or "").strip()
    if not google_sub:
        raise HTTPException(status_code=401, detail="Invalid Google credential.")
    enforce_allowed_google_domain(email, id_info.get("hd"))

    user = db.get_user_by_google_sub(google_sub)
    if user:
        existing_by_email = db.get_user_by_email(email)
        if existing_by_email and existing_by_email["id"] != user["id"]:
            raise HTTPException(
                status_code=409,
                detail="This Google email is already assigned to another user.",
            )
        user = db.update_user_google_identity(
            user["id"],
            email=email,
            display_name=id_info.get("name") or user.get("display_name"),
        )
    else:
        user = db.get_user_by_email(email) or db.get_user_by_username(email)
        if not user and GOOGLE_AUTO_CREATE:
            user = db.create_google_user(
                email,
                google_sub=google_sub,
                display_name=id_info.get("name"),
            )
        if not user:
            db.record_audit(email, "google_login_unprovisioned")
            raise HTTPException(
                status_code=403,
                detail="This Google account has not been granted CodeAtlas access.",
            )
        if user.get("google_sub") and user["google_sub"] != google_sub:
            raise HTTPException(
                status_code=409,
                detail="This user is already linked to a different Google account.",
            )
        user = db.update_user_google_identity(
            user["id"],
            email=email,
            google_sub=google_sub,
            display_name=id_info.get("name") or user.get("display_name"),
        )

    token = db.create_session(user["id"])
    set_session_cookie(response, token)
    db.record_audit(user["username"], "google_login", email)
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
    users = []
    for user in db.list_users():
        users.append({**user, "auth_status": user_auth_status(user)})
    return {"users": users}


@router.get("/admin/audit")
def list_audit(admin: dict = Depends(require_admin), limit: int = 100):
    return {"audit": db.list_audit(limit)}


@router.post("/admin/users")
def create_user(req: CreateUserRequest, admin: dict = Depends(require_admin)):
    if req.role not in ("admin", "user"):
        raise HTTPException(status_code=400, detail="role must be 'admin' or 'user'.")
    if req.user_type not in ("product_team", "dev_team"):
        raise HTTPException(
            status_code=400,
            detail="user_type must be 'product_team' or 'dev_team'.",
        )
    if db.get_user_by_username(req.username):
        raise HTTPException(status_code=409, detail="username already exists.")
    user = db.create_user(
        req.username,
        hash_password(req.password),
        role=req.role,
        user_type=req.user_type,
    )
    db.record_audit(
        admin["username"],
        "create_user",
        req.username,
        f"role={req.role}, user_type={req.user_type}",
    )

    granted = []
    for slug in req.grant_slugs or []:
        repo = db.get_repo_by_slug(slug)
        if repo:
            db.grant_access(user["id"], repo["id"])
            db.record_audit(admin["username"], "grant", slug, req.username)
            granted.append(slug)
    return {"user": _public_user(user), "granted": granted}


@router.post("/admin/google-access")
def grant_google_access(
    req: GrantGoogleAccessRequest,
    admin: dict = Depends(require_admin),
):
    email = validate_email(req.email)
    confirm_email = validate_email(req.confirm_email)
    if email != confirm_email:
        raise HTTPException(status_code=400, detail="Email addresses do not match.")
    enforce_allowed_google_domain(email)
    if req.role not in ("admin", "user"):
        raise HTTPException(status_code=400, detail="role must be 'admin' or 'user'.")
    if req.user_type not in ("product_team", "dev_team"):
        raise HTTPException(
            status_code=400,
            detail="user_type must be 'product_team' or 'dev_team'.",
        )
    slugs = [slug.strip() for slug in req.grant_slugs if slug and slug.strip()]
    if not slugs:
        raise HTTPException(status_code=400, detail="Select at least one repository.")

    repos = []
    missing = []
    unavailable = []
    for slug in slugs:
        repo = db.get_repo_by_slug(slug)
        if not repo:
            missing.append(slug)
        elif repo["status"] == "new":
            unavailable.append(slug)
        else:
            repos.append(repo)
    if missing:
        raise HTTPException(
            status_code=404,
            detail=f"No repo with slug '{missing[0]}'.",
        )
    if unavailable:
        raise HTTPException(
            status_code=409,
            detail=f"Repository '{unavailable[0]}' has not been cloned yet.",
        )

    user = db.get_user_by_email(email) or db.get_user_by_username(email)
    if user:
        if not user.get("email"):
            user = db.update_user_google_identity(user["id"], email=email)
        role_change = req.role != user["role"]
        user_type_change = req.user_type != (user.get("user_type") or "dev_team")
        if role_change:
            if user["id"] == admin["id"] and req.role != "admin":
                raise HTTPException(
                    status_code=400,
                    detail="You cannot change your own admin role.",
                )
            if user["role"] == "admin" and req.role != "admin" and db.admin_count() <= 1:
                raise HTTPException(
                    status_code=400,
                    detail="Cannot remove the last admin.",
                )
        if role_change or user_type_change:
            user = db.update_user_role_and_type(
                user["id"],
                role=req.role if role_change else None,
                user_type=req.user_type,
            )
    else:
        user = db.create_google_user(email, role=req.role, user_type=req.user_type)

    granted = []
    for repo in repos:
        db.grant_access(user["id"], repo["id"])
        db.record_audit(admin["username"], "grant", repo["slug"], email)
        granted.append(repo["slug"])
    db.record_audit(
        admin["username"],
        "provision_google_user",
        email,
        f"role={req.role}; user_type={req.user_type}; repos={', '.join(granted)}",
    )
    return {"user": _public_user(user), "granted": granted}


@router.patch("/admin/users/{user_id}")
def update_user_credentials(
    user_id: int,
    req: UpdateUserRequest,
    admin: dict = Depends(require_admin),
):
    target = db.get_user_by_id(user_id)
    if not target:
        raise HTTPException(status_code=404, detail="User not found.")
    if req.username is None and req.password is None and req.user_type is None:
        raise HTTPException(status_code=400, detail="No user changes provided.")

    username = req.username.strip() if req.username is not None else target["username"]
    if not username:
        raise HTTPException(status_code=400, detail="username is required.")
    if req.password == "":
        raise HTTPException(status_code=400, detail="password cannot be empty.")
    user_type = req.user_type or target.get("user_type") or "dev_team"
    if user_type not in ("product_team", "dev_team"):
        raise HTTPException(
            status_code=400,
            detail="user_type must be 'product_team' or 'dev_team'.",
        )
    if (
        username == target["username"]
        and req.password is None
        and user_type == (target.get("user_type") or "dev_team")
    ):
        raise HTTPException(status_code=400, detail="No user changes provided.")

    existing = db.get_user_by_username(username)
    if existing and existing["id"] != user_id:
        raise HTTPException(status_code=409, detail="username already exists.")

    password_hash = hash_password(req.password) if req.password is not None else None
    updated = db.update_user_credentials(
        user_id,
        username,
        password_hash,
        user_type=user_type,
    )
    if not updated:
        raise HTTPException(status_code=404, detail="User not found.")

    changes = []
    if username != target["username"]:
        changes.append(f"username={username}")
    if req.password is not None:
        changes.append("password=updated")
    if user_type != (target.get("user_type") or "dev_team"):
        changes.append(f"user_type={user_type}")
    db.record_audit(
        admin["username"],
        "update_user",
        target["username"],
        ", ".join(changes),
    )
    clear_login_failures(target["username"])
    clear_login_failures(username)
    return {"user": _public_user(updated)}


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
    validate_llm_key_endpoint(creds["api_key"], creds["base_url"])
    db.set_user_llm_creds(user["id"], crypto.encrypt(json.dumps(creds)))
    db.record_audit(user["username"], "set_llm_key", None, creds["provider"])
    return {"configured": True, "provider": creds["provider"],
            "base_url": creds["base_url"], "model": creds["model"]}


@router.delete("/me/llm")
def clear_my_llm(user: dict = Depends(require_user)):
    db.set_user_llm_creds(user["id"], None)
    db.record_audit(user["username"], "clear_llm_key")
    return {"configured": False}
