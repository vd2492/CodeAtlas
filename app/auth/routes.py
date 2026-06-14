"""Authentication & user-management routes.

SKELETON (Phase 2): endpoints are declared so the surface exists and is
documented, but return 501 until sessions and the user store are wired to
app/db.py and app/auth/security.py.
"""

from fastapi import APIRouter, HTTPException

router = APIRouter(prefix="/auth", tags=["auth"])

_NOT_YET = "Not implemented yet (Phase 2: auth & user management)."


@router.post("/login")
def login():
    """Exchange username/password for a session. -> Phase 2."""
    raise HTTPException(status_code=501, detail=_NOT_YET)


@router.post("/logout")
def logout():
    raise HTTPException(status_code=501, detail=_NOT_YET)


@router.get("/me")
def me():
    """Return the current user and their authorized repositories."""
    raise HTTPException(status_code=501, detail=_NOT_YET)


@router.post("/admin/users")
def create_user():
    """Admin-only: create a user and (optionally) grant repo access."""
    raise HTTPException(status_code=501, detail=_NOT_YET)
