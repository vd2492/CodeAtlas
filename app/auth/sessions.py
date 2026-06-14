"""Session cookie helpers and FastAPI auth dependencies.

Sessions are DB-backed (app/db.py) and carried in an HttpOnly cookie. Same-origin
fetches send it automatically, so the vanilla-JS UI needs no token plumbing.
"""

from typing import Optional

from fastapi import Depends, HTTPException, Request, Response

from .. import db

COOKIE_NAME = "ca_session"
# 30 days; sessions also live until logout / row deletion.
COOKIE_MAX_AGE = 60 * 60 * 24 * 30


def set_session_cookie(response: Response, token: str) -> None:
    response.set_cookie(
        COOKIE_NAME,
        token,
        max_age=COOKIE_MAX_AGE,
        httponly=True,
        samesite="lax",
        path="/",
    )


def clear_session_cookie(response: Response) -> None:
    response.delete_cookie(COOKIE_NAME, path="/")


def get_current_user(request: Request) -> Optional[dict]:
    """Resolve the logged-in user from the session cookie, or None."""
    token = request.cookies.get(COOKIE_NAME)
    return db.get_session_user(token)


def require_user(request: Request) -> dict:
    user = get_current_user(request)
    if not user:
        raise HTTPException(status_code=401, detail="Authentication required.")
    return user


def require_admin(user: dict = Depends(require_user)) -> dict:
    if user.get("role") != "admin":
        raise HTTPException(status_code=403, detail="Admin access required.")
    return user
