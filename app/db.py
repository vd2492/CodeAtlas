"""SQLite storage for users, repositories, sessions, and per-user repo access.

Phase 2 wires the auth and repo-lifecycle routers to these helpers. Per-repo
retrieval tuning is stored as JSON on disk (see app/retrieval/config_schema.py),
not here.
"""

import secrets
import sqlite3
from contextlib import contextmanager
from typing import List, Optional

from .config import DB_PATH, DEFAULT_WORKSPACE, SESSION_MAX_AGE_SECONDS

SCHEMA = """
CREATE TABLE IF NOT EXISTS users (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    username      TEXT UNIQUE NOT NULL,
    password_hash TEXT NOT NULL,
    role          TEXT NOT NULL DEFAULT 'user' CHECK (role IN ('admin', 'user')),
    llm_creds     TEXT,                        -- optional encrypted BYOK creds (JSON)
    created_at    TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS repos (
    id                    INTEGER PRIMARY KEY AUTOINCREMENT,
    slug                  TEXT UNIQUE NOT NULL,
    name                  TEXT NOT NULL,
    source_url            TEXT,
    clone_method          TEXT CHECK (clone_method IN ('https', 'ssh', 'gh')),
    workspace             TEXT UNIQUE NOT NULL, -- folder under data/workspaces/
    status                TEXT NOT NULL DEFAULT 'new'
                          CHECK (status IN ('new', 'cloned', 'indexed', 'published')),
    allow_shared_fallback INTEGER NOT NULL DEFAULT 0,  -- 0 = never use shared Kimi tier
    created_at            TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS repo_access (
    user_id    INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    repo_id    INTEGER NOT NULL REFERENCES repos(id) ON DELETE CASCADE,
    granted_at TEXT NOT NULL DEFAULT (datetime('now')),
    PRIMARY KEY (user_id, repo_id)
);

CREATE TABLE IF NOT EXISTS sessions (
    token      TEXT PRIMARY KEY,
    user_id    INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    created_at TEXT NOT NULL DEFAULT (datetime('now')),
    expires_at TEXT
);

CREATE TABLE IF NOT EXISTS audit_log (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    actor_username  TEXT,                        -- who performed the action
    action          TEXT NOT NULL,               -- e.g. login, grant, revoke, delete_repo
    target          TEXT,                        -- repo slug / username acted upon
    detail          TEXT,                        -- optional extra context
    created_at      TEXT NOT NULL DEFAULT (datetime('now'))
);
"""


@contextmanager
def connect():
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode = WAL")
    conn.execute("PRAGMA foreign_keys = ON")
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()


def init_db() -> None:
    with connect() as conn:
        conn.executescript(SCHEMA)
        session_columns = {
            row["name"] for row in conn.execute("PRAGMA table_info(sessions)").fetchall()
        }
        if "expires_at" not in session_columns:
            conn.execute("ALTER TABLE sessions ADD COLUMN expires_at TEXT")


def user_count() -> int:
    with connect() as conn:
        return conn.execute("SELECT COUNT(*) FROM users").fetchone()[0]


# --- Users -------------------------------------------------------------------

def create_user(username: str, password_hash: str, role: str = "user") -> dict:
    with connect() as conn:
        cur = conn.execute(
            "INSERT INTO users (username, password_hash, role) VALUES (?, ?, ?)",
            (username, password_hash, role),
        )
        row = conn.execute("SELECT * FROM users WHERE id = ?", (cur.lastrowid,)).fetchone()
        return dict(row)


def get_user_by_username(username: str) -> Optional[dict]:
    with connect() as conn:
        row = conn.execute("SELECT * FROM users WHERE username = ?", (username,)).fetchone()
        return dict(row) if row else None


def get_user_by_id(user_id: int) -> Optional[dict]:
    with connect() as conn:
        row = conn.execute("SELECT * FROM users WHERE id = ?", (user_id,)).fetchone()
        return dict(row) if row else None


def list_users() -> List[dict]:
    with connect() as conn:
        rows = conn.execute(
            "SELECT id, username, role, created_at FROM users ORDER BY id"
        ).fetchall()
        return [dict(r) for r in rows]


def admin_count() -> int:
    with connect() as conn:
        return conn.execute("SELECT COUNT(*) FROM users WHERE role = 'admin'").fetchone()[0]


def delete_user(user_id: int) -> None:
    """Delete a user; FK cascade removes their sessions and repo_access grants."""
    with connect() as conn:
        conn.execute("DELETE FROM users WHERE id = ?", (user_id,))


def set_user_llm_creds(user_id: int, blob: Optional[str]) -> None:
    """Store the user's encrypted BYOK creds JSON (or NULL to clear)."""
    with connect() as conn:
        conn.execute("UPDATE users SET llm_creds = ? WHERE id = ?", (blob, user_id))


def get_user_llm_creds(user_id: int) -> Optional[str]:
    with connect() as conn:
        row = conn.execute("SELECT llm_creds FROM users WHERE id = ?", (user_id,)).fetchone()
        return row[0] if row and row[0] else None


# --- Sessions ----------------------------------------------------------------

def create_session(user_id: int) -> str:
    token = secrets.token_urlsafe(32)
    with connect() as conn:
        conn.execute(
            "INSERT INTO sessions (token, user_id, expires_at) "
            "VALUES (?, ?, datetime('now', ?))",
            (token, user_id, f"+{SESSION_MAX_AGE_SECONDS} seconds"),
        )
    return token


def get_session_user(token: str) -> Optional[dict]:
    if not token:
        return None
    with connect() as conn:
        row = conn.execute(
            "SELECT u.* FROM sessions s JOIN users u ON u.id = s.user_id "
            "WHERE s.token = ? "
            "AND (s.expires_at IS NULL OR s.expires_at > datetime('now'))",
            (token,),
        ).fetchone()
        return dict(row) if row else None


def delete_session(token: str) -> None:
    if not token:
        return
    with connect() as conn:
        conn.execute("DELETE FROM sessions WHERE token = ?", (token,))


# --- Repos -------------------------------------------------------------------

def create_repo(slug: str, name: str, source_url: str, clone_method: str,
                workspace: str, status: str = "new") -> dict:
    with connect() as conn:
        cur = conn.execute(
            "INSERT INTO repos "
            "(slug, name, source_url, clone_method, workspace, status, allow_shared_fallback) "
            "VALUES (?, ?, ?, ?, ?, ?, ?)",
            (slug, name, source_url, clone_method, workspace, status, 0),
        )
        row = conn.execute("SELECT * FROM repos WHERE id = ?", (cur.lastrowid,)).fetchone()
        return dict(row)


def get_repo_by_slug(slug: str) -> Optional[dict]:
    with connect() as conn:
        row = conn.execute("SELECT * FROM repos WHERE slug = ?", (slug,)).fetchone()
        return dict(row) if row else None


def get_repo_by_workspace(workspace: str) -> Optional[dict]:
    with connect() as conn:
        row = conn.execute("SELECT * FROM repos WHERE workspace = ?", (workspace,)).fetchone()
        return dict(row) if row else None


def list_repos() -> List[dict]:
    with connect() as conn:
        rows = conn.execute("SELECT * FROM repos ORDER BY id").fetchall()
        return [dict(r) for r in rows]


def set_repo_status(slug: str, status: str) -> None:
    with connect() as conn:
        conn.execute("UPDATE repos SET status = ? WHERE slug = ?", (status, slug))


def update_repo(slug: str, name: Optional[str] = None,
                source_url: Optional[str] = None) -> None:
    """Update editable repo details. slug/workspace are immutable (on-disk
    identity) and not changed here."""
    sets, vals = [], []
    if name is not None:
        sets.append("name = ?"); vals.append(name)
    if source_url is not None:
        sets.append("source_url = ?"); vals.append(source_url)
    if not sets:
        return
    vals.append(slug)
    with connect() as conn:
        conn.execute(f"UPDATE repos SET {', '.join(sets)} WHERE slug = ?", vals)


def set_repo_shared_fallback(slug: str, allow: bool) -> None:
    with connect() as conn:
        conn.execute(
            "UPDATE repos SET allow_shared_fallback = ? WHERE slug = ?",
            (1 if allow else 0, slug),
        )


def delete_repo(slug: str) -> None:
    """Delete a repo row; FK cascade clears its repo_access grants. The caller
    is responsible for removing the workspace directory on disk."""
    with connect() as conn:
        conn.execute("DELETE FROM repos WHERE slug = ?", (slug,))


def seed_default_repo() -> None:
    """Register the seeded default workspace as a published repo so access
    control is uniform across all repos."""
    if get_repo_by_workspace(DEFAULT_WORKSPACE):
        return
    with connect() as conn:
        conn.execute(
            "INSERT INTO repos "
            "(slug, name, source_url, clone_method, workspace, status, allow_shared_fallback) "
            "VALUES (?, ?, ?, ?, ?, ?, ?)",
            ("default", "Default (demo)", None, None, DEFAULT_WORKSPACE, "published", 0),
        )


# --- Access ------------------------------------------------------------------

def grant_access(user_id: int, repo_id: int) -> None:
    with connect() as conn:
        conn.execute(
            "INSERT OR IGNORE INTO repo_access (user_id, repo_id) VALUES (?, ?)",
            (user_id, repo_id),
        )


def revoke_access(user_id: int, repo_id: int) -> None:
    with connect() as conn:
        conn.execute(
            "DELETE FROM repo_access WHERE user_id = ? AND repo_id = ?",
            (user_id, repo_id),
        )


def list_repos_for_user(user_id: int) -> List[dict]:
    with connect() as conn:
        rows = conn.execute(
            "SELECT r.* FROM repos r JOIN repo_access a ON a.repo_id = r.id "
            "WHERE a.user_id = ? ORDER BY r.id",
            (user_id,),
        ).fetchall()
        return [dict(r) for r in rows]


def user_has_repo(user_id: int, workspace: str) -> bool:
    with connect() as conn:
        row = conn.execute(
            "SELECT 1 FROM repo_access a JOIN repos r ON r.id = a.repo_id "
            "WHERE a.user_id = ? AND r.workspace = ?",
            (user_id, workspace),
        ).fetchone()
        return row is not None


def list_repo_members(repo_id: int) -> List[dict]:
    """Users (non-admins are the typical case) granted access to a repo."""
    with connect() as conn:
        rows = conn.execute(
            "SELECT u.id, u.username, u.role, a.granted_at "
            "FROM repo_access a JOIN users u ON u.id = a.user_id "
            "WHERE a.repo_id = ? ORDER BY u.username",
            (repo_id,),
        ).fetchall()
        return [dict(r) for r in rows]


# --- Audit log ---------------------------------------------------------------

def record_audit(actor: Optional[str], action: str,
                 target: str = None, detail: str = None) -> None:
    with connect() as conn:
        conn.execute(
            "INSERT INTO audit_log (actor_username, action, target, detail) "
            "VALUES (?, ?, ?, ?)",
            (actor, action, target, detail),
        )


def list_audit(limit: int = 100) -> List[dict]:
    with connect() as conn:
        rows = conn.execute(
            "SELECT actor_username, action, target, detail, created_at "
            "FROM audit_log ORDER BY id DESC LIMIT ?",
            (limit,),
        ).fetchall()
        return [dict(r) for r in rows]
