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

CREATE TABLE IF NOT EXISTS repo_branches (
    id                         INTEGER PRIMARY KEY AUTOINCREMENT,
    repo_id                    INTEGER NOT NULL REFERENCES repos(id) ON DELETE CASCADE,
    name                       TEXT NOT NULL,
    workspace                  TEXT UNIQUE,
    indexed_commit_sha         TEXT,
    remote_commit_sha          TEXT,
    indexed_at                 TEXT,
    last_checked_at            TEXT,
    index_status               TEXT NOT NULL DEFAULT 'never_indexed'
                               CHECK (index_status IN ('never_indexed', 'ready', 'indexing', 'failed')),
    job_stage                  TEXT NOT NULL DEFAULT 'idle',
    freshness_status           TEXT NOT NULL DEFAULT 'unknown'
                               CHECK (freshness_status IN
                                      ('unknown', 'checking', 'up_to_date', 'behind',
                                       'diverged', 'remote_unavailable')),
    behind_count               INTEGER NOT NULL DEFAULT 0,
    last_error                 TEXT,
    allow_user_sync            INTEGER NOT NULL DEFAULT 1,
    auto_sync                  INTEGER NOT NULL DEFAULT 0,
    strict_freshness           INTEGER NOT NULL DEFAULT 0,
    freshness_interval_seconds INTEGER NOT NULL DEFAULT 300,
    is_legacy                  INTEGER NOT NULL DEFAULT 0,
    created_at                 TEXT NOT NULL DEFAULT (datetime('now')),
    updated_at                 TEXT NOT NULL DEFAULT (datetime('now')),
    UNIQUE (repo_id, name)
);

CREATE TABLE IF NOT EXISTS repo_branch_versions (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    branch_id   INTEGER NOT NULL REFERENCES repo_branches(id) ON DELETE CASCADE,
    commit_sha  TEXT,
    workspace   TEXT UNIQUE NOT NULL,
    active      INTEGER NOT NULL DEFAULT 0,
    indexed_at  TEXT NOT NULL DEFAULT (datetime('now')),
    created_at  TEXT NOT NULL DEFAULT (datetime('now'))
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
        branch_columns = {
            row["name"] for row in conn.execute(
                "PRAGMA table_info(repo_branches)"
            ).fetchall()
        }
        if "job_stage" not in branch_columns:
            conn.execute(
                "ALTER TABLE repo_branches "
                "ADD COLUMN job_stage TEXT NOT NULL DEFAULT 'idle'"
            )
        if "behind_count" not in branch_columns:
            conn.execute(
                "ALTER TABLE repo_branches "
                "ADD COLUMN behind_count INTEGER NOT NULL DEFAULT 0"
            )


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
        row = conn.execute(
            "SELECT DISTINCT r.* FROM repos r "
            "LEFT JOIN repo_branches b ON b.repo_id = r.id "
            "WHERE r.workspace = ? OR b.workspace = ? "
            "ORDER BY CASE WHEN r.workspace = ? THEN 0 ELSE 1 END LIMIT 1",
            (workspace, workspace, workspace),
        ).fetchone()
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


# --- Repository branches -----------------------------------------------------

def create_repo_branch(
    repo_id: int,
    name: str,
    workspace: str = None,
    indexed_commit_sha: str = None,
    index_status: str = "never_indexed",
    freshness_status: str = "unknown",
    indexed_at: str = None,
    is_legacy: bool = False,
) -> dict:
    with connect() as conn:
        cur = conn.execute(
            "INSERT INTO repo_branches "
            "(repo_id, name, workspace, indexed_commit_sha, index_status, "
            "freshness_status, indexed_at, is_legacy) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (
                repo_id,
                name,
                workspace,
                indexed_commit_sha,
                index_status,
                freshness_status,
                indexed_at,
                1 if is_legacy else 0,
            ),
        )
        row = conn.execute(
            "SELECT * FROM repo_branches WHERE id = ?", (cur.lastrowid,)
        ).fetchone()
        return dict(row)


def get_repo_branch(branch_id: int) -> Optional[dict]:
    with connect() as conn:
        row = conn.execute(
            "SELECT b.*, r.slug AS repo_slug, r.name AS repo_name, "
            "r.workspace AS repo_workspace, r.status AS repo_status "
            "FROM repo_branches b JOIN repos r ON r.id = b.repo_id "
            "WHERE b.id = ?",
            (branch_id,),
        ).fetchone()
        return dict(row) if row else None


def get_repo_branch_by_name(repo_id: int, name: str) -> Optional[dict]:
    with connect() as conn:
        row = conn.execute(
            "SELECT * FROM repo_branches WHERE repo_id = ? AND name = ?",
            (repo_id, name),
        ).fetchone()
        return dict(row) if row else None


def get_repo_branch_by_workspace(workspace: str) -> Optional[dict]:
    with connect() as conn:
        row = conn.execute(
            "SELECT b.*, r.slug AS repo_slug, r.name AS repo_name, "
            "r.workspace AS repo_workspace "
            "FROM repo_branches b JOIN repos r ON r.id = b.repo_id "
            "WHERE b.workspace = ?",
            (workspace,),
        ).fetchone()
        return dict(row) if row else None


def get_legacy_repo_branch(repo_id: int) -> Optional[dict]:
    with connect() as conn:
        row = conn.execute(
            "SELECT * FROM repo_branches WHERE repo_id = ? "
            "ORDER BY is_legacy DESC, id LIMIT 1",
            (repo_id,),
        ).fetchone()
        return dict(row) if row else None


def list_repo_branches(repo_id: int) -> List[dict]:
    with connect() as conn:
        rows = conn.execute(
            "SELECT * FROM repo_branches WHERE repo_id = ? "
            "ORDER BY is_legacy DESC, name",
            (repo_id,),
        ).fetchall()
        return [dict(row) for row in rows]


def list_all_repo_branches() -> List[dict]:
    with connect() as conn:
        rows = conn.execute(
            "SELECT b.*, r.slug AS repo_slug, r.workspace AS repo_workspace "
            "FROM repo_branches b JOIN repos r ON r.id = b.repo_id ORDER BY b.id"
        ).fetchall()
        return [dict(row) for row in rows]


def update_repo_branch_state(
    branch_id: int,
    *,
    index_status: str = None,
    job_stage: str = None,
    freshness_status: str = None,
    remote_commit_sha: str = None,
    behind_count: int = None,
    last_error: str = None,
    checked: bool = False,
) -> None:
    sets, values = [], []
    if index_status is not None:
        sets.append("index_status = ?"); values.append(index_status)
    if job_stage is not None:
        sets.append("job_stage = ?"); values.append(job_stage)
    if freshness_status is not None:
        sets.append("freshness_status = ?"); values.append(freshness_status)
    if remote_commit_sha is not None:
        sets.append("remote_commit_sha = ?"); values.append(remote_commit_sha)
    if behind_count is not None:
        sets.append("behind_count = ?"); values.append(behind_count)
    if last_error is not None:
        sets.append("last_error = ?"); values.append(last_error)
    if checked:
        sets.append("last_checked_at = datetime('now')")
    sets.append("updated_at = datetime('now')")
    values.append(branch_id)
    with connect() as conn:
        conn.execute(
            f"UPDATE repo_branches SET {', '.join(sets)} WHERE id = ?", values
        )


def clear_repo_branch_error(branch_id: int) -> None:
    with connect() as conn:
        conn.execute(
            "UPDATE repo_branches SET last_error = NULL, "
            "updated_at = datetime('now') WHERE id = ?",
            (branch_id,),
        )


def update_repo_branch_settings(
    branch_id: int,
    *,
    allow_user_sync: bool,
    auto_sync: bool,
    strict_freshness: bool,
    freshness_interval_seconds: int,
) -> None:
    with connect() as conn:
        conn.execute(
            "UPDATE repo_branches SET allow_user_sync = ?, auto_sync = ?, "
            "strict_freshness = ?, freshness_interval_seconds = ?, "
            "updated_at = datetime('now') WHERE id = ?",
            (
                1 if allow_user_sync else 0,
                1 if auto_sync else 0,
                1 if strict_freshness else 0,
                freshness_interval_seconds,
                branch_id,
            ),
        )


def activate_repo_branch_version(
    branch_id: int,
    commit_sha: str,
    workspace: str,
    verified_remote: bool = True,
) -> None:
    with connect() as conn:
        conn.execute(
            "UPDATE repo_branch_versions SET active = 0 WHERE branch_id = ?",
            (branch_id,),
        )
        conn.execute(
            "INSERT OR IGNORE INTO repo_branch_versions "
            "(branch_id, commit_sha, workspace, active) VALUES (?, ?, ?, 1)",
            (branch_id, commit_sha, workspace),
        )
        conn.execute(
            "UPDATE repo_branch_versions SET active = 1, commit_sha = ?, "
            "indexed_at = datetime('now') "
            "WHERE branch_id = ? AND workspace = ?",
            (commit_sha, branch_id, workspace),
        )
        conn.execute(
            "UPDATE repo_branches SET workspace = ?, indexed_commit_sha = ?, "
            "remote_commit_sha = ?, indexed_at = datetime('now'), "
            "last_checked_at = CASE WHEN ? THEN datetime('now') ELSE NULL END, "
            "index_status = 'ready', job_stage = 'idle', freshness_status = ?, "
            "behind_count = 0, last_error = NULL, "
            "updated_at = datetime('now') WHERE id = ?",
            (
                workspace,
                commit_sha,
                commit_sha if verified_remote else None,
                1 if verified_remote else 0,
                "up_to_date" if verified_remote else "unknown",
                branch_id,
            ),
        )


def add_legacy_repo_branch_version(
    branch_id: int,
    commit_sha: str,
    workspace: str,
) -> None:
    with connect() as conn:
        conn.execute(
            "INSERT OR IGNORE INTO repo_branch_versions "
            "(branch_id, commit_sha, workspace, active) VALUES (?, ?, ?, 1)",
            (branch_id, commit_sha, workspace),
        )


def find_repo_branch_version(branch_id: int, commit_sha: str) -> Optional[dict]:
    with connect() as conn:
        row = conn.execute(
            "SELECT * FROM repo_branch_versions "
            "WHERE branch_id = ? AND commit_sha = ? ORDER BY id DESC LIMIT 1",
            (branch_id, commit_sha),
        ).fetchone()
        return dict(row) if row else None


def list_repo_branch_versions(branch_id: int) -> List[dict]:
    with connect() as conn:
        rows = conn.execute(
            "SELECT * FROM repo_branch_versions WHERE branch_id = ? ORDER BY id DESC",
            (branch_id,),
        ).fetchall()
        return [dict(row) for row in rows]


def list_expired_repo_branch_versions(retention_seconds: int) -> List[dict]:
    modifier = f"-{retention_seconds} seconds"
    with connect() as conn:
        rows = conn.execute(
            "SELECT v.*, b.repo_id, r.workspace AS repo_workspace "
            "FROM repo_branch_versions v "
            "JOIN repo_branches b ON b.id = v.branch_id "
            "JOIN repos r ON r.id = b.repo_id "
            "WHERE v.active = 0 AND v.workspace != r.workspace "
            "AND v.indexed_at < datetime('now', ?)",
            (modifier,),
        ).fetchall()
        return [dict(row) for row in rows]


def delete_repo_branch_version(version_id: int) -> None:
    with connect() as conn:
        conn.execute(
            "DELETE FROM repo_branch_versions WHERE id = ? AND active = 0",
            (version_id,),
        )


def delete_repo_branch(branch_id: int) -> None:
    with connect() as conn:
        conn.execute("DELETE FROM repo_branches WHERE id = ?", (branch_id,))


def recover_interrupted_repo_branch_jobs() -> None:
    with connect() as conn:
        conn.execute(
            "UPDATE repo_branches SET "
            "index_status = CASE WHEN workspace IS NULL THEN 'failed' ELSE 'ready' END, "
            "job_stage = 'idle', "
            "last_error = 'Indexing was interrupted by a server restart', "
            "updated_at = datetime('now') WHERE index_status = 'indexing'"
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
            "WHERE a.user_id = ? AND "
            "(r.workspace = ? OR EXISTS ("
            "SELECT 1 FROM repo_branches b "
            "WHERE b.repo_id = r.id AND b.workspace = ?))",
            (user_id, workspace, workspace),
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
