"""SQLite storage for users, repositories, and per-user repo access.

Phase 1 provides the schema and helpers; the auth and repo routers wire up to
it in later phases. Per-repo retrieval tuning is stored as JSON on disk (see
app/retrieval/config_schema.py), not here.
"""

import sqlite3
from contextlib import contextmanager

from .config import DB_PATH

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
    allow_shared_fallback INTEGER NOT NULL DEFAULT 1,  -- 0 = never use shared Kimi tier
    created_at            TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS repo_access (
    user_id    INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    repo_id    INTEGER NOT NULL REFERENCES repos(id) ON DELETE CASCADE,
    granted_at TEXT NOT NULL DEFAULT (datetime('now')),
    PRIMARY KEY (user_id, repo_id)
);
"""


@contextmanager
def connect():
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()


def init_db() -> None:
    with connect() as conn:
        conn.executescript(SCHEMA)


def user_count() -> int:
    with connect() as conn:
        return conn.execute("SELECT COUNT(*) FROM users").fetchone()[0]
