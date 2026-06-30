import os
import sqlite3
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from fastapi import HTTPException

from app import db, main
from app.auth import routes as auth_routes
from app.llm import client
from app.repos.cloning import sanitize_clone_url


class SecurityGuardTests(unittest.TestCase):
    def test_eager_source_scan_skips_symlink_escapes_and_secrets(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            base = Path(temp_dir)
            source_root = base / "repo"
            source_root.mkdir()
            (source_root / "inside.py").write_text("inside = True\n")
            (source_root / "credentials.json").write_text('{"secret": true}\n')
            outside = base / "outside.py"
            outside.write_text("outside = True\n")
            try:
                os.symlink(outside, source_root / "escape.py")
            except (OSError, NotImplementedError):
                self.skipTest("symlinks are unavailable")

            source_root = source_root.resolve()
            main._source_file_cache.pop(str(source_root), None)
            paths = [
                relative for relative, _ in main._iter_source_files(source_root) or []
            ]
            main._source_file_cache.pop(str(source_root), None)

            self.assertEqual(paths, ["inside.py"])

    def test_llm_base_url_rejects_private_resolution(self):
        with patch(
            "app.llm.client.socket.getaddrinfo",
            return_value=[(None, None, None, None, ("127.0.0.1", 443))],
        ), patch.object(client, "LLM_ALLOW_LOCAL_BASE_URLS", False):
            with self.assertRaisesRegex(RuntimeError, "non-public"):
                client._validate_outbound_base_url("https://example.test/v1")

    def test_llm_base_url_allows_explicit_local_resolution(self):
        with patch(
            "app.llm.client.socket.getaddrinfo",
            return_value=[(None, None, None, None, ("127.0.0.1", 443))],
        ), patch.object(client, "LLM_ALLOW_LOCAL_BASE_URLS", True):
            client._validate_outbound_base_url("http://localhost:11434/v1")

    def test_existing_sessions_survive_expiry_column_migration(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            database = Path(temp_dir) / "codeatlas.db"
            connection = sqlite3.connect(database)
            connection.executescript(
                """
                CREATE TABLE users (
                    id INTEGER PRIMARY KEY,
                    username TEXT UNIQUE NOT NULL,
                    password_hash TEXT NOT NULL,
                    role TEXT NOT NULL,
                    llm_creds TEXT,
                    created_at TEXT NOT NULL DEFAULT (datetime('now'))
                );
                CREATE TABLE sessions (
                    token TEXT PRIMARY KEY,
                    user_id INTEGER NOT NULL,
                    created_at TEXT NOT NULL DEFAULT (datetime('now'))
                );
                INSERT INTO users (id, username, password_hash, role)
                VALUES (1, 'existing', 'hash', 'user');
                INSERT INTO sessions (token, user_id) VALUES ('old-token', 1);
                """
            )
            connection.close()

            with patch.object(db, "DB_PATH", database):
                db.init_db()
                self.assertEqual(db.get_session_user("old-token")["username"], "existing")

                token = db.create_session(1)
                with db.connect() as current:
                    expires_at = current.execute(
                        "SELECT expires_at FROM sessions WHERE token = ?", (token,)
                    ).fetchone()[0]
                    current.execute(
                        "UPDATE sessions SET expires_at = datetime('now', '-1 second') "
                        "WHERE token = ?",
                        (token,),
                    )
                self.assertIsNotNone(expires_at)
                self.assertIsNone(db.get_session_user(token))

                with db.connect() as current:
                    current.execute(
                        "INSERT INTO repos "
                        "(slug, name, source_url, clone_method, workspace, status, "
                        "allow_shared_fallback) VALUES (?, ?, ?, ?, ?, ?, ?)",
                        ("existing", "Existing", None, None, "existing", "published", 1),
                    )
                created = db.create_repo(
                    "new", "New", "https://example.test/repo.git", "https", "new"
                )
                self.assertEqual(created["allow_shared_fallback"], 0)
                self.assertEqual(
                    db.get_repo_by_slug("existing")["allow_shared_fallback"], 1
                )

            connection = sqlite3.connect(database)
            journal_mode = connection.execute("PRAGMA journal_mode").fetchone()[0]
            connection.close()
            self.assertEqual(journal_mode.lower(), "wal")

    def test_clone_url_sanitizer_removes_credentials(self):
        self.assertEqual(
            sanitize_clone_url("https://user:token@example.com/org/repo.git"),
            "https://example.com/org/repo.git",
        )
        self.assertEqual(
            sanitize_clone_url("git@github.com:org/repo.git"),
            "git@github.com:org/repo.git",
        )
        self.assertEqual(
            sanitize_clone_url(
                "https://example.com/org/repo.git?ref=main&access_token=secret"
            ),
            "https://example.com/org/repo.git?ref=main&access_token=%5Bredacted%5D",
        )

    def test_login_throttle_clears_after_success(self):
        auth_routes._login_failures.clear()
        with patch.object(auth_routes, "LOGIN_RATE_LIMIT", 2), patch.object(
            auth_routes, "LOGIN_RATE_WINDOW_SECONDS", 300
        ):
            auth_routes.record_login_failure("Admin")
            auth_routes.record_login_failure("admin")
            with self.assertRaises(HTTPException) as raised:
                auth_routes.enforce_login_rate_limit("ADMIN")
            self.assertEqual(raised.exception.status_code, 429)

            auth_routes.clear_login_failures("admin")
            auth_routes.enforce_login_rate_limit("admin")
        auth_routes._login_failures.clear()


if __name__ == "__main__":
    unittest.main()
