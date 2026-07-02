import os
import sqlite3
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from fastapi import HTTPException, Response

from app import db, main
from app.auth import routes as auth_routes
from app.auth.security import hash_password, verify_password
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
                self.assertEqual(
                    db.get_session_user("old-token")["user_type"], "dev_team"
                )

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

    def test_admin_can_update_username_and_password(self):
        with tempfile.TemporaryDirectory() as temp_dir, patch.object(
            db, "DB_PATH", Path(temp_dir) / "codeatlas.db"
        ):
            db.init_db()
            admin = db.create_user("admin", hash_password("admin-pass"), role="admin")
            target = db.create_user("reader", hash_password("old-pass"), role="user")
            repo = db.create_repo(
                "sample",
                "Sample",
                "https://example.test/sample.git",
                "https",
                "sample",
                status="published",
            )
            db.grant_access(target["id"], repo["id"])

            result = auth_routes.update_user_credentials(
                target["id"],
                auth_routes.UpdateUserRequest(
                    username="renamed-reader",
                    password="new-pass",
                ),
                admin,
            )

            self.assertEqual(result["user"]["username"], "renamed-reader")
            self.assertEqual(result["user"]["role"], "user")
            self.assertEqual(result["user"]["user_type"], "dev_team")
            self.assertIsNone(db.get_user_by_username("reader"))
            updated = db.get_user_by_username("renamed-reader")
            self.assertTrue(verify_password("new-pass", updated["password_hash"]))
            self.assertFalse(verify_password("old-pass", updated["password_hash"]))
            self.assertEqual(
                [repo["slug"] for repo in db.list_repos_for_user(target["id"])],
                ["sample"],
            )

            logged_in = auth_routes.login(
                auth_routes.Credentials(
                    username="renamed-reader",
                    password="new-pass",
                ),
                Response(),
            )
            self.assertEqual(logged_in["user"]["id"], target["id"])
            with self.assertRaises(HTTPException) as raised:
                auth_routes.login(
                    auth_routes.Credentials(
                        username="reader",
                        password="old-pass",
                    ),
                    Response(),
                )
            self.assertEqual(raised.exception.status_code, 401)
            auth_routes._login_failures.clear()

    def test_admin_can_create_and_edit_product_team_user(self):
        with tempfile.TemporaryDirectory() as temp_dir, patch.object(
            db, "DB_PATH", Path(temp_dir) / "codeatlas.db"
        ):
            db.init_db()
            admin = db.create_user("admin", hash_password("admin-pass"), role="admin")

            created = auth_routes.create_user(
                auth_routes.CreateUserRequest(
                    username="product-reader",
                    password="reader-pass",
                    user_type="product_team",
                ),
                admin,
            )
            self.assertEqual(created["user"]["user_type"], "product_team")

            updated = auth_routes.update_user_credentials(
                created["user"]["id"],
                auth_routes.UpdateUserRequest(user_type="dev_team"),
                admin,
            )
            self.assertEqual(updated["user"]["username"], "product-reader")
            self.assertEqual(updated["user"]["user_type"], "dev_team")

    def test_product_team_type_reaches_llm_answer_pipeline(self):
        request = main.AskRequest(question="What happens during checkout?")
        user = {"id": 7, "user_type": "product_team"}
        with patch.object(main, "enforce_rate_limit"), patch.object(
            main, "enforce_strict_branch_freshness"
        ), patch.object(
            main.db,
            "get_repo_by_workspace",
            return_value={"allow_shared_fallback": 0},
        ), patch.object(
            main, "load_user_llm", return_value=None
        ), patch.object(
            main, "answer_question", return_value={"answer": "A simple answer."}
        ) as answer:
            result = main.ask_llm_endpoint(request, "sample", user)

        self.assertEqual(result["answer"], "A simple answer.")
        self.assertEqual(answer.call_args.kwargs["user_type"], "product_team")


if __name__ == "__main__":
    unittest.main()
