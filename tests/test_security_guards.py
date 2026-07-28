import os
import sqlite3
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

from fastapi import HTTPException, Request, Response

from app import db, main
from app.auth import routes as auth_routes
from app.auth.security import hash_password, verify_password
from app.llm import client
from app.repos.cloning import sanitize_clone_url


class SecurityGuardTests(unittest.TestCase):
    def test_health_check_verifies_database_connectivity(self):
        connection = MagicMock()
        context = MagicMock()
        context.__enter__.return_value = connection
        with patch.object(main.db, "connect", return_value=context):
            self.assertEqual(main.healthz(), {"status": "ok"})
        connection.execute.assert_called_once_with("SELECT 1")

    def test_health_check_reports_database_failure(self):
        with patch.object(main.db, "connect", side_effect=OSError("disk unavailable")):
            with self.assertRaises(HTTPException) as raised:
                main.healthz()
        self.assertEqual(raised.exception.status_code, 503)

    def test_mimo_key_must_match_endpoint_type(self):
        auth_routes.validate_llm_key_endpoint(
            "tp-token",
            "https://token-plan-sgp.xiaomimimo.com/v1",
        )
        auth_routes.validate_llm_key_endpoint(
            "sk-token",
            "https://api.xiaomimimo.com/v1",
        )

        with self.assertRaisesRegex(HTTPException, "Token Plan keys"):
            auth_routes.validate_llm_key_endpoint(
                "tp-token",
                "https://api.xiaomimimo.com/v1",
            )
        with self.assertRaisesRegex(HTTPException, "require a tp-"):
            auth_routes.validate_llm_key_endpoint(
                "sk-token",
                "https://token-plan-sgp.xiaomimimo.com/v1",
            )

    def test_visiting_home_deletes_the_active_session(self):
        request = Request(
            {
                "type": "http",
                "method": "GET",
                "path": "/",
                "headers": [(b"cookie", b"ca_session=session-token")],
                "query_string": b"",
                "scheme": "http",
                "server": ("testserver", 80),
                "client": ("testclient", 50000),
            }
        )

        with patch.object(main.db, "delete_session") as delete_session:
            response = main.root(request)

        delete_session.assert_called_once_with("session-token")
        self.assertIn("ca_session=", response.headers["set-cookie"])
        self.assertIn("Max-Age=0", response.headers["set-cookie"])
        self.assertEqual(response.headers["cache-control"], "no-store")

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

    def test_admin_credential_create_form_confirms_password_with_visibility_toggles(self):
        html = (Path(__file__).resolve().parents[1] / "app/static/admin.html").read_text()
        self.assertIn('id="uPass"', html)
        self.assertIn('id="uPassConfirm"', html)
        self.assertIn('id="uPassToggle"', html)
        self.assertIn('id="uPassConfirmToggle"', html)
        self.assertIn("toggleCreateUserPasswordVisibility", html)
        self.assertIn("Password and re-entered password do not match.", html)
        self.assertIn('id="editEmail"', html)
        self.assertIn("Associated Gmail ID", html)

    def test_admin_can_provision_google_login_access(self):
        with tempfile.TemporaryDirectory() as temp_dir, patch.object(
            db, "DB_PATH", Path(temp_dir) / "codeatlas.db"
        ), patch.object(auth_routes, "GOOGLE_ALLOWED_DOMAINS", set()):
            db.init_db()
            admin = db.create_user("admin", hash_password("admin-pass"), role="admin")
            admin = db.update_user_google_identity(
                admin["id"],
                email="admin@example.test",
            )
            repo = db.create_repo(
                "sample",
                "Sample",
                "https://example.test/sample.git",
                "https",
                "sample",
                status="cloned",
            )

            result = auth_routes.grant_google_access(
                auth_routes.GrantGoogleAccessRequest(
                    email="Reader@Gmail.com",
                    confirm_email="reader@gmail.com",
                    grant_slugs=["sample"],
                    role="admin",
                    user_type="product_team",
                ),
                admin,
            )

            self.assertEqual(result["user"]["username"], "reader@gmail.com")
            self.assertEqual(result["user"]["email"], "reader@gmail.com")
            self.assertEqual(result["user"]["role"], "admin")
            self.assertEqual(result["user"]["user_type"], "product_team")
            self.assertEqual(result["user"]["auth_status"], "google_pending")
            user = db.get_user_by_email("reader@gmail.com")
            self.assertEqual(user["role"], "admin")
            self.assertEqual(user["user_type"], "product_team")
            self.assertFalse(verify_password("anything", user["password_hash"]))
            self.assertEqual(
                [granted["slug"] for granted in db.list_repos_for_user(user["id"])],
                [repo["slug"]],
            )

            updated = auth_routes.grant_google_access(
                auth_routes.GrantGoogleAccessRequest(
                    email="reader@gmail.com",
                    confirm_email="reader@gmail.com",
                    grant_slugs=["sample"],
                    role="user",
                    user_type="dev_team",
                ),
                admin,
            )
            self.assertEqual(updated["user"]["id"], user["id"])
            self.assertEqual(updated["user"]["role"], "user")
            self.assertEqual(updated["user"]["user_type"], "dev_team")

            with self.assertRaises(HTTPException) as raised:
                auth_routes.grant_google_access(
                    auth_routes.GrantGoogleAccessRequest(
                        email="admin@example.test",
                        confirm_email="admin@example.test",
                        grant_slugs=["sample"],
                        role="user",
                        user_type="dev_team",
                    ),
                    admin,
                )
            self.assertEqual(raised.exception.status_code, 400)

    def test_google_login_links_pending_user_and_preserves_grants(self):
        with tempfile.TemporaryDirectory() as temp_dir, patch.object(
            db, "DB_PATH", Path(temp_dir) / "codeatlas.db"
        ), patch.object(auth_routes, "AUTH_MODE", "mixed"), patch.object(
            auth_routes, "GOOGLE_ALLOWED_DOMAINS", set()
        ), patch.object(
            auth_routes,
            "verify_google_credential",
            return_value={
                "sub": "google-sub-1",
                "email": "reader@gmail.com",
                "email_verified": True,
                "name": "Reader Person",
            },
        ):
            db.init_db()
            user = db.create_google_user("reader@gmail.com")
            repo = db.create_repo(
                "sample",
                "Sample",
                "https://example.test/sample.git",
                "https",
                "sample",
                status="published",
            )
            db.grant_access(user["id"], repo["id"])

            response = Response()
            result = auth_routes.google_login(
                auth_routes.GoogleCredentialRequest(credential="token"),
                response,
            )

            self.assertEqual(result["user"]["id"], user["id"])
            self.assertEqual(result["user"]["auth_status"], "google_linked")
            self.assertIn("ca_session=", response.headers["set-cookie"])
            linked = db.get_user_by_google_sub("google-sub-1")
            self.assertEqual(linked["id"], user["id"])
            self.assertEqual(linked["display_name"], "Reader Person")
            self.assertEqual(
                [granted["slug"] for granted in db.list_repos_for_user(user["id"])],
                ["sample"],
            )

    def test_google_login_rejects_unprovisioned_user_by_default(self):
        with tempfile.TemporaryDirectory() as temp_dir, patch.object(
            db, "DB_PATH", Path(temp_dir) / "codeatlas.db"
        ), patch.object(auth_routes, "AUTH_MODE", "mixed"), patch.object(
            auth_routes, "GOOGLE_AUTO_CREATE", False
        ), patch.object(
            auth_routes, "GOOGLE_ALLOWED_DOMAINS", set()
        ), patch.object(
            auth_routes,
            "verify_google_credential",
            return_value={
                "sub": "google-sub-2",
                "email": "unknown@gmail.com",
                "email_verified": True,
            },
        ):
            db.init_db()

            with self.assertRaises(HTTPException) as raised:
                auth_routes.google_login(
                    auth_routes.GoogleCredentialRequest(credential="token"),
                    Response(),
                )

            self.assertEqual(raised.exception.status_code, 403)
            self.assertIsNone(db.get_user_by_email("unknown@gmail.com"))

    def test_google_only_bootstraps_allowed_admin(self):
        with tempfile.TemporaryDirectory() as temp_dir, patch.object(
            db, "DB_PATH", Path(temp_dir) / "codeatlas.db"
        ), patch.object(auth_routes, "AUTH_MODE", "google"), patch.object(
            auth_routes, "GOOGLE_BOOTSTRAP_ADMIN_EMAILS", {"admin@gmail.com"}
        ), patch.object(
            auth_routes, "GOOGLE_ALLOWED_DOMAINS", set()
        ), patch.object(
            auth_routes,
            "verify_google_credential",
            return_value={
                "sub": "bootstrap-google-sub",
                "email": "admin@gmail.com",
                "email_verified": True,
                "name": "Admin Person",
            },
        ):
            db.init_db()

            response = Response()
            result = auth_routes.google_login(
                auth_routes.GoogleCredentialRequest(credential="token"),
                response,
            )

            self.assertEqual(result["user"]["role"], "admin")
            self.assertEqual(result["user"]["email"], "admin@gmail.com")
            self.assertEqual(result["user"]["auth_status"], "google_linked")
            self.assertIn("ca_session=", response.headers["set-cookie"])
            self.assertEqual(db.admin_count(), 1)
            linked = db.get_user_by_google_sub("bootstrap-google-sub")
            self.assertEqual(linked["display_name"], "Admin Person")

    def test_google_only_blocks_credentials_but_allows_gmail_mapping(self):
        with tempfile.TemporaryDirectory() as temp_dir, patch.object(
            db, "DB_PATH", Path(temp_dir) / "codeatlas.db"
        ), patch.object(auth_routes, "AUTH_MODE", "google"), patch.object(
            auth_routes, "GOOGLE_ALLOWED_DOMAINS", set()
        ):
            db.init_db()
            admin = db.create_user("admin", hash_password("admin-pass"), role="admin")
            target = db.create_user("reader", hash_password("old-pass"), role="user")

            with self.assertRaises(HTTPException) as raised:
                auth_routes.create_user(
                    auth_routes.CreateUserRequest(
                        username="new-reader",
                        password="reader-pass",
                    ),
                    admin,
                )
            self.assertEqual(raised.exception.status_code, 403)

            with self.assertRaises(HTTPException) as raised:
                auth_routes.update_user_credentials(
                    target["id"],
                    auth_routes.UpdateUserRequest(password="new-pass"),
                    admin,
                )
            self.assertEqual(raised.exception.status_code, 403)

            result = auth_routes.update_user_credentials(
                target["id"],
                auth_routes.UpdateUserRequest(
                    email="Reader@Gmail.com",
                    user_type="product_team",
                ),
                admin,
            )

            self.assertEqual(result["user"]["username"], "reader")
            self.assertEqual(result["user"]["email"], "reader@gmail.com")
            self.assertEqual(result["user"]["user_type"], "product_team")
            updated = db.get_user_by_id(target["id"])
            self.assertTrue(verify_password("old-pass", updated["password_hash"]))

    def test_admin_can_update_username_and_password(self):
        with tempfile.TemporaryDirectory() as temp_dir, patch.object(
            db, "DB_PATH", Path(temp_dir) / "codeatlas.db"
        ), patch.object(auth_routes, "GOOGLE_ALLOWED_DOMAINS", set()):
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
                    email="Reader@Gmail.com",
                ),
                admin,
            )

            self.assertEqual(result["user"]["username"], "renamed-reader")
            self.assertEqual(result["user"]["email"], "reader@gmail.com")
            self.assertEqual(result["user"]["role"], "user")
            self.assertEqual(result["user"]["user_type"], "dev_team")
            self.assertIsNone(db.get_user_by_username("reader"))
            updated = db.get_user_by_username("renamed-reader")
            self.assertEqual(updated["email"], "reader@gmail.com")
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

            with patch.object(auth_routes, "AUTH_MODE", "mixed"), patch.object(
                auth_routes,
                "verify_google_credential",
                return_value={
                    "sub": "credential-user-google-sub",
                    "email": "reader@gmail.com",
                    "email_verified": True,
                },
            ):
                google_result = auth_routes.google_login(
                    auth_routes.GoogleCredentialRequest(credential="token"),
                    Response(),
                )
            self.assertEqual(google_result["user"]["id"], target["id"])
            self.assertEqual(
                db.get_user_by_google_sub("credential-user-google-sub")["id"],
                target["id"],
            )

            relinked = auth_routes.update_user_credentials(
                target["id"],
                auth_routes.UpdateUserRequest(email="reader2@gmail.com"),
                admin,
            )
            self.assertEqual(relinked["user"]["email"], "reader2@gmail.com")
            self.assertIsNone(db.get_user_by_google_sub("credential-user-google-sub"))

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

    def test_normal_dev_user_me_exposes_answer_style_type(self):
        user = {
            "id": 7,
            "username": "dev-reader",
            "role": "user",
            "user_type": "dev_team",
        }

        with patch.object(auth_routes.db, "list_repos_for_user", return_value=[]):
            result = auth_routes.me(user)

        self.assertEqual(result["user"]["role"], "user")
        self.assertEqual(result["user"]["user_type"], "dev_team")

    def test_product_team_type_reaches_llm_answer_pipeline(self):
        request = main.AskRequest(
            question="What happens during checkout?",
            deep_investigation=True,
        )
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

    def test_dev_team_can_request_product_style_answer(self):
        request = main.AskRequest(
            question="What happens during checkout?",
            answer_user_type="product_team",
            deep_investigation=True,
        )
        user = {"id": 7, "user_type": "dev_team"}
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

        self.assertEqual(result["answer_user_type"], "product_team")
        self.assertEqual(answer.call_args.kwargs["user_type"], "product_team")

    def test_image_attachment_is_normalized_for_web_question(self):
        image_data = "iVBORw0KGgo="
        request = main.AskRequest(
            question="What is visible in this screenshot?",
            image_attachments=[{
                "name": "screen.png",
                "mime_type": "image/png",
                "data": image_data,
            }],
        )
        user = {"id": 7, "user_type": "dev_team", "_session_key": "session"}
        response = {
            "answer": "The screenshot shows a login screen.",
            "provider_used": "shared:model",
            "retrieval_mode": "agentic",
            "context": {"llm_context_preview": {"question": request.question}},
        }
        with patch.object(main, "enforce_rate_limit"), patch.object(
            main, "enforce_strict_branch_freshness"
        ), patch.object(
            main.db,
            "get_repo_by_workspace",
            return_value={"allow_shared_fallback": 1},
        ), patch.object(
            main, "load_user_llm", return_value=None
        ), patch.object(
            main, "repository_revision", return_value="rev"
        ), patch.object(
            main.conversation_store, "get_cached_answer"
        ) as session_cache, patch.object(
            main.conversation_store, "get_repo_cached_answer"
        ) as repo_cache, patch.object(
            main, "_remember_session_answer"
        ) as remember_session, patch.object(
            main, "answer_question", return_value=response
        ) as answer:
            result = main.ask_llm_endpoint(request, "sample", user)

        self.assertEqual(result["answer"], response["answer"])
        image_attachments = answer.call_args.kwargs["image_attachments"]
        self.assertEqual(image_attachments[0]["name"], "screen.png")
        self.assertEqual(image_attachments[0]["mime_type"], "image/png")
        self.assertEqual(image_attachments[0]["data"], image_data)
        session_cache.assert_not_called()
        repo_cache.assert_not_called()
        remember_session.assert_not_called()

    def test_invalid_image_attachment_type_is_rejected(self):
        with self.assertRaises(HTTPException) as raised:
            main.normalize_image_attachments([{
                "name": "notes.txt",
                "mime_type": "text/plain",
                "data": "aGVsbG8=",
            }])

        self.assertEqual(raised.exception.status_code, 400)
        self.assertIn("PNG", raised.exception.detail)

    def test_product_team_cannot_request_dev_style_answer(self):
        request = main.AskRequest(
            question="What happens during checkout?",
            answer_user_type="dev_team",
            deep_investigation=True,
        )
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

        self.assertEqual(result["answer_user_type"], "product_team")
        self.assertEqual(answer.call_args.kwargs["user_type"], "product_team")

    def test_invalid_answer_user_type_is_rejected(self):
        request = main.AskRequest(
            question="What happens during checkout?",
            answer_user_type="sales_team",
        )
        user = {"id": 7, "user_type": "dev_team"}
        with patch.object(main, "enforce_rate_limit"), patch.object(
            main, "enforce_strict_branch_freshness"
        ), patch.object(
            main.db,
            "get_repo_by_workspace",
            return_value={"allow_shared_fallback": 0},
        ), patch.object(
            main, "load_user_llm", return_value=None
        ), self.assertRaises(HTTPException) as raised:
            main.ask_llm_endpoint(request, "sample", user)

        self.assertEqual(raised.exception.status_code, 400)
        self.assertIn("answer_user_type", raised.exception.detail)


if __name__ == "__main__":
    unittest.main()
