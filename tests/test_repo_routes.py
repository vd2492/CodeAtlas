import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from fastapi import HTTPException, Response

from app import config, db
from app.conversations import ConversationStore
from app.repos import routes as repo_routes


class RepositoryCloneRetryTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.db_patch = patch.object(db, "DB_PATH", self.root / "codeatlas.db")
        self.workspace_patch = patch.object(
            config, "WORKSPACES_DIR", self.root / "workspaces"
        )
        self.db_patch.start()
        self.workspace_patch.start()
        db.init_db()
        self.admin = {"username": "admin"}

    def tearDown(self):
        self.workspace_patch.stop()
        self.db_patch.stop()
        self.temp.cleanup()

    def test_failed_clone_can_be_retried_with_the_same_slug(self):
        attempts = 0

        def clone(_source_url, _method, workspace):
            nonlocal attempts
            attempts += 1
            clone_path = config.repo_clone_dir(workspace)
            clone_path.mkdir(parents=True)
            if attempts == 1:
                (clone_path / "partial-clone").write_text("incomplete")
                raise RuntimeError("authentication failed")
            (clone_path / ".git").mkdir()

        request = repo_routes.AddRepoRequest(
            slug="sortbuddy",
            name="Sortbuddy",
            source_url="https://example.test/sortbuddy.git",
            clone_method="https",
        )
        with patch.object(repo_routes, "clone_repo", side_effect=clone), patch.object(
            repo_routes, "ensure_repo_branch"
        ):
            with self.assertRaises(HTTPException) as first_failure:
                repo_routes.add_repo(request, self.admin)

            self.assertEqual(first_failure.exception.status_code, 400)
            self.assertIn("Cloning failed", first_failure.exception.detail)
            self.assertEqual(db.get_repo_by_slug("sortbuddy")["status"], "new")
            self.assertFalse(config.repo_clone_dir("sortbuddy").exists())

            retried = repo_routes.add_repo(request, self.admin)

            self.assertTrue(retried["retried"])
            self.assertEqual(retried["repo"]["status"], "cloned")
            self.assertTrue(
                config.repo_clone_dir("sortbuddy").joinpath(".git").is_dir()
            )

            with self.assertRaises(HTTPException) as duplicate:
                repo_routes.add_repo(request, self.admin)

        self.assertEqual(duplicate.exception.status_code, 409)
        self.assertIn("already exists", duplicate.exception.detail)
        self.assertEqual(attempts, 2)

    def test_add_repo_rejects_embedded_credentials_before_clone(self):
        request = repo_routes.AddRepoRequest(
            slug="private",
            name="Private",
            source_url="https://user:secret@github.com/org/private.git",
            clone_method="https",
        )

        with patch.object(repo_routes, "clone_repo") as clone:
            with self.assertRaises(HTTPException) as raised:
                repo_routes.add_repo(request, self.admin)

        self.assertEqual(raised.exception.status_code, 400)
        self.assertIn("must not include embedded credentials", raised.exception.detail)
        clone.assert_not_called()
        self.assertIsNone(db.get_repo_by_slug("private"))

    def test_reclone_rejects_embedded_credentials_before_clone(self):
        db.create_repo(
            "private",
            "Private",
            "https://github.com/org/private.git",
            "https",
            "private",
            status="published",
        )
        request = repo_routes.RecloneRepoRequest(
            source_url="https://user:secret@github.com/org/private.git",
            clone_method="https",
        )

        with patch.object(repo_routes, "clone_repo") as clone:
            with self.assertRaises(HTTPException) as raised:
                repo_routes.reclone_repo("private", request, self.admin)

        self.assertEqual(raised.exception.status_code, 400)
        self.assertIn("must not include embedded credentials", raised.exception.detail)
        clone.assert_not_called()


class RepositoryGrantTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.db_patch = patch.object(db, "DB_PATH", self.root / "codeatlas.db")
        self.workspace_patch = patch.object(
            config, "WORKSPACES_DIR", self.root / "workspaces"
        )
        self.db_patch.start()
        self.workspace_patch.start()
        db.init_db()
        self.admin = {"username": "admin"}
        self.repo = db.create_repo(
            "roadmap",
            "Roadmap",
            "https://example.test/roadmap.git",
            "https",
            "roadmap",
            status="published",
        )

    def tearDown(self):
        self.workspace_patch.stop()
        self.db_patch.stop()
        self.temp.cleanup()

    def create_user(self, username, user_type="dev_team", role="user"):
        return db.create_user(
            username,
            "pbkdf2_sha256$stub",
            role=role,
            user_type=user_type,
        )

    def test_grant_product_users_adds_current_product_users_only(self):
        product = self.create_user("product", user_type="product_team")
        self.create_user("developer", user_type="dev_team")
        self.create_user("product-admin", user_type="product_team", role="admin")

        result = repo_routes.grant_product_users("roadmap", self.admin)

        self.assertEqual(result["matched_count"], 1)
        self.assertEqual(result["granted_count"], 1)
        self.assertEqual(result["already_granted_count"], 0)
        self.assertTrue(db.user_has_repo(product["id"], "roadmap"))
        members = db.list_repo_members(self.repo["id"])
        self.assertEqual([member["username"] for member in members], ["product"])

    def test_grant_product_users_is_repeatable_for_new_product_users(self):
        first = self.create_user("first-product", user_type="product_team")

        first_result = repo_routes.grant_product_users("roadmap", self.admin)
        second = self.create_user("second-product", user_type="product_team")
        second_result = repo_routes.grant_product_users("roadmap", self.admin)

        self.assertEqual(first_result["granted_count"], 1)
        self.assertEqual(second_result["matched_count"], 2)
        self.assertEqual(second_result["granted_count"], 1)
        self.assertEqual(second_result["already_granted_count"], 1)
        self.assertTrue(db.user_has_repo(first["id"], "roadmap"))
        self.assertTrue(db.user_has_repo(second["id"], "roadmap"))

    def test_clear_repo_answer_cache_clears_repo_and_branch_workspaces(self):
        store = ConversationStore(ttl_seconds=30, max_states=10)
        db.create_repo_branch(
            self.repo["id"],
            "release/1.0",
            workspace="roadmap--branch-1",
        )
        response = {
            "question": "How does login work?",
            "answer": "Login is verified in src/auth.py:L1-L20.",
            "provider_used": "shared:mimo-v2.5",
        }
        for workspace in ("roadmap", "roadmap--branch-1", "other-repo"):
            store.store_repo_cached_answer(
                workspace=workspace,
                user_type="dev_team",
                repository_revision="branch:abc123",
                question="How does login work?",
                response=response,
            )

        with patch.object(repo_routes, "conversation_store", store):
            result = repo_routes.clear_repo_answer_cache(
                "roadmap",
                Response(),
                admin=self.admin,
            )

        self.assertEqual(result["cleared"], 2)
        self.assertIsNone(store.get_repo_cached_answer(
            workspace="roadmap",
            user_type="dev_team",
            repository_revision="branch:abc123",
            question="How does login work?",
        ))
        self.assertIsNone(store.get_repo_cached_answer(
            workspace="roadmap--branch-1",
            user_type="dev_team",
            repository_revision="branch:abc123",
            question="How does login work?",
        ))
        self.assertIsNotNone(store.get_repo_cached_answer(
            workspace="other-repo",
            user_type="dev_team",
            repository_revision="branch:abc123",
            question="How does login work?",
        ))


if __name__ == "__main__":
    unittest.main()
