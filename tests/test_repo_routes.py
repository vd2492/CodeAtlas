import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from fastapi import HTTPException

from app import config, db
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


if __name__ == "__main__":
    unittest.main()
