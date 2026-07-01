import shutil
import subprocess
import tempfile
import threading
import time
import unittest
from pathlib import Path
from unittest.mock import patch

from fastapi import HTTPException

from app import config, db, main
from app.repos import branches
from app.repos import branch_routes
from app.repos import routes as repo_routes


def run_git(cwd: Path, *args: str) -> str:
    result = subprocess.run(
        ["git", *args],
        cwd=str(cwd),
        capture_output=True,
        text=True,
        check=True,
    )
    return result.stdout.strip()


class BranchIndexingTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.database = self.root / "codeatlas.db"
        self.workspaces = self.root / "workspaces"
        self.db_patch = patch.object(db, "DB_PATH", self.database)
        self.workspace_patch = patch.object(
            config, "WORKSPACES_DIR", self.workspaces
        )
        self.db_patch.start()
        self.workspace_patch.start()
        db.init_db()

        self.remote = self.root / "remote.git"
        self.seed = self.root / "seed"
        run_git(self.root, "init", "--bare", str(self.remote))
        run_git(self.root, "init", str(self.seed))
        run_git(self.seed, "config", "user.email", "test@example.com")
        run_git(self.seed, "config", "user.name", "CodeAtlas Test")
        run_git(self.seed, "checkout", "-b", "main")
        (self.seed / "app.py").write_text("VALUE = 'main-v1'\n")
        run_git(self.seed, "add", "app.py")
        run_git(self.seed, "commit", "-m", "main v1")
        run_git(self.seed, "remote", "add", "origin", str(self.remote))
        run_git(self.seed, "push", "-u", "origin", "main")
        subprocess.run(
            ["git", "--git-dir", str(self.remote), "symbolic-ref", "HEAD", "refs/heads/main"],
            check=True,
            capture_output=True,
            text=True,
        )

        clone = config.repo_clone_dir("sample")
        clone.parent.mkdir(parents=True, exist_ok=True)
        run_git(self.root, "clone", str(self.remote), str(clone))
        graph = config.graph_path("sample")
        graph.parent.mkdir(parents=True, exist_ok=True)
        graph.write_text('{"nodes": [], "links": []}')

        self.repo = db.create_repo(
            "sample",
            "Sample",
            str(self.remote),
            "https",
            "sample",
            status="published",
        )
        self.legacy = branches.ensure_repo_branch(self.repo)

    def tearDown(self):
        self.workspace_patch.stop()
        self.db_patch.stop()
        self.temp.cleanup()

    def add_remote_branch_commit(self, branch: str, value: str) -> str:
        current = run_git(self.seed, "branch", "--show-current")
        if branch not in run_git(self.seed, "branch", "--format=%(refname:short)").splitlines():
            run_git(self.seed, "checkout", "-b", branch)
        else:
            run_git(self.seed, "checkout", branch)
        (self.seed / "app.py").write_text(f"VALUE = '{value}'\n")
        run_git(self.seed, "add", "app.py")
        run_git(self.seed, "commit", "-m", value)
        run_git(self.seed, "push", "-u", "origin", branch)
        commit = run_git(self.seed, "rev-parse", "HEAD")
        run_git(self.seed, "checkout", current)
        return commit

    @staticmethod
    def fake_index(workspace: str):
        graph = config.graph_path(workspace)
        graph.parent.mkdir(parents=True, exist_ok=True)
        graph.write_text('{"nodes": [], "links": []}')
        return graph

    def test_legacy_repo_is_registered_without_reindexing(self):
        branch = db.get_repo_branch(self.legacy["id"])
        self.assertEqual(branch["name"], "main")
        self.assertEqual(branch["workspace"], "sample")
        self.assertEqual(branch["index_status"], "ready")
        self.assertTrue(config.graph_path("sample").exists())

    def test_reclone_restores_missing_clone_without_replacing_graph(self):
        clone = config.repo_clone_dir("sample")
        graph = config.graph_path("sample")
        graph_before = graph.read_text()
        with db.connect() as connection:
            connection.execute(
                "UPDATE repo_branches SET name = 'default', "
                "indexed_commit_sha = NULL, remote_commit_sha = NULL, "
                "allow_user_sync = 0, freshness_status = 'remote_unavailable', "
                "last_error = 'Repository clone is not available' WHERE id = ?",
                (self.legacy["id"],),
            )
        shutil.rmtree(clone)

        result = repo_routes.reclone_repo(
            self.repo["slug"],
            repo_routes.RecloneRepoRequest(),
            {"username": "admin"},
        )

        self.assertTrue(result["repo"]["clone_available"])
        self.assertTrue((clone / ".git").exists())
        self.assertEqual(graph.read_text(), graph_before)
        restored = db.get_repo_branch(self.legacy["id"])
        self.assertEqual(restored["name"], "main")
        self.assertEqual(restored["freshness_status"], "unknown")
        self.assertIsNone(restored["last_error"])
        self.assertTrue(restored["allow_user_sync"])
        self.assertIn(
            "main",
            {item["name"] for item in branches.discover_remote_branches(self.repo)},
        )

        with self.assertRaises(HTTPException) as raised:
            repo_routes.reclone_repo(
                self.repo["slug"],
                repo_routes.RecloneRepoRequest(),
                {"username": "admin"},
            )
        self.assertEqual(raised.exception.status_code, 409)

    def test_remote_branch_discovery_and_atomic_version_activation(self):
        first_commit = self.add_remote_branch_commit("develop", "develop-v1")
        discovered = branches.discover_remote_branches(self.repo)
        self.assertIn("develop", {item["name"] for item in discovered})

        approved = branches.approve_repo_branch(self.repo, "develop")
        with patch("app.repos.branches.index_repo", side_effect=self.fake_index):
            synced = branches.sync_and_index_branch(approved["id"])

        self.assertEqual(synced["indexed_commit_sha"], first_commit)
        self.assertEqual(synced["freshness_status"], "up_to_date")
        self.assertNotEqual(synced["workspace"], self.repo["workspace"])
        self.assertEqual(
            config.repo_clone_dir(synced["workspace"]).joinpath("app.py").read_text(),
            "VALUE = 'develop-v1'\n",
        )
        self.assertTrue(config.graph_path(synced["workspace"]).is_file())
        self.assertTrue(db.user_has_repo(
            self._grant_user(), synced["workspace"]
        ))

        resolved = main.authorized_workspace(
            self.repo["workspace"],
            approved["id"],
            {"id": self.user_id, "role": "user"},
        )
        self.assertEqual(resolved, synced["workspace"])

    def _grant_user(self) -> int:
        user = db.create_user("reader", "hash", role="user")
        db.grant_access(user["id"], self.repo["id"])
        self.user_id = user["id"]
        return user["id"]

    def test_failed_refresh_keeps_previous_active_version(self):
        self.add_remote_branch_commit("develop", "develop-v1")
        approved = branches.approve_repo_branch(self.repo, "develop")
        with patch("app.repos.branches.index_repo", side_effect=self.fake_index):
            first = branches.sync_and_index_branch(approved["id"])
        previous_workspace = first["workspace"]

        self.add_remote_branch_commit("develop", "develop-v2")
        with patch(
            "app.repos.branches.index_repo",
            side_effect=RuntimeError("graphify failed"),
        ):
            failed = branches.sync_and_index_branch(approved["id"])

        self.assertEqual(failed["workspace"], previous_workspace)
        self.assertEqual(failed["index_status"], "ready")
        self.assertIn("graphify failed", failed["last_error"])
        self.assertTrue(config.graph_path(previous_workspace).is_file())

    def test_up_to_date_sync_skips_reindexing(self):
        self.add_remote_branch_commit("develop", "develop-v1")
        approved = branches.approve_repo_branch(self.repo, "develop")
        with patch("app.repos.branches.index_repo", side_effect=self.fake_index):
            branches.sync_and_index_branch(approved["id"])

        with patch("app.repos.branches.index_repo") as index:
            current = branches.sync_and_index_branch(approved["id"])
        index.assert_not_called()
        self.assertEqual(current["freshness_status"], "up_to_date")

    def test_legacy_workspace_follows_new_active_default_branch_version(self):
        self.add_remote_branch_commit("main", "main-v2")
        with patch("app.repos.branches.index_repo", side_effect=self.fake_index):
            refreshed = branches.sync_and_index_branch(self.legacy["id"])
        user_id = self._grant_user()
        resolved = main.authorized_workspace(
            self.repo["workspace"],
            None,
            {"id": user_id, "role": "user"},
        )
        self.assertEqual(resolved, refreshed["workspace"])
        self.assertNotEqual(resolved, self.repo["workspace"])

    def test_answer_includes_exact_repository_version(self):
        self.add_remote_branch_commit("develop", "develop-v1")
        approved = branches.approve_repo_branch(self.repo, "develop")
        with patch("app.repos.branches.index_repo", side_effect=self.fake_index):
            synced = branches.sync_and_index_branch(approved["id"])
        with patch(
            "app.main.generate",
            return_value={
                "answer": "verified",
                "provider_used": "test",
                "retrieval_mode": "agentic",
            },
        ):
            answer = main.answer_question("What branch?", workspace=synced["workspace"])
        self.assertEqual(answer["repository_version"]["branch"], "develop")
        self.assertEqual(
            answer["repository_version"]["commit_sha"],
            synced["indexed_commit_sha"],
        )

    def test_force_pushed_branch_is_reported_as_diverged(self):
        self.add_remote_branch_commit("develop", "develop-v1")
        approved = branches.approve_repo_branch(self.repo, "develop")
        with patch("app.repos.branches.index_repo", side_effect=self.fake_index):
            branches.sync_and_index_branch(approved["id"])

        run_git(self.seed, "checkout", "--orphan", "rewritten-develop")
        run_git(self.seed, "rm", "-rf", ".")
        (self.seed / "app.py").write_text("VALUE = 'rewritten'\n")
        run_git(self.seed, "add", "app.py")
        run_git(self.seed, "commit", "-m", "rewritten")
        run_git(self.seed, "push", "--force", "origin", "HEAD:develop")
        run_git(self.seed, "checkout", "main")

        checked = branches.check_branch_freshness(approved["id"])
        self.assertEqual(checked["freshness_status"], "diverged")

    def test_deleted_remote_branch_keeps_last_index_available(self):
        self.add_remote_branch_commit("develop", "develop-v1")
        approved = branches.approve_repo_branch(self.repo, "develop")
        with patch("app.repos.branches.index_repo", side_effect=self.fake_index):
            indexed = branches.sync_and_index_branch(approved["id"])
        run_git(self.seed, "push", "origin", "--delete", "develop")

        checked = branches.check_branch_freshness(approved["id"])
        self.assertEqual(checked["freshness_status"], "remote_unavailable")
        self.assertEqual(checked["workspace"], indexed["workspace"])
        self.assertEqual(checked["index_status"], "ready")

    def test_user_sync_permission_is_enforced(self):
        user_id = self._grant_user()
        db.update_repo_branch_settings(
            self.legacy["id"],
            allow_user_sync=False,
            auto_sync=False,
            strict_freshness=False,
            freshness_interval_seconds=300,
        )
        with self.assertRaises(HTTPException) as raised:
            branch_routes.user_sync_branch(
                self.legacy["id"],
                self.repo["workspace"],
                {"id": user_id, "username": "reader", "role": "user"},
            )
        self.assertEqual(raised.exception.status_code, 403)

    def test_strict_freshness_blocks_only_questions(self):
        branch = db.get_repo_branch(self.legacy["id"])
        db.update_repo_branch_settings(
            branch["id"],
            allow_user_sync=True,
            auto_sync=False,
            strict_freshness=True,
            freshness_interval_seconds=300,
        )
        db.update_repo_branch_state(
            branch["id"],
            freshness_status="behind",
            behind_count=1,
        )
        with self.assertRaises(HTTPException) as raised:
            main.enforce_strict_branch_freshness(branch["workspace"])
        self.assertEqual(raised.exception.status_code, 409)

    def test_authorized_user_lists_only_approved_repo_branches(self):
        user_id = self._grant_user()
        payload = branch_routes.user_list_branches(
            self.repo["workspace"],
            {"id": user_id, "username": "reader", "role": "user"},
        )
        self.assertEqual([item["name"] for item in payload["branches"]], ["main"])
        self.assertTrue(payload["branches"][0]["can_sync"])

    def test_concurrent_branch_jobs_are_deduplicated(self):
        started = threading.Event()
        release = threading.Event()

        def blocking_sync(branch_id):
            started.set()
            release.wait(2)
            return db.get_repo_branch(branch_id)

        with patch("app.repos.branches.sync_and_index_branch", side_effect=blocking_sync):
            self.assertTrue(branches.submit_branch_job(self.legacy["id"], sync=True))
            self.assertTrue(started.wait(1))
            self.assertFalse(branches.submit_branch_job(self.legacy["id"], sync=True))
            release.set()
            for _ in range(50):
                if not branches.branch_job_running(self.legacy["id"]):
                    break
                time.sleep(0.01)
        self.assertFalse(branches.branch_job_running(self.legacy["id"]))

    def test_invalid_branch_name_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "Invalid"):
            branches.approve_repo_branch(self.repo, "../escape")


if __name__ == "__main__":
    unittest.main()
