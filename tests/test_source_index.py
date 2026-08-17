import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from app import config, main
from app.retrieval.source_index import build_source_index, search_source_index
from app.repos import indexing


class SourceIndexTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)

    def tearDown(self):
        self.temp.cleanup()

    def test_build_source_index_excludes_sensitive_files_and_searches_persisted_lines(self):
        source_root = self.root / "repo"
        source_root.mkdir()
        (source_root / "auth.py").write_text(
            "def login_user():\n"
            "    return create_session()\n"
        )
        (source_root / ".env").write_text("API_KEY=secret\n")
        index_path = self.root / "source_index.json"

        payload = build_source_index(source_root, index_path)
        hits = search_source_index(index_path, ["login"], limit=5)

        self.assertEqual(payload["file_count"], 1)
        self.assertEqual(hits[0]["path"], "auth.py")
        self.assertIn("login_user", hits[0]["snippets"][0]["code"])
        self.assertNotIn(".env", index_path.read_text())
        self.assertNotIn("API_KEY", index_path.read_text())

    def test_query_search_uses_persisted_index_when_source_file_is_unavailable(self):
        source_root = self.root / "repo"
        source_root.mkdir()
        source_file = source_root / "auth.py"
        source_file.write_text("def login_user():\n    return True\n")
        index_path = self.root / "source_index.json"
        build_source_index(source_root, index_path)
        source_file.unlink()

        hits = main._search_source_files(
            source_root,
            ["login"],
            limit=5,
            index_path=index_path,
        )

        self.assertEqual(hits[0]["path"], "auth.py")

    def test_index_repo_builds_persisted_source_index_after_graphify(self):
        workspace_patch = patch.object(
            config,
            "WORKSPACES_DIR",
            self.root / "workspaces",
        )
        workspace_patch.start()
        self.addCleanup(workspace_patch.stop)
        workspace = "sample"
        repo = config.repo_clone_dir(workspace)
        repo.mkdir(parents=True)
        (repo / "app.py").write_text("def checkout():\n    return 'ok'\n")
        produced_graph = repo / "graphify-out" / "graph.json"
        produced_graph.parent.mkdir()
        produced_graph.write_text(json.dumps({"nodes": [], "links": []}))

        with patch.object(
            indexing.subprocess,
            "run",
            return_value=SimpleNamespace(returncode=0, stderr="", stdout=""),
        ):
            target = indexing.index_repo(workspace)

        self.assertTrue(target.is_file())
        source_index = config.source_index_path(workspace)
        self.assertTrue(source_index.is_file())
        self.assertEqual(
            search_source_index(source_index, ["checkout"], limit=5)[0]["path"],
            "app.py",
        )


if __name__ == "__main__":
    unittest.main()
