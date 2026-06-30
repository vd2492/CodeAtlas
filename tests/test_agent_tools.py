import json
import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from app.agent.tools import RepositoryToolbox


class RepositoryToolboxTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name) / "repo"
        (self.root / "src").mkdir(parents=True)
        (self.root / "src" / "auth.py").write_text(
            "def login(user):\n"
            "    token = issue_token(user)\n"
            "    return token\n\n"
            "def issue_token(user):\n"
            "    return user + '-token'\n"
        )
        (self.root / ".env").write_text("API_KEY=secret\n")
        (Path(self.temp.name) / "outside.txt").write_text("outside\n")
        (self.root / "src" / "outside-link.py").symlink_to(
            Path(self.temp.name) / "outside.txt"
        )

        nodes = [
            {
                "id": "auth_login",
                "label": "login",
                "source_file": "src/auth.py",
                "source_location": "L1",
            },
            {
                "id": "auth_issue_token",
                "label": "issue_token",
                "source_file": "src/auth.py",
                "source_location": "L5",
            },
        ]
        links = [
            {
                "source": "auth_login",
                "target": "auth_issue_token",
                "relation": "calls",
                "context": "call",
                "source_file": "src/auth.py",
                "source_location": "L2",
            }
        ]
        config = SimpleNamespace(
            stopwords=[],
            synonyms={},
            keyword_boosts={},
        )
        env = patch.dict(os.environ, {"CODEATLAS_SOURCE_ROOT": str(self.root)})
        graph = patch("app.agent.tools.load_graph", return_value=(nodes, links))
        retrieval_config = patch(
            "app.agent.tools.load_retrieval_config", return_value=config
        )
        env.start()
        graph.start()
        retrieval_config.start()
        self.addCleanup(env.stop)
        self.addCleanup(graph.stop)
        self.addCleanup(retrieval_config.stop)
        self.box = RepositoryToolbox("default")

    def tearDown(self):
        self.temp.cleanup()

    def call(self, name, arguments):
        return json.loads(self.box.call(name, arguments))

    def test_search_combines_source_and_graph(self):
        result = self.call("search_code", {"query": "login token", "limit": 5})
        self.assertTrue(result["ok"])
        self.assertGreaterEqual(result["source_hit_count"], 1)
        self.assertGreaterEqual(result["graph_hit_count"], 1)
        self.assertEqual(result["source_hits"][0]["path"], "src/auth.py")
        self.assertIn("L1:", result["source_hits"][0]["snippets"][0]["code"])

    def test_read_file_returns_numbered_bounded_lines(self):
        result = self.call(
            "read_file",
            {"path": "src/auth.py", "start_line": 2, "end_line": 3},
        )
        self.assertTrue(result["ok"])
        self.assertEqual(result["start_line"], 2)
        self.assertEqual(result["end_line"], 3)
        self.assertIn("L2:     token", result["content"])

    def test_read_file_rejects_traversal_and_secrets(self):
        traversal = self.call("read_file", {"path": "../outside.txt"})
        secret = self.call("read_file", {"path": ".env"})
        self.assertFalse(traversal["ok"])
        self.assertIn("outside", traversal["error"])
        self.assertFalse(secret["ok"])
        self.assertIn("credential", secret["error"])
        self.assertEqual(self.box._source_matches_fallback(["outside"]), {})

    def test_graph_tools_follow_calls(self):
        definitions = self.call("find_definition", {"symbol": "login"})
        references = self.call("find_references", {"symbol": "login"})
        callers = self.call("get_callers", {"symbol": "issue token"})
        self.assertGreaterEqual(definitions["definition_count"], 1)
        self.assertGreaterEqual(references["reference_count"], 1)
        self.assertEqual(callers["caller_count"], 1)
        self.assertEqual(callers["callers"][0]["source_location"], "L2")


if __name__ == "__main__":
    unittest.main()
