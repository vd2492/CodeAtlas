import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from fastapi import HTTPException, Response

from app import config, db
from app.repos import routes as repo_routes
from app.retrieval.config_schema import (
    DEFAULT_PRE_SEARCH_INSTRUCTION,
    DEFAULT_STOPWORDS,
    RetrievalConfigValidationError,
    validate_retrieval_config,
)


class RetrievalConfigValidationTests(unittest.TestCase):
    def test_valid_partial_config_is_normalized_with_defaults(self):
        parsed = validate_retrieval_config(
            {
                "pre_search_instruction": "Map business terms before searching.",
                "synonyms": {"signin": ["login", "auth"]},
                "node_limit": 20,
                "allow_shared_fallback": False,
            }
        )

        self.assertEqual(parsed.stopwords, DEFAULT_STOPWORDS)
        self.assertEqual(
            parsed.pre_search_instruction,
            "Map business terms before searching.",
        )
        self.assertEqual(parsed.synonyms["signin"], ["login", "auth"])
        self.assertEqual(parsed.node_limit, 20)
        self.assertFalse(parsed.allow_shared_fallback)

    def test_unknown_fields_are_rejected(self):
        with self.assertRaisesRegex(
            RetrievalConfigValidationError, "Unknown config field.*node_limits"
        ):
            validate_retrieval_config({"node_limits": 20})

    def test_field_types_are_checked_strictly(self):
        invalid_configs = [
            ({"stopwords": "the,and"}, "array of strings"),
            ({"pre_search_instruction": ["map terms"]}, "must be a string"),
            (
                {"pre_search_instruction": "x" * 4001},
                "4000 characters or fewer",
            ),
            ({"synonyms": {"signin": "login"}}, "arrays of strings"),
            ({"keyword_boosts": {"auth": -1}}, "positive numeric"),
            ({"node_limit": True}, "positive integer"),
            ({"allow_shared_fallback": 1}, "true or false"),
        ]
        for value, message in invalid_configs:
            with self.subTest(value=value):
                with self.assertRaisesRegex(RetrievalConfigValidationError, message):
                    validate_retrieval_config(value)

    def test_default_config_includes_pre_search_instruction(self):
        parsed = validate_retrieval_config({})

        self.assertEqual(
            parsed.pre_search_instruction,
            DEFAULT_PRE_SEARCH_INSTRUCTION,
        )


class RetrievalConfigRouteTests(unittest.TestCase):
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
        db.create_repo(
            "sample",
            "Sample",
            "https://example.test/sample.git",
            "https",
            "sample",
            status="published",
        )
        self.admin = {"username": "admin"}

    def tearDown(self):
        self.workspace_patch.stop()
        self.db_patch.stop()
        self.temp.cleanup()

    def test_save_then_load_returns_the_saved_config(self):
        saved = repo_routes.update_retrieval_config(
            "sample",
            {"synonyms": {"delivery": ["forward"]}, "node_limit": 31},
            Response(),
            self.admin,
        )
        loaded = repo_routes.get_retrieval_config(
            "sample", Response(), self.admin
        )

        self.assertEqual(loaded, saved)
        self.assertEqual(loaded["source"], "saved")
        self.assertEqual(loaded["config"]["node_limit"], 31)
        self.assertEqual(
            loaded["config"]["synonyms"], {"delivery": ["forward"]}
        )

    def test_validation_route_does_not_save(self):
        validated = repo_routes.validate_retrieval_config_route(
            "sample", {"node_limit": 27}, Response(), self.admin
        )
        loaded = repo_routes.get_retrieval_config(
            "sample", Response(), self.admin
        )

        self.assertEqual(validated["config"]["node_limit"], 27)
        self.assertEqual(loaded["config"]["node_limit"], 16)

    def test_save_rejects_invalid_config(self):
        with self.assertRaises(HTTPException) as raised:
            repo_routes.update_retrieval_config(
                "sample", {"node_limit": "many"}, Response(), self.admin
            )

        self.assertEqual(raised.exception.status_code, 400)
        self.assertIn("positive integer", raised.exception.detail)

    def test_saved_config_survives_logout_and_login(self):
        user = db.create_user("console-admin", "password-hash", role="admin")
        first_session = db.create_session(user["id"])
        repo_routes.update_retrieval_config(
            "sample",
            {"synonyms": {"checkout": ["payment"]}, "node_limit": 29},
            Response(),
            self.admin,
        )

        db.delete_session(first_session)
        second_session = db.create_session(user["id"])
        self.assertIsNotNone(db.get_session_user(second_session))
        response = Response()
        loaded = repo_routes.get_retrieval_config(
            "sample", response, self.admin
        )

        self.assertEqual(response.headers["cache-control"], "no-store, max-age=0")
        self.assertEqual(loaded["source"], "saved")
        self.assertEqual(loaded["config"]["node_limit"], 29)
        self.assertEqual(
            loaded["config"]["synonyms"], {"checkout": ["payment"]}
        )
