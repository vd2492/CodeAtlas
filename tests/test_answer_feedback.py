import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from fastapi import HTTPException

from app import ask_service, db
from app.auth import routes as auth_routes


class AnonymousAnswerFeedbackTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.db_patch = patch.object(
            db,
            "DB_PATH",
            Path(self.temp.name) / "codeatlas.db",
        )
        self.db_patch.start()
        db.init_db()
        self.admin = db.create_user("admin", "unused", role="admin")
        self.user = db.create_user(
            "alice",
            "unused",
            email="alice@example.com",
        )
        self.outsider = db.create_user(
            "bob",
            "unused",
            email="bob@example.com",
        )
        self.repo = db.create_repo(
            "atlas",
            "CodeAtlas",
            "https://example.test/codeatlas.git",
            "https",
            "atlas-main",
            "published",
        )
        db.grant_access(self.user["id"], self.repo["id"])

    def tearDown(self):
        self.db_patch.stop()
        self.temp.cleanup()

    @staticmethod
    def request(
        satisfaction="unrated",
        reason=None,
        feedback_id="feedback-123456789012",
        workspace="atlas-main",
        question="How does authentication work?",
    ):
        return auth_routes.AnswerFeedbackRequest(
            feedback_id=feedback_id,
            workspace=workspace,
            question=question,
            satisfaction=satisfaction,
            reason=reason,
        )

    def test_every_question_is_stored_before_it_is_rated(self):
        result = auth_routes.save_answer_feedback(self.request(), self.user)

        self.assertEqual(result["satisfaction"], "unrated")
        rows = db.list_answer_feedback()
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["question"], "How does authentication work?")
        self.assertEqual(rows[0]["satisfaction"], "unrated")
        self.assertEqual(rows[0]["repo_slug"], "atlas")

    def test_shared_ask_flow_tracks_questions_with_an_anonymous_feedback_id(self):
        feedback_id = ask_service._record_anonymous_question(
            SimpleNamespace(
                feedback_id="feedback-sharedflow123",
                question="Which service owns checkout?",
            ),
            self.repo,
        )

        self.assertTrue(feedback_id.startswith("feedback-"))
        rows = db.list_answer_feedback()
        self.assertEqual(rows[0]["question"], "Which service owns checkout?")
        self.assertEqual(rows[0]["satisfaction"], "unrated")

    def test_feedback_storage_failure_does_not_break_the_answer_path(self):
        request = SimpleNamespace(
            feedback_id="feedback-storagefail1",
            question="Can this still be answered?",
        )
        with self.assertLogs(ask_service.logger, level="ERROR"), patch.object(
            ask_service.db,
            "upsert_answer_feedback",
            side_effect=RuntimeError("database temporarily unavailable"),
        ):
            feedback_id = ask_service._record_anonymous_question(request, self.repo)

        self.assertEqual(feedback_id, "feedback-storagefail1")

    def test_rating_updates_the_question_without_creating_a_duplicate(self):
        auth_routes.save_answer_feedback(self.request(), self.user)
        auth_routes.save_answer_feedback(self.request(satisfaction="liked"), self.user)
        # A delayed tracking request must not overwrite an explicit rating.
        auth_routes.save_answer_feedback(self.request(), self.user)

        rows = db.list_answer_feedback()
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["satisfaction"], "liked")
        self.assertIsNone(rows[0]["reason"])

    def test_schema_upgrade_preserves_existing_ratings_and_adds_unrated(self):
        with db.connect() as connection:
            connection.executescript(
                "DROP INDEX idx_answer_feedback_repo;"
                "DROP TABLE answer_feedback;"
                "CREATE TABLE answer_feedback ("
                "feedback_id TEXT PRIMARY KEY, repo_slug TEXT NOT NULL, "
                "repo_name TEXT NOT NULL, question TEXT NOT NULL, "
                "satisfaction TEXT NOT NULL "
                "CHECK (satisfaction IN ('liked', 'disliked')), reason TEXT, "
                "created_at TEXT NOT NULL DEFAULT (datetime('now')), "
                "updated_at TEXT NOT NULL DEFAULT (datetime('now')));"
                "INSERT INTO answer_feedback "
                "(feedback_id, repo_slug, repo_name, question, satisfaction) "
                "VALUES ('feedback-legacyschema1', 'atlas', 'CodeAtlas', "
                "'Was this useful?', 'liked');"
            )

        db.init_db()
        auth_routes.save_answer_feedback(self.request(), self.user)

        rows = db.list_answer_feedback()
        self.assertEqual(len(rows), 2)
        self.assertEqual(
            {row["satisfaction"] for row in rows},
            {"liked", "unrated"},
        )

    def test_dislike_requires_one_of_the_supported_reasons(self):
        with self.assertRaises(HTTPException) as missing:
            auth_routes.save_answer_feedback(
                self.request(satisfaction="disliked"),
                self.user,
            )
        self.assertEqual(missing.exception.status_code, 400)

        saved = auth_routes.save_answer_feedback(
            self.request(satisfaction="disliked", reason="too_vague"),
            self.user,
        )
        self.assertEqual(saved["reason"], "too_vague")
        self.assertEqual(db.list_answer_feedback()[0]["reason"], "too_vague")

    def test_user_cannot_submit_feedback_for_an_ungranted_repository(self):
        with self.assertRaises(HTTPException) as denied:
            auth_routes.save_answer_feedback(self.request(), self.outsider)
        self.assertEqual(denied.exception.status_code, 403)

    def test_admin_payload_and_table_contain_no_user_identity(self):
        auth_routes.save_answer_feedback(
            self.request(question="<script>alert('xss')</script>"),
            self.user,
        )

        payload = auth_routes.get_answer_feedback(self.admin)
        self.assertEqual(len(payload["feedback"]), 1)
        self.assertEqual(
            payload["feedback"][0]["question"],
            "<script>alert('xss')</script>",
        )
        identity_fields = {"user_id", "username", "email"}
        self.assertTrue(identity_fields.isdisjoint(payload["feedback"][0]))
        with db.connect() as connection:
            columns = {
                row["name"]
                for row in connection.execute(
                    "PRAGMA table_info(answer_feedback)"
                ).fetchall()
            }
        self.assertTrue(identity_fields.isdisjoint(columns))

    def test_feedback_remains_anonymous_and_groupable_after_user_deletion(self):
        second_repo = db.create_repo(
            "payments",
            "Payments",
            "https://example.test/payments.git",
            "https",
            "payments-main",
            "published",
        )
        db.grant_access(self.user["id"], second_repo["id"])
        auth_routes.save_answer_feedback(self.request(), self.user)
        auth_routes.save_answer_feedback(
            self.request(
                feedback_id="feedback-abcdefghijkl",
                workspace="payments-main",
                question="How are refunds processed?",
            ),
            self.user,
        )

        db.delete_user(self.user["id"])

        rows = db.list_answer_feedback()
        self.assertEqual({row["repo_slug"] for row in rows}, {"atlas", "payments"})
        self.assertTrue(all("user_id" not in row for row in rows))

    def test_feedback_controls_and_admin_sheet_are_wired(self):
        root = Path(__file__).resolve().parents[1]
        ask_html = (root / "app/static/index.html").read_text()
        admin_html = (root / "app/static/admin.html").read_text()

        self.assertIn("/auth/me/answer-feedback", ask_html)
        self.assertIn("feedback_id: feedbackId", ask_html)
        self.assertIn("Too vague", ask_html)
        self.assertIn("Too simple", ask_html)
        self.assertIn("Too complex", ask_html)
        self.assertIn("Out of context", ask_html)
        self.assertLess(
            admin_html.index('id="answerFeedbackBtn"'),
            admin_html.index('id="repoActivityBtn"'),
        )
        self.assertIn("<th>Question</th><th>Satisfaction</th>", admin_html)
        self.assertIn("esc(row.question || \"\")", admin_html)


if __name__ == "__main__":
    unittest.main()
