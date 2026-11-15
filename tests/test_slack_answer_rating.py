import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from app import db
from app.auth import routes as auth_routes
from app.slack import routes as slack_routes

FEEDBACK_ID = "feedback-12345678-aaaa-bbbb-cccc-1234567890ab"


class SlackAnswerRatingTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.db_patch = patch.object(db, "DB_PATH", Path(self.temp.name) / "codeatlas.db")
        self.db_patch.start()
        db.init_db()
        db.upsert_answer_feedback(FEEDBACK_ID, "atlas", "CodeAtlas", "How does login work?", "unrated")
        # Run the message update inline so the test can inspect it.
        self.executor = patch.object(
            slack_routes._executor, "submit", side_effect=lambda fn, *a, **k: fn(*a, **k)
        )
        self.executor.start()
        self.response = {"answer": "Login creates a session."}
        self.topic = {"question": "How does login work?", "ask_type": slack_routes.ASK_SINGLE}

    def tearDown(self):
        self.executor.stop()
        self.db_patch.stop()
        self.temp.cleanup()

    @staticmethod
    def row():
        (row,) = db.list_answer_feedback()
        return row

    def payload(self, action_id, value):
        return {
            "response_url": "https://hooks.slack.test/actions/1",
            "message": {
                "text": "CodeAtlas answer",
                "blocks": [
                    {"type": "section", "block_id": "b1", "text": {"type": "mrkdwn", "text": "a"}},
                    slack_routes._rating_buttons_block(FEEDBACK_ID),
                ],
            },
            "actions": [{"action_id": action_id, "value": value}],
        }

    def test_answer_gets_a_rating_row_only_when_a_feedback_id_is_given(self):
        plain = slack_routes._answer_text_blocks(self.response, self.topic)
        self.assertNotIn(
            slack_routes.ACTION_RATE_UP,
            [e["action_id"] for b in plain if b["type"] == "actions" for e in b["elements"]],
        )
        # The existing buttons stay the last block when no id is given.
        self.assertEqual(plain[-1]["elements"][0]["action_id"], slack_routes.ACTION_FOLLOW_UP)

        rated = slack_routes._answer_text_blocks(self.response, self.topic, FEEDBACK_ID)
        self.assertEqual(rated[:-1], plain)
        self.assertEqual(
            [e["action_id"] for e in rated[-1]["elements"]],
            [slack_routes.ACTION_RATE_UP, slack_routes.ACTION_RATE_DOWN],
        )
        self.assertEqual(rated[-1]["elements"][0]["value"], FEEDBACK_ID)

    def test_a_malformed_feedback_id_adds_no_rating_row(self):
        blocks = slack_routes._answer_text_blocks(self.response, self.topic, "nope")
        self.assertEqual(blocks, slack_routes._answer_text_blocks(self.response, self.topic))

    def test_thumbs_up_is_saved_and_the_buttons_are_replaced(self):
        with patch.object(slack_routes.requests, "post") as post:
            slack_routes._handle_block_actions(self.payload(slack_routes.ACTION_RATE_UP, FEEDBACK_ID))
        self.assertEqual(self.row()["satisfaction"], "liked")
        body = post.call_args.kwargs["json"]
        self.assertTrue(body["replace_original"])
        self.assertEqual(body["blocks"][0]["block_id"], "b1")
        self.assertEqual(len(body["blocks"]), 2)
        self.assertIn("Thanks", body["blocks"][-1]["elements"][0]["text"])

    def test_thumbs_down_is_saved_then_asks_for_a_reason(self):
        with patch.object(slack_routes.requests, "post") as post:
            slack_routes._handle_block_actions(self.payload(slack_routes.ACTION_RATE_DOWN, FEEDBACK_ID))
            self.assertEqual(self.row()["satisfaction"], "disliked")
            self.assertIsNone(self.row()["reason"])
            reasons = post.call_args.kwargs["json"]["blocks"][-1]["elements"]
            self.assertEqual(
                {e["value"].split("|")[1] for e in reasons},
                set(auth_routes.ANSWER_FEEDBACK_REASONS),
            )
            self.assertEqual(len({e["action_id"] for e in reasons}), len(reasons))
            slack_routes._handle_block_actions(
                self.payload(reasons[1]["action_id"], reasons[1]["value"])
            )
        self.assertEqual(self.row()["satisfaction"], "disliked")
        self.assertEqual(self.row()["reason"], "too_simple")

    def test_unknown_ids_and_reasons_change_nothing(self):
        with patch.object(slack_routes.requests, "post") as post:
            slack_routes._handle_block_actions(
                self.payload(slack_routes.ACTION_RATE_UP, "feedback-unknown-id-123456789")
            )
            slack_routes._handle_block_actions(
                self.payload(slack_routes.ACTION_RATE_UP, "not-a-feedback-id")
            )
            slack_routes._handle_block_actions(
                self.payload(f"{slack_routes.ACTION_RATE_REASON}_x", f"{FEEDBACK_ID}|made_up")
            )
        post.assert_not_called()
        self.assertEqual(db.answer_feedback_count(), 1)
        self.assertEqual(self.row()["satisfaction"], "unrated")

    def test_a_failed_message_update_keeps_the_rating(self):
        with patch.object(slack_routes.requests, "post", side_effect=RuntimeError("boom")):
            slack_routes._handle_block_actions(self.payload(slack_routes.ACTION_RATE_UP, FEEDBACK_ID))
        self.assertEqual(self.row()["satisfaction"], "liked")


class ProviderAuthBadgeTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.db_patch = patch.object(db, "DB_PATH", Path(self.temp.name) / "codeatlas.db")
        self.db_patch.start()
        db.init_db()

    def tearDown(self):
        self.db_patch.stop()
        self.temp.cleanup()

    def test_counts_only_recent_provider_auth_failures(self):
        db.record_answer_failure(question="a", error_kind="provider_auth", error="[401]")
        db.record_answer_failure(question="b", error_kind="provider_auth", error="[401]")
        db.record_answer_failure(question="c", error_kind="rate_limited", error="[429]")
        with db.connect() as conn:
            conn.execute(
                "UPDATE answer_failures SET created_at = datetime('now', '-3 days') WHERE question = 'b'"
            )
        self.assertEqual(db.recent_answer_failure_count("provider_auth", 24), 1)
        self.assertEqual(db.recent_answer_failure_count("rate_limited", 24), 1)

    def test_admin_listing_reports_the_count_without_changing_other_fields(self):
        db.record_answer_failure(question="a", error_kind="provider_auth", error="[401]")
        admin = db.create_user("admin", "unused", role="admin")
        result = auth_routes.get_answer_failures(admin)
        self.assertEqual(result["provider_auth_24h"], 1)
        self.assertEqual(result["total"], 1)
        self.assertEqual(len(result["failures"]), 1)


if __name__ == "__main__":
    unittest.main()
