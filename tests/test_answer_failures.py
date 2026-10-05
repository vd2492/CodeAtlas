import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from fastapi import HTTPException

from app import ask_service, config, db
from app.auth import routes as auth_routes


class AnswerFailureTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        root = Path(self.temp.name)
        self.patches = [
            patch.object(db, "DB_PATH", root / "codeatlas.db"),
            patch.object(config, "WORKSPACES_DIR", root / "workspaces"),
            # Run the background write inline so the test can read it back.
            patch.object(
                ask_service._ANALYTICS_EXECUTOR,
                "submit",
                side_effect=lambda fn, *a, **k: fn(*a, **k),
            ),
        ]
        for item in self.patches:
            item.start()
        db.init_db()
        self.repo = db.create_repo(
            "alpha", "Alpha", "https://example.test/alpha.git", "https", "alpha",
            status="published",
        )
        self.user = {"id": 7, "username": "daksh", "role": "user"}

    def tearDown(self):
        for item in reversed(self.patches):
            item.stop()
        self.temp.cleanup()

    def test_failed_ask_is_saved_with_the_user_and_still_raises(self):
        @ask_service.records_failures("repo.ask")
        def flow(request, workspace, user):
            raise HTTPException(status_code=500, detail="LLM request failed: boom")

        request = SimpleNamespace(question="map column x", llm_mode="auto")
        with self.assertRaises(HTTPException) as raised:
            flow(request, "alpha", self.user)
        self.assertEqual(raised.exception.status_code, 500)
        (row,) = db.list_answer_failures()
        self.assertEqual(row["username"], "daksh")
        self.assertEqual(row["user_id"], 7)
        self.assertEqual(row["repo_slug"], "alpha")
        self.assertEqual(row["question"], "map column x")
        self.assertEqual(row["endpoint"], "repo.ask")
        self.assertEqual(row["status_code"], 500)
        self.assertEqual(row["error_kind"], "provider_error")
        self.assertEqual(row["source"], "web")
        self.assertIsNotNone(row["duration_ms"])

    def test_success_records_nothing_and_returns_the_result(self):
        @ask_service.records_failures("repo.ask")
        def flow(request, workspace, user):
            return {"answer": "ok"}

        self.assertEqual(flow(SimpleNamespace(question="q"), "alpha", self.user), {"answer": "ok"})
        self.assertEqual(db.answer_failure_count(), 0)

    def test_unexpected_errors_and_group_asks_are_recorded(self):
        @ask_service.records_failures("repo.group")
        def flow(request, group, user, *, analytics_context=None):
            raise RuntimeError("kaput")

        group = {"slug": "stack", "name": "Stack", "members": [
            {"repo_slug": "alpha"}, {"repo_slug": "beta"},
        ]}
        with self.assertRaises(RuntimeError):
            flow(SimpleNamespace(question="q"), group, self.user,
                 analytics_context={"source": "slack"})
        (row,) = db.list_answer_failures()
        self.assertEqual(row["workspace"], "group:stack")
        self.assertEqual(row["repo_slug"], "alpha,beta")
        self.assertEqual(row["status_code"], 500)
        self.assertEqual(row["source"], "slack")

    def test_recording_problems_never_change_the_error(self):
        @ask_service.records_failures("repo.ask")
        def flow(request, workspace, user):
            raise HTTPException(status_code=409, detail="not ready")

        with patch.object(db, "record_answer_failure", side_effect=RuntimeError("db down")):
            with self.assertRaises(HTTPException) as raised:
                flow(SimpleNamespace(question="q"), "alpha", self.user)
        self.assertEqual(raised.exception.status_code, 409)

    def test_failures_are_bucketed(self):
        classify = ask_service.classify_answer_failure
        self.assertEqual(classify(504, ""), "timeout")
        self.assertEqual(classify(500, "request timed out"), "timeout")
        self.assertEqual(classify(429, "slow down"), "rate_limited")
        self.assertEqual(classify(503, "queue full"), "capacity")
        self.assertEqual(classify(500, "No LLM provider succeeded. Tried -> x"), "no_provider")
        self.assertEqual(classify(409, "branch not ready"), "not_ready")
        self.assertEqual(classify(502, "bad gateway"), "provider_error")
        self.assertEqual(classify(400, "bad input"), "other")

    def test_browser_report_uses_the_session_user(self):
        auth_routes.report_answer_failure(
            auth_routes.AnswerFailureReport(
                question="why", workspace="alpha", endpoint="repo.ask",
                kind="timeout", message="gateway cut it off", status_code=504,
                duration_ms=61000,
            ),
            self.user,
        )
        auth_routes.report_answer_failure(
            auth_routes.AnswerFailureReport(question="x", kind="made-up"), self.user
        )
        newest, oldest = db.list_answer_failures()
        self.assertEqual((oldest["username"], oldest["source"], oldest["error_kind"]),
                         ("daksh", "web-client", "timeout"))
        self.assertEqual(oldest["repo_slug"], "alpha")
        self.assertEqual(newest["error_kind"], "no_response")
        with self.assertRaises(HTTPException):
            auth_routes.report_answer_failure(
                auth_routes.AnswerFailureReport(question="  "), self.user
            )

    def test_browser_reports_are_capped_per_user(self):
        report = auth_routes.AnswerFailureReport(question="again", kind="network")
        results = [
            auth_routes.report_answer_failure(report, self.user)["recorded"]
            for _ in range(db.CLIENT_FAILURE_REPORTS_PER_HOUR + 3)
        ]
        self.assertEqual(results.count(True), db.CLIENT_FAILURE_REPORTS_PER_HOUR)
        other = {"id": 8, "username": "ishaan", "role": "user"}
        self.assertTrue(auth_routes.report_answer_failure(report, other)["recorded"])

    def test_old_failures_are_pruned(self):
        db.record_answer_failure(username="u", question="ancient")
        with db.connect() as conn:
            conn.execute(
                "UPDATE answer_failures SET created_at = datetime('now', '-91 days')"
            )
        db.record_answer_failure(username="u", question="fresh")
        self.assertEqual([f["question"] for f in db.list_answer_failures()], ["fresh"])

    def test_admin_listing_pages_newest_first(self):
        for index in range(3):
            db.record_answer_failure(username="u", question=f"q{index}")
        listing = auth_routes.get_answer_failures({"role": "admin"}, limit=2, offset=0)
        self.assertEqual(listing["total"], 3)
        self.assertEqual([f["question"] for f in listing["failures"]], ["q2", "q1"])


if __name__ == "__main__":
    unittest.main()
