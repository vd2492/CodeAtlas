import tempfile
import threading
import unittest
from pathlib import Path
from unittest.mock import patch

from fastapi import HTTPException

from app import db
from app import main
from app.llm.admission import LLMAdmissionController, LLMCapacityError


class LLMAdmissionControllerTests(unittest.TestCase):
    def test_rejects_when_active_and_queue_capacity_are_full(self):
        controller = LLMAdmissionController(
            max_active=1,
            max_queued=0,
            queue_timeout_seconds=1,
        )
        controller.acquire()
        try:
            with self.assertRaisesRegex(LLMCapacityError, "at capacity"):
                controller.acquire()
        finally:
            controller.release()

        self.assertEqual(controller.snapshot().active, 0)

    def test_queued_request_times_out_and_is_removed(self):
        controller = LLMAdmissionController(
            max_active=1,
            max_queued=1,
            queue_timeout_seconds=0.01,
        )
        controller.acquire()
        try:
            with self.assertRaisesRegex(LLMCapacityError, "busy"):
                controller.acquire()
        finally:
            controller.release()

        snapshot = controller.snapshot()
        self.assertEqual(snapshot.active, 0)
        self.assertEqual(snapshot.queued, 0)

    def test_release_allows_the_first_queued_request_to_run(self):
        controller = LLMAdmissionController(
            max_active=1,
            max_queued=1,
            queue_timeout_seconds=1,
        )
        controller.acquire()
        completed = threading.Event()
        errors = []

        def queued_request():
            try:
                with controller.slot():
                    completed.set()
            except Exception as error:  # pragma: no cover - asserted below
                errors.append(error)

        thread = threading.Thread(target=queued_request)
        thread.start()
        for _ in range(100):
            if controller.snapshot().queued == 1:
                break
            completed.wait(0.005)

        self.assertEqual(controller.snapshot().queued, 1)
        controller.release()
        thread.join(timeout=1)

        self.assertFalse(thread.is_alive())
        self.assertTrue(completed.is_set())
        self.assertEqual(errors, [])
        self.assertEqual(controller.snapshot().active, 0)

    def test_context_manager_releases_after_pipeline_error(self):
        controller = LLMAdmissionController(max_active=1, max_queued=0)
        with self.assertRaisesRegex(RuntimeError, "generation failed"):
            with controller.slot():
                raise RuntimeError("generation failed")
        self.assertEqual(controller.snapshot().active, 0)


class SQLiteConcurrencyConfigurationTests(unittest.TestCase):
    def test_connection_applies_configured_busy_timeout(self):
        with tempfile.TemporaryDirectory() as temporary_directory, patch.object(
            db, "DB_PATH", Path(temporary_directory) / "codeatlas.db"
        ), patch.object(db, "SQLITE_BUSY_TIMEOUT_MS", 1234):
            with db.connect() as connection:
                configured = connection.execute("PRAGMA busy_timeout").fetchone()[0]

        self.assertEqual(configured, 1234)


class LLMAdmissionEndpointTests(unittest.TestCase):
    def test_ask_returns_retryable_service_busy_without_running_pipeline(self):
        controller = LLMAdmissionController(max_active=1, max_queued=0)
        controller.acquire()
        try:
            with patch.object(main, "llm_admission", controller), patch.object(
                main, "enforce_rate_limit"
            ), patch.object(
                main, "enforce_strict_branch_freshness"
            ), patch.object(
                main.db,
                "get_repo_by_workspace",
                return_value={"allow_shared_fallback": 1},
            ), patch.object(
                main, "load_user_llm", return_value=None
            ), patch.object(
                main, "repository_revision", return_value="branch:abc123"
            ), patch.object(
                main, "answer_question"
            ) as answer:
                with self.assertRaises(HTTPException) as raised:
                    main.ask_llm_endpoint(
                        main.AskRequest(question="How does login work?"),
                        "repo-main",
                        {"id": 7, "user_type": "dev_team"},
                    )
        finally:
            controller.release()

        self.assertEqual(raised.exception.status_code, 503)
        self.assertEqual(raised.exception.headers, {"Retry-After": "5"})
        answer.assert_not_called()


if __name__ == "__main__":
    unittest.main()
