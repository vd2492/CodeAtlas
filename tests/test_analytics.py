import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from app import ask_service, db
from app.auth import routes as auth_routes
from app.auth.security import hash_password


class TokenAnalyticsTests(unittest.TestCase):
    def test_records_and_summarizes_token_usage_by_user_day_and_provider(self):
        with tempfile.TemporaryDirectory() as temp_dir, patch.object(
            db, "DB_PATH", Path(temp_dir) / "codeatlas.db"
        ):
            db.init_db()
            admin = db.create_user("admin", hash_password("admin-pass"), role="admin")
            reader = db.create_user("reader", hash_password("reader-pass"), role="user")

            self.assertTrue(db.record_token_usage(
                user_id=admin["id"],
                username=admin["username"],
                repo_slug="sample",
                workspace="sample",
                endpoint="repo.ask",
                provider_used="shared:mimo",
                token_usage={
                    "available": True,
                    "input_tokens": 100,
                    "output_tokens": 40,
                    "cached_input_tokens": 10,
                    "total_tokens": 150,
                    "requests": 1,
                },
            ))
            self.assertTrue(db.record_token_usage(
                user_id=reader["id"],
                username=reader["username"],
                repo_slug="sample",
                workspace="sample",
                endpoint="repo.compare",
                provider_used="user:openai",
                token_usage={
                    "available": True,
                    "input_tokens": 180,
                    "output_tokens": 70,
                    "cached_input_tokens": 0,
                    "total_tokens": 250,
                    "requests": 2,
                },
            ))
            self.assertFalse(db.record_token_usage(
                user_id=reader["id"],
                username=reader["username"],
                token_usage={"available": False},
            ))

            analytics = db.token_usage_analytics(days=7)

            self.assertEqual(analytics["totals"]["total_tokens"], 400)
            self.assertEqual(analytics["totals"]["input_tokens"], 280)
            self.assertEqual(analytics["totals"]["output_tokens"], 110)
            self.assertEqual(analytics["totals"]["cached_input_tokens"], 10)
            self.assertEqual(analytics["totals"]["llm_requests"], 3)
            self.assertEqual(analytics["totals"]["active_users"], 2)
            users = {row["username"]: row for row in analytics["by_user"]}
            self.assertEqual(users["admin"]["total_tokens"], 150)
            self.assertEqual(users["reader"]["total_tokens"], 250)
            self.assertEqual(sum(row["total_tokens"] for row in analytics["by_day"]), 400)
            providers = {row["provider"]: row for row in analytics["by_provider"]}
            self.assertEqual(providers["shared:mimo"]["total_tokens"], 150)
            self.assertEqual(providers["user:openai"]["llm_requests"], 2)

    def test_admin_analytics_route_returns_summary(self):
        with tempfile.TemporaryDirectory() as temp_dir, patch.object(
            db, "DB_PATH", Path(temp_dir) / "codeatlas.db"
        ):
            db.init_db()
            admin = db.create_user("admin", hash_password("admin-pass"), role="admin")
            db.record_token_usage(
                user_id=admin["id"],
                username=admin["username"],
                endpoint="repo.flow_summary",
                provider_used="shared:mimo",
                token_usage={
                    "available": True,
                    "input_tokens": 20,
                    "output_tokens": 5,
                    "cached_input_tokens": 0,
                    "total_tokens": 25,
                    "requests": 1,
                },
            )

            result = auth_routes.admin_analytics(admin=admin, days=30)

            self.assertEqual(result["totals"]["total_tokens"], 25)
            self.assertEqual(result["by_user"][0]["username"], "admin")
            self.assertEqual(result["by_provider"][0]["provider"], "shared:mimo")

    def test_admin_analytics_route_supports_last_24h(self):
        with tempfile.TemporaryDirectory() as temp_dir, patch.object(
            db, "DB_PATH", Path(temp_dir) / "codeatlas.db"
        ):
            db.init_db()
            admin = db.create_user("admin", hash_password("admin-pass"), role="admin")
            db.record_token_usage(
                user_id=admin["id"],
                username=admin["username"],
                endpoint="repo.ask",
                provider_used="shared:mimo",
                token_usage={
                    "available": True,
                    "input_tokens": 10,
                    "output_tokens": 5,
                    "total_tokens": 15,
                    "requests": 1,
                },
                created_at="2000-01-01 00:00:00",
            )
            db.record_token_usage(
                user_id=admin["id"],
                username=admin["username"],
                endpoint="repo.ask",
                provider_used="shared:mimo",
                token_usage={
                    "available": True,
                    "input_tokens": 20,
                    "output_tokens": 5,
                    "total_tokens": 25,
                    "requests": 1,
                },
            )

            result = auth_routes.admin_analytics(admin=admin, range="24h")

            self.assertEqual(result["hours"], 24)
            self.assertEqual(result["bucket"], "hour")
            self.assertEqual(len(result["by_day"]), 24)
            self.assertEqual(result["totals"]["total_tokens"], 25)
            self.assertEqual(result["by_user"][0]["event_count"], 1)

    def test_answer_token_usage_storage_is_deferred(self):
        response = {
            "provider_used": "shared:mimo",
            "token_usage": {
                "available": True,
                "input_tokens": 10,
                "output_tokens": 5,
                "cached_input_tokens": 0,
                "total_tokens": 15,
                "requests": 1,
            },
        }
        with patch.object(ask_service._ANALYTICS_EXECUTOR, "submit") as submit, \
             patch.object(db, "record_token_usage") as record_token_usage:
            scheduled = ask_service.schedule_answer_token_usage(
                {"id": 7, "username": "reader"},
                "sample-workspace",
                "repo.ask",
                response,
                repo={"slug": "sample"},
            )

        self.assertTrue(scheduled)
        record_token_usage.assert_not_called()
        submit.assert_called_once()
        payload = submit.call_args.args[1]
        self.assertEqual(payload["token_usage"]["total_tokens"], 15)
        self.assertEqual(payload["repo_slug"], "sample")

    def test_admin_page_contains_analytics_dashboard_entry_point(self):
        html = (Path(__file__).resolve().parents[1] / "app/static/admin.html").read_text()
        self.assertIn('id="analyticsBtn"', html)
        self.assertIn('id="analyticsDashboard"', html)
        self.assertIn("/auth/admin/analytics", html)
        self.assertIn('<option value="24h">Last 24h</option>', html)
        self.assertIn("dailyTokenChart", html)
        self.assertIn("analyticsUserTable", html)
        self.assertIn("<th>Queries</th>", html)
        self.assertIn("provider-total", html)
        self.assertIn('id="dailyBarChartBtn"', html)
        self.assertIn('id="dailyLineChartBtn"', html)
        self.assertIn("renderDailyLineChart", html)
        self.assertIn("Hourly tokens", html)


if __name__ == "__main__":
    unittest.main()
