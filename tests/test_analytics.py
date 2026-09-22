import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from app import ask_service, config, db, main
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

    def test_records_slack_token_usage_with_source_metadata(self):
        with tempfile.TemporaryDirectory() as temp_dir, patch.object(
            db, "DB_PATH", Path(temp_dir) / "codeatlas.db"
        ):
            db.init_db()

            self.assertTrue(db.record_token_usage(
                user_id=-42,
                username="slack:T123:U123",
                repo_slug="sample",
                workspace="sample",
                endpoint="repo.ask",
                source="slack",
                slack_user_id="U123",
                slack_team_id="T123",
                slack_channel_id="C123",
                ask_type="single_branch",
                branch="main",
                provider_used="shared:mimo",
                token_usage={
                    "available": True,
                    "input_tokens": 120,
                    "output_tokens": 30,
                    "total_tokens": 150,
                    "requests": 1,
                },
            ))

            analytics = db.token_usage_analytics(days=7)

            self.assertEqual(analytics["totals"]["total_tokens"], 150)
            self.assertEqual(analytics["by_user"][0]["username"], "slack:T123:U123")
            self.assertEqual(analytics["by_user"][0]["source"], "slack")
            self.assertEqual(analytics["by_source"], [{
                "source": "slack",
                "total_tokens": 150,
                "llm_requests": 1,
                "event_count": 1,
            }])

    def test_records_unique_marketing_page_visitors(self):
        with tempfile.TemporaryDirectory() as temp_dir, patch.object(
            db, "DB_PATH", Path(temp_dir) / "codeatlas.db"
        ):
            db.init_db()

            db.record_site_visit("visitor-a")
            db.record_site_visit("visitor-a")
            db.record_site_visit("visitor-b")

            self.assertEqual(db.site_visitor_count(), 2)

    def test_admin_analytics_route_returns_summary(self):
        with tempfile.TemporaryDirectory() as temp_dir, patch.object(
            db, "DB_PATH", Path(temp_dir) / "codeatlas.db"
        ):
            db.init_db()
            admin = db.create_user("admin", hash_password("admin-pass"), role="admin")
            db.record_site_visit("visitor-a")
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
            self.assertEqual(result["site_visitors"], 1)

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

    def test_admin_analytics_route_passes_timezone_offset(self):
        admin = {"id": 1, "username": "admin", "role": "admin"}
        with patch.object(auth_routes.db, "token_usage_analytics") as analytics, \
             patch.object(auth_routes.db, "site_visitor_count", return_value=3):
            analytics.return_value = {"ok": True}

            self.assertEqual(
                auth_routes.admin_analytics(
                    admin=admin,
                    range="24h",
                    tz_offset_minutes=330,
                ),
                {"ok": True, "site_visitors": 3},
            )

        analytics.assert_called_once_with(hours=24, tz_offset_minutes=330)

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

    def test_answer_token_usage_schedules_slack_metadata(self):
        response = {
            "provider_used": "shared:mimo",
            "token_usage": {
                "available": True,
                "input_tokens": 10,
                "output_tokens": 5,
                "total_tokens": 15,
                "requests": 1,
            },
        }
        with patch.object(ask_service._ANALYTICS_EXECUTOR, "submit") as submit:
            scheduled = ask_service.schedule_answer_token_usage(
                {"id": -7, "username": "slack:T123:U123"},
                "sample-workspace",
                "repo.ask",
                response,
                repo={"slug": "sample"},
                analytics_context={
                    "source": "slack",
                    "slack_user_id": "U123",
                    "slack_team_id": "T123",
                    "slack_channel_id": "C123",
                    "ask_type": "single_branch",
                    "branch": "main",
                },
            )

        self.assertTrue(scheduled)
        payload = submit.call_args.args[1]
        self.assertEqual(payload["source"], "slack")
        self.assertEqual(payload["slack_user_id"], "U123")
        self.assertEqual(payload["slack_team_id"], "T123")
        self.assertEqual(payload["slack_channel_id"], "C123")
        self.assertEqual(payload["ask_type"], "single_branch")
        self.assertEqual(payload["branch"], "main")

    def test_admin_page_contains_analytics_dashboard_entry_point(self):
        html = (Path(__file__).resolve().parents[1] / "app/static/admin.html").read_text()
        self.assertIn('id="analyticsBtn"', html)
        self.assertIn('id="analyticsDashboard"', html)
        self.assertIn("/auth/admin/analytics", html)
        self.assertIn('<option value="24h">Last 24h</option>', html)
        self.assertIn("tz_offset_minutes", html)
        self.assertIn("dailyTokenChart", html)
        self.assertIn('id="dailyTokenChartViewport"', html)
        self.assertIn("bindDailyChartPan", html)
        self.assertIn("analyticsUserTable", html)
        self.assertIn('id="siteVisitedUsers"', html)
        self.assertIn("Total site visitors", html)
        self.assertIn("<th>Source</th>", html)
        self.assertIn("formatSourceLabel", html)
        self.assertIn("<th>Queries</th>", html)
        self.assertIn("provider-total", html)
        self.assertIn('id="dailyBarChartBtn"', html)
        self.assertIn('id="dailyLineChartBtn"', html)
        self.assertIn("renderDailyLineChart", html)
        self.assertIn("Hourly tokens", html)
        self.assertIn('id="insightsPanel"', html)
        self.assertIn('id="insightsClearCacheBtn"', html)
        self.assertIn("/admin/repos/${curInsightsSlug}/answer-cache/clear", html)


class AnswerMarkdownTests(unittest.TestCase):
    def test_ask_page_renders_markdown_blocks(self):
        """Headings, rules and lists were falling through as literal text, so
        answers showed '##' and '---' instead of formatting."""
        html = (Path(__file__).resolve().parents[1] / "app/static/index.html").read_text()
        for helper in ("headingMatch", "isHorizontalRule", "listItemMatch"):
            self.assertIn(f"function {helper}(", html)
        self.assertIn("answer-heading", html)
        self.assertIn('<hr class="answer-rule">', html)
        self.assertIn("answer-list", html)

    def test_ask_page_renders_bold_instead_of_stripping_it(self):
        html = (Path(__file__).resolve().parents[1] / "app/static/index.html").read_text()
        self.assertIn("function renderEmphasis(", html)
        self.assertIn("<strong>$1</strong>", html)
        # The old stripping helper is gone.
        self.assertNotIn("cleanAnswerText", html)
        # Inline code is split out first so ** inside it stays literal, and
        # __ is left alone so names like __init__ survive.
        self.assertIn("split(/(`[^`\\n]+`)/g)", html)
        self.assertNotIn("__([^_\\n]+)__", html)


class UiFeatureFlagTests(unittest.TestCase):
    def test_optional_surfaces_are_hidden_by_default(self):
        with patch.dict(os.environ, {}, clear=False):
            for name in config.UI_FEATURE_ENV.values():
                os.environ.pop(name, None)
            flags = config.ui_feature_flags()
        self.assertEqual(
            flags,
            {
                "repo_summary_hidden": True,
                "flow_explorer_hidden": True,
                "graph_search_hidden": True,
            },
        )

    def test_env_var_can_bring_a_surface_back(self):
        with patch.dict(os.environ, {"CODEATLAS_HIDE_FLOW_EXPLORER": "false"}):
            flags = config.ui_feature_flags()
        self.assertFalse(flags["flow_explorer_hidden"])
        # The other surfaces are unaffected.
        self.assertTrue(flags["repo_summary_hidden"])
        self.assertTrue(flags["graph_search_hidden"])

    def test_ui_config_endpoint_serves_the_flags(self):
        self.assertEqual(main.ui_config(), {"features": config.ui_feature_flags()})

    def test_ask_page_reads_the_remote_config(self):
        html = (Path(__file__).resolve().parents[1] / "app/static/index.html").read_text()
        self.assertIn("/ui-config", html)
        self.assertIn(".feature-off { display: none !important; }", html)
        for element_id in (
            "repoSummaryCard",
            "flowExplorerCard",
            "graphSearchCard",
            "secondaryTools",
        ):
            self.assertIn(f'id="{element_id}"', html)


if __name__ == "__main__":
    unittest.main()
