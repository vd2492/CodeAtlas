import asyncio
import hmac
import json
import os
import time
import unittest
from hashlib import sha256
from unittest.mock import patch
from urllib.parse import urlencode

from fastapi import HTTPException

from app.slack import routes as slack_routes


def signed_headers(secret: str, body: bytes, timestamp: int = None) -> dict:
    timestamp = timestamp or int(time.time())
    base = f"v0:{timestamp}:{body.decode('utf-8')}".encode("utf-8")
    signature = "v0=" + hmac.new(secret.encode("utf-8"), base, sha256).hexdigest()
    return {
        "content-type": "application/x-www-form-urlencoded",
        "x-slack-request-timestamp": str(timestamp),
        "x-slack-signature": signature,
    }


class FakeRequest:
    def __init__(self, body: bytes, headers: dict):
        self._body = body
        self.headers = headers

    async def body(self) -> bytes:
        return self._body


class FakeResponse:
    def __init__(self, status_code: int = 200, text: str = "ok"):
        self.status_code = status_code
        self.text = text


class SlackIntegrationTests(unittest.TestCase):
    def setUp(self):
        self.secret = "test-signing-secret"
        self.env = patch.dict(os.environ, {
            "CODEATLAS_SLACK_ENABLED": "true",
            "CODEATLAS_SLACK_SIGNING_SECRET": self.secret,
            "CODEATLAS_SLACK_BOT_TOKEN": "xoxb-test-token",
            "CODEATLAS_SLACK_ALLOWED_TEAM_IDS": "T123",
        })
        self.env.start()

    def tearDown(self):
        self.env.stop()

    def test_verify_slack_request_accepts_valid_signature(self):
        body = b"team_id=T123&text=hello"
        slack_routes.verify_slack_request(signed_headers(self.secret, body), body)

    def test_verify_slack_request_accepts_valid_relay_secret(self):
        body = b"team_id=T123&text=hello"
        with patch.dict(os.environ, {
            "CODEATLAS_SLACK_RELAY_SECRET": "relay-secret",
            "CODEATLAS_SLACK_SIGNING_SECRET": "",
        }):
            slack_routes.verify_slack_request({
                "x-codeatlas-relay-secret": "relay-secret",
            }, body)

    def test_verify_slack_request_can_require_relay_secret(self):
        body = b"team_id=T123&text=hello"
        with patch.dict(os.environ, {
            "CODEATLAS_SLACK_RELAY_SECRET": "relay-secret",
            "CODEATLAS_SLACK_REQUIRE_RELAY_SECRET": "true",
        }):
            with self.assertRaises(HTTPException) as raised:
                slack_routes.verify_slack_request(signed_headers(self.secret, body), body)

        self.assertEqual(raised.exception.status_code, 401)

    def test_verify_slack_request_rejects_stale_signature(self):
        body = b"team_id=T123&text=hello"
        stale = int(time.time()) - 600
        with self.assertRaises(HTTPException) as raised:
            slack_routes.verify_slack_request(signed_headers(self.secret, body, stale), body)
        self.assertEqual(raised.exception.status_code, 401)

    def test_slash_command_dispatches_codeatlas_modal_open(self):
        body = urlencode({
            "team_id": "T123",
            "enterprise_id": "",
            "channel_id": "C123",
            "user_id": "U123",
            "trigger_id": "trigger-1",
            "response_url": "https://hooks.slack.test/response",
            "text": "explain auth",
        }).encode("utf-8")
        repos = [{
            "name": "Payments",
            "slug": "payments",
            "status": "published",
        }]
        with patch.object(slack_routes.ask_service, "published_repos", return_value=repos), \
                patch.object(slack_routes, "_open_ask_modal") as modal:
            response = asyncio.run(
                slack_routes.slash_command(
                    FakeRequest(body, signed_headers(self.secret, body))
                )
            )

        self.assertEqual(response.status_code, 200)
        modal.assert_called_once()
        metadata, trigger_id = modal.call_args.args
        self.assertEqual(trigger_id, "trigger-1")
        self.assertEqual(metadata["question"], "explain auth")
        self.assertEqual(metadata["team_id"], "T123")
        self.assertEqual(metadata["user_id"], "U123")
        self.assertEqual(metadata["slack_user_id"], "U123")

    def test_slash_command_accepts_relay_secret(self):
        body = urlencode({
            "team_id": "T123",
            "enterprise_id": "",
            "channel_id": "C123",
            "user_id": "U123",
            "trigger_id": "trigger-1",
            "response_url": "https://hooks.slack.test/response",
            "text": "explain auth",
        }).encode("utf-8")
        headers = {
            "content-type": "application/x-www-form-urlencoded",
            "x-codeatlas-relay-secret": "relay-secret",
        }
        with patch.dict(os.environ, {
            "CODEATLAS_SLACK_RELAY_SECRET": "relay-secret",
            "CODEATLAS_SLACK_SIGNING_SECRET": "",
        }), patch.object(slack_routes, "_open_ask_modal") as modal:
            response = asyncio.run(slack_routes.slash_command(FakeRequest(body, headers)))

        self.assertEqual(response.status_code, 200)
        modal.assert_called_once()

    def test_send_user_message_accepts_user_id_alias(self):
        values = {
            "channel_id": "C123",
            "user_id": "U123",
        }
        with patch.object(slack_routes, "_post_ephemeral") as post:
            slack_routes._send_user_message(values, "hello")

        post.assert_called_once_with("C123", "U123", "hello", None)

    def test_open_ask_modal_posts_expected_view(self):
        metadata = {
            "team_id": "T123",
            "channel_id": "C123",
            "slack_user_id": "U123",
            "repo_slug": "payments",
            "question": "explain auth",
        }
        repo = {
            "name": "Payments",
            "slug": "payments",
            "status": "published",
        }
        with patch.object(slack_routes.ask_service, "published_repos", return_value=[repo]), \
                patch.object(slack_routes, "_slack_api", return_value={"ok": True}) as api:
            slack_routes._open_ask_modal(metadata, "trigger-1")

        api.assert_called_once()
        method, payload = api.call_args.args
        self.assertEqual(method, "views.open")
        self.assertEqual(payload["trigger_id"], "trigger-1")
        view = payload["view"]
        self.assertEqual(view["callback_id"], slack_routes.CALLBACK_ASK)
        labels = [block.get("label", {}).get("text") for block in view["blocks"] if block.get("label")]
        self.assertEqual(labels[:3], ["Repository", "Ask type", "Branch"])
        stored_metadata = json.loads(view["private_metadata"])
        self.assertEqual(stored_metadata["question"], "explain auth")
        self.assertEqual(stored_metadata["team_id"], "T123")

    def test_branch_selection_starts_background_prepare_and_updates_modal(self):
        payload = {
            "type": "block_actions",
            "team": {"id": "T123"},
            "user": {"id": "U123"},
            "trigger_id": "trigger-2",
            "actions": [{
                "action_id": slack_routes.ACTION_BRANCH,
                "selected_option": {"value": "feature/auth"},
            }],
            "view": {
                "id": "view-1",
                "hash": "hash-1",
                "private_metadata": json.dumps({
                    "team_id": "T123",
                    "channel_id": "C123",
                    "slack_user_id": "U123",
                    "repo_slug": "payments",
                    "ask_type": slack_routes.ASK_SINGLE,
                }),
                "state": {
                    "values": {
                        slack_routes.BLOCK_REPO: {
                            slack_routes.ACTION_REPO: {
                                "selected_option": {"value": "payments"}
                            }
                        },
                        slack_routes.BLOCK_ASK_TYPE: {
                            slack_routes.ACTION_ASK_TYPE: {
                                "selected_option": {"value": slack_routes.ASK_SINGLE}
                            }
                        },
                        slack_routes.BLOCK_BRANCH: {
                            slack_routes.ACTION_BRANCH: {
                                "selected_option": {"value": "feature/auth"}
                            }
                        },
                    }
                },
            },
        }
        body = urlencode({"payload": json.dumps(payload)}).encode("utf-8")
        repo = {"id": 1, "name": "Payments", "slug": "payments", "status": "published"}
        branch = {
            "id": 11,
            "name": "feature/auth",
            "workspace": None,
            "index_status": "indexing",
        }
        with patch.object(slack_routes, "_repo_by_slug", return_value=repo), \
                patch.object(slack_routes.ask_service, "prepare_existing_repo_branch", return_value=branch) as prepare, \
                patch.object(
                    slack_routes.ask_service,
                    "approved_branch_options",
                    return_value=[{
                        "id": 11,
                        "name": "feature/auth",
                        "commit_sha": "abc123",
                        "is_default": False,
                    }],
                ), \
                patch.object(slack_routes.ask_service, "published_repos", return_value=[repo]), \
                patch.object(slack_routes.ask_service, "remote_branch_options") as remote, \
                patch.object(slack_routes, "_slack_api", return_value={"ok": True}) as api:
            response = asyncio.run(
                slack_routes.interactions(
                    FakeRequest(body, signed_headers(self.secret, body))
                )
            )

        self.assertEqual(response, {})
        prepare.assert_called_once_with(
            repo,
            "feature/auth",
            actor="slack:T123:U123",
        )
        remote.assert_not_called()
        self.assertEqual(api.call_args.args[0], "views.update")
        updated_view = api.call_args.args[1]["view"]
        self.assertIn("Syncing and indexing", updated_view["private_metadata"])
        branch_block = next(
            block for block in updated_view["blocks"]
            if block.get("block_id") == slack_routes.BLOCK_BRANCH
        )
        self.assertEqual(branch_block["element"]["type"], "static_select")
        self.assertEqual(
            branch_block["element"]["options"][0]["value"],
            "feature/auth",
        )

    def test_collect_view_values_preserves_user_id_alias(self):
        payload = {
            "user": {"id": "U123"},
            "view": {
                "private_metadata": json.dumps({
                    "team_id": "T123",
                    "channel_id": "C123",
                    "user_id": "U123",
                    "repo_slug": "payments",
                }),
                "state": {"values": {}},
            },
        }
        repo = {"id": 1, "name": "Payments", "slug": "payments", "status": "published"}
        with patch.object(slack_routes, "_repo_by_slug", return_value=repo):
            values = slack_routes._collect_view_values(payload)

        self.assertEqual(values["user_id"], "U123")
        self.assertEqual(values["slack_user_id"], "U123")

    def test_repo_initial_option_matches_static_option_shape(self):
        repo = {
            "id": 1,
            "name": "Payments",
            "slug": "payments",
            "status": "published",
        }
        with patch.object(slack_routes, "_repo_by_slug", return_value=repo), \
                patch.object(slack_routes.ask_service, "published_repos", return_value=[repo]), \
                patch.object(
                    slack_routes.ask_service,
                    "approved_branch_options",
                    return_value=[{
                        "id": 11,
                        "name": "main",
                        "commit_sha": "abc123",
                        "is_default": True,
                    }],
                ):
            view = slack_routes.build_ask_view({
                "repo_slug": "payments",
                "repo_name": "Payments",
                "branch": "main",
            })

        repo_block = next(
            block for block in view["blocks"]
            if block.get("block_id") == slack_routes.BLOCK_REPO
        )
        self.assertEqual(
            repo_block["element"]["initial_option"],
            repo_block["element"]["options"][0],
        )

    def test_send_user_message_prefers_response_url(self):
        values = {
            "response_url": "https://hooks.slack.test/response",
            "channel_id": "C123",
            "slack_user_id": "U123",
        }
        blocks = [{"type": "section", "text": {"type": "mrkdwn", "text": "Working"}}]
        with patch.object(slack_routes.requests, "post", return_value=FakeResponse()) as post, \
                patch.object(slack_routes, "_post_ephemeral") as ephemeral:
            slack_routes._send_user_message(values, "Working", blocks)

        post.assert_called_once()
        self.assertEqual(post.call_args.kwargs["json"]["response_type"], "ephemeral")
        self.assertEqual(post.call_args.kwargs["json"]["blocks"], blocks)
        ephemeral.assert_not_called()

    def test_send_user_message_falls_back_to_ephemeral(self):
        values = {
            "response_url": "https://hooks.slack.test/response",
            "channel_id": "C123",
            "slack_user_id": "U123",
        }
        with patch.object(
            slack_routes.requests,
            "post",
            return_value=FakeResponse(status_code=500, text="bad"),
        ), patch.object(slack_routes, "_post_ephemeral") as ephemeral:
            slack_routes._send_user_message(values, "Working")

        ephemeral.assert_called_once_with("C123", "U123", "Working", None)

    def test_view_submission_dispatches_answer_job(self):
        payload = {
            "type": "view_submission",
            "team": {"id": "T123"},
            "user": {"id": "U123"},
            "view": {
                "callback_id": slack_routes.CALLBACK_ASK,
                "private_metadata": json.dumps({
                    "team_id": "T123",
                    "channel_id": "C123",
                    "slack_user_id": "U123",
                }),
                "state": {
                    "values": {
                        slack_routes.BLOCK_REPO: {
                            slack_routes.ACTION_REPO: {
                                "selected_option": {"value": "payments"}
                            }
                        },
                        slack_routes.BLOCK_ASK_TYPE: {
                            slack_routes.ACTION_ASK_TYPE: {
                                "selected_option": {"value": slack_routes.ASK_SINGLE}
                            }
                        },
                        slack_routes.BLOCK_BRANCH: {
                            slack_routes.ACTION_BRANCH: {
                                "selected_option": {"value": "main"}
                            }
                        },
                        slack_routes.BLOCK_USER_TYPE: {
                            slack_routes.ACTION_USER_TYPE: {
                                "selected_option": {"value": slack_routes.USER_PRODUCT}
                            }
                        },
                        slack_routes.BLOCK_QUESTION: {
                            slack_routes.ACTION_QUESTION: {
                                "value": "How does login work?"
                            }
                        },
                    }
                },
            },
        }
        body = urlencode({"payload": json.dumps(payload)}).encode("utf-8")
        repo = {"id": 1, "name": "Payments", "slug": "payments", "status": "published"}
        with patch.object(slack_routes, "_repo_by_slug", return_value=repo), \
                patch.object(slack_routes, "_start_answer_job") as start:
            response = asyncio.run(
                slack_routes.interactions(
                    FakeRequest(body, signed_headers(self.secret, body))
                )
            )

        self.assertEqual(response, {})
        start.assert_called_once()
        values = start.call_args.args[0]
        self.assertEqual(values["repo_slug"], "payments")
        self.assertEqual(values["branch"], "main")
        self.assertEqual(values["user_type"], slack_routes.USER_PRODUCT)
        self.assertEqual(values["question"], "How does login work?")
