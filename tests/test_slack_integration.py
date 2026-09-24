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
        self.assertIn("Branch is still being prepared", updated_view["private_metadata"])
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

    def test_answer_failure_detail_hides_shared_llm_auth_errors(self):
        error = RuntimeError(
            "No LLM provider succeeded. Tried -> shared: [401] "
            "{\"error\":{\"message\":\"Invalid API Key\",\"code\":\"401\",\"type\":\"invalid_key\"}}"
        )

        detail = slack_routes._answer_failure_detail(error)

        self.assertEqual(
            detail,
            "CodeAtlas could not generate an answer because the shared LLM quota "
            "is unavailable. Please contact an admin.",
        )
        self.assertNotIn("Invalid API Key", detail)

    def test_answer_failure_detail_keeps_non_llm_errors(self):
        detail = slack_routes._answer_failure_detail(
            HTTPException(status_code=404, detail="Repository not found.")
        )

        self.assertEqual(detail, "Repository not found.")

    def test_single_answer_passes_slack_analytics_context(self):
        values = {
            "team_id": "T123",
            "channel_id": "C123",
            "slack_user_id": "U123",
            "user_id": "U123",
            "repo_slug": "payments",
            "repo_name": "Payments",
            "branch": "main",
            "branch_workspace": "payments-main",
            "ask_type": slack_routes.ASK_SINGLE,
            "user_type": slack_routes.USER_PRODUCT,
            "question": "How does login work?",
        }
        repo = {
            "id": 1,
            "name": "Payments",
            "slug": "payments",
            "workspace": "payments",
            "status": "published",
        }
        response = {
            "answer": "Login uses sessions.",
            "question": values["question"],
            "conversation_id": "conv-1",
            "investigate_deeply_available": False,
        }
        actor = {"id": -1, "username": "slack:T123:U123", "role": "admin"}
        with patch.object(slack_routes, "_repo_by_slug", return_value=repo), \
                patch.object(slack_routes.ask_service, "slack_actor_user", return_value=actor), \
                patch.object(slack_routes, "_send_user_message"), \
                patch.object(
                    slack_routes.ask_service,
                    "answer_single_request",
                    return_value=response,
                ) as answer:
            slack_routes._run_single_answer(values)

        analytics_context = answer.call_args.kwargs["analytics_context"]
        self.assertEqual(analytics_context["source"], "slack")
        self.assertEqual(analytics_context["slack_user_id"], "U123")
        self.assertEqual(analytics_context["slack_team_id"], "T123")
        self.assertEqual(analytics_context["slack_channel_id"], "C123")
        self.assertEqual(analytics_context["ask_type"], slack_routes.ASK_SINGLE)
        self.assertEqual(analytics_context["branch"], "main")

    def test_single_answer_uses_ready_branch_status_message(self):
        values = {
            "team_id": "T123",
            "channel_id": "C123",
            "slack_user_id": "U123",
            "user_id": "U123",
            "repo_slug": "payments",
            "repo_name": "Payments",
            "branch": "main",
            "ask_type": slack_routes.ASK_SINGLE,
            "user_type": slack_routes.USER_PRODUCT,
            "question": "How does login work?",
        }
        repo = {
            "id": 1,
            "name": "Payments",
            "slug": "payments",
            "workspace": "payments",
            "status": "published",
        }
        branch = {
            "id": 11,
            "name": "main",
            "workspace": "payments-main",
            "index_status": "ready",
        }
        response = {
            "answer": "Login uses sessions.",
            "question": values["question"],
            "conversation_id": "conv-1",
            "investigate_deeply_available": False,
        }
        actor = {"id": -1, "username": "slack:T123:U123", "role": "admin"}
        with patch.object(slack_routes, "_repo_by_slug", return_value=repo), \
                patch.object(slack_routes, "_current_branch", return_value=branch), \
                patch.object(
                    slack_routes.ask_service,
                    "resolve_existing_ready_branch",
                    return_value=branch,
                ), \
                patch.object(slack_routes.ask_service, "slack_actor_user", return_value=actor), \
                patch.object(
                    slack_routes.ask_service,
                    "answer_single_request",
                    return_value=response,
                ), \
                patch.object(slack_routes, "_send_user_message") as send:
            slack_routes._run_single_answer(values)

        messages = [
            call.args[2][0]["text"]["text"]
            for call in send.call_args_list
            if call.args[2]
        ]
        self.assertIn(
            "Branch is ready. Searching repository context and generating the answer...",
            messages,
        )
        self.assertNotIn(
            "Searching repository context and generating the answer...",
            messages,
        )

    def test_branch_preparation_notice_covers_index_states(self):
        self.assertEqual(
            slack_routes._single_branch_preparation_notice({
                "workspace": "payments-main",
                "index_status": "ready",
            }),
            (
                "Branch is ready. Searching repository context and generating the answer...",
                True,
            ),
        )
        self.assertEqual(
            slack_routes._single_branch_preparation_notice({
                "workspace": None,
                "index_status": "never_indexed",
            })[0],
            "Indexing the selected branch for the first time...",
        )
        self.assertEqual(
            slack_routes._single_branch_preparation_notice({
                "workspace": None,
                "index_status": "failed",
            })[0],
            "Re-indexing the selected branch because the previous index failed or is stale...",
        )
        self.assertEqual(
            slack_routes._single_branch_preparation_notice({
                "workspace": "payments-main",
                "index_status": "ready",
                "freshness_status": "behind",
            })[0],
            "Updating the selected branch index because new commits were found...",
        )

    def test_answer_blocks_include_question_asked(self):
        topic = {
            "repo_name": "Payments",
            "branch": "main",
            "user_type": slack_routes.USER_PRODUCT,
            "question": "How does login work?",
        }
        response = {
            "answer": "Login creates a session.",
            "investigate_deeply_available": False,
        }

        blocks = slack_routes._answer_text_blocks(response, topic)

        self.assertEqual(
            blocks[1]["text"]["text"],
            "*Question asked:*\nHow does login work?",
        )
        self.assertEqual(blocks[2]["text"]["text"], "Login creates a session.")
        action_ids = [item["action_id"] for item in blocks[-1]["elements"]]
        self.assertEqual(action_ids, [
            slack_routes.ACTION_FOLLOW_UP,
            slack_routes.ACTION_NEW,
        ])

    def test_answer_blocks_include_deep_action_when_available(self):
        topic = {
            "repo_name": "Payments",
            "branch": "main",
            "user_type": slack_routes.USER_PRODUCT,
            "question": "How does login work?",
        }
        response = {
            "answer": "Login creates a session.",
            "investigate_deeply_available": True,
        }

        blocks = slack_routes._answer_text_blocks(response, topic)

        action_ids = [item["action_id"] for item in blocks[-1]["elements"]]
        self.assertEqual(action_ids, [
            slack_routes.ACTION_FOLLOW_UP,
            slack_routes.ACTION_DEEP,
            slack_routes.ACTION_NEW,
        ])

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


class SlackEventsTests(unittest.TestCase):
    """@codeatlas mention entry point (Phase A): Events API handshake, auth,
    and dedupe. Repo/branch resolution is covered in SlackMentionJobTests."""

    def setUp(self):
        self.secret = "test-signing-secret"
        self.env = patch.dict(os.environ, {
            "CODEATLAS_SLACK_ENABLED": "true",
            "CODEATLAS_SLACK_SIGNING_SECRET": self.secret,
            "CODEATLAS_SLACK_BOT_TOKEN": "xoxb-test-token",
            "CODEATLAS_SLACK_ALLOWED_TEAM_IDS": "T123",
        })
        self.env.start()
        slack_routes._SEEN_EVENT_IDS.clear()
        slack_routes._SEEN_MESSAGE_KEYS.clear()

    def tearDown(self):
        self.env.stop()

    def _post_events(self, payload: dict):
        body = json.dumps(payload).encode("utf-8")
        headers = {**signed_headers(self.secret, body), "content-type": "application/json"}
        return asyncio.run(slack_routes.slack_events(FakeRequest(body, headers)))

    def test_url_verification_echoes_challenge(self):
        result = self._post_events({"type": "url_verification", "challenge": "abc123"})
        self.assertEqual(result, {"challenge": "abc123"})

    def test_app_mention_dispatches_mention_job(self):
        payload = {
            "type": "event_callback",
            "team_id": "T123",
            "event_id": "Ev1",
            "event": {
                "type": "app_mention",
                "channel": "C1",
                "user": "U1",
                "ts": "111.222",
                "text": "<@B1> how do I go online?",
            },
        }
        with patch.object(slack_routes, "_start_mention_job") as start:
            response = self._post_events(payload)
        self.assertEqual(response.status_code, 200)
        start.assert_called_once()
        _, event = start.call_args.args
        self.assertEqual(event["text"], "<@B1> how do I go online?")

    def test_duplicate_event_id_is_processed_once(self):
        payload = {
            "type": "event_callback",
            "team_id": "T123",
            "event_id": "Ev1",
            "event": {"type": "app_mention", "channel": "C1", "user": "U1", "ts": "1", "text": "hi"},
        }
        with patch.object(slack_routes, "_start_mention_job") as start:
            self._post_events(payload)
            self._post_events(payload)
        start.assert_called_once()

    def test_bot_originated_mention_is_ignored(self):
        payload = {
            "type": "event_callback",
            "team_id": "T123",
            "event_id": "Ev2",
            "event": {
                "type": "app_mention",
                "channel": "C1",
                "user": "U1",
                "ts": "1",
                "text": "hi",
                "bot_id": "B123",
            },
        }
        with patch.object(slack_routes, "_start_mention_job") as start:
            self._post_events(payload)
        start.assert_not_called()

    def test_disallowed_team_is_rejected(self):
        payload = {
            "type": "event_callback",
            "team_id": "T999",
            "event_id": "Ev3",
            "event": {"type": "app_mention", "channel": "C1", "user": "U1", "ts": "1", "text": "hi"},
        }
        with self.assertRaises(HTTPException) as raised:
            self._post_events(payload)
        self.assertEqual(raised.exception.status_code, 403)

    def test_dm_message_dispatches_mention_job_without_a_mention(self):
        payload = {
            "type": "event_callback",
            "team_id": "T123",
            "event_id": "Ev4",
            "event": {
                "type": "message",
                "channel": "D1",
                "channel_type": "im",
                "user": "U1",
                "ts": "1",
                "text": "how do I go online?",
            },
        }
        with patch.object(slack_routes, "_start_mention_job") as start:
            response = self._post_events(payload)
        self.assertEqual(response.status_code, 200)
        start.assert_called_once()

    def test_dm_message_with_subtype_is_ignored(self):
        payload = {
            "type": "event_callback",
            "team_id": "T123",
            "event_id": "Ev5",
            "event": {
                "type": "message",
                "channel": "D1",
                "channel_type": "im",
                "user": "U1",
                "ts": "1",
                "text": "edited text",
                "subtype": "message_changed",
            },
        }
        with patch.object(slack_routes, "_start_mention_job") as start:
            self._post_events(payload)
        start.assert_not_called()

    def test_plain_channel_message_without_mention_is_ignored(self):
        payload = {
            "type": "event_callback",
            "team_id": "T123",
            "event_id": "Ev6",
            "event": {
                "type": "message",
                "channel": "C1",
                "channel_type": "channel",
                "user": "U1",
                "ts": "1",
                "text": "just chatting, not asking codeatlas",
            },
        }
        with patch.object(slack_routes, "_start_mention_job") as start:
            self._post_events(payload)
        start.assert_not_called()

    def test_same_message_is_not_answered_twice_across_event_types(self):
        """A DM @mention can fire both app_mention and message.im for the
        same physical message; only one of them should be answered."""
        mention_payload = {
            "type": "event_callback",
            "team_id": "T123",
            "event_id": "Ev7a",
            "event": {
                "type": "app_mention",
                "channel": "D1",
                "user": "U1",
                "ts": "100.001",
                "text": "<@B1> how do I go online?",
            },
        }
        dm_payload = {
            "type": "event_callback",
            "team_id": "T123",
            "event_id": "Ev7b",
            "event": {
                "type": "message",
                "channel": "D1",
                "channel_type": "im",
                "user": "U1",
                "ts": "100.001",
                "text": "<@B1> how do I go online?",
            },
        }
        with patch.object(slack_routes, "_start_mention_job") as start:
            self._post_events(mention_payload)
            self._post_events(dm_payload)
        start.assert_called_once()


class SlackMentionRoutingTests(unittest.TestCase):
    """Deterministic (Phase A) mention parsing and repo inference."""

    def test_strip_mention_removes_leading_tokens(self):
        self.assertEqual(
            slack_routes._strip_mention("<@U123>  how do I go online?"),
            "how do I go online?",
        )
        self.assertEqual(slack_routes._strip_mention("<@U1><@U2> hi"), "hi")
        self.assertEqual(slack_routes._strip_mention("no mention here"), "no mention here")

    def test_infer_repo_matches_explicit_name_or_slug(self):
        repos = [
            {"slug": "gandalf", "name": "Gandalf Android"},
            {"slug": "frodo", "name": "Frodo Backend"},
        ]
        self.assertEqual(
            slack_routes._infer_repo_from_text("how do I go online in gandalf", repos)["slug"],
            "gandalf",
        )
        self.assertEqual(
            slack_routes._infer_repo_from_text("frodo supply flow question", repos)["slug"],
            "frodo",
        )

    def test_infer_repo_returns_none_when_ambiguous_or_unmatched(self):
        repos = [
            {"slug": "gandalf", "name": "Gandalf Android"},
            {"slug": "frodo", "name": "Frodo Backend"},
        ]
        self.assertIsNone(slack_routes._infer_repo_from_text("how does login work?", repos))

    def test_infer_repo_defaults_to_sole_published_repo(self):
        repos = [{"slug": "gandalf", "name": "Gandalf Android"}]
        self.assertEqual(
            slack_routes._infer_repo_from_text("how does login work?", repos)["slug"],
            "gandalf",
        )

    def test_infer_repo_matches_name_with_punctuated_edges(self):
        """\\b fails to match a candidate that itself starts/ends with
        punctuation (e.g. a repo named "Payments API (EU)"); the fix uses
        (?<!\\w)/(?!\\w) instead, which matches correctly either way."""
        repos = [{"slug": "payments", "name": "Payments API (EU)"}]
        self.assertEqual(
            slack_routes._infer_repo_from_text(
                "does codeatlas support payments api (eu) yet?", repos
            )["slug"],
            "payments",
        )

    def test_strip_mention_removes_mid_sentence_token(self):
        self.assertEqual(
            slack_routes._strip_mention("hey <@B123>, how do I go online?"),
            "hey , how do I go online?",
        )


class SlackMentionJobTests(unittest.TestCase):
    """End-to-end _run_mention_job behavior, with the answer pipeline mocked
    out (it's exercised separately by the existing modal-flow tests)."""

    def setUp(self):
        slack_routes._PENDING_REPO_QUESTIONS.clear()

    def _event(self, text="how do I go online?", **overrides):
        event = {
            "type": "app_mention",
            "channel": "C1",
            "user": "U1",
            "ts": "111.222",
            "text": f"<@B1> {text}",
        }
        event.update(overrides)
        return event

    def _payload(self, event):
        return {"type": "event_callback", "team_id": "T123", "event_id": "Ev1", "event": event}

    def test_answers_when_repo_and_branch_resolve(self):
        repo = {"id": 1, "slug": "gandalf", "name": "Gandalf Android", "status": "published"}
        event = self._event()
        with patch.object(slack_routes.ask_service, "published_repos", return_value=[repo]), \
                patch.object(slack_routes, "_default_branch_name", return_value="main"), \
                patch.object(slack_routes.db, "get_repo_branch_by_name", return_value=None), \
                patch.object(slack_routes.ask_service, "prepare_repo_branch") as prepare, \
                patch.object(slack_routes, "_run_answer_job") as run_job:
            slack_routes._run_mention_job(self._payload(event), event)

        prepare.assert_called_once()
        run_job.assert_called_once()
        values = run_job.call_args.args[0]
        self.assertEqual(values["repo_slug"], "gandalf")
        self.assertEqual(values["branch"], "main")
        self.assertEqual(values["ask_type"], slack_routes.ASK_SINGLE)
        self.assertEqual(values["user_type"], slack_routes.USER_PRODUCT)
        self.assertEqual(values["thread_ts"], "111.222")
        self.assertEqual(values["question"], "how do I go online?")

    def test_skips_redundant_prepare_when_branch_already_approved(self):
        """Once a repo's default branch is already approved, re-preparing it
        on every later question only races the answer pipeline's own
        freshness check and flickers a stale 'still preparing' notice."""
        repo = {"id": 1, "slug": "gandalf", "name": "Gandalf Android", "status": "published"}
        event = self._event()
        with patch.object(slack_routes.ask_service, "published_repos", return_value=[repo]), \
                patch.object(slack_routes, "_default_branch_name", return_value="main"), \
                patch.object(slack_routes.db, "get_repo_branch_by_name", return_value={"id": 5}), \
                patch.object(slack_routes.ask_service, "prepare_repo_branch") as prepare, \
                patch.object(slack_routes, "_run_answer_job") as run_job:
            slack_routes._run_mention_job(self._payload(event), event)

        prepare.assert_not_called()
        run_job.assert_called_once()
        self.assertEqual(run_job.call_args.args[0]["branch"], "main")

    def test_bare_repo_name_reply_resumes_the_original_question(self):
        """After 'which repository?', replying with just the repo name in
        the SAME thread should re-ask the original question against that
        repo, not answer the bare repo name as a new one-word question."""
        repos = [
            {"id": 1, "slug": "gandalf", "name": "Gandalf Android", "status": "published"},
            {"id": 2, "slug": "frodo", "name": "Frodo Backend", "status": "published"},
        ]
        ambiguous_event = self._event(text="how do I go online?", ts="100.001")
        with patch.object(slack_routes.ask_service, "published_repos", return_value=repos), \
                patch.object(slack_routes, "_send_user_message"):
            slack_routes._run_mention_job(self._payload(ambiguous_event), ambiguous_event)

        # A real threaded reply carries thread_ts = the original message's ts.
        reply_event = self._event(text="gandalf", ts="100.002", thread_ts="100.001")
        with patch.object(slack_routes.ask_service, "published_repos", return_value=[repos[0]]), \
                patch.object(slack_routes, "_default_branch_name", return_value="main"), \
                patch.object(slack_routes.db, "get_repo_branch_by_name", return_value={"id": 5}), \
                patch.object(slack_routes.ask_service, "prepare_repo_branch"), \
                patch.object(slack_routes, "_run_answer_job") as run_job:
            slack_routes._run_mention_job(self._payload(reply_event), reply_event)

        run_job.assert_called_once()
        values = run_job.call_args.args[0]
        self.assertEqual(values["repo_slug"], "gandalf")
        self.assertEqual(values["question"], "how do I go online?")

    def test_bare_repo_name_reply_in_different_thread_is_not_resumed(self):
        """Two concurrent ambiguous questions in the same channel/DM must
        not clobber each other's pending entry: a bare repo-name reply in a
        DIFFERENT thread than the one that asked 'which repository?' should
        not resume that other thread's question."""
        repos = [
            {"id": 1, "slug": "gandalf", "name": "Gandalf Android", "status": "published"},
            {"id": 2, "slug": "frodo", "name": "Frodo Backend", "status": "published"},
        ]
        ambiguous_event = self._event(text="how do I go online?", ts="100.001")
        with patch.object(slack_routes.ask_service, "published_repos", return_value=repos), \
                patch.object(slack_routes, "_send_user_message"):
            slack_routes._run_mention_job(self._payload(ambiguous_event), ambiguous_event)

        # A bare "gandalf" in an unrelated thread is its own new question.
        reply_event = self._event(text="gandalf", ts="200.001")
        with patch.object(slack_routes.ask_service, "published_repos", return_value=[repos[0]]), \
                patch.object(slack_routes, "_default_branch_name", return_value="main"), \
                patch.object(slack_routes.db, "get_repo_branch_by_name", return_value={"id": 5}), \
                patch.object(slack_routes.ask_service, "prepare_repo_branch"), \
                patch.object(slack_routes, "_run_answer_job") as run_job:
            slack_routes._run_mention_job(self._payload(reply_event), reply_event)

        run_job.assert_called_once()
        self.assertEqual(run_job.call_args.args[0]["question"], "gandalf")

    def test_bare_repo_name_without_pending_question_is_answered_literally(self):
        """No prior 'which repository?' prompt in this channel/user means a
        bare repo-name message is just answered as-is (existing behavior)."""
        repo = {"id": 1, "slug": "gandalf", "name": "Gandalf Android", "status": "published"}
        event = self._event(text="gandalf")
        with patch.object(slack_routes.ask_service, "published_repos", return_value=[repo]), \
                patch.object(slack_routes, "_default_branch_name", return_value="main"), \
                patch.object(slack_routes.db, "get_repo_branch_by_name", return_value={"id": 5}), \
                patch.object(slack_routes.ask_service, "prepare_repo_branch"), \
                patch.object(slack_routes, "_run_answer_job") as run_job:
            slack_routes._run_mention_job(self._payload(event), event)

        run_job.assert_called_once()
        self.assertEqual(run_job.call_args.args[0]["question"], "gandalf")

    def test_concurrent_prepare_race_is_not_reported_as_a_failure(self):
        """Two near-simultaneous first-time questions about the same repo
        can both see 'not approved yet' and both call prepare_repo_branch;
        if a concurrent call already won and created the branch, the loser
        should proceed normally instead of erroring out to the user."""
        repo = {"id": 1, "slug": "gandalf", "name": "Gandalf Android", "status": "published"}
        event = self._event()
        # First call (inside prepare_repo_branch) says "not approved yet";
        # by the time we recover from the raised exception, a concurrent
        # call has already created it.
        branch_lookup = iter([None, {"id": 5}])
        with patch.object(slack_routes.ask_service, "published_repos", return_value=[repo]), \
                patch.object(slack_routes, "_default_branch_name", return_value="main"), \
                patch.object(
                    slack_routes.db, "get_repo_branch_by_name",
                    side_effect=lambda *a, **k: next(branch_lookup),
                ), \
                patch.object(
                    slack_routes.ask_service, "prepare_repo_branch",
                    side_effect=RuntimeError("UNIQUE constraint failed"),
                ), \
                patch.object(slack_routes, "_run_answer_job") as run_job, \
                patch.object(slack_routes, "_send_user_message") as send:
            slack_routes._run_mention_job(self._payload(event), event)

        send.assert_not_called()
        run_job.assert_called_once()
        self.assertEqual(run_job.call_args.args[0]["branch"], "main")

    def test_unhandled_error_in_repo_inference_notifies_user_and_does_not_raise(self):
        """A previously-unguarded exception (e.g. a transient DB error from
        published_repos()) must not vanish silently in the executor thread
        with no log and no reply."""
        event = self._event()
        with patch.object(
                slack_routes.ask_service, "published_repos",
                side_effect=RuntimeError("db unavailable"),
        ), patch.object(slack_routes, "_send_user_message") as send:
            slack_routes._run_mention_job(self._payload(event), event)  # must not raise

        send.assert_called_once()
        blocks = send.call_args.args[2]
        self.assertIn("db unavailable", blocks[0]["text"]["text"])

    def test_empty_question_prompts_for_one(self):
        event = self._event(text="")
        with patch.object(slack_routes, "_send_user_message") as send:
            slack_routes._run_mention_job(self._payload(event), event)
        send.assert_called_once()
        blocks = send.call_args.args[2]
        self.assertIn("Ask me something", blocks[0]["text"]["text"])

    def test_ambiguous_repo_asks_user_to_specify(self):
        repos = [
            {"id": 1, "slug": "gandalf", "name": "Gandalf Android", "status": "published"},
            {"id": 2, "slug": "frodo", "name": "Frodo Backend", "status": "published"},
        ]
        event = self._event(text="how does login work?")
        with patch.object(slack_routes.ask_service, "published_repos", return_value=repos), \
                patch.object(slack_routes, "_send_user_message") as send:
            slack_routes._run_mention_job(self._payload(event), event)
        send.assert_called_once()
        blocks = send.call_args.args[2]
        self.assertIn("couldn't tell which repository", blocks[0]["text"]["text"])


class SlackThreadReplyTests(unittest.TestCase):
    """_send_user_message must keep every existing surface working (Slack
    routing regression guard) while adding the mention flow's threaded reply."""

    def test_thread_reply_used_when_mention_originated(self):
        with patch.object(slack_routes, "_slack_api") as api:
            slack_routes._send_user_message(
                {"channel_id": "C1", "thread_ts": "111.222"},
                "hello",
                [{"type": "section", "text": {"type": "mrkdwn", "text": "hi"}}],
            )
        api.assert_called_once()
        method, payload = api.call_args.args
        self.assertEqual(method, "chat.postMessage")
        self.assertEqual(payload["channel"], "C1")
        self.assertEqual(payload["thread_ts"], "111.222")

    def test_ephemeral_fallback_unchanged_when_no_thread_or_response_url(self):
        with patch.object(slack_routes, "_post_ephemeral") as ephemeral:
            slack_routes._send_user_message(
                {"channel_id": "C1", "slack_user_id": "U1"},
                "hello",
            )
        ephemeral.assert_called_once()


class SlackAnswerFormattingTests(unittest.TestCase):
    """Slack renders mrkdwn, not Markdown: it has no headings and uses a single
    asterisk for bold, so an answer posted verbatim showed literal ## and **."""

    def test_headings_rules_bullets_and_bold_become_mrkdwn(self):
        converted = slack_routes.markdown_to_mrkdwn(
            "## Answer\n\n---\n\n### Step 1\n\n"
            "The **CTA** text is not hardcoded.\n"
            "- **Early Withdrawal** - withdraw early\n"
            "See [the docs](https://example.com/docs)."
        )
        self.assertIn("*Answer*", converted)
        self.assertIn("*Step 1*", converted)
        self.assertIn("The *CTA* text", converted)
        self.assertIn("• *Early Withdrawal*", converted)
        self.assertIn("<https://example.com/docs|the docs>", converted)
        self.assertNotIn("##", converted)
        self.assertNotIn("**", converted)
        self.assertNotIn("---", converted)

    def test_code_is_passed_through_untouched(self):
        converted = slack_routes.markdown_to_mrkdwn(
            "before\n```python\n# a comment\nx = a ** b\n```\n"
            "inline `foo(**kwargs)` and `__init__` stay literal"
        )
        self.assertIn("# a comment", converted)
        self.assertIn("x = a ** b", converted)
        self.assertIn("`foo(**kwargs)`", converted)
        self.assertIn("`__init__`", converted)

    def test_dunder_names_are_not_treated_as_emphasis(self):
        converted = slack_routes.markdown_to_mrkdwn("the __init__ method runs first")
        self.assertIn("__init__", converted)

    def test_chunks_split_on_line_boundaries(self):
        text = "\n".join(f"line {index} with *bold {index}* text" for index in range(200))
        chunks = slack_routes._mrkdwn_chunks(text, limit=300)
        self.assertTrue(all(len(chunk) <= 300 for chunk in chunks))
        # A chunk boundary must not land inside a bold pair.
        self.assertTrue(all(chunk.count("*") % 2 == 0 for chunk in chunks))
        self.assertEqual("\n".join(chunks).replace("\n", ""), text.replace("\n", ""))

    def test_oversized_single_line_is_still_split(self):
        self.assertEqual(len(slack_routes._mrkdwn_chunks("x" * 700, limit=300)), 3)
