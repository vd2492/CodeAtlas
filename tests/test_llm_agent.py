import json
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from app import main
from app.conversations import ConversationStore
from app.llm import client


class FakeResponse:
    def __init__(self, payload, status_code=200, text="", headers=None):
        self.payload = payload
        self.status_code = status_code
        self.text = text
        self.headers = headers or {}

    def json(self):
        return self.payload


class FakeToolbox:
    def __init__(self, pre_search_instruction="", response_style_instruction=""):
        self.trace = []
        self.config = SimpleNamespace(
            pre_search_instruction=pre_search_instruction
        )
        self.response_style_instruction = response_style_instruction

    def call(self, name, arguments):
        self.trace.append({"tool": name, "arguments": arguments, "result": {"ok": True}})
        return json.dumps({"ok": True, "evidence": "src/auth.py:L1-L3"})


class AgentLoopTests(unittest.TestCase):
    def test_collects_token_usage_across_provider_requests(self):
        responses = [
            FakeResponse({
                "usage": {
                    "prompt_tokens": 100,
                    "completion_tokens": 20,
                    "total_tokens": 120,
                },
            }),
            FakeResponse({
                "usage": {
                    "input_tokens": 40,
                    "output_tokens": 10,
                    "cache_read_input_tokens": 30,
                },
            }),
        ]
        with patch(
            "app.llm.client.requests.post",
            side_effect=responses,
        ), client.collect_token_usage() as usage:
            client._post_with_retries("https://example.test/first")
            client._post_with_retries("https://example.test/second")

        self.assertEqual(
            client.token_usage_payload(usage),
            {
                "input_tokens": 140,
                "output_tokens": 30,
                "total_tokens": 200,
                "cached_input_tokens": 30,
                "requests": 2,
                "available": True,
            },
        )

    def test_token_usage_is_unavailable_when_provider_omits_it(self):
        with patch(
            "app.llm.client.requests.post",
            return_value=FakeResponse({"choices": []}),
        ), client.collect_token_usage() as usage:
            client._post_with_retries("https://example.test")

        self.assertEqual(client.token_usage_payload(usage)["available"], False)

    def test_openai_chat_sends_image_content_parts(self):
        response = FakeResponse({
            "choices": [{
                "message": {"content": "The screenshot shows login."}
            }]
        })
        context = {"llm_context_preview": {"question": "What is visible?"}}
        image = {
            "name": "screen.png",
            "mime_type": "image/png",
            "data": "iVBORw0KGgo=",
        }
        with patch("app.llm.client.requests.post", return_value=response) as post:
            answer = client._openai_chat(
                "https://example.test/v1",
                "key",
                "gpt-vision",
                context,
                [image],
            )

        self.assertEqual(answer, "The screenshot shows login.")
        payload = post.call_args.kwargs["json"]
        user_content = payload["messages"][1]["content"]
        self.assertEqual(user_content[0]["type"], "text")
        self.assertIn("attached 1 image", user_content[0]["text"])
        self.assertEqual(user_content[1]["type"], "image_url")
        self.assertEqual(
            user_content[1]["image_url"]["url"],
            "data:image/png;base64,iVBORw0KGgo=",
        )

    def test_anthropic_chat_sends_image_blocks(self):
        response = FakeResponse({
            "content": [{"type": "text", "text": "The screenshot shows login."}]
        })
        context = {"llm_context_preview": {"question": "What is visible?"}}
        image = {
            "name": "screen.webp",
            "mime_type": "image/webp",
            "data": "UklGRg==",
        }
        with patch("app.llm.client.requests.post", return_value=response) as post:
            answer = client._anthropic_chat(
                "https://anthropic.example.test",
                "key",
                "claude-vision",
                context,
                [image],
            )

        self.assertEqual(answer, "The screenshot shows login.")
        payload = post.call_args.kwargs["json"]
        user_content = payload["messages"][0]["content"]
        self.assertEqual(user_content[0]["type"], "text")
        self.assertIn("attached 1 image", user_content[0]["text"])
        self.assertEqual(user_content[1]["type"], "image")
        self.assertEqual(user_content[1]["source"]["media_type"], "image/webp")
        self.assertEqual(user_content[1]["source"]["data"], "UklGRg==")

    def test_image_rejection_uses_clear_error_message(self):
        response = FakeResponse(
            {"error": "unsupported"},
            status_code=400,
            text="This model does not support image input.",
        )
        context = {"llm_context_preview": {"question": "What is visible?"}}
        with patch("app.llm.client.requests.post", return_value=response):
            with self.assertRaises(client.ImageInputUnsupported) as raised:
                client._openai_chat(
                    "https://example.test/v1",
                    "key",
                    "text-only-model",
                    context,
                    [{
                        "name": "screen.png",
                        "mime_type": "image/png",
                        "data": "iVBORw0KGgo=",
                    }],
                )

        self.assertEqual(str(raised.exception), client.IMAGE_INPUT_UNSUPPORTED_MESSAGE)

    def test_provider_post_retries_temporary_failure_then_succeeds(self):
        unavailable = FakeResponse(
            {"error": "busy"},
            status_code=503,
            text="busy",
            headers={"Retry-After": "0"},
        )
        success = FakeResponse({"ok": True})
        with patch.object(client, "PROVIDER_RETRIES", 2), patch(
            "app.llm.client.requests.post",
            side_effect=[unavailable, success],
        ) as post, patch("app.llm.client.time.sleep") as sleep:
            response = client._post_with_retries("https://example.test")

        self.assertIs(response, success)
        self.assertEqual(post.call_count, 2)
        sleep.assert_called_once()

    def test_provider_post_does_not_retry_authentication_failure(self):
        unauthorized = FakeResponse(
            {"error": "invalid key"},
            status_code=401,
            text="invalid key",
        )
        with patch.object(client, "PROVIDER_RETRIES", 2), patch(
            "app.llm.client.requests.post",
            return_value=unauthorized,
        ) as post, patch("app.llm.client.time.sleep") as sleep:
            response = client._post_with_retries("https://example.test")

        self.assertIs(response, unauthorized)
        post.assert_called_once()
        sleep.assert_not_called()

    def test_provider_post_retries_connection_failure(self):
        success = FakeResponse({"ok": True})
        with patch.object(client, "PROVIDER_RETRIES", 1), patch(
            "app.llm.client.requests.post",
            side_effect=[client.requests.ConnectionError("unavailable"), success],
        ) as post, patch("app.llm.client.time.sleep") as sleep:
            response = client._post_with_retries("https://example.test")

        self.assertIs(response, success)
        self.assertEqual(post.call_count, 2)
        sleep.assert_called_once()

    def test_openai_agent_executes_tool_then_answers(self):
        toolbox = FakeToolbox(
            "Map customer-facing terms to canonical symbols before searching."
        )
        responses = [
            FakeResponse({
                "choices": [{
                    "message": {
                        "role": "assistant",
                        "content": None,
                        "tool_calls": [{
                            "id": "call_1",
                            "type": "function",
                            "function": {
                                "name": "search_code",
                                "arguments": '{"query":"login"}',
                            },
                        }],
                    }
                }]
            }),
            FakeResponse({
                "choices": [{
                    "message": {
                        "role": "assistant",
                        "content": "Login is handled in src/auth.py:L1-L3.",
                    }
                }]
            }),
        ]
        with patch("app.llm.client.requests.post", side_effect=responses) as post:
            result = client._openai_agent(
                "https://example.test/v1",
                "key",
                "model",
                "How does login work?",
                toolbox,
                client.TOOL_DEFINITIONS,
            )
        self.assertEqual(result["tool_calls"], 1)
        self.assertEqual(result["rounds"], 2)
        self.assertIn("src/auth.py", result["answer"])
        first_messages = post.call_args_list[0].kwargs["json"]["messages"]
        self.assertIn(
            "Map customer-facing terms to canonical symbols",
            first_messages[0]["content"],
        )
        second_messages = post.call_args_list[1].kwargs["json"]["messages"]
        self.assertEqual(second_messages[-1]["role"], "tool")
        self.assertEqual(second_messages[-1]["tool_call_id"], "call_1")

    def test_openai_agent_stops_and_returns_ask_user_question(self):
        toolbox = FakeToolbox()
        responses = [
            FakeResponse({
                "choices": [{
                    "message": {
                        "role": "assistant",
                        "content": None,
                        "tool_calls": [{
                            "id": "call_1",
                            "type": "function",
                            "function": {
                                "name": "search_code",
                                "arguments": '{"query":"validation"}',
                            },
                        }],
                    }
                }]
            }),
            FakeResponse({
                "choices": [{
                    "message": {
                        "role": "assistant",
                        "content": None,
                        "tool_calls": [{
                            "id": "call_2",
                            "type": "function",
                            "function": {
                                "name": "ask_user",
                                "arguments": (
                                    '{"question":"Did you mean the habit flow\'s '
                                    'validation or the revision flow\'s validation?"}'
                                ),
                            },
                        }],
                    }
                }]
            }),
        ]
        with patch("app.llm.client.requests.post", side_effect=responses) as post:
            result = client._openai_agent(
                "https://example.test/v1",
                "key",
                "model",
                "How does validation work?",
                toolbox,
                client.TOOL_DEFINITIONS,
            )
        self.assertEqual(post.call_count, 2)
        self.assertEqual(result["rounds"], 2)
        self.assertEqual(result["tool_calls"], 1)
        self.assertIn("habit flow", result["answer"])
        self.assertEqual(toolbox.trace[-1]["tool"], "ask_user")
        self.assertTrue(toolbox.trace[-1]["result"]["needs_clarification"])
        self.assertTrue(result["needs_clarification"])

    def test_answer_response_surfaces_needs_clarification_flag(self):
        clarifying = main._answer_response(
            "How does validation work?",
            {
                "answer": "Did you mean the habit flow or the revision flow?",
                "provider_used": "shared:mimo-v2.5",
                "retrieval_mode": "agentic",
                "rounds": 2,
                "tool_calls": 1,
                "needs_clarification": True,
            },
            context={},
            workspace="default",
        )
        self.assertTrue(clarifying["needs_clarification"])

        normal = main._answer_response(
            "How does login work?",
            {
                "answer": "Login is handled in src/auth.py.",
                "provider_used": "shared:mimo-v2.5",
                "retrieval_mode": "agentic",
                "rounds": 2,
                "tool_calls": 1,
            },
            context={},
            workspace="default",
        )
        self.assertFalse(normal["needs_clarification"])

    def test_anthropic_agent_uses_tool_result_blocks(self):
        toolbox = FakeToolbox()
        responses = [
            FakeResponse({
                "content": [{
                    "type": "tool_use",
                    "id": "toolu_1",
                    "name": "read_file",
                    "input": {"path": "src/auth.py"},
                }],
                "stop_reason": "tool_use",
            }),
            FakeResponse({
                "content": [{
                    "type": "text",
                    "text": "Verified in src/auth.py:L1-L3.",
                }],
                "stop_reason": "end_turn",
            }),
        ]
        with patch("app.llm.client.requests.post", side_effect=responses) as post:
            result = client._anthropic_agent(
                "https://api.anthropic.test",
                "key",
                "model",
                "How does login work?",
                toolbox,
                client.TOOL_DEFINITIONS,
            )
        self.assertEqual(result["tool_calls"], 1)
        second_messages = post.call_args_list[1].kwargs["json"]["messages"]
        self.assertEqual(second_messages[-1]["role"], "user")
        self.assertEqual(second_messages[-1]["content"][0]["type"], "tool_result")
        self.assertEqual(second_messages[-1]["content"][0]["tool_use_id"], "toolu_1")

    def test_anthropic_prompt_cache_is_enabled_only_for_official_api(self):
        self.assertEqual(
            client._anthropic_cache_settings("https://api.anthropic.com"),
            {"cache_control": {"type": "ephemeral"}},
        )
        self.assertEqual(
            client._anthropic_cache_settings("https://anthropic.example.test"),
            {},
        )

    def test_fast_follow_up_request_omits_tools_and_uses_small_output_budget(self):
        response = FakeResponse({
            "choices": [{
                "message": {
                    "role": "assistant",
                    "content": "Cached answer from src/auth.py:L8-L12.",
                }
            }]
        })
        with patch("app.llm.client.requests.post", return_value=response) as post:
            answer = client._openai_fast_follow_up(
                "https://example.test/v1",
                "key",
                "model",
                {},
                "What happens when it fails?",
                "Verified evidence",
            )

        payload = post.call_args.kwargs["json"]
        self.assertEqual(answer, "Cached answer from src/auth.py:L8-L12.")
        self.assertNotIn("tools", payload)
        self.assertEqual(payload["max_tokens"], client.FOLLOW_UP_MAX_TOKENS)

    def test_fast_follow_up_sentinel_requests_full_evidence(self):
        response = FakeResponse({
            "choices": [{
                "message": {
                    "role": "assistant",
                    "content": client.FOLLOW_UP_NEEDS_EVIDENCE,
                }
            }]
        })
        with patch("app.llm.client.requests.post", return_value=response):
            with self.assertRaises(client.FollowUpNeedsEvidence):
                client._openai_fast_follow_up(
                    "https://example.test/v1",
                    "key",
                    "model",
                    {},
                    "Question",
                    "Insufficient evidence",
                )

    def test_fast_follow_up_uses_selected_shared_provider(self):
        shared = {
            "provider": "openai_compatible",
            "base_url": "https://example.test/v1",
            "api_key": "key",
            "model": "mimo-v2.5",
        }
        with patch.object(
            client, "_configured_shared_creds", return_value=shared
        ), patch.object(
            client,
            "_call_fast_follow_up_with_creds",
            return_value="Fast grounded answer.",
        ) as call:
            result = client.generate_fast_follow_up(
                {},
                "Verified evidence",
                llm_mode="mimo",
                question="Follow-up?",
            )

        call.assert_called_once()
        self.assertEqual(result["provider_used"], "shared:mimo-v2.5")
        self.assertEqual(result["retrieval_mode"], "follow_up_cache")
        self.assertEqual(result["tool_calls"], 0)

    def test_fast_follow_up_preserves_product_team_answer_style(self):
        prompt = client._fast_follow_up_system_prompt({
            "response_style_instruction": client.PRODUCT_TEAM_RESPONSE_INSTRUCTION,
        })
        self.assertIn("Do not include technical terms", prompt)
        self.assertIn("do not expose technical evidence", prompt)
        self.assertNotIn("Preserve valid source citations", prompt)

    def test_ollama_agent_executes_object_arguments(self):
        toolbox = FakeToolbox()
        responses = [
            FakeResponse({
                "message": {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [{
                        "type": "function",
                        "function": {
                            "name": "search_code",
                            "arguments": {"query": "login"},
                        },
                    }],
                }
            }),
            FakeResponse({
                "message": {
                    "role": "assistant",
                    "content": "Verified in src/auth.py:L1-L3.",
                }
            }),
        ]
        with patch("app.llm.client.requests.post", side_effect=responses) as post:
            result = client._ollama_agent(
                "http://localhost:11434",
                "model",
                "How does login work?",
                toolbox,
                client.TOOL_DEFINITIONS,
            )
        self.assertEqual(result["tool_calls"], 1)
        second_messages = post.call_args_list[1].kwargs["json"]["messages"]
        self.assertEqual(second_messages[-1]["role"], "tool")
        self.assertEqual(second_messages[-1]["tool_name"], "search_code")

    def test_ollama_mode_is_disabled_placeholder_by_default(self):
        with patch.object(client, "OLLAMA_ENABLED", False), patch.object(
            client, "_ollama_available"
        ) as available:
            with self.assertRaisesRegex(RuntimeError, "currently disabled"):
                client.generate(
                    {"llm_context_preview": {"question": "How does login work?"}},
                    llm_mode="ollama",
                )
        available.assert_not_called()

    def test_provider_falls_back_to_one_shot_when_model_skips_tools(self):
        toolbox = FakeToolbox()
        responses = [
            FakeResponse({
                "choices": [{
                    "message": {"role": "assistant", "content": "A guess."}
                }]
            }),
            FakeResponse({
                "choices": [{
                    "message": {"role": "assistant", "content": "Grounded fallback answer."}
                }]
            }),
        ]
        creds = {
            "provider": "openai",
            "base_url": "https://example.test/v1",
            "api_key": "key",
            "model": "model",
        }
        context = {"llm_context_preview": {"question": "How does login work?"}}
        with patch(
            "app.llm.client.socket.getaddrinfo",
            return_value=[(None, None, None, None, ("93.184.216.34", 443))],
        ), patch("app.llm.client.requests.post", side_effect=responses):
            result = client._attempt_with_creds(
                creds,
                context,
                question="How does login work?",
                toolbox=toolbox,
            )
        self.assertEqual(result["retrieval_mode"], "one_shot")
        self.assertIn("without using repository tools", result["agent_fallback_reason"])
        self.assertEqual(result["answer"], "Grounded fallback answer.")

    def test_follow_up_can_answer_from_cached_evidence_without_tool_call(self):
        toolbox = FakeToolbox()
        response = FakeResponse({
            "choices": [{
                "message": {
                    "role": "assistant",
                    "content": "It rejects the request in src/auth.py:L8-L12.",
                }
            }]
        })
        with patch("app.llm.client.requests.post", return_value=response) as post:
            result = client._openai_agent(
                "https://example.test/v1",
                "key",
                "model",
                "What happens when it fails?",
                toolbox,
                client.TOOL_DEFINITIONS,
                agent_context="Login validation is in src/auth.py:L1-L12.",
                require_tool=False,
            )

        self.assertEqual(result["tool_calls"], 0)
        self.assertEqual(result["rounds"], 1)
        self.assertEqual(toolbox.trace, [])
        user_prompt = post.call_args.kwargs["json"]["messages"][1]["content"]
        self.assertIn("Previously verified", user_prompt)
        self.assertIn("src/auth.py:L1-L12", user_prompt)

    def test_product_team_style_is_added_to_agent_and_one_shot_prompts(self):
        toolbox = FakeToolbox(
            response_style_instruction=client.PRODUCT_TEAM_RESPONSE_INSTRUCTION
        )
        agent_prompt = client._agent_system_prompt(toolbox)
        one_shot_prompt = client.build_prompt({
            "llm_context_preview": {"question": "How does checkout work?"},
            "response_style_instruction": client.PRODUCT_TEAM_RESPONSE_INSTRUCTION,
        })

        self.assertIn("everyday language only", agent_prompt)
        self.assertIn("Do not include technical terms", agent_prompt)
        self.assertIn("everyday language only", one_shot_prompt)
        self.assertNotIn("Include file paths and line numbers", one_shot_prompt)
        self.assertNotIn("Cite concrete claims", agent_prompt)

    def test_product_team_comparison_prompt_hides_source_locations(self):
        toolbox = FakeToolbox(
            response_style_instruction=client.PRODUCT_TEAM_RESPONSE_INSTRUCTION
        )
        toolbox.comparison_mode = True
        agent_prompt = client._agent_system_prompt(toolbox)
        one_shot_prompt = client.build_prompt({
            "comparison_mode": True,
            "response_style_instruction": client.PRODUCT_TEAM_RESPONSE_INSTRUCTION,
            "llm_context_preview": {
                "question": "Compare checkout",
                "branches": [
                    {"label": "Branch A", "name": "main"},
                    {"label": "Branch B", "name": "release"},
                ],
            },
        })

        self.assertIn("product-team reader", agent_prompt)
        self.assertIn("Do not include technical terms", one_shot_prompt)
        self.assertNotIn("Cite source files and line numbers", one_shot_prompt)
        self.assertNotIn("branch label plus file path", agent_prompt)
        self.assertEqual(
            client._system_prompt({
                "comparison_mode": True,
                "response_style_instruction": client.PRODUCT_TEAM_RESPONSE_INSTRUCTION,
            }),
            client.PRODUCT_TEAM_COMPARISON_SYSTEM_PROMPT,
        )

    def test_product_team_answer_guard_removes_source_references(self):
        answer = (
            "The login screen validates the user in src/auth.py:L1-L3 and "
            "then continues from LoginViewModel.kt L12-L20."
        )

        cleaned = client._final_answer(
            answer,
            "test",
            {"response_style_instruction": client.PRODUCT_TEAM_RESPONSE_INSTRUCTION},
        )

        self.assertNotIn("src/auth.py", cleaned)
        self.assertNotIn("LoginViewModel.kt", cleaned)
        self.assertNotIn("L12", cleaned)
        self.assertIn("login screen", cleaned)

    def test_product_team_suffix_is_appended_only_to_llm_facing_question(self):
        context = {
            "llm_context_preview": {"question": "How does checkout work?"}
        }
        toolbox = SimpleNamespace()
        with patch.object(
            main, "build_context", return_value=context
        ), patch.object(
            main, "RepositoryToolbox", return_value=toolbox
        ), patch.object(
            main,
            "generate",
            return_value={
                "answer": "Customers can complete their purchase.",
                "provider_used": "test",
            },
        ) as generate, patch.object(
            main.db, "get_repo_branch_by_workspace", return_value=None
        ):
            result = main.answer_question(
                "How does checkout work?",
                workspace="sample",
                user_type="product_team",
            )

        llm_question = generate.call_args.kwargs["question"]
        self.assertTrue(llm_question.endswith(client.PRODUCT_TEAM_QUERY_SUFFIX))
        self.assertEqual(context["llm_context_preview"]["question"], llm_question)
        self.assertEqual(result["question"], "How does checkout work?")

    def test_dev_team_question_is_not_modified(self):
        context = {
            "llm_context_preview": {"question": "How does checkout work?"}
        }
        toolbox = SimpleNamespace()
        with patch.object(
            main, "build_context", return_value=context
        ), patch.object(
            main, "RepositoryToolbox", return_value=toolbox
        ), patch.object(
            main,
            "generate",
            return_value={"answer": "Technical answer.", "provider_used": "test"},
        ) as generate, patch.object(
            main.db, "get_repo_branch_by_workspace", return_value=None
        ):
            main.answer_question(
                "How does checkout work?",
                workspace="sample",
                user_type="dev_team",
            )

        self.assertEqual(
            generate.call_args.kwargs["question"],
            "How does checkout work?",
        )

    def test_flow_summary_questions_are_audience_aware(self):
        product_question = main.flow_summary_question(
            "CreatebadgeView flow",
            "product_team",
        )
        developer_question = main.flow_summary_question(
            "CreatebadgeView flow",
            "dev_team",
        )

        self.assertIn("brief product-friendly summary", product_question)
        self.assertIn("Do not include technical terms", product_question)
        self.assertIn("Do not repeat the internal flow identifier", product_question)
        self.assertIn("brief developer-focused summary", developer_question)
        self.assertIn("components, methods, files, and endpoints", developer_question)
        self.assertIn("do not return a raw inventory", developer_question)

    def test_product_flow_summary_prompt_keeps_technical_evidence_internal(self):
        toolbox = FakeToolbox(
            response_style_instruction=client.PRODUCT_TEAM_RESPONSE_INSTRUCTION
        )
        toolbox.product_flow_summary = True
        agent_prompt = client._agent_system_prompt(toolbox)
        one_shot_prompt = client.build_prompt({
            "llm_context_preview": {
                "question": "Summarize CreatebadgeView flow",
            },
            "response_style_instruction": client.PRODUCT_TEAM_RESPONSE_INSTRUCTION,
            "product_flow_summary": True,
        })

        self.assertIn("keep all implementation details private", agent_prompt)
        self.assertNotIn("Cite concrete claims", agent_prompt)
        self.assertIn("brief summary", one_shot_prompt)
        self.assertNotIn("Include file paths and line numbers", one_shot_prompt)
        self.assertEqual(
            client._system_prompt({"product_flow_summary": True}),
            client.PRODUCT_FLOW_SUMMARY_SYSTEM_PROMPT,
        )

    def test_flow_summary_endpoint_uses_authenticated_user_type(self):
        request = main.FlowSummaryRequest(llm_mode="mimo")
        generated = {
            "question": "internal prompt",
            "answer": "A brief flow summary.",
            "provider_used": "test",
        }
        with patch.object(
            main, "enforce_rate_limit"
        ), patch.object(
            main, "enforce_strict_branch_freshness"
        ), patch.object(
            main,
            "_flow",
            return_value={
                "topic": "createbadgeview",
                "title": "CreatebadgeView flow",
            },
        ), patch.object(
            main.db,
            "get_repo_by_workspace",
            return_value={"allow_shared_fallback": 1},
        ), patch.object(
            main,
            "load_user_llm",
            return_value={"provider": "openai", "api_key": "saved"},
        ), patch.object(
            main,
            "generate",
            return_value=generated,
        ) as generate:
            result = main.flow_summary_endpoint(
                "createbadgeview",
                request,
                workspace="sample",
                user={"id": 7, "user_type": "product_team"},
            )

        context = generate.call_args.args[0]
        question = context["llm_context_preview"]["question"]
        self.assertIn("brief product-friendly summary", question)
        self.assertTrue(context["product_flow_summary"])
        self.assertEqual(generate.call_args.kwargs["llm_mode"], "mimo")
        self.assertIsNone(generate.call_args.kwargs.get("toolbox"))
        self.assertEqual(result["question"], "CreatebadgeView flow")
        self.assertEqual(result["flow_topic"], "createbadgeview")

    def test_flow_summary_context_uses_selected_flow_evidence(self):
        flow_data = {
            "topic": "createbadgeview",
            "title": "CreatebadgeView flow",
            "high_level_flow": "Entry point -> related components -> data/persistence",
            "entry_points": [
                {
                    "name": "CreatebadgeView",
                    "node": "ui_createbadgeview_createbadgeview",
                    "source_file": "app/src/main/java/CreatebadgeView.kt",
                    "source_location": "L12",
                },
            ],
            "viewmodels": [
                {
                    "name": "BadgeViewModel",
                    "node": "viewmodel_badgeviewmodel_badgeviewmodel",
                    "source_file": "app/src/main/java/BadgeViewModel.kt",
                    "source_location": "L20",
                },
            ],
            "repositories": [],
            "important_methods": [],
        }

        with patch.object(main, "_safe_source_root", return_value=Path("/missing")):
            context = main.build_flow_summary_context(
                "Summarize the flow.",
                flow_data,
                "sample",
                "dev_team",
            )

        preview = context["llm_context_preview"]
        self.assertEqual(preview["flow"]["title"], "CreatebadgeView flow")
        self.assertEqual(
            [node["name"] for node in context["context_nodes"]],
            ["CreatebadgeView", "BadgeViewModel"],
        )
        self.assertEqual(context["source_hits"], [])

    def test_answer_compare_combines_context_from_both_branches(self):
        left = {
            "repo": {
                "id": 1,
                "name": "Repo",
                "slug": "repo",
                "allow_shared_fallback": 1,
            },
            "branch": {"id": 11, "name": "main"},
            "workspace": "repo-main-workspace",
        }
        right = {
            "repo": {
                "id": 1,
                "name": "Repo",
                "slug": "repo",
                "allow_shared_fallback": 1,
            },
            "branch": {"id": 12, "name": "release"},
            "workspace": "repo-release-workspace",
        }
        contexts = {
            "repo-main-workspace": {
                "context_nodes": [{"name": "LoginA"}],
                "context_relations": [],
                "source_hits": [{"path": "a/login.py", "snippets": []}],
                "llm_context_preview": {"question": "Compare login", "nodes": []},
            },
            "repo-release-workspace": {
                "context_nodes": [{"name": "LoginB"}],
                "context_relations": [],
                "source_hits": [{"path": "b/login.py", "snippets": []}],
                "llm_context_preview": {"question": "Compare login", "nodes": []},
            },
        }

        with patch.object(
            main,
            "build_context",
            side_effect=lambda question, limit, workspace: contexts[workspace],
        ) as build_context, patch.object(
            main,
            "repository_version_payload",
            return_value=None,
        ), patch.object(
            main,
            "generate",
            return_value={
                "answer": "main and release handle login differently.",
                "provider_used": "shared:mimo-v2.5",
            },
        ) as generate:
            result = main.answer_compare(
                "Compare login",
                left,
                right,
                llm_mode="mimo",
            )

        self.assertEqual(build_context.call_count, 2)
        generated_context = generate.call_args.args[0]
        self.assertTrue(generated_context["comparison_mode"])
        self.assertEqual(
            [repo["name"] for repo in generated_context["comparison_repositories"]],
            ["main", "release"],
        )
        self.assertEqual(
            [branch["branch_id"] for branch in generated_context["comparison_repositories"]],
            [11, 12],
        )
        self.assertEqual(result["retrieval_mode"], "compare_one_shot")
        self.assertEqual(len(result["comparison_repositories"]), 2)

    def test_compare_endpoint_uses_repo_shared_fallback_setting_for_branches(self):
        repo = {
            "id": 1,
            "name": "Repo",
            "slug": "repo",
            "workspace": "repo-workspace",
            "allow_shared_fallback": 0,
        }
        left = {
            "repo": repo,
            "branch": {"id": 11, "name": "main"},
            "workspace": "repo-main-workspace",
        }
        right = {
            "repo": repo,
            "branch": {"id": 12, "name": "release"},
            "workspace": "repo-release-workspace",
        }
        response = {
            "question": "Compare login",
            "answer": "Compared.",
            "provider_used": "user:openai",
            "retrieval_mode": "compare_one_shot",
            "comparison_repositories": [],
            "context": {},
        }

        with patch.object(main, "enforce_rate_limit"), patch.object(
            main,
            "_resolve_compare_base_repo",
            return_value=repo,
        ), patch.object(
            main,
            "_resolve_compare_branch",
            side_effect=[left, right],
        ), patch.object(
            main,
            "enforce_strict_branch_freshness",
        ), patch.object(
            main,
            "load_user_llm",
            return_value={"provider": "openai", "api_key": "saved"},
        ), patch.object(
            main,
            "answer_compare",
            return_value=response,
        ) as answer:
            result = main.compare_repos_endpoint(
                main.CompareRequest(
                    question="Compare login",
                    left_branch=11,
                    right_branch=12,
                    llm_mode="auto",
                ),
                workspace="repo-workspace",
                user={"id": 7, "role": "user", "user_type": "dev_team"},
            )

        self.assertEqual(result["answer"], "Compared.")
        self.assertFalse(answer.call_args.kwargs["allow_shared_fallback"])
        self.assertEqual(result["token_usage"]["available"], False)

    def test_compare_endpoint_allows_dev_user_to_request_product_style(self):
        repo = {
            "id": 1,
            "name": "Repo",
            "slug": "repo",
            "workspace": "repo-workspace",
            "allow_shared_fallback": 1,
        }
        left = {
            "repo": repo,
            "branch": {"id": 11, "name": "main"},
            "workspace": "repo-main-workspace",
        }
        right = {
            "repo": repo,
            "branch": {"id": 12, "name": "release"},
            "workspace": "repo-release-workspace",
        }
        response = {
            "question": "Compare login",
            "answer": "Compared simply.",
            "provider_used": "shared:mimo-v2.5",
            "retrieval_mode": "compare_one_shot",
            "comparison_repositories": [],
            "context": {},
        }

        with patch.object(main, "enforce_rate_limit"), patch.object(
            main,
            "_resolve_compare_base_repo",
            return_value=repo,
        ), patch.object(
            main,
            "_resolve_compare_branch",
            side_effect=[left, right],
        ), patch.object(
            main,
            "enforce_strict_branch_freshness",
        ), patch.object(
            main,
            "load_user_llm",
            return_value=None,
        ), patch.object(
            main,
            "answer_compare",
            return_value=response,
        ) as answer:
            result = main.compare_repos_endpoint(
                main.CompareRequest(
                    question="Compare login",
                    left_branch=11,
                    right_branch=12,
                    llm_mode="mimo",
                    answer_user_type="product_team",
                ),
                workspace="repo-workspace",
                user={"id": 7, "role": "user", "user_type": "dev_team"},
            )

        self.assertEqual(result["answer_user_type"], "product_team")
        self.assertEqual(answer.call_args.kwargs["user_type"], "product_team")

    def test_compare_endpoint_rejects_same_branch(self):
        base_repo = {
            "id": 1,
            "name": "Repo",
            "slug": "repo",
            "workspace": "repo-workspace",
            "allow_shared_fallback": 1,
        }
        repo = {
            "repo": base_repo,
            "branch": {"id": 11, "name": "main"},
            "workspace": "repo-main-workspace",
        }

        with patch.object(main, "enforce_rate_limit"), patch.object(
            main,
            "_resolve_compare_base_repo",
            return_value=base_repo,
        ), patch.object(
            main,
            "_resolve_compare_branch",
            side_effect=[repo, repo],
        ), self.assertRaises(main.HTTPException) as raised:
            main.compare_repos_endpoint(
                main.CompareRequest(
                    question="Compare login",
                    left_branch=11,
                    right_branch=11,
                    llm_mode="mimo",
                ),
                workspace="repo-workspace",
                user={"id": 7, "role": "user", "user_type": "dev_team"},
            )

        self.assertEqual(raised.exception.status_code, 400)
        self.assertIn("different branches", raised.exception.detail)

    def test_compare_endpoint_returns_repeated_question_from_session_cache(self):
        store = ConversationStore(ttl_seconds=300, max_states=10, max_cached_answers=10)
        repo = {
            "id": 1,
            "name": "Repo",
            "slug": "repo",
            "workspace": "repo-workspace",
            "allow_shared_fallback": 1,
        }
        left = {
            "repo": repo,
            "branch": {"id": 11, "name": "main"},
            "workspace": "repo-main-workspace",
        }
        right = {
            "repo": repo,
            "branch": {"id": 12, "name": "release"},
            "workspace": "repo-release-workspace",
        }
        response = {
            "question": "Compare login",
            "answer": "Compared.",
            "provider_used": "shared:mimo-v2.5",
            "retrieval_mode": "compare_one_shot",
            "comparison_repositories": [],
            "context": {},
        }

        with patch.object(main, "conversation_store", store), patch.object(
            main, "enforce_rate_limit"
        ), patch.object(
            main,
            "_resolve_compare_base_repo",
            return_value=repo,
        ), patch.object(
            main,
            "_resolve_compare_branch",
            side_effect=[left, right, left, right],
        ), patch.object(
            main,
            "_comparison_revision",
            return_value="comparison-revision",
        ), patch.object(
            main,
            "enforce_strict_branch_freshness",
        ), patch.object(
            main,
            "load_user_llm",
            return_value=None,
        ), patch.object(
            main,
            "answer_compare",
            return_value=response,
        ) as answer:
            request = main.CompareRequest(
                question="Compare login",
                left_branch=11,
                right_branch=12,
                llm_mode="mimo",
            )
            first = main.compare_repos_endpoint(
                request,
                workspace="repo-workspace",
                user={
                    "id": 7,
                    "role": "user",
                    "user_type": "dev_team",
                    "_session_key": "session",
                },
            )
            second = main.compare_repos_endpoint(
                request,
                workspace="repo-workspace",
                user={
                    "id": 7,
                    "role": "user",
                    "user_type": "dev_team",
                    "_session_key": "session",
                },
            )

        self.assertEqual(first["answer"], "Compared.")
        self.assertEqual(answer.call_count, 1)
        self.assertTrue(second["session_cache_hit"])
        self.assertEqual(second["retrieval_mode"], "session_cache")
        self.assertEqual(second["token_usage"]["available"], True)
        self.assertIn("conversation_id", second)

    def test_compare_follow_up_reuses_comparison_conversation(self):
        store = ConversationStore(ttl_seconds=300, max_states=10, max_cached_answers=10)
        repo = {
            "id": 1,
            "name": "Repo",
            "slug": "repo",
            "workspace": "repo-workspace",
            "allow_shared_fallback": 1,
        }
        left = {
            "repo": repo,
            "branch": {"id": 11, "name": "main"},
            "workspace": "repo-main-workspace",
        }
        right = {
            "repo": repo,
            "branch": {"id": 12, "name": "release"},
            "workspace": "repo-release-workspace",
        }
        initial = {
            "question": "Compare login",
            "answer": "Initial comparison.",
            "provider_used": "shared:mimo-v2.5",
            "retrieval_mode": "compare_one_shot",
            "comparison_repositories": [],
            "context": {"llm_context_preview": {"question": "Compare login"}},
        }
        follow_up = {
            "question": "what about this flow?",
            "answer": "Follow-up comparison.",
            "provider_used": "shared:mimo-v2.5",
            "retrieval_mode": "compare_follow_up_cache",
            "comparison_repositories": [],
            "context": {},
            "follow_up_reused": True,
            "follow_up_fallback": False,
            "deep_investigation": False,
            "investigate_deeply_available": True,
        }

        with patch.object(main, "conversation_store", store), patch.object(
            main, "enforce_rate_limit"
        ), patch.object(
            main,
            "_resolve_compare_base_repo",
            return_value=repo,
        ), patch.object(
            main,
            "_resolve_compare_branch",
            side_effect=[left, right, left, right],
        ), patch.object(
            main,
            "_comparison_revision",
            return_value="comparison-revision",
        ), patch.object(
            main,
            "enforce_strict_branch_freshness",
        ), patch.object(
            main,
            "load_user_llm",
            return_value=None,
        ), patch.object(
            main,
            "answer_compare",
            return_value=initial,
        ), patch.object(
            main,
            "answer_compare_follow_up",
            return_value=follow_up,
        ) as follow:
            first = main.compare_repos_endpoint(
                main.CompareRequest(
                    question="Compare login",
                    left_branch=11,
                    right_branch=12,
                    llm_mode="mimo",
                ),
                workspace="repo-workspace",
                user={
                    "id": 7,
                    "role": "user",
                    "user_type": "dev_team",
                    "_session_key": "session",
                },
            )
            second = main.compare_repos_endpoint(
                main.CompareRequest(
                    question="what about this flow?",
                    left_branch=11,
                    right_branch=12,
                    llm_mode="mimo",
                    conversation_id=first["conversation_id"],
                    follow_up=True,
                ),
                workspace="repo-workspace",
                user={
                    "id": 7,
                    "role": "user",
                    "user_type": "dev_team",
                    "_session_key": "session",
                },
            )

        follow.assert_called_once()
        self.assertTrue(second["follow_up_reused"])
        self.assertEqual(second["conversation_id"], first["conversation_id"])

    def test_compare_deep_investigation_bypasses_repeated_question_cache(self):
        store = ConversationStore(ttl_seconds=300, max_states=10, max_cached_answers=10)
        repo = {
            "id": 1,
            "name": "Repo",
            "slug": "repo",
            "workspace": "repo-workspace",
            "allow_shared_fallback": 1,
        }
        left = {
            "repo": repo,
            "branch": {"id": 11, "name": "main"},
            "workspace": "repo-main-workspace",
        }
        right = {
            "repo": repo,
            "branch": {"id": 12, "name": "release"},
            "workspace": "repo-release-workspace",
        }
        initial = {
            "question": "Compare login",
            "answer": "Initial comparison.",
            "provider_used": "shared:mimo-v2.5",
            "retrieval_mode": "compare_one_shot",
            "comparison_repositories": [],
            "context": {"llm_context_preview": {"question": "Compare login"}},
        }
        deep = {
            "question": "Compare login",
            "answer": "Deep comparison.",
            "provider_used": "shared:mimo-v2.5",
            "retrieval_mode": "compare_agentic",
            "comparison_repositories": [],
            "context": {},
            "follow_up_reused": False,
            "follow_up_fallback": True,
            "deep_investigation": True,
            "investigate_deeply_available": False,
        }

        with patch.object(main, "conversation_store", store), patch.object(
            main, "enforce_rate_limit"
        ), patch.object(
            main,
            "_resolve_compare_base_repo",
            return_value=repo,
        ), patch.object(
            main,
            "_resolve_compare_branch",
            side_effect=[left, right, left, right],
        ), patch.object(
            main,
            "_comparison_revision",
            return_value="comparison-revision",
        ), patch.object(
            main,
            "enforce_strict_branch_freshness",
        ), patch.object(
            main,
            "load_user_llm",
            return_value=None,
        ), patch.object(
            main,
            "answer_compare",
            return_value=initial,
        ), patch.object(
            main,
            "answer_compare_follow_up",
            return_value=deep,
        ) as deep_follow_up:
            first = main.compare_repos_endpoint(
                main.CompareRequest(
                    question="Compare login",
                    left_branch=11,
                    right_branch=12,
                    llm_mode="mimo",
                ),
                workspace="repo-workspace",
                user={
                    "id": 7,
                    "role": "user",
                    "user_type": "dev_team",
                    "_session_key": "session",
                },
            )
            second = main.compare_repos_endpoint(
                main.CompareRequest(
                    question="Compare login",
                    left_branch=11,
                    right_branch=12,
                    llm_mode="mimo",
                    conversation_id=first["conversation_id"],
                    follow_up=True,
                    deep_investigation=True,
                ),
                workspace="repo-workspace",
                user={
                    "id": 7,
                    "role": "user",
                    "user_type": "dev_team",
                    "_session_key": "session",
                },
            )

        deep_follow_up.assert_called_once()
        self.assertNotIn("session_cache_hit", second)
        self.assertTrue(second["deep_investigation"])
        self.assertFalse(second["investigate_deeply_available"])

    def test_comparison_prompt_is_used_for_compare_context(self):
        prompt = client.build_prompt({
            "comparison_mode": True,
            "llm_context_preview": {
                "question": "Compare login",
                "branches": [
                    {"label": "Branch A", "name": "main"},
                    {"label": "Branch B", "name": "release"},
                ],
            },
        })

        self.assertIn("Comparison evidence", prompt)
        self.assertIn("Branch-by-branch findings", prompt)
        self.assertEqual(
            client._system_prompt({"comparison_mode": True}),
            client.COMPARISON_SYSTEM_PROMPT,
        )


class QueryRoutingTests(unittest.TestCase):
    def retrieval_config(self):
        return SimpleNamespace(
            stopwords=["what", "is", "the", "of", "to", "for", "and", "how"],
            synonyms={},
            keyword_boosts={},
            preferred_components=[],
            preferred_methods=[],
            node_limit=16,
            relation_limit=24,
            excerpt_nodes=6,
            excerpt_max_lines=22,
            excerpt_max_chars=1100,
            pre_search_instruction="Map terms first.",
        )

    def test_routes_common_query_shapes(self):
        self.assertEqual(
            main.route_query_type("createSession"),
            main.QUERY_EXACT_SYMBOL,
        )
        self.assertEqual(
            main.route_query_type("where is createSession defined?"),
            main.QUERY_DEFINITION,
        )
        self.assertEqual(
            main.route_query_type("who calls createSession?"),
            main.QUERY_CALLERS,
        )
        self.assertEqual(
            main.route_query_type("what does createSession call?"),
            main.QUERY_CALLEES,
        )
        self.assertEqual(
            main.route_query_type("show references to createSession"),
            main.QUERY_REFERENCES,
        )
        self.assertEqual(
            main.route_query_type("explain checkout flow"),
            main.QUERY_FLOW,
        )
        self.assertEqual(
            main.route_query_type("debug login crash"),
            main.QUERY_DEBUG,
        )
        self.assertEqual(
            main.route_query_type("how does login work?"),
            main.QUERY_CONCEPT,
        )

    def test_exact_symbol_fast_path_skips_source_search_when_sufficient(self):
        nodes = [
            {
                "id": "createSession",
                "source_file": "app/auth.py",
                "source_location": "L10-L20",
            }
        ]
        links = [
            {
                "source": "loginRoute",
                "target": "createSession",
                "relation": "calls",
                "source_file": "app/auth.py",
                "source_location": "L18",
            }
        ]

        with patch.object(
            main, "load_graph", return_value=(nodes, links)
        ), patch.object(
            main, "load_retrieval_config", return_value=self.retrieval_config()
        ), patch.object(
            main,
            "read_source_excerpt",
            return_value={"start_line": 10, "end_line": 20, "code": "def createSession(): pass"},
        ), patch.object(
            main,
            "_search_source_files",
            side_effect=AssertionError("fast path should not run source search"),
        ):
            context = main.build_context("createSession", workspace="repo-main")

        self.assertEqual(context["context_nodes"][0]["node"], "createSession")
        self.assertEqual(context["source_hits"][0]["path"], "app/auth.py")
        self.assertEqual(
            context["llm_context_preview"]["nodes"][0]["name"],
            "createsession",
        )

    def test_fast_path_falls_back_when_evidence_is_not_sufficient(self):
        nodes = [
            {
                "id": "createSession",
                "source_file": "app/auth.py",
                "source_location": "L10-L20",
            }
        ]
        source_search_calls = []

        def fake_source_search(*args, **kwargs):
            source_search_calls.append((args, kwargs))
            return []

        with patch.object(
            main, "load_graph", return_value=(nodes, [])
        ), patch.object(
            main, "load_retrieval_config", return_value=self.retrieval_config()
        ), patch.object(
            main, "_search_source_files", side_effect=fake_source_search
        ):
            context = main.build_context("who calls createSession?", workspace="repo-main")

        self.assertTrue(source_search_calls)
        self.assertEqual(context["context_nodes"][0]["node"], "createSession")


class AnswerActivityTests(unittest.TestCase):
    def tearDown(self):
        with main._ask_activity_lock:
            main._ask_activity.clear()

    def test_answer_activity_returns_capped_human_readable_context(self):
        request_id = "activity-test-123"
        context = {
            "context_nodes": [
                {
                    "name": "create_session",
                    "node": "func_create_session",
                    "source_file": "app/db.py",
                    "source_location": "L433-L441",
                }
            ],
            "context_relations": [
                {
                    "source_name": "create_session",
                    "relation_label": "queries",
                    "target_name": "sessions",
                    "source_file": "app/db.py",
                    "source_location": "L433-L441",
                }
            ],
            "source_hits": [
                {
                    "path": "app/db.py",
                    "snippets": [{"start_line": 433, "end_line": 441}],
                }
            ],
        }

        main.record_answer_activity(
            request_id,
            user_id=7,
            workspace="repo-main",
            question="How does create session work?",
            context=context,
        )
        payload = main.answer_activity_endpoint(request_id, {"id": 7})

        self.assertEqual(payload["status"], "generating_answer")
        self.assertEqual(payload["candidate_node_count"], 1)
        self.assertEqual(payload["node_count"], 1)
        self.assertEqual(payload["nodes"][0]["name"], "create_session")
        self.assertEqual(payload["nodes"][0]["type"], "Function")
        self.assertEqual(payload["relations"][0]["relation"], "queries")
        self.assertEqual(payload["source_files"][0]["path"], "app/db.py")
        self.assertNotIn("user_id", payload)

    def test_answer_activity_hides_unrelated_filled_context(self):
        request_id = "activity-test-earnings"
        context = {
            "context_nodes": [
                {
                    "name": "analytics",
                    "node": "analytics",
                    "source_file": "core/analytics/Analytics.kt",
                    "source_location": "L68",
                },
                {
                    "name": "EarningsScreen",
                    "node": "features_earnings_earningsscreen",
                    "source_file": "features/earnings/EarningsScreen.kt",
                    "source_location": "L12-L90",
                },
            ],
            "context_relations": [],
            "source_hits": [
                {
                    "path": "core/analytics/Analytics.kt",
                    "score": 700,
                    "snippets": [{"start_line": 68, "end_line": 80}],
                },
                {
                    "path": "features/earnings/EarningsScreen.kt",
                    "score": 900,
                    "snippets": [{"start_line": 12, "end_line": 90}],
                },
            ],
        }

        main.record_answer_activity(
            request_id,
            user_id=7,
            workspace="repo-main",
            question="What are the functionalities of earnings screen?",
            context=context,
        )
        payload = main.answer_activity_endpoint(request_id, {"id": 7})

        self.assertEqual(payload["candidate_node_count"], 2)
        self.assertEqual(payload["node_count"], 1)
        self.assertEqual([node["name"] for node in payload["nodes"]], ["EarningsScreen"])
        self.assertEqual(payload["source_files"][0]["path"], "features/earnings/EarningsScreen.kt")
        self.assertNotIn(
            "core/analytics/Analytics.kt",
            [item["path"] for item in payload["source_files"]],
        )

    def test_answer_activity_is_scoped_to_user(self):
        request_id = "activity-test-456"
        main.record_answer_activity(
            request_id,
            user_id=7,
            workspace="repo-main",
            question="How does login work?",
            context={"context_nodes": []},
        )

        with self.assertRaises(main.HTTPException) as raised:
            main.answer_activity_endpoint(request_id, {"id": 8})

        self.assertEqual(raised.exception.status_code, 404)

    def test_answer_question_updates_activity_from_retrieval_progress(self):
        request_id = "activity-progress-123"

        def fake_build_context(question, limit=16, workspace="repo-main", activity_callback=None):
            partial_context = {
                "context_nodes": [
                    {
                        "name": "CheckoutScreen",
                        "node": "features_checkout_checkoutscreen",
                        "source_file": "features/checkout/CheckoutScreen.kt",
                        "source_location": "L12-L90",
                    }
                ],
                "context_relations": [],
                "source_hits": [
                    {
                        "path": "features/checkout/CheckoutScreen.kt",
                        "score": 900,
                        "snippets": [{"start_line": 12, "end_line": 90}],
                    }
                ],
                "llm_context_preview": {"question": question},
            }
            if activity_callback:
                activity_callback("matching_source_files", {
                    "question": question,
                    "source_hits": partial_context["source_hits"],
                })
                activity_callback("ranking_graph_nodes", partial_context)
            return partial_context

        with patch.object(
            main, "build_context", side_effect=fake_build_context
        ), patch.object(
            main, "RepositoryToolbox", return_value=SimpleNamespace()
        ), patch.object(
            main,
            "generate",
            return_value={"answer": "Checkout uses CheckoutScreen.", "provider_used": "test"},
        ), patch.object(
            main,
            "repository_version_payload",
            return_value={"workspace": "repo-main", "revision": "test"},
        ):
            main.answer_question(
                "What does checkout screen do?",
                workspace="repo-main",
                activity_request_id=request_id,
                activity_user_id=7,
            )

        payload = main.answer_activity_endpoint(request_id, {"id": 7})

        self.assertEqual(payload["stage"], "generating_answer")
        self.assertEqual(payload["candidate_node_count"], 1)
        self.assertEqual(payload["node_count"], 1)
        self.assertEqual(payload["nodes"][0]["name"], "CheckoutScreen")
        self.assertEqual(
            payload["source_files"][0]["path"],
            "features/checkout/CheckoutScreen.kt",
        )


if __name__ == "__main__":
    unittest.main()
