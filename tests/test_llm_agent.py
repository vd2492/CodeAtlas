import json
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from app import main
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
            "answer_question",
            return_value=generated,
        ) as answer:
            result = main.flow_summary_endpoint(
                "createbadgeview",
                request,
                workspace="sample",
                user={"id": 7, "user_type": "product_team"},
            )

        question = answer.call_args.args[0]
        self.assertIn("brief product-friendly summary", question)
        self.assertEqual(answer.call_args.kwargs["user_type"], "product_team")
        self.assertEqual(answer.call_args.kwargs["llm_mode"], "mimo")
        self.assertEqual(answer.call_args.kwargs["answer_mode"], "flow_summary")
        self.assertEqual(result["question"], "CreatebadgeView flow")
        self.assertEqual(result["flow_topic"], "createbadgeview")


if __name__ == "__main__":
    unittest.main()
