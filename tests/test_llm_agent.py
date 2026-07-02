import json
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from app import main
from app.llm import client


class FakeResponse:
    def __init__(self, payload, status_code=200, text=""):
        self.payload = payload
        self.status_code = status_code
        self.text = text

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


if __name__ == "__main__":
    unittest.main()
