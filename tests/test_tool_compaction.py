import copy
import unittest
from unittest.mock import patch

from app.llm import client

BIG = "x" * 5000


def openai_history(rounds):
    messages = [{"role": "system", "content": "sys"}, {"role": "user", "content": "q"}]
    for n in range(rounds):
        messages.append({
            "role": "assistant",
            "content": None,
            "tool_calls": [{"id": f"c{n}", "function": {"name": "t", "arguments": "{}"}}],
        })
        messages.append({"role": "tool", "tool_call_id": f"c{n}", "content": f"r{n}:{BIG}"})
    return messages


class ToolCompactionTests(unittest.TestCase):
    def test_recent_rounds_stay_full_and_older_are_stubbed(self):
        history = openai_history(5)
        with patch.object(client, "AGENT_KEEP_RECENT_TOOL_ROUNDS", 2), patch.object(
            client, "AGENT_CONTEXT_CHARS", 10**9
        ):
            out = client._compact_openai_messages(history)
        tools = [m for m in out if m["role"] == "tool"]
        self.assertEqual([len(m["content"]) > 4000 for m in tools], [False] * 3 + [True] * 2)
        self.assertTrue(tools[0]["content"].startswith("r0:"))
        self.assertIn("trimmed", tools[0]["content"])
        self.assertEqual([m["tool_call_id"] for m in tools], [f"c{n}" for n in range(5)])

    def test_input_is_not_mutated(self):
        history = openai_history(5)
        snapshot = copy.deepcopy(history)
        client._compact_openai_messages(history)
        self.assertEqual(history, snapshot)

    def test_total_budget_trims_oldest_but_never_latest_round(self):
        history = openai_history(4)
        with patch.object(client, "AGENT_KEEP_RECENT_TOOL_ROUNDS", 4), patch.object(
            client, "AGENT_CONTEXT_CHARS", 12000
        ):
            out = client._compact_openai_messages(history)
        tools = [m for m in out if m["role"] == "tool"]
        self.assertIn("trimmed", tools[0]["content"])
        self.assertNotIn("trimmed", tools[-1]["content"])
        self.assertEqual(len(tools[-1]["content"]), len(BIG) + 3)

    def test_single_round_is_untouched(self):
        history = openai_history(1)
        self.assertEqual(client._compact_openai_messages(history), history)

    def test_responses_items(self):
        items = [{"role": "user", "content": "q"}]
        for n in range(4):
            items.append({"type": "function_call", "call_id": f"c{n}", "name": "t", "arguments": "{}"})
            items.append({"type": "function_call_output", "call_id": f"c{n}", "output": f"r{n}:{BIG}"})
        with patch.object(client, "AGENT_KEEP_RECENT_TOOL_ROUNDS", 1), patch.object(
            client, "AGENT_CONTEXT_CHARS", 10**9
        ):
            out = client._compact_responses_items(items)
        outputs = [i for i in out if i.get("type") == "function_call_output"]
        self.assertIn("trimmed", outputs[0]["output"])
        self.assertNotIn("trimmed", outputs[-1]["output"])
        self.assertEqual([i["call_id"] for i in outputs], ["c0", "c1", "c2", "c3"])

    def test_anthropic_tool_result_blocks_keep_structure(self):
        messages = [{"role": "user", "content": "q"}]
        for n in range(4):
            messages.append({"role": "assistant", "content": [{"type": "tool_use", "id": f"u{n}", "name": "t", "input": {}}]})
            messages.append({"role": "user", "content": [
                {"type": "tool_result", "tool_use_id": f"u{n}", "content": f"r{n}:{BIG}"},
            ]})
        with patch.object(client, "AGENT_KEEP_RECENT_TOOL_ROUNDS", 1), patch.object(
            client, "AGENT_CONTEXT_CHARS", 10**9
        ):
            out = client._compact_anthropic_messages(messages)
        results = [m["content"][0] for m in out if client._is_anthropic_tool_results(m)]
        self.assertEqual([r["tool_use_id"] for r in results], ["u0", "u1", "u2", "u3"])
        self.assertIn("trimmed", results[0]["content"])
        self.assertNotIn("trimmed", results[-1]["content"])

    def test_short_results_are_never_altered(self):
        history = openai_history(3)
        for m in history:
            if m["role"] == "tool":
                m["content"] = "short"
        self.assertEqual(client._compact_openai_messages(history), history)

    def test_tool_output_cap_defaults(self):
        from app.agent import tools
        self.assertEqual(tools.MAX_SEARCH_RESULT_CHARS, 16000)
        self.assertEqual(tools.MAX_READ_CHARS, 16000)


if __name__ == "__main__":
    unittest.main()
