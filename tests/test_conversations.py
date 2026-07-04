import unittest
from types import SimpleNamespace
from unittest.mock import patch

from app import main
from app.conversations import ConversationStore


class ConversationStoreTests(unittest.TestCase):
    def setUp(self):
        self.store = ConversationStore(ttl_seconds=30, max_states=2)
        self.state = self.store.create(
            user_id=7,
            workspace="repo-main",
            llm_mode="mimo",
            user_type="dev_team",
            repository_revision="branch:abc123",
            context={"llm_context_preview": {"question": "How does login work?"}},
            question="How does login work?",
            answer="Login is verified in src/auth.py:L1-L20.",
        )

    def get_state(self, **overrides):
        values = {
            "user_id": 7,
            "workspace": "repo-main",
            "llm_mode": "mimo",
            "user_type": "dev_team",
            "repository_revision": "branch:abc123",
        }
        values.update(overrides)
        return self.store.get(self.state.conversation_id, **values)

    def test_state_is_scoped_to_user_workspace_mode_and_revision(self):
        self.assertIsNotNone(self.get_state())
        self.assertIsNone(self.get_state(user_id=8))
        self.assertIsNone(self.get_state(workspace="other"))
        self.assertIsNone(self.get_state(llm_mode="personal"))
        self.assertIsNone(self.get_state(repository_revision="branch:def456"))

    def test_state_expires_without_affecting_normal_requests(self):
        store = ConversationStore(ttl_seconds=1, max_states=2)
        with patch("app.conversations.time.monotonic", side_effect=[100.0, 102.0]):
            state = store.create(
                user_id=1,
                workspace="repo",
                llm_mode="mimo",
                user_type="dev_team",
                repository_revision="rev",
                context={},
                question="Question",
                answer="Answer",
            )
            loaded = store.get(
                state.conversation_id,
                user_id=1,
                workspace="repo",
                llm_mode="mimo",
                user_type="dev_team",
                repository_revision="rev",
            )
        self.assertIsNone(loaded)

    def test_related_follow_up_detects_references_and_topic_overlap(self):
        self.assertTrue(main.is_related_follow_up(
            self.state,
            "What happens when it fails?",
        ))
        self.assertTrue(main.is_related_follow_up(
            self.state,
            "Where is login validation performed?",
        ))
        self.assertFalse(main.is_related_follow_up(
            self.state,
            "Explain the payment settlement scheduler.",
        ))
        self.assertFalse(main.is_related_follow_up(
            self.state,
            "How does payment settlement work?",
        ))


class ConversationEndpointTests(unittest.TestCase):
    def test_follow_up_skips_context_rebuild_and_keeps_tools_optional(self):
        state = ConversationStore(ttl_seconds=30, max_states=10).create(
            user_id=7,
            workspace="repo-main",
            llm_mode="mimo",
            user_type="dev_team",
            repository_revision="branch:abc123",
            context={
                "llm_context_preview": {
                    "question": "How does login work?",
                    "nodes": [{"name": "Auth", "source": "src/auth.py L1-L20"}],
                }
            },
            question="How does login work?",
            answer="Login is verified in src/auth.py:L1-L20.",
        )
        toolbox = SimpleNamespace()
        generated = {
            "answer": "It rejects invalid users in src/auth.py:L8-L12.",
            "provider_used": "shared:mimo-v2.5",
            "tool_calls": 0,
        }
        with patch.object(main, "build_context") as build_context, patch.object(
            main, "RepositoryToolbox", return_value=toolbox
        ), patch.object(
            main, "generate", return_value=generated
        ) as generate, patch.object(
            main, "repository_version_payload", return_value=None
        ):
            result = main.answer_follow_up(
                "What happens when it fails?",
                state,
                workspace="repo-main",
                llm_mode="mimo",
            )

        build_context.assert_not_called()
        self.assertFalse(generate.call_args.kwargs["require_tool"])
        self.assertIn("src/auth.py L1-L20", generate.call_args.kwargs["agent_context"])
        self.assertTrue(result["follow_up_reused"])

    def test_related_follow_up_reuses_server_evidence(self):
        store = ConversationStore(ttl_seconds=30, max_states=10)
        user = {"id": 7, "user_type": "dev_team"}
        first_answer = {
            "question": "How does login work?",
            "answer": "Login is verified in src/auth.py:L1-L20.",
            "provider_used": "shared:mimo-v2.5",
            "context": {
                "llm_context_preview": {
                    "question": "How does login work?",
                    "nodes": [{"name": "Auth", "source": "src/auth.py L1-L20"}],
                }
            },
        }
        follow_up_answer = {
            "question": "What happens when it fails?",
            "answer": "The request is rejected in src/auth.py:L8-L12.",
            "provider_used": "shared:mimo-v2.5",
            "context": first_answer["context"],
            "follow_up_reused": True,
        }

        common_patches = (
            patch.object(main, "conversation_store", store),
            patch.object(main, "enforce_rate_limit"),
            patch.object(main, "enforce_strict_branch_freshness"),
            patch.object(
                main.db,
                "get_repo_by_workspace",
                return_value={"allow_shared_fallback": 1},
            ),
            patch.object(main, "load_user_llm", return_value=None),
            patch.object(main, "repository_revision", return_value="branch:abc123"),
        )
        for item in common_patches:
            item.start()
            self.addCleanup(item.stop)

        with patch.object(main, "answer_question", return_value=first_answer):
            first = main.ask_llm_endpoint(
                main.AskRequest(
                    question="How does login work?",
                    llm_mode="mimo",
                ),
                "repo-main",
                user,
            )

        with patch.object(main, "answer_follow_up", return_value=follow_up_answer) as fast, \
                patch.object(main, "answer_question") as full:
            second = main.ask_llm_endpoint(
                main.AskRequest(
                    question="What happens when it fails?",
                    llm_mode="mimo",
                    conversation_id=first["conversation_id"],
                    follow_up=True,
                ),
                "repo-main",
                user,
            )

        fast.assert_called_once()
        full.assert_not_called()
        self.assertTrue(second["follow_up_reused"])
        self.assertEqual(second["conversation_id"], first["conversation_id"])

    def test_unrelated_follow_up_uses_full_retrieval_and_new_conversation(self):
        store = ConversationStore(ttl_seconds=30, max_states=10)
        original = store.create(
            user_id=7,
            workspace="repo-main",
            llm_mode="mimo",
            user_type="dev_team",
            repository_revision="branch:abc123",
            context={"llm_context_preview": {"question": "How does login work?"}},
            question="How does login work?",
            answer="Login is verified in src/auth.py:L1-L20.",
        )
        full_answer = {
            "question": "Explain payment settlement.",
            "answer": "Settlement runs in src/payments.py:L1-L20.",
            "provider_used": "shared:mimo-v2.5",
            "context": {"llm_context_preview": {"question": "Explain payment settlement."}},
        }
        user = {"id": 7, "user_type": "dev_team"}

        with patch.object(main, "conversation_store", store), patch.object(
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
            main, "answer_follow_up"
        ) as fast, patch.object(
            main, "answer_question", return_value=full_answer
        ) as full:
            result = main.ask_llm_endpoint(
                main.AskRequest(
                    question="Explain the payment settlement scheduler.",
                    llm_mode="mimo",
                    conversation_id=original.conversation_id,
                    follow_up=True,
                ),
                "repo-main",
                user,
            )

        fast.assert_not_called()
        full.assert_called_once()
        self.assertFalse(result["follow_up_reused"])
        self.assertNotEqual(result["conversation_id"], original.conversation_id)


if __name__ == "__main__":
    unittest.main()
