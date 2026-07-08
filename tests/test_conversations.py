import unittest
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

    def test_cached_answer_is_scoped_and_uses_normalized_question(self):
        response = {
            "question": "How does login work?",
            "answer": "Login is verified in src/auth.py:L1-L20.",
            "provider_used": "shared:mimo-v2.5",
            "retrieval_mode": "agentic",
            "context": {"llm_context_preview": {"question": "How does login work?"}},
        }
        self.store.store_cached_answer(
            session_key="session-a",
            user_id=7,
            workspace="repo-main",
            llm_mode="mimo",
            user_type="dev_team",
            repository_revision="branch:abc123",
            question="How does\nlogin   work?",
            response=response,
        )

        cached = self.store.get_cached_answer(
            session_key="session-a",
            user_id=7,
            workspace="repo-main",
            llm_mode="mimo",
            user_type="dev_team",
            repository_revision="branch:abc123",
            question="how does login work?",
        )

        self.assertIsNotNone(cached)
        self.assertEqual(cached["answer"], response["answer"])
        self.assertIsNone(self.store.get_cached_answer(
            session_key="session-b",
            user_id=7,
            workspace="repo-main",
            llm_mode="mimo",
            user_type="dev_team",
            repository_revision="branch:abc123",
            question="How does login work?",
        ))
        self.assertIsNone(self.store.get_cached_answer(
            session_key="session-a",
            user_id=7,
            workspace="repo-main",
            llm_mode="mimo",
            user_type="dev_team",
            repository_revision="branch:def456",
            question="How does login work?",
        ))

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
    def test_follow_up_uses_compact_no_tool_path_without_context_rebuild(self):
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
        generated = {
            "answer": "It rejects invalid users in src/auth.py:L8-L12.",
            "provider_used": "shared:mimo-v2.5",
            "retrieval_mode": "follow_up_cache",
            "tool_calls": 0,
        }
        with patch.object(main, "build_context") as build_context, patch.object(
            main, "RepositoryToolbox"
        ) as toolbox, patch.object(
            main, "generate_fast_follow_up", return_value=generated
        ) as generate_fast, patch.object(
            main, "repository_version_payload", return_value=None
        ):
            result = main.answer_follow_up(
                "What happens when it fails?",
                state,
                workspace="repo-main",
                llm_mode="mimo",
            )

        build_context.assert_not_called()
        toolbox.assert_not_called()
        self.assertIn("src/auth.py L1-L20", generate_fast.call_args.args[1])
        self.assertTrue(result["follow_up_reused"])
        self.assertTrue(result["investigate_deeply_available"])
        self.assertIn("follow_up_generation", result["timings_ms"])

    def test_deep_investigation_bypasses_cached_follow_up_generation(self):
        state = ConversationStore(ttl_seconds=30, max_states=10).create(
            user_id=7,
            workspace="repo-main",
            llm_mode="mimo",
            user_type="dev_team",
            repository_revision="branch:abc123",
            context={"llm_context_preview": {"question": "How does login work?"}},
            question="How does login work?",
            answer="Login is verified in src/auth.py:L1-L20.",
        )
        full_response = {
            "question": "Does it handle expired tokens?",
            "answer": "A fresh investigation found the expiry path.",
            "provider_used": "shared:mimo-v2.5",
            "context": {"llm_context_preview": {"question": "expired tokens"}},
            "timings_ms": {
                "retrieval": 10.0,
                "generation": 20.0,
                "total": 30.0,
            },
        }
        with patch.object(
            main, "generate_fast_follow_up"
        ) as generate_fast, patch.object(
            main, "answer_question", return_value=full_response
        ) as full:
            result = main.answer_follow_up(
                "Does it handle expired tokens?",
                state,
                workspace="repo-main",
                llm_mode="mimo",
                deep_investigation=True,
            )

        generate_fast.assert_not_called()
        full.assert_called_once_with(
            "Does it handle expired tokens?",
            workspace="repo-main",
            user_llm=None,
            allow_shared_fallback=True,
            llm_mode="mimo",
            user_type="dev_team",
        )
        self.assertTrue(result["deep_investigation"])
        self.assertFalse(result["follow_up_reused"])
        self.assertFalse(result["investigate_deeply_available"])

    def test_insufficient_compact_evidence_runs_full_retrieval(self):
        state = ConversationStore(ttl_seconds=30, max_states=10).create(
            user_id=7,
            workspace="repo-main",
            llm_mode="mimo",
            user_type="dev_team",
            repository_revision="branch:abc123",
            context={"llm_context_preview": {"question": "How does login work?"}},
            question="How does login work?",
            answer="Login is verified in src/auth.py:L1-L20.",
        )
        full_response = {
            "question": "What happens after the token expires?",
            "answer": "A fresh investigation found the expiry path.",
            "provider_used": "shared:mimo-v2.5",
            "context": {"llm_context_preview": {"question": "expiry"}},
            "timings_ms": {
                "retrieval": 10.0,
                "generation": 20.0,
                "total": 30.0,
            },
        }
        with patch.object(
            main,
            "generate_fast_follow_up",
            side_effect=main.FollowUpNeedsEvidence("more evidence"),
        ), patch.object(
            main, "answer_question", return_value=full_response
        ) as full:
            result = main.answer_follow_up(
                "What happens after the token expires?",
                state,
                workspace="repo-main",
                llm_mode="mimo",
            )

        full.assert_called_once()
        self.assertFalse(result["follow_up_reused"])
        self.assertTrue(result["follow_up_fallback"])
        self.assertIn("follow_up_gate", result["timings_ms"])

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

    def test_repeated_question_returns_from_session_cache_without_llm(self):
        store = ConversationStore(ttl_seconds=30, max_states=10)
        user = {
            "id": 7,
            "user_type": "dev_team",
            "_session_key": "session-a",
        }
        first_answer = {
            "question": "How does login work?",
            "answer": "Login is verified in src/auth.py:L1-L20.",
            "provider_used": "shared:mimo-v2.5",
            "retrieval_mode": "agentic",
            "context": {
                "llm_context_preview": {
                    "question": "How does login work?",
                    "nodes": [{"name": "Auth", "source": "src/auth.py L1-L20"}],
                }
            },
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
            patch.object(main, "repository_version_payload", return_value=None),
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

        with patch.object(main, "answer_question") as full, patch.object(
            main,
            "answer_follow_up",
        ) as follow_up:
            second = main.ask_llm_endpoint(
                main.AskRequest(
                    question="how does\nlogin   work?",
                    llm_mode="mimo",
                ),
                "repo-main",
                user,
            )

        full.assert_not_called()
        follow_up.assert_not_called()
        self.assertEqual(second["answer"], first_answer["answer"])
        self.assertTrue(second["session_cache_hit"])
        self.assertEqual(second["retrieval_mode"], "session_cache")
        self.assertTrue(second["investigate_deeply_available"])
        self.assertTrue(second["token_usage"]["available"])
        self.assertEqual(second["token_usage"]["total_tokens"], 0)
        self.assertNotEqual(second["conversation_id"], first["conversation_id"])

    def test_repeated_question_cache_is_scoped_to_session(self):
        store = ConversationStore(ttl_seconds=30, max_states=10)
        first_user = {
            "id": 7,
            "user_type": "dev_team",
            "_session_key": "session-a",
        }
        second_user_session = {
            "id": 7,
            "user_type": "dev_team",
            "_session_key": "session-b",
        }
        first_answer = {
            "question": "How does login work?",
            "answer": "Login is verified in src/auth.py:L1-L20.",
            "provider_used": "shared:mimo-v2.5",
            "context": {"llm_context_preview": {"question": "How does login work?"}},
        }
        second_answer = {
            "question": "How does login work?",
            "answer": "A fresh answer was generated for the new session.",
            "provider_used": "shared:mimo-v2.5",
            "context": {"llm_context_preview": {"question": "How does login work?"}},
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
            main.ask_llm_endpoint(
                main.AskRequest(
                    question="How does login work?",
                    llm_mode="mimo",
                ),
                "repo-main",
                first_user,
            )

        with patch.object(
            main,
            "answer_question",
            return_value=second_answer,
        ) as full:
            result = main.ask_llm_endpoint(
                main.AskRequest(
                    question="How does login work?",
                    llm_mode="mimo",
                ),
                "repo-main",
                second_user_session,
            )

        full.assert_called_once()
        self.assertNotIn("session_cache_hit", result)
        self.assertEqual(result["answer"], second_answer["answer"])

    def test_deep_investigation_bypasses_repeated_question_cache(self):
        store = ConversationStore(ttl_seconds=30, max_states=10)
        user = {
            "id": 7,
            "user_type": "dev_team",
            "_session_key": "session-a",
        }
        first_answer = {
            "question": "How does login work?",
            "answer": "Login is verified in src/auth.py:L1-L20.",
            "provider_used": "shared:mimo-v2.5",
            "context": {"llm_context_preview": {"question": "How does login work?"}},
        }
        deep_answer = {
            "question": "How does login work?",
            "answer": "A deep investigation refreshed the answer.",
            "provider_used": "shared:mimo-v2.5",
            "context": {"llm_context_preview": {"question": "How does login work?"}},
            "follow_up_reused": False,
            "follow_up_fallback": True,
            "deep_investigation": True,
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

        with patch.object(main, "answer_follow_up", return_value=deep_answer) as deep:
            result = main.ask_llm_endpoint(
                main.AskRequest(
                    question="How does login work?",
                    llm_mode="mimo",
                    conversation_id=first["conversation_id"],
                    follow_up=True,
                    deep_investigation=True,
                ),
                "repo-main",
                user,
            )

        deep.assert_called_once()
        self.assertTrue(deep.call_args.kwargs["deep_investigation"])
        self.assertNotIn("session_cache_hit", result)
        self.assertTrue(result["deep_investigation"])

    def test_endpoint_forwards_explicit_deep_investigation(self):
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
        deep_answer = {
            "question": "Does it handle expired tokens?",
            "answer": "The deep investigation found the expiry path.",
            "provider_used": "shared:mimo-v2.5",
            "context": {"llm_context_preview": {"question": "expired tokens"}},
            "follow_up_reused": False,
            "follow_up_fallback": True,
            "deep_investigation": True,
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
            main, "answer_follow_up", return_value=deep_answer
        ) as follow_up, patch.object(
            main, "answer_question"
        ) as full:
            result = main.ask_llm_endpoint(
                main.AskRequest(
                    question="Does it handle expired tokens?",
                    llm_mode="mimo",
                    conversation_id=original.conversation_id,
                    follow_up=True,
                    deep_investigation=True,
                ),
                "repo-main",
                user,
            )

        follow_up.assert_called_once()
        self.assertTrue(follow_up.call_args.kwargs["deep_investigation"])
        full.assert_not_called()
        self.assertTrue(result["deep_investigation"])
        self.assertEqual(result["conversation_id"], original.conversation_id)

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
        self.assertTrue(result["follow_up_fallback"])
        self.assertNotEqual(result["conversation_id"], original.conversation_id)


if __name__ == "__main__":
    unittest.main()
