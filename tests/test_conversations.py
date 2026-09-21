import unittest
import time
from unittest.mock import patch

from app import main
from app.conversations import ConversationStore
from app.llm import client


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

    def test_malformed_tool_call_text_is_not_cached(self):
        bad_response = {
            "question": "What are the features?",
            "answer": (
                "<tool_call>\n"
                "<function=list_directory>\n"
                "<parameter=path>app/src/main</parameter>\n"
                "</function>\n"
                "</tool_call>"
            ),
            "provider_used": "shared:mimo-v2.5",
            "retrieval_mode": "agentic",
            "context": {},
        }
        cache_args = {
            "session_key": "session-a",
            "user_id": 7,
            "workspace": "repo-main",
            "llm_mode": "mimo",
            "user_type": "dev_team",
            "repository_revision": "branch:abc123",
            "question": "What are the features?",
        }

        self.store.store_cached_answer(**cache_args, response=bad_response)
        self.assertIsNone(self.store.get_cached_answer(**cache_args))

        cache_key = self.store._answer_cache_key(**cache_args)
        self.store._answer_cache[cache_key] = {
            "response": bad_response,
            "updated_at": time.monotonic(),
        }
        self.assertIsNone(self.store.get_cached_answer(**cache_args))
        self.assertNotIn(cache_key, self.store._answer_cache)

        self.store.store_repo_cached_answer(
            workspace="repo-main",
            user_type="dev_team",
            repository_revision="branch:abc123",
            question="What are the features?",
            response=bad_response,
        )
        self.assertIsNone(
            self.store.get_repo_cached_answer(
                workspace="repo-main",
                user_type="dev_team",
                repository_revision="branch:abc123",
                question="What are the features?",
            )
        )

    def test_llm_rejects_tool_call_text_as_final_answer(self):
        with self.assertRaisesRegex(RuntimeError, "tool call instead of a final answer"):
            client._require_answer(
                "<tool_call><function=list_directory></function></tool_call>",
                "shared:mimo-v2.5",
            )

    def test_repo_cached_answer_is_shared_across_sessions_but_scoped_to_revision_and_audience(self):
        response = {
            "question": "How does login work?",
            "answer": "Login is verified in src/auth.py:L1-L20.",
            "provider_used": "shared:mimo-v2.5",
            "retrieval_mode": "agentic",
            "context": {"llm_context_preview": {"question": "How does login work?"}},
        }
        self.store.store_repo_cached_answer(
            workspace="repo-main",
            user_type="dev_team",
            repository_revision="branch:abc123",
            question="How does\nlogin   work?",
            response=response,
        )

        # A completely different session/user gets the cached answer — that's the point.
        cached = self.store.get_repo_cached_answer(
            workspace="repo-main",
            user_type="dev_team",
            repository_revision="branch:abc123",
            question="how does login work?",
        )
        self.assertIsNotNone(cached)
        self.assertEqual(cached["answer"], response["answer"])
        self.assertNotIn("conversation_id", cached)
        self.assertNotIn("token_usage", cached)

        # But a different indexed revision or answer audience is a real cache miss.
        self.assertIsNone(self.store.get_repo_cached_answer(
            workspace="repo-main",
            user_type="dev_team",
            repository_revision="branch:def456",
            question="How does login work?",
        ))
        self.assertIsNone(self.store.get_repo_cached_answer(
            workspace="repo-main",
            user_type="product_team",
            repository_revision="branch:abc123",
            question="How does login work?",
        ))
        self.assertIsNone(self.store.get_repo_cached_answer(
            workspace="other-repo",
            user_type="dev_team",
            repository_revision="branch:abc123",
            question="How does login work?",
        ))

    def test_repo_answer_cache_can_be_cleared_by_workspace(self):
        response = {
            "question": "How does login work?",
            "answer": "Login is verified in src/auth.py:L1-L20.",
            "provider_used": "shared:mimo-v2.5",
        }
        self.store.store_repo_cached_answer(
            workspace="repo-main",
            user_type="dev_team",
            repository_revision="branch:abc123",
            question="How does login work?",
            response=response,
        )
        self.store.store_repo_cached_answer(
            workspace="other-repo",
            user_type="dev_team",
            repository_revision="branch:abc123",
            question="How does login work?",
            response=response,
        )

        cleared = self.store.clear_repo_cached_answers(workspaces={"repo-main"})

        self.assertEqual(cleared, 1)
        self.assertIsNone(self.store.get_repo_cached_answer(
            workspace="repo-main",
            user_type="dev_team",
            repository_revision="branch:abc123",
            question="How does login work?",
        ))
        self.assertIsNotNone(self.store.get_repo_cached_answer(
            workspace="other-repo",
            user_type="dev_team",
            repository_revision="branch:abc123",
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

    def test_follow_up_answers_are_scoped_to_their_own_conversation(self):
        """A follow-up's text only means something inside its own thread."""
        base = {
            "session_key": "session-a",
            "user_id": 7,
            "workspace": "repo-main",
            "llm_mode": "mimo",
            "user_type": "dev_team",
            "repository_revision": "branch:abc123",
        }
        response = {
            "question": "what about failures?",
            "answer": "Login failures are rejected in src/auth.py:L8-L12.",
            "provider_used": "shared:mimo-v2.5",
            "context": {},
        }
        self.store.store_cached_answer(
            **base,
            question="what about failures?",
            response=response,
            conversation_id="thread-a",
        )

        self.assertIsNotNone(self.store.get_cached_answer(
            **base,
            question="What about   failures?",
            conversation_id="thread-a",
        ))
        # The same follow-up text in a different thread is a different question.
        self.assertIsNone(self.store.get_cached_answer(
            **base,
            question="what about failures?",
            conversation_id="thread-b",
        ))
        # And it must not leak into standalone, session-wide lookups either.
        self.assertIsNone(self.store.get_cached_answer(
            **base,
            question="what about failures?",
        ))

    def test_root_turn_outlives_the_recent_turn_window(self):
        context = {
            "llm_context_preview": {
                "question": "How does login work?",
                "nodes": [{"name": "Auth", "source": "src/auth.py L1-L20"}],
            }
        }
        state = self.store.create(
            user_id=7,
            workspace="repo-main",
            llm_mode="mimo",
            user_type="dev_team",
            repository_revision="branch:abc123",
            context=context,
            question="How does login work?",
            answer="Login is verified in src/auth.py:L1-L20.",
        )
        self.store.append(
            state.conversation_id,
            question="What if it fails?",
            answer="Rejected in src/auth.py:L8-L12.",
            context=context,
        )
        latest = self.store.append(
            state.conversation_id,
            question="And on retry?",
            answer="Backoff in src/auth.py:L30-L36.",
            context=context,
        )

        self.assertNotIn(
            "How does login work?",
            [turn["question"] for turn in latest.turns[-2:]],
        )
        evidence = main.compact_follow_up_evidence(latest)
        self.assertIn("How does login work?", evidence)
        self.assertIn("Login is verified in src/auth.py:L1-L20.", evidence)

    def test_root_evidence_is_restored_only_after_a_full_retrieval_fallback(self):
        root_context = {
            "llm_context_preview": {
                "question": "How does login work?",
                "nodes": [{"name": "Auth", "source": "src/auth.py L1-L20"}],
            }
        }
        state = self.store.create(
            user_id=7,
            workspace="repo-main",
            llm_mode="mimo",
            user_type="dev_team",
            repository_revision="branch:abc123",
            context=root_context,
            question="How does login work?",
            answer="Login is verified in src/auth.py:L1-L20.",
        )

        # Reused-evidence follow-up: the thread still stands on its own
        # evidence, so no anchor block is added and the prompt is unchanged.
        reused = self.store.append(
            state.conversation_id,
            question="What if it fails?",
            answer="Rejected in src/auth.py:L8-L12.",
            context=root_context,
        )
        self.assertNotIn(
            "that started this conversation, from the same indexed commit",
            main.compact_follow_up_evidence(reused),
        )

        # A fallback to full retrieval replaces the thread's evidence.
        after_fallback = self.store.append(
            state.conversation_id,
            question="Where are tokens stored?",
            answer="In src/token.py:L5-L9.",
            context={
                "llm_context_preview": {
                    "question": "Where are tokens stored?",
                    "nodes": [{"name": "Token", "source": "src/token.py L5-L9"}],
                }
            },
        )
        evidence = main.compact_follow_up_evidence(after_fallback)
        self.assertIn("src/token.py L5-L9", evidence)
        self.assertIn("src/auth.py L1-L20", evidence)

    def test_related_follow_up_matches_the_root_not_only_the_last_turn(self):
        state = self.store.create(
            user_id=7,
            workspace="repo-main",
            llm_mode="mimo",
            user_type="dev_team",
            repository_revision="branch:abc123",
            context={"llm_context_preview": {"question": "How does login work?"}},
            question="How does login work?",
            answer="Login is verified in src/auth.py:L1-L20.",
        )
        moved_on = self.store.append(
            state.conversation_id,
            question="Where are tokens stored?",
            answer="In src/token.py:L5-L9.",
            context={"llm_context_preview": {"question": "Where are tokens stored?"}},
        )

        # Shares nothing with the previous turn, but is plainly about the
        # question that started the thread.
        self.assertTrue(main.is_related_follow_up(
            moved_on,
            "Is login validation cached?",
        ))
        self.assertFalse(main.is_related_follow_up(
            moved_on,
            "Explain the payment settlement scheduler.",
        ))

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
        # The deep investigation still runs, but it now carries the thread so a
        # referential follow-up is not retrieved and answered from scratch.
        full.assert_called_once_with(
            "Does it handle expired tokens?",
            workspace="repo-main",
            user_llm=None,
            allow_shared_fallback=True,
            llm_mode="mimo",
            user_type="dev_team",
            conversation_state=state,
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

    def test_full_retrieval_fallback_keeps_the_thread_topic(self):
        """A referential follow-up must not be retrieved or answered cold.

        Reproduces the reported bug: asking "what are the main features of the
        earnings screen?" then "do we have ads in this screen?" came back asking
        which screen the user meant, because the fallback dropped the thread."""
        state = ConversationStore(ttl_seconds=30, max_states=10).create(
            user_id=7,
            workspace="riderapp",
            llm_mode="mimo",
            user_type="product_team",
            repository_revision="branch:abc123",
            context={
                "llm_context_preview": {
                    "question": "what are the main features of earnings screen?",
                    "nodes": [{"name": "Earnings", "source": "ui/Earnings.kt L1-L80"}],
                }
            },
            question="what are the main features of earnings screen?",
            answer="The earnings screen shows daily payout, incentives and trip history.",
        )
        seen = {}

        def fake_build_context(question, limit=16, workspace=None, activity_callback=None):
            seen["retrieval_question"] = question
            return {
                "question": question,
                "llm_context_preview": {"question": question, "nodes": []},
            }

        def fake_generate(context, **kwargs):
            seen["agent_context"] = kwargs.get("agent_context", "")
            return {
                "answer": "Ads are not present on that screen.",
                "provider_used": "shared:mimo-v2.5",
                "retrieval_mode": "agentic",
                "agent_trace": [],
                "rounds": 1,
                "tool_calls": 1,
            }

        for deep in (False, True):
            seen.clear()
            with patch.object(main, "build_context", fake_build_context), patch.object(
                main, "generate", fake_generate
            ), patch.object(main, "RepositoryToolbox"), patch.object(
                main, "repository_version_payload", return_value=None
            ), patch.object(
                main,
                "generate_fast_follow_up",
                side_effect=main.FollowUpNeedsEvidence("more evidence"),
            ):
                response = main.answer_follow_up(
                    "do we have ads in this screen?",
                    state,
                    workspace="riderapp",
                    llm_mode="mimo",
                    user_type="product_team",
                    deep_investigation=deep,
                )

            # Retrieval can reach the screen the thread is actually about.
            self.assertIn("earnings", seen["retrieval_question"].lower())
            # And the model can resolve "this screen" without asking.
            self.assertIn("earnings screen", seen["agent_context"].lower())
            # The question the user asked is untouched.
            self.assertEqual(response["question"], "do we have ads in this screen?")

    def test_plain_question_retrieval_and_prompt_are_untouched(self):
        seen = {}

        def fake_build_context(question, limit=16, workspace=None, activity_callback=None):
            seen["retrieval_question"] = question
            return {
                "question": question,
                "llm_context_preview": {"question": question, "nodes": []},
            }

        def fake_generate(context, **kwargs):
            seen["agent_context"] = kwargs.get("agent_context", "")
            return {
                "answer": "Login is verified in src/auth.py:L1-L20.",
                "provider_used": "shared:mimo-v2.5",
                "retrieval_mode": "agentic",
                "agent_trace": [],
                "rounds": 1,
                "tool_calls": 1,
            }

        with patch.object(main, "build_context", fake_build_context), patch.object(
            main, "generate", fake_generate
        ), patch.object(main, "RepositoryToolbox"), patch.object(
            main, "repository_version_payload", return_value=None
        ):
            main.answer_question("How does login work?", workspace="repo-main")

        self.assertEqual(seen["retrieval_question"], "How does login work?")
        self.assertEqual(seen["agent_context"], "")
        self.assertEqual(
            client._agent_question("How does login work?", ""),
            "How does login work?",
        )

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

    def test_repeated_shared_tier_question_is_served_from_repo_cache_across_sessions(self):
        """A fresh, non-follow-up question answered by the shared tier is
        reusable across different users/sessions asking the same thing
        against the same indexed revision — the repo-scoped cache."""
        store = ConversationStore(ttl_seconds=30, max_states=10)
        first_user = {
            "id": 7,
            "user_type": "dev_team",
            "_session_key": "session-a",
        }
        second_user_session = {
            "id": 8,
            "user_type": "dev_team",
            "_session_key": "session-b",
        }
        first_answer = {
            "question": "How does login work?",
            "answer": "Login is verified in src/auth.py:L1-L20.",
            "provider_used": "shared:mimo-v2.5",
            "context": {
                "query_terms": ["login"],
                "context_nodes": [
                    {"name": "Auth", "source_file": "src/auth.py", "source_location": "L1-L20"},
                ],
                "source_hits": [
                    {
                        "path": "src/auth.py",
                        "snippets": [
                            {"start_line": 1, "end_line": 20, "code": "fun login() = verifyCredentials()"},
                        ],
                    },
                ],
                "llm_context_preview": {"question": "How does login work?"},
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

        with patch.object(main, "answer_question") as full:
            result = main.ask_llm_endpoint(
                main.AskRequest(
                    question="How does\nlogin   work?",
                    llm_mode="mimo",
                ),
                "repo-main",
                second_user_session,
            )

        full.assert_not_called()
        self.assertEqual(result["answer"], first_answer["answer"])
        self.assertTrue(result["repo_cache_hit"])
        self.assertEqual(result["retrieval_mode"], "repo_cache")
        self.assertNotIn("session_cache_hit", result)

    def test_ungrounded_shared_answer_is_not_promoted_to_repo_cache(self):
        store = ConversationStore(ttl_seconds=30, max_states=10)
        first_user = {
            "id": 7,
            "user_type": "dev_team",
            "_session_key": "session-a",
        }
        second_user_session = {
            "id": 8,
            "user_type": "dev_team",
            "_session_key": "session-b",
        }
        weak_answer = {
            "question": "What does this app do?",
            "answer": "This app processes image buffers.",
            "provider_used": "shared:mimo-v2.5",
            "context": {
                "query_terms": ["app"],
                "context_nodes": [],
                "source_hits": [],
                "llm_context_preview": {"question": "What does this app do?"},
            },
        }
        fresh_answer = {
            "question": "What does this app do?",
            "answer": "Fresh grounded answer.",
            "provider_used": "shared:mimo-v2.5",
            "context": weak_answer["context"],
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

        with patch.object(main, "answer_question", side_effect=[weak_answer, fresh_answer]) as full:
            main.ask_llm_endpoint(
                main.AskRequest(
                    question="What does this app do?",
                    llm_mode="mimo",
                ),
                "repo-main",
                first_user,
            )
            result = main.ask_llm_endpoint(
                main.AskRequest(
                    question="what does this app do?",
                    llm_mode="mimo",
                ),
                "repo-main",
                second_user_session,
            )

        self.assertEqual(full.call_count, 2)
        self.assertEqual(result["answer"], fresh_answer["answer"])
        self.assertNotIn("repo_cache_hit", result)

    def test_product_answer_reuses_cached_dev_evidence_without_full_retrieval(self):
        store = ConversationStore(ttl_seconds=30, max_states=10)
        user = {
            "id": 7,
            "user_type": "dev_team",
            "_session_key": "session-a",
        }
        dev_answer = {
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
        product_answer = {
            "question": "How does login work?",
            "answer": "The app checks the person's sign-in details before continuing.",
            "provider_used": "shared:mimo-v2.5",
            "retrieval_mode": "audience_cache",
            "audience_cache_hit": True,
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
            patch.object(main, "repository_version_payload", return_value=None),
        )
        for item in common_patches:
            item.start()
            self.addCleanup(item.stop)

        with patch.object(main, "answer_question", return_value=dev_answer):
            main.ask_llm_endpoint(
                main.AskRequest(
                    question="How does login work?",
                    llm_mode="mimo",
                ),
                "repo-main",
                user,
            )

        with patch.object(main, "answer_question") as full, patch.object(
            main,
            "answer_from_cached_audience_evidence",
            return_value=product_answer,
        ) as audience:
            result = main.ask_llm_endpoint(
                main.AskRequest(
                    question="How does login work?",
                    llm_mode="mimo",
                    answer_user_type="product_team",
                ),
                "repo-main",
                user,
            )

        full.assert_not_called()
        audience.assert_called_once()
        cached_source = audience.call_args.args[1]
        self.assertEqual(cached_source["answer"], dev_answer["answer"])
        self.assertEqual(result["answer"], product_answer["answer"])
        self.assertTrue(result["audience_cache_hit"])
        self.assertEqual(result["answer_user_type"], "product_team")

    def test_product_cached_evidence_fallback_runs_full_retrieval_when_needed(self):
        store = ConversationStore(ttl_seconds=30, max_states=10)
        user = {
            "id": 7,
            "user_type": "dev_team",
            "_session_key": "session-a",
        }
        dev_answer = {
            "question": "How does login work?",
            "answer": "Login is verified in src/auth.py:L1-L20.",
            "provider_used": "shared:mimo-v2.5",
            "context": {"llm_context_preview": {"question": "How does login work?"}},
        }
        full_product_answer = {
            "question": "How does login work?",
            "answer": "A fresh product answer was generated.",
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

        with patch.object(main, "answer_question", return_value=dev_answer):
            main.ask_llm_endpoint(
                main.AskRequest(
                    question="How does login work?",
                    llm_mode="mimo",
                ),
                "repo-main",
                user,
            )

        with patch.object(
            main,
            "answer_from_cached_audience_evidence",
            side_effect=main.FollowUpNeedsEvidence("more evidence"),
        ) as audience, patch.object(
            main,
            "answer_question",
            return_value=full_product_answer,
        ) as full:
            result = main.ask_llm_endpoint(
                main.AskRequest(
                    question="How does login work?",
                    llm_mode="mimo",
                    answer_user_type="product_team",
                ),
                "repo-main",
                user,
            )

        audience.assert_called_once()
        full.assert_called_once()
        self.assertEqual(full.call_args.kwargs["user_type"], "product_team")
        self.assertEqual(result["answer"], full_product_answer["answer"])
        self.assertNotIn("audience_cache_hit", result)

    def test_repeated_personal_key_question_stays_scoped_to_session(self):
        """BYOK answers must never be shared with a different user/session —
        that would hand another person content generated by someone else's
        personal key. Only the shared tier is eligible for the repo cache."""
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
            "provider_used": "user:anthropic",
            "context": {"llm_context_preview": {"question": "How does login work?"}},
        }
        second_answer = {
            "question": "How does login work?",
            "answer": "A fresh answer was generated for the new session.",
            "provider_used": "user:anthropic",
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
            patch.object(
                main,
                "load_user_llm",
                return_value={
                    "provider": "anthropic",
                    "api_key": "sk-ant-test",
                    "base_url": "https://api.anthropic.com",
                    "model": "claude-sonnet-4-5",
                },
            ),
            patch.object(main, "repository_revision", return_value="branch:abc123"),
        )
        for item in common_patches:
            item.start()
            self.addCleanup(item.stop)

        with patch.object(main, "answer_question", return_value=first_answer):
            main.ask_llm_endpoint(
                main.AskRequest(
                    question="How does login work?",
                    llm_mode="auto",
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
                    llm_mode="auto",
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

    def test_request_uses_shared_tier_only(self):
        self.assertTrue(main._request_uses_shared_tier_only("mimo", None))
        self.assertTrue(main._request_uses_shared_tier_only(
            "mimo", {"api_key": "sk-ant-test"},
        ))
        self.assertTrue(main._request_uses_shared_tier_only("auto", None))
        self.assertTrue(main._request_uses_shared_tier_only("auto", {}))
        self.assertFalse(main._request_uses_shared_tier_only(
            "auto", {"api_key": "sk-ant-test"},
        ))
        self.assertFalse(main._request_uses_shared_tier_only(
            "personal", {"api_key": "sk-ant-test"},
        ))
        self.assertFalse(main._request_uses_shared_tier_only("ollama", None))


if __name__ == "__main__":
    unittest.main()
