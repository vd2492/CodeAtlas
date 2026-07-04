"""Short-lived, server-side state for grounded follow-up questions.

Conversation state is intentionally in-process: CodeAtlas currently runs as a
single application process, and losing this cache only makes the next follow-up
take the normal full-retrieval path. State is always scoped to an authenticated
user, workspace, provider mode, audience type, and indexed repository revision.
"""

from __future__ import annotations

import copy
import os
import threading
import time
import uuid
from dataclasses import dataclass, field
from typing import Optional


CONVERSATION_TTL_SECONDS = max(
    60, int(os.environ.get("CODEATLAS_CONVERSATION_TTL_SECONDS", "1800"))
)
CONVERSATION_MAX_STATES = max(
    10, int(os.environ.get("CODEATLAS_CONVERSATION_MAX_STATES", "200"))
)
CONVERSATION_MAX_TURNS = max(
    1, int(os.environ.get("CODEATLAS_CONVERSATION_MAX_TURNS", "4"))
)
CONVERSATION_MAX_ANSWER_CHARS = max(
    1000, int(os.environ.get("CODEATLAS_CONVERSATION_MAX_ANSWER_CHARS", "8000"))
)


@dataclass
class ConversationState:
    conversation_id: str
    user_id: int
    workspace: str
    llm_mode: str
    user_type: str
    repository_revision: str
    context: dict
    turns: list[dict] = field(default_factory=list)
    updated_at: float = field(default_factory=time.monotonic)


class ConversationStore:
    """Thread-safe TTL/LRU store containing no API keys or credentials."""

    def __init__(
        self,
        ttl_seconds: int = CONVERSATION_TTL_SECONDS,
        max_states: int = CONVERSATION_MAX_STATES,
    ):
        self.ttl_seconds = max(1, int(ttl_seconds))
        self.max_states = max(1, int(max_states))
        self._states: dict[str, ConversationState] = {}
        self._lock = threading.Lock()

    @staticmethod
    def _bounded_turn(question: str, answer: str) -> dict:
        return {
            "question": str(question or "")[:2000],
            "answer": str(answer or "")[:CONVERSATION_MAX_ANSWER_CHARS],
        }

    def _prune_locked(self, now: float) -> None:
        expired = [
            conversation_id
            for conversation_id, state in self._states.items()
            if now - state.updated_at > self.ttl_seconds
        ]
        for conversation_id in expired:
            self._states.pop(conversation_id, None)

        overflow = len(self._states) - self.max_states
        if overflow > 0:
            oldest = sorted(
                self._states.values(),
                key=lambda state: state.updated_at,
            )[:overflow]
            for state in oldest:
                self._states.pop(state.conversation_id, None)

    def create(
        self,
        *,
        user_id: int,
        workspace: str,
        llm_mode: str,
        user_type: str,
        repository_revision: str,
        context: dict,
        question: str,
        answer: str,
    ) -> ConversationState:
        now = time.monotonic()
        state = ConversationState(
            conversation_id=uuid.uuid4().hex,
            user_id=int(user_id),
            workspace=str(workspace),
            llm_mode=str(llm_mode),
            user_type=str(user_type),
            repository_revision=str(repository_revision),
            context=copy.deepcopy(context or {}),
            turns=[self._bounded_turn(question, answer)],
            updated_at=now,
        )
        with self._lock:
            self._prune_locked(now)
            self._states[state.conversation_id] = state
            self._prune_locked(now)
        return copy.deepcopy(state)

    def get(
        self,
        conversation_id: Optional[str],
        *,
        user_id: int,
        workspace: str,
        llm_mode: str,
        user_type: str,
        repository_revision: str,
    ) -> Optional[ConversationState]:
        if not conversation_id:
            return None
        now = time.monotonic()
        with self._lock:
            self._prune_locked(now)
            state = self._states.get(str(conversation_id))
            if not state:
                return None
            if (
                state.user_id != int(user_id)
                or state.workspace != str(workspace)
                or state.llm_mode != str(llm_mode)
                or state.user_type != str(user_type)
                or state.repository_revision != str(repository_revision)
            ):
                return None
            state.updated_at = now
            return copy.deepcopy(state)

    def append(
        self,
        conversation_id: str,
        *,
        question: str,
        answer: str,
        context: Optional[dict] = None,
    ) -> Optional[ConversationState]:
        now = time.monotonic()
        with self._lock:
            self._prune_locked(now)
            state = self._states.get(str(conversation_id))
            if not state:
                return None
            state.turns.append(self._bounded_turn(question, answer))
            state.turns = state.turns[-CONVERSATION_MAX_TURNS:]
            if context is not None:
                state.context = copy.deepcopy(context)
            state.updated_at = now
            return copy.deepcopy(state)

    def clear(self) -> None:
        with self._lock:
            self._states.clear()


conversation_store = ConversationStore()
