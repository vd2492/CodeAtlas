"""Per-repository retrieval configuration — the *safe, config-only* knobs an
admin can tune from the UI. No code execution: every field below is data that
the retrieval pipeline reads.

Phase 1 defines the schema, defaults, and load/save. Phase 3 threads a loaded
RetrievalConfig through the context builder so these actually take effect per
workspace (today the builder uses equivalent hardcoded defaults).
"""

import json
import math
import uuid
from dataclasses import asdict, dataclass, field
from typing import Dict, List, Mapping

from ..config import DEFAULT_WORKSPACE, retrieval_config_path

# Generic stopwords stripped from questions before keyword matching.
DEFAULT_STOPWORDS: List[str] = [
    "how", "does", "do", "what", "where", "when", "why", "which", "who",
    "is", "are", "the", "a", "an", "work", "works", "working", "use",
    "uses", "used", "using", "tell", "explain", "show", "me", "in",
    "of", "to", "for", "and", "or", "with", "this", "that", "about", "flow",
]

DEFAULT_PRE_SEARCH_INSTRUCTION = (
    "First, translate the user's terminology into the codebase's canonical "
    "terminology. Users often use business or colloquial terms that don't match "
    "actual class, feature, module, or file names. Before searching or making "
    "changes, identify the most likely internal name(s), including synonyms, "
    "abbreviations, legacy names, and business terms. Use the mapped codebase "
    "names for all subsequent search and reasoning. If multiple mappings are "
    "possible, consider all likely candidates before proceeding."
)


@dataclass
class RetrievalConfig:
    # Free-text guidance for the read-only agent's terminology/search planning.
    pre_search_instruction: str = DEFAULT_PRE_SEARCH_INSTRUCTION

    # Query understanding
    stopwords: List[str] = field(default_factory=lambda: list(DEFAULT_STOPWORDS))
    synonyms: Dict[str, List[str]] = field(default_factory=dict)        # term -> expansions
    keyword_boosts: Dict[str, float] = field(default_factory=dict)      # term -> score multiplier

    # Deterministic anchors for important questions (repo-specific)
    preferred_components: List[str] = field(default_factory=list)
    preferred_methods: List[str] = field(default_factory=list)

    # Size / shape of the context handed to the LLM
    node_limit: int = 16          # max nodes in context
    relation_limit: int = 24      # max relations in context
    excerpt_nodes: int = 6        # how many top nodes get source code attached
    excerpt_max_lines: int = 22   # lines per source excerpt
    excerpt_max_chars: int = 1100 # hard cap per source excerpt

    # Privacy: when False, the shared "Kimi" LLM tier is never used for this repo.
    allow_shared_fallback: bool = True

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "RetrievalConfig":
        known = {f for f in cls.__dataclass_fields__}  # ignore unknown keys safely
        return cls(**{k: v for k, v in (data or {}).items() if k in known})


class RetrievalConfigValidationError(ValueError):
    """Raised when an admin-supplied retrieval config has an invalid shape."""


def validate_retrieval_config(data: object) -> RetrievalConfig:
    """Validate JSON data strictly and return the canonical config.

    Missing fields retain their defaults, while unknown fields and mismatched
    types are rejected so a typo cannot be silently saved and ignored.
    """
    if not isinstance(data, dict):
        raise RetrievalConfigValidationError("Config must be a JSON object.")

    known = set(RetrievalConfig.__dataclass_fields__)
    unknown = sorted(set(data) - known)
    if unknown:
        names = ", ".join(unknown)
        raise RetrievalConfigValidationError(f"Unknown config field(s): {names}.")

    if "pre_search_instruction" in data:
        instruction = data["pre_search_instruction"]
        if not isinstance(instruction, str):
            raise RetrievalConfigValidationError(
                "'pre_search_instruction' must be a string."
            )
        if len(instruction) > 4000:
            raise RetrievalConfigValidationError(
                "'pre_search_instruction' must be 4000 characters or fewer."
            )

    def string_list(field_name: str) -> None:
        value = data.get(field_name)
        if value is None and field_name not in data:
            return
        if not isinstance(value, list) or any(not isinstance(item, str) for item in value):
            raise RetrievalConfigValidationError(
                f"'{field_name}' must be an array of strings."
            )

    for field_name in ("stopwords", "preferred_components", "preferred_methods"):
        string_list(field_name)

    synonyms = data.get("synonyms")
    if synonyms is not None or "synonyms" in data:
        if not isinstance(synonyms, Mapping) or any(
            not isinstance(term, str)
            or not isinstance(expansions, list)
            or any(not isinstance(expansion, str) for expansion in expansions)
            for term, expansions in (synonyms.items() if isinstance(synonyms, Mapping) else ())
        ):
            raise RetrievalConfigValidationError(
                "'synonyms' must be an object whose values are arrays of strings."
            )

    boosts = data.get("keyword_boosts")
    if boosts is not None or "keyword_boosts" in data:
        if not isinstance(boosts, Mapping) or any(
            not isinstance(term, str)
            or isinstance(multiplier, bool)
            or not isinstance(multiplier, (int, float))
            or not math.isfinite(multiplier)
            or multiplier <= 0
            for term, multiplier in (boosts.items() if isinstance(boosts, Mapping) else ())
        ):
            raise RetrievalConfigValidationError(
                "'keyword_boosts' must be an object with positive numeric values."
            )

    for field_name in (
        "node_limit",
        "relation_limit",
        "excerpt_nodes",
        "excerpt_max_lines",
        "excerpt_max_chars",
    ):
        if field_name not in data:
            continue
        value = data[field_name]
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise RetrievalConfigValidationError(
                f"'{field_name}' must be a positive integer."
            )

    if "allow_shared_fallback" in data and not isinstance(
        data["allow_shared_fallback"], bool
    ):
        raise RetrievalConfigValidationError(
            "'allow_shared_fallback' must be true or false."
        )

    return RetrievalConfig.from_dict(data)


def load_retrieval_config(workspace: str) -> RetrievalConfig:
    path = retrieval_config_path(workspace)
    if path.exists():
        return RetrievalConfig.from_dict(json.loads(path.read_text()))
    return RetrievalConfig()


def save_retrieval_config(workspace: str, config: RetrievalConfig) -> None:
    path = retrieval_config_path(workspace)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        temporary_path.write_text(json.dumps(config.to_dict(), indent=2))
        temporary_path.replace(path)
    finally:
        temporary_path.unlink(missing_ok=True)


# --- Default workspace seed --------------------------------------------------
# The demo "destiny" anchors, migrated out of build_context() and into data.
# Anchor names are display names (must match relation_utils.readable_name);
# anchors that don't resolve in a given graph are skipped gracefully.

DEFAULT_DESTINY_CONFIG = RetrievalConfig(
    synonyms={
        "habit": ["habits"],
        "habits": ["habit"],
        "completion": ["complete", "completed", "state", "today", "toggle", "streak"],
        "complete": ["completion", "completed", "state"],
        "completed": ["completion", "complete", "state"],
        "toggle": ["completion", "state", "today"],
        "streak": ["completion", "state"],
        "login": ["auth", "authentication", "signin"],
        "auth": ["login", "authentication", "signin"],
        "signin": ["login", "auth"],
        "revision": ["revisions", "spaced", "repetition"],
        "revisions": ["revision"],
        "spaced": ["revision", "repetition"],
    },
    preferred_components=[
        "HabitsScreen", "HomeScreen", "RevisionsScreen", "LoginScreen",
        "HabitsViewModel", "HomeViewModel", "RevisionsViewModel",
        "HabitRepository", "AuthRepository", "HabitDao", "HabitEntity",
        "HabitCompletionEntity",
    ],
    preferred_methods=[
        # Habit
        "HomeViewModel.toggleHabit", "HabitsViewModel.continueHabitStreak",
        "HabitRepository.getTodayHabitsWithCompletion", "HabitRepository.setHabitStateToday",
        "HabitRepository.computeDisplayedHabitStreak", "HabitRepository.computeHabitStreak",
        "HabitRepository.currentUserHabitsCollection", "HabitRepository.calculateHabitHistory",
        "HabitDao.iscompletedon",
        # Revision
        "HabitRepository.addRevisionTopic", "HabitRepository.deleteRevisionTopic",
        "HabitRepository.calculateRevisionDueAt", "HabitRepository.findMissedRevisionDay",
        "HabitRepository.currentUserRevisionsCollection", "RevisionsViewModel.startRevision",
        "RevisionsViewModel.completeRevision", "RevisionsViewModel.completeActiveRevision",
        "RevisionsViewModel.restartRevisionTopic",
        # Login / auth
        "AuthRepository.login", "AuthRepository.loginWithGoogleIdToken",
        "AuthRepository.register", "AuthRepository.logout",
        "AuthRepository.saveUserProfile", "AuthRepository.requestGoogleIdToken",
        "AuthRepository.extractGoogleIdToken", "AuthRepository.mapGoogleSignInError",
    ],
)


def seed_default_retrieval_config() -> None:
    """Write the migrated destiny anchors as the default workspace's config on
    first boot (data/ is gitignored, so this can't be a committed file). Other
    workspaces get RetrievalConfig() defaults until an admin tunes them."""
    if retrieval_config_path(DEFAULT_WORKSPACE).exists():
        return
    save_retrieval_config(DEFAULT_WORKSPACE, DEFAULT_DESTINY_CONFIG)
