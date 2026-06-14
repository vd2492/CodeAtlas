"""Per-repository retrieval configuration — the *safe, config-only* knobs an
admin can tune from the UI. No code execution: every field below is data that
the retrieval pipeline reads.

Phase 1 defines the schema, defaults, and load/save. Phase 3 threads a loaded
RetrievalConfig through the context builder so these actually take effect per
workspace (today the builder uses equivalent hardcoded defaults).
"""

import json
from dataclasses import asdict, dataclass, field
from typing import Dict, List

from ..config import retrieval_config_path

# Generic stopwords stripped from questions before keyword matching.
DEFAULT_STOPWORDS: List[str] = [
    "how", "does", "do", "what", "where", "when", "why", "which", "who",
    "is", "are", "the", "a", "an", "work", "works", "working", "use",
    "uses", "used", "using", "tell", "explain", "show", "me", "in",
    "of", "to", "for", "and", "or", "with", "this", "that", "about", "flow",
]


@dataclass
class RetrievalConfig:
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


def load_retrieval_config(workspace: str) -> RetrievalConfig:
    path = retrieval_config_path(workspace)
    if path.exists():
        return RetrievalConfig.from_dict(json.loads(path.read_text()))
    return RetrievalConfig()


def save_retrieval_config(workspace: str, config: RetrievalConfig) -> None:
    path = retrieval_config_path(workspace)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(config.to_dict(), indent=2))
