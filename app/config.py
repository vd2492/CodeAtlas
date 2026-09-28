"""Central configuration and filesystem layout for CodeAtlas.

All tunable behavior is read from environment variables so the same code runs
locally (single default workspace) and, later, as a multi-tenant service.
"""

import os
import re
from pathlib import Path

# Repo root = two levels up from this file (app/config.py -> app -> <root>).
ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = Path(os.environ.get("CODEATLAS_DATA_DIR", ROOT / "data"))
WORKSPACES_DIR = DATA_DIR / "workspaces"
DB_PATH = Path(os.environ.get("CODEATLAS_DB_PATH", DATA_DIR / "codeatlas.db"))
SQLITE_BUSY_TIMEOUT_MS = max(
    0, int(os.environ.get("CODEATLAS_SQLITE_BUSY_TIMEOUT_MS", "10000"))
)
SESSION_MAX_AGE_SECONDS = int(
    os.environ.get("CODEATLAS_SESSION_MAX_AGE_SECONDS", str(60 * 60 * 24 * 30))
)
BRANCH_SYNC_MAX_WORKERS = max(
    1, int(os.environ.get("CODEATLAS_BRANCH_SYNC_MAX_WORKERS", "2"))
)
BRANCH_SYNC_POLL_SECONDS = max(
    15, int(os.environ.get("CODEATLAS_BRANCH_SYNC_POLL_SECONDS", "60"))
)
BRANCH_FRESHNESS_INTERVAL_SECONDS = max(
    60, int(os.environ.get("CODEATLAS_BRANCH_FRESHNESS_INTERVAL_SECONDS", "300"))
)
BRANCH_USER_SYNC_COOLDOWN_SECONDS = max(
    0, int(os.environ.get("CODEATLAS_BRANCH_USER_SYNC_COOLDOWN_SECONDS", "60"))
)
BRANCH_VERSION_RETENTION_SECONDS = max(
    3600, int(os.environ.get("CODEATLAS_BRANCH_VERSION_RETENTION_SECONDS", "86400"))
)


def _env_flag(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


# Optional UI surfaces, hidden by default because they are not in use yet.
# Each is remote config: set the env var to false and restart to bring one
# back, without a code or frontend change.
UI_FEATURE_ENV = {
    "repo_summary_hidden": "CODEATLAS_HIDE_REPO_SUMMARY",
    "flow_explorer_hidden": "CODEATLAS_HIDE_FLOW_EXPLORER",
    "graph_search_hidden": "CODEATLAS_HIDE_GRAPH_SEARCH",
}


def ui_feature_flags() -> dict:
    """Which optional UI surfaces the client should hide. Read per call so the
    served value always reflects the current environment."""
    return {
        key: _env_flag(env_name, True)
        for key, env_name in UI_FEATURE_ENV.items()
    }


# --- Shared LLM registry -----------------------------------------------------
# Shared ("admin-provided") LLMs are configured entirely from the environment so
# swapping Mimo for another model, or offering several, needs no code change:
#
#   CODEATLAS_SHARED_LLMS=mimo,luna
#   CODEATLAS_SHARED_LLM_MIMO_MODEL=mimo-v2.5
#   CODEATLAS_SHARED_LLM_MIMO_API_KEY=...
#   CODEATLAS_SHARED_LLM_MIMO_BASE_URL=https://api.xiaomimimo.com/v1
#   CODEATLAS_SHARED_LLM_MIMO_PROVIDER=openai_compatible      (optional)
#   CODEATLAS_SHARED_LLM_MIMO_NAME=Mimo v2.5                  (optional)
#   CODEATLAS_DEFAULT_SHARED_LLM=mimo                         (optional)
#
# The model is declared rather than discovered: an endpoint's model list is
# optional in the OpenAI-compatible spec and often returns many entries, so
# probing it would be both unreliable and a startup dependency on the provider.
# The model string is already required to call the API at all, so declaring it
# adds no configuration that was not needed anyway.
#
# When CODEATLAS_SHARED_LLMS is unset the pre-registry variables still define a
# single shared LLM, so existing deployments keep working untouched.
SHARED_LLM_ENV_PREFIX = "CODEATLAS_SHARED_LLM"
# The two names below exist only so deployments predating the registry keep
# working unchanged. Nothing outside the legacy path may reference a model.
LEGACY_SHARED_LLM_ID = "mimo"
LEGACY_SHARED_LLM_MODEL = "mimo-v2.5"


def _shared_llm_env_segment(llm_id: str) -> str:
    return re.sub(r"[^A-Za-z0-9]+", "_", str(llm_id or "")).strip("_").upper()


def _shared_llm_env(llm_id: str, suffix: str, default: str = "") -> str:
    key = f"{SHARED_LLM_ENV_PREFIX}_{_shared_llm_env_segment(llm_id)}_{suffix}"
    return str(os.environ.get(key, default) or "").strip()


def _configured_shared_llm_ids() -> list[str]:
    raw = os.environ.get("CODEATLAS_SHARED_LLMS", "")
    ids = []
    for part in re.split(r"[,\s]+", raw):
        candidate = part.strip().lower()
        if candidate and candidate not in ids:
            ids.append(candidate)
    return ids


def _legacy_shared_llm() -> "dict | None":
    """The single shared LLM described by the pre-registry variables."""
    base_url = str(os.environ.get("CODEATLAS_LLM_BASE_URL", "") or "").strip()
    api_key = str(
        os.environ.get("CODEATLAS_LLM_API_KEY")
        or os.environ.get("OPENAI_API_KEY")
        or os.environ.get("ANTHROPIC_API_KEY")
        or ""
    ).strip()
    model = str(
        os.environ.get("CODEATLAS_MIMO_MODEL")
        or os.environ.get("CODEATLAS_LLM_MODEL")
        or LEGACY_SHARED_LLM_MODEL
    ).strip()
    if not base_url and not api_key:
        return None
    return {
        "id": LEGACY_SHARED_LLM_ID,
        "name": model,
        "model": model,
        "provider": str(
            os.environ.get("CODEATLAS_LLM_PROVIDER", "openai_compatible") or ""
        ).strip() or "openai_compatible",
        "base_url": base_url,
        "api_key": api_key,
    }


def shared_llms() -> list[dict]:
    """Every configured shared LLM, in declaration order.

    Entries include the API key, so this is server-side only; the client is
    served public_shared_llms() instead."""
    entries = []
    for llm_id in _configured_shared_llm_ids():
        model = _shared_llm_env(llm_id, "MODEL")
        base_url = _shared_llm_env(llm_id, "BASE_URL")
        api_key = _shared_llm_env(llm_id, "API_KEY")
        if not model or not base_url or not api_key:
            # Half-configured entries are skipped rather than offered and then
            # failing at request time.
            continue
        entries.append({
            "id": llm_id,
            "name": _shared_llm_env(llm_id, "NAME") or model,
            "model": model,
            "provider": _shared_llm_env(llm_id, "PROVIDER") or "openai_compatible",
            "base_url": base_url,
            "api_key": api_key,
        })
    if entries:
        return entries
    legacy = _legacy_shared_llm()
    return [legacy] if legacy else []


def default_shared_llm_id() -> str:
    """The shared LLM selected when none is chosen. Falls back to the first."""
    entries = shared_llms()
    if not entries:
        return ""
    requested = str(os.environ.get("CODEATLAS_DEFAULT_SHARED_LLM", "") or "").strip().lower()
    if requested and any(entry["id"] == requested for entry in entries):
        return requested
    return entries[0]["id"]


def shared_llm(llm_id: str = None) -> "dict | None":
    """One shared LLM by id, or the default when no id is given."""
    entries = shared_llms()
    if not entries:
        return None
    wanted = str(llm_id or "").strip().lower() or default_shared_llm_id()
    for entry in entries:
        if entry["id"] == wanted:
            return entry
    return None


def public_shared_llms() -> list[dict]:
    """Shared LLMs as the client may see them: never includes an API key."""
    return [
        {"id": entry["id"], "name": entry["name"], "model": entry["model"]}
        for entry in shared_llms()
    ]


def shared_llm_mode(llm_id: str) -> str:
    """The llm_mode string that selects one shared LLM."""
    return f"shared:{str(llm_id or '').strip().lower()}"


def shared_llm_id_for_mode(mode: str) -> "str | None":
    """Resolve an llm_mode to a shared LLM id, or None if it is not a shared mode.

    "mimo" is the pre-registry name for "the shared tier" and still selects the
    default, so saved client preferences and Slack settings keep working."""
    value = str(mode or "").strip().lower()
    if not value:
        return None
    if value in {LEGACY_SHARED_LLM_ID, "shared"}:
        # Prefer an entry actually named this; otherwise it just means
        # "the shared tier" and resolves to the default.
        entry = shared_llm(value) if value == LEGACY_SHARED_LLM_ID else None
        if entry:
            return entry["id"]
        return default_shared_llm_id() or None
    if value.startswith("shared:"):
        requested = value.split(":", 1)[1].strip()
        if not requested:
            return default_shared_llm_id() or None
        entry = shared_llm(requested)
        return entry["id"] if entry else None
    return None


def is_shared_llm_mode(mode: str) -> bool:
    value = str(mode or "").strip().lower()
    return value in {LEGACY_SHARED_LLM_ID, "shared"} or value.startswith("shared:")


# The default workspace lets the tool run as the current single-repo app until
# the multi-tenant repo registry (Phase 2) is wired in.
DEFAULT_WORKSPACE = os.environ.get("CODEATLAS_DEFAULT_WORKSPACE", "default")


def workspace_dir(workspace: str = DEFAULT_WORKSPACE) -> Path:
    return WORKSPACES_DIR / workspace


def graph_path(workspace: str = DEFAULT_WORKSPACE) -> Path:
    """Path to a workspace's graph.json. An explicit CODEATLAS_GRAPH_PATH wins
    (handy for pointing the default workspace at an existing graph)."""
    override = os.environ.get("CODEATLAS_GRAPH_PATH")
    if override and workspace == DEFAULT_WORKSPACE:
        return Path(override)
    return workspace_dir(workspace) / "graph" / "graph.json"


def repo_clone_dir(workspace: str) -> Path:
    return workspace_dir(workspace) / "repo"


def retrieval_config_path(workspace: str = DEFAULT_WORKSPACE) -> Path:
    return workspace_dir(workspace) / "retrieval_config.json"


def source_index_path(workspace: str = DEFAULT_WORKSPACE) -> Path:
    return workspace_dir(workspace) / "source_index.json"


def branch_version_workspace(repo_workspace: str, branch_id: int, commit_sha: str) -> str:
    """Stable, server-generated workspace name for an immutable branch version."""
    return f"{repo_workspace}--branch-{branch_id}--{commit_sha.lower()}"
