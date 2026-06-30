"""Central configuration and filesystem layout for CodeAtlas.

All tunable behavior is read from environment variables so the same code runs
locally (single default workspace) and, later, as a multi-tenant service.
"""

import os
from pathlib import Path

# Repo root = two levels up from this file (app/config.py -> app -> <root>).
ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = Path(os.environ.get("CODEATLAS_DATA_DIR", ROOT / "data"))
WORKSPACES_DIR = DATA_DIR / "workspaces"
DB_PATH = Path(os.environ.get("CODEATLAS_DB_PATH", DATA_DIR / "codeatlas.db"))
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


def branch_version_workspace(repo_workspace: str, branch_id: int, commit_sha: str) -> str:
    """Stable, server-generated workspace name for an immutable branch version."""
    return f"{repo_workspace}--branch-{branch_id}--{commit_sha.lower()}"
