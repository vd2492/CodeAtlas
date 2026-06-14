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
