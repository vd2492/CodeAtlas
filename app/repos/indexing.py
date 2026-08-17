"""Index a cloned repository with graphify and store its graph in the workspace.

graphify is a structural extractor (no LLM needed): `graphify update . --no-cluster`.
"""

import shutil
import subprocess
import logging

from ..config import graph_path, repo_clone_dir, source_index_path
from ..retrieval.source_index import build_source_index

INDEX_TIMEOUT = 1800
logger = logging.getLogger(__name__)


def index_repo(workspace: str):
    """Run graphify over the workspace repo and place graph.json in the workspace."""
    repo = repo_clone_dir(workspace)
    if not repo.exists():
        raise RuntimeError(f"nothing to index: {repo} does not exist")

    result = subprocess.run(
        ["graphify", "update", ".", "--no-cluster"],
        cwd=str(repo),
        capture_output=True,
        text=True,
        timeout=INDEX_TIMEOUT,
    )
    if result.returncode != 0:
        raise RuntimeError(f"indexing failed: {result.stderr.strip() or result.stdout.strip()}")

    produced = repo / "graphify-out" / "graph.json"
    if not produced.exists():
        raise RuntimeError("graphify did not produce graphify-out/graph.json")

    target = graph_path(workspace)
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy(produced, target)
    try:
        build_source_index(repo, source_index_path(workspace))
    except Exception:
        logger.exception("Failed to build persisted source index for workspace %s.", workspace)
    return target
