"""Index a cloned repository with graphify and store its graph in the workspace.

graphify is a structural extractor (no LLM needed): `graphify update . --no-cluster`.
"""

import shutil
import subprocess

from ..config import graph_path, repo_clone_dir

INDEX_TIMEOUT = 1800


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
    return target
