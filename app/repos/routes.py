"""Admin repository-management routes.

SKELETON (Phase 2): the full lifecycle is declared here — add, index, test,
tune retrieval config, publish, and grant access. Handlers return 501 until the
clone/index helpers (already present in this package) and the DB are wired up,
gated behind admin auth.
"""

from fastapi import APIRouter, HTTPException

router = APIRouter(prefix="/admin/repos", tags=["admin-repos"])

_NOT_YET = "Not implemented yet (Phase 2: admin repo workspace lifecycle)."


@router.get("")
def list_repos():
    raise HTTPException(status_code=501, detail=_NOT_YET)


@router.post("")
def add_repo():
    """Register + clone a repo (https/ssh/gh) into a new workspace."""
    raise HTTPException(status_code=501, detail=_NOT_YET)


@router.post("/{slug}/index")
def index_repo_route(slug: str):
    """Run graphify indexing for the workspace."""
    raise HTTPException(status_code=501, detail=_NOT_YET)


@router.get("/{slug}/config")
def get_retrieval_config(slug: str):
    """Read the safe, config-only retrieval settings for this repo."""
    raise HTTPException(status_code=501, detail=_NOT_YET)


@router.put("/{slug}/config")
def update_retrieval_config(slug: str):
    """Update retrieval settings (stopwords, synonyms, boosts, limits, ...)."""
    raise HTTPException(status_code=501, detail=_NOT_YET)


@router.post("/{slug}/publish")
def publish_repo(slug: str):
    """Mark the workspace published and exposable to granted users."""
    raise HTTPException(status_code=501, detail=_NOT_YET)


@router.post("/{slug}/grant")
def grant_access(slug: str):
    """Grant a user access to this published repo."""
    raise HTTPException(status_code=501, detail=_NOT_YET)
