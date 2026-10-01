"""Repo groups: admin-defined sets of repositories that are asked about together.

A group never grants anything by itself. A user can use a group only when they
can already reach every member repo, so it cannot widen read access.
"""

from typing import Optional

from fastapi import HTTPException

from .. import db
from ..config import graph_path

MIN_GROUP_MEMBERS = 2
MAX_GROUP_MEMBERS = 6


def member_workspace(member: dict) -> Optional[str]:
    """The workspace a member contributes: its pinned branch when the admin
    chose one, otherwise the same default the single-repo path uses."""
    if member.get("branch_id"):
        branch = db.get_repo_branch(member["branch_id"])
        if branch and branch["repo_id"] == member["repo_id"]:
            if branch.get("workspace") and branch["index_status"] in {"ready", "indexing"}:
                return branch["workspace"]
            return None
    workspace = member["repo_workspace"]
    legacy = db.get_legacy_repo_branch(member["repo_id"])
    if legacy and legacy.get("workspace"):
        return legacy["workspace"]
    return workspace


def resolve_group_members(group: dict) -> list[dict]:
    """Validate that every member is queryable and return ready-to-use targets."""
    members = group.get("members") or []
    if len(members) < MIN_GROUP_MEMBERS:
        raise HTTPException(
            status_code=409,
            detail=f"Group '{group['name']}' needs at least {MIN_GROUP_MEMBERS} repositories.",
        )
    resolved = []
    for member in members:
        name = member["repo_name"]
        if member["repo_status"] != "published":
            raise HTTPException(
                status_code=409,
                detail=f"Repository '{name}' in this group is not published.",
            )
        workspace = member_workspace(member)
        if not workspace or not graph_path(workspace).exists():
            raise HTTPException(
                status_code=409,
                detail=f"Repository '{name}' in this group has no indexed data available.",
            )
        repo = db.get_repo_by_slug(member["repo_slug"])
        resolved.append({
            "repo": repo,
            "name": name,
            "slug": member["repo_slug"],
            "workspace": workspace,
        })
    return resolved


def require_group_for_user(slug: str, user: dict) -> dict:
    group = db.get_repo_group_by_slug(slug)
    if not group:
        raise HTTPException(status_code=404, detail="Repository group not found.")
    if not db.user_can_access_repo_group(user, group):
        raise HTTPException(
            status_code=403,
            detail="You do not have access to every repository in this group.",
        )
    return group


def public_group(group: dict) -> dict:
    return {
        "slug": group["slug"],
        "name": group["name"],
        "description": group.get("description") or "",
        "repos": [
            {
                "slug": member["repo_slug"],
                "name": member["repo_name"],
                "branch": member.get("branch_name"),
            }
            for member in group.get("members") or []
        ],
    }
