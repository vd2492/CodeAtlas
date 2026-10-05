"""Repo groups: admin-defined sets of repositories that are asked about together.

A group never grants anything by itself. A user can use a group only when they
can already reach every member repo, so it cannot widen read access.
"""

import hashlib
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


ADHOC_GROUP_NAME = "Selected repositories"


def _parse_adhoc_selection(repos_csv: str) -> list[tuple[str, Optional[int]]]:
    """Split "slug,slug@12" into (slug, pinned branch id or None) pairs."""
    picked: dict[str, Optional[int]] = {}
    for part in str(repos_csv or "").split(","):
        token = part.strip()
        if not token:
            continue
        slug, _, branch_text = token.partition("@")
        slug = slug.strip()
        branch_id = None
        if branch_text:
            if not branch_text.strip().isdigit():
                raise HTTPException(status_code=400, detail="Invalid branch selection.")
            branch_id = int(branch_text)
        if slug in picked and picked[slug] != branch_id:
            raise HTTPException(
                status_code=400,
                detail=f"Select one branch for repository '{slug}'.",
            )
        picked[slug] = branch_id
    return sorted(picked.items())


def adhoc_group_for_user(repos_csv: str, user: dict) -> dict:
    """An unsaved group built from repos the user picked for one question.

    Each entry is a repo slug, optionally pinned to one of its branches as
    "slug@<branch id>"; unpinned repos use their default branch. It is shaped
    like a stored group so the normal group ask flow can run it, but nothing is
    written to the database. Each repo is checked against the user's own
    grants, so picking repos or branches never widens what they can read.
    """
    selection = _parse_adhoc_selection(repos_csv)
    if not MIN_GROUP_MEMBERS <= len(selection) <= MAX_GROUP_MEMBERS:
        raise HTTPException(
            status_code=400,
            detail=(
                f"Select between {MIN_GROUP_MEMBERS} and {MAX_GROUP_MEMBERS} "
                "repositories."
            ),
        )
    is_admin = user.get("role") == "admin"
    members = []
    for slug, branch_id in selection:
        repo = db.get_repo_by_slug(slug)
        if not repo:
            # Non-admins get the same answer for "missing" and "not granted".
            if is_admin:
                raise HTTPException(status_code=404, detail=f"Repository '{slug}' not found.")
            raise HTTPException(
                status_code=403,
                detail="You do not have access to every selected repository.",
            )
        branch_name = None
        if branch_id is not None:
            branch = db.get_repo_branch(branch_id)
            if not branch or branch["repo_id"] != repo["id"]:
                if is_admin or db.user_has_repo(user["id"], repo["workspace"]):
                    raise HTTPException(
                        status_code=404,
                        detail=f"Branch not found in repository '{repo['name']}'.",
                    )
                raise HTTPException(
                    status_code=403,
                    detail="You do not have access to every selected repository.",
                )
            branch_name = branch["name"]
        members.append({
            "repo_id": repo["id"],
            "branch_id": branch_id,
            "position": len(members),
            "repo_slug": repo["slug"],
            "repo_name": repo["name"],
            "repo_workspace": repo["workspace"],
            "repo_status": repo["status"],
            "allow_shared_fallback": repo["allow_shared_fallback"],
            "branch_name": branch_name,
        })
    identity = ",".join(
        f"{m['repo_slug']}@{m['branch_id']}" if m["branch_id"] else m["repo_slug"]
        for m in members
    )
    group = {
        "id": None,
        "slug": "adhoc-" + hashlib.sha1(identity.encode("utf-8")).hexdigest()[:12],
        "name": ADHOC_GROUP_NAME,
        "description": "",
        "members": members,
    }
    if not db.user_can_access_repo_group(user, group):
        raise HTTPException(
            status_code=403,
            detail="You do not have access to every selected repository.",
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
