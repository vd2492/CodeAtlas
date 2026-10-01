"""Admin management and user listing for repo groups."""

import re
import sqlite3
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel

from .. import db
from ..auth.sessions import require_admin, require_user
from .groups import MAX_GROUP_MEMBERS, MIN_GROUP_MEMBERS, public_group

admin_router = APIRouter(prefix="/admin/groups", tags=["admin-groups"])
user_router = APIRouter(tags=["repo-groups"])

SLUG_RE = re.compile(r"^[a-z0-9][a-z0-9-]{0,62}$")


class GroupMember(BaseModel):
    repo_slug: str
    branch_id: Optional[int] = None


class CreateGroupRequest(BaseModel):
    slug: str
    name: str
    description: str = ""
    members: list[GroupMember]


class UpdateGroupRequest(BaseModel):
    name: Optional[str] = None
    description: Optional[str] = None
    members: Optional[list[GroupMember]] = None


class GroupGrantRequest(BaseModel):
    username: str


def _require_group(slug: str) -> dict:
    group = db.get_repo_group_by_slug(slug)
    if not group:
        raise HTTPException(status_code=404, detail=f"No group with slug '{slug}'.")
    return group


def _validated_members(members: list[GroupMember]) -> list[dict]:
    if not (MIN_GROUP_MEMBERS <= len(members) <= MAX_GROUP_MEMBERS):
        raise HTTPException(
            status_code=400,
            detail=(
                f"A group needs between {MIN_GROUP_MEMBERS} and "
                f"{MAX_GROUP_MEMBERS} repositories."
            ),
        )
    resolved, seen = [], set()
    for member in members:
        repo = db.get_repo_by_slug(member.repo_slug)
        if not repo:
            raise HTTPException(
                status_code=404, detail=f"No repo with slug '{member.repo_slug}'."
            )
        if repo["id"] in seen:
            raise HTTPException(
                status_code=400,
                detail=f"Repository '{repo['name']}' is listed more than once.",
            )
        seen.add(repo["id"])
        if member.branch_id is not None:
            branch = db.get_repo_branch(member.branch_id)
            if not branch or branch["repo_id"] != repo["id"]:
                raise HTTPException(
                    status_code=404,
                    detail=f"Branch not found for repository '{repo['name']}'.",
                )
        resolved.append({"repo_id": repo["id"], "branch_id": member.branch_id})
    return resolved


@admin_router.get("")
def list_groups(admin: dict = Depends(require_admin)):
    return {"groups": db.list_repo_groups()}


@admin_router.post("")
def create_group(req: CreateGroupRequest, admin: dict = Depends(require_admin)):
    if not SLUG_RE.match(req.slug):
        raise HTTPException(
            status_code=400,
            detail="slug must be lowercase letters, digits, and dashes.",
        )
    name = req.name.strip()
    if not name:
        raise HTTPException(status_code=400, detail="name is required.")
    members = _validated_members(req.members)
    try:
        group = db.create_repo_group(req.slug, name, req.description.strip(), members)
    except sqlite3.IntegrityError:
        raise HTTPException(
            status_code=409, detail=f"A group with slug '{req.slug}' already exists."
        )
    db.record_audit(admin["username"], "create_group", req.slug)
    return {"group": group}


@admin_router.put("/{slug}")
def update_group(
    slug: str, req: UpdateGroupRequest, admin: dict = Depends(require_admin)
):
    _require_group(slug)
    name = req.name.strip() if req.name is not None else None
    if name is not None and not name:
        raise HTTPException(status_code=400, detail="name cannot be empty.")
    members = _validated_members(req.members) if req.members is not None else None
    group = db.update_repo_group(
        slug,
        name=name,
        description=req.description.strip() if req.description is not None else None,
        members=members,
    )
    db.record_audit(admin["username"], "update_group", slug)
    return {"group": group}


@admin_router.delete("/{slug}")
def delete_group(slug: str, admin: dict = Depends(require_admin)):
    _require_group(slug)
    db.delete_repo_group(slug)
    db.record_audit(admin["username"], "delete_group", slug)
    return {"deleted": slug}


@admin_router.post("/{slug}/grant")
def grant_group_access(
    slug: str, req: GroupGrantRequest, admin: dict = Depends(require_admin)
):
    """Give a user access to every member repo, which is what makes the group
    usable for them. Each repo grant is the same one /admin/repos/{slug}/grant makes."""
    group = _require_group(slug)
    user = db.get_user_by_login_identifier(req.username)
    if not user:
        raise HTTPException(status_code=404, detail=f"No user '{req.username}'.")
    for member in group["members"]:
        db.grant_access(user["id"], member["repo_id"])
    db.record_audit(admin["username"], "grant_group", slug, req.username)
    return {"granted": {"username": req.username, "slug": slug}}


@user_router.get("/repo/groups")
def user_list_groups(user: dict = Depends(require_user)):
    """Groups this user can ask about: those whose member repos they can all reach."""
    groups = [
        public_group(group)
        for group in db.list_repo_groups()
        if db.user_can_access_repo_group(user, group)
        and all(m["repo_status"] == "published" for m in group["members"])
        and len(group["members"]) >= MIN_GROUP_MEMBERS
    ]
    return {"groups": groups}
