"""Shared CodeAtlas ask orchestration for web and Slack surfaces.

The low-level retrieval/generation helpers still live in ``app.main`` for now;
this module centralizes the request flow so additional entry points can reuse
the same cache, follow-up, and deep-investigation behavior as the browser UI.
"""

from __future__ import annotations

import hashlib
import logging
import os
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Optional

from fastapi import HTTPException

from . import db
from .config import graph_path
from .llm.admission import LLMCapacityError
from .llm.client import collect_token_usage, token_usage_payload
from .repos.branches import (
    approve_repo_branch,
    branch_job_running,
    discover_remote_branches,
    submit_branch_job,
)

logger = logging.getLogger(__name__)
_ANALYTICS_EXECUTOR = ThreadPoolExecutor(
    max_workers=max(1, int(os.environ.get("CODEATLAS_ANALYTICS_MAX_WORKERS", "1"))),
    thread_name_prefix="codeatlas-analytics",
)


def _main():
    from . import main

    return main


def _record_token_usage_event(payload: dict) -> None:
    try:
        db.record_token_usage(**payload)
    except Exception:
        logger.exception("Failed to record token usage analytics event.")


def schedule_answer_token_usage(
    user: dict,
    workspace: str,
    endpoint: str,
    response: dict,
    *,
    repo: Optional[dict] = None,
    analytics_context: Optional[dict] = None,
) -> bool:
    """Schedule analytics storage from the response's existing usage payload."""
    usage = (response or {}).get("token_usage") or {}
    if not usage.get("available"):
        return False
    context = analytics_context or {}
    payload = {
        "user_id": user.get("id"),
        "username": user.get("username"),
        "repo_slug": repo.get("slug") if repo else None,
        "workspace": workspace,
        "endpoint": endpoint,
        "source": context.get("source") or "web",
        "slack_user_id": context.get("slack_user_id"),
        "slack_team_id": context.get("slack_team_id"),
        "slack_channel_id": context.get("slack_channel_id"),
        "ask_type": context.get("ask_type"),
        "branch": context.get("branch"),
        "provider_used": (response or {}).get("provider_used"),
        "token_usage": dict(usage),
    }
    try:
        _ANALYTICS_EXECUTOR.submit(_record_token_usage_event, payload)
        return True
    except Exception:
        logger.exception("Failed to schedule token usage analytics event.")
        return False


def slack_actor_user(team_id: str, user_id: str, user_type: str = "dev_team") -> dict:
    """Stable synthetic user identity for Slack conversation scoping.

    CodeAtlas DB user IDs are positive AUTOINCREMENT values, so a negative hash
    keeps Slack topic state isolated without creating local users.
    """
    key = f"{team_id}:{user_id}"
    digest = hashlib.sha256(key.encode("utf-8")).hexdigest()
    synthetic_id = -(int(digest[:12], 16) % 2_000_000_000 + 1)
    return {
        "id": synthetic_id,
        "username": f"slack:{key}",
        "role": "admin",
        "user_type": user_type if user_type in {"dev_team", "product_team"} else "dev_team",
        "_session_key": f"slack:{key}",
    }


def answer_single_request(
    request,
    workspace: str,
    user: dict,
    *,
    enforce_limit: bool = True,
    analytics_context: Optional[dict] = None,
) -> dict:
    """Run the existing single-repository ask flow for any authenticated surface."""
    main = _main()
    if enforce_limit:
        main.enforce_rate_limit(user["id"])
    main.enforce_strict_branch_freshness(workspace)
    repo = db.get_repo_by_workspace(workspace)
    allow_shared = bool(repo["allow_shared_fallback"]) if repo else True
    user_llm = request.user_llm or main.load_user_llm(user["id"])
    llm_mode = (request.llm_mode or "auto").lower()
    user_type = main._effective_answer_user_type(user, request.answer_user_type)
    image_attachments = main.normalize_image_attachments(
        getattr(request, "image_attachments", None)
    )
    if image_attachments and request.follow_up:
        raise HTTPException(
            status_code=400,
            detail="Image attachments are only supported on new web questions.",
        )
    revision = main.repository_revision(workspace)
    session_key = str(user.get("_session_key") or "")
    use_session_cache = not (
        request.deep_investigation
        or (llm_mode == "mimo" and not allow_shared)
        or image_attachments
    )
    if use_session_cache:
        cached_response = main.conversation_store.get_cached_answer(
            session_key=session_key,
            user_id=user["id"],
            workspace=workspace,
            llm_mode=llm_mode,
            user_type=user_type,
            repository_revision=revision,
            question=request.question,
        )
        if cached_response:
            response = main._session_cached_answer_response(
                cached_response,
                request.question,
                workspace,
            )
            main.record_answer_activity(
                getattr(request, "activity_request_id", None),
                user_id=user["id"],
                workspace=workspace,
                question=request.question,
                status="answered_from_cache",
                context=response.get("context"),
            )
            state = main._create_conversation_from_response(
                user=user,
                workspace=workspace,
                llm_mode=llm_mode,
                user_type=user_type,
                repository_revision=revision,
                question=request.question,
                response=response,
            )
            response["conversation_id"] = state.conversation_id
            response["answer_user_type"] = user_type
            return response

    use_repo_cache = (
        allow_shared
        and not request.follow_up
        and not request.deep_investigation
        and not image_attachments
        and main._request_uses_shared_tier_only(llm_mode, user_llm)
    )
    if use_repo_cache:
        repo_cached_response = main.conversation_store.get_repo_cached_answer(
            workspace=workspace,
            user_type=user_type,
            repository_revision=revision,
            question=request.question,
        )
        if repo_cached_response:
            response = main._session_cached_answer_response(
                repo_cached_response,
                request.question,
                workspace,
            )
            main.record_answer_activity(
                getattr(request, "activity_request_id", None),
                user_id=user["id"],
                workspace=workspace,
                question=request.question,
                status="answered_from_cache",
                context=response.get("context"),
            )
            state = main._create_conversation_from_response(
                user=user,
                workspace=workspace,
                llm_mode=llm_mode,
                user_type=user_type,
                repository_revision=revision,
                question=request.question,
                response=response,
            )
            response["conversation_id"] = state.conversation_id
            response["answer_user_type"] = user_type
            return response

    try:
        with main.llm_admission.slot(), collect_token_usage() as token_usage:
            state = None
            if request.follow_up and request.conversation_id:
                state = main.conversation_store.get(
                    request.conversation_id,
                    user_id=user["id"],
                    session_key=session_key,
                    workspace=workspace,
                    llm_mode=llm_mode,
                    user_type=user_type,
                    repository_revision=revision,
                )
            if state and (
                request.deep_investigation
                or main.is_related_follow_up(state, request.question)
            ):
                response = main.answer_follow_up(
                    request.question,
                    state,
                    workspace=workspace,
                    user_llm=user_llm,
                    allow_shared_fallback=allow_shared,
                    llm_mode=llm_mode,
                    user_type=user_type,
                    deep_investigation=request.deep_investigation,
                    activity_request_id=getattr(request, "activity_request_id", None),
                    activity_user_id=user["id"],
                )
                main.conversation_store.append(
                    state.conversation_id,
                    question=request.question,
                    answer=response["answer"],
                    context=response.get("context"),
                )
                response["conversation_id"] = state.conversation_id
                response["token_usage"] = token_usage_payload(token_usage)
                response["answer_user_type"] = user_type
                main._remember_session_answer(
                    user=user,
                    workspace=workspace,
                    llm_mode=llm_mode,
                    user_type=user_type,
                    repository_revision=revision,
                    question=request.question,
                    response=response,
                )
                schedule_answer_token_usage(
                    user,
                    workspace,
                    "repo.ask.follow_up",
                    response,
                    repo=repo,
                    analytics_context=analytics_context,
                )
                return response

            response = main.answer_question(
                request.question,
                workspace=workspace,
                user_llm=user_llm,
                allow_shared_fallback=allow_shared,
                llm_mode=llm_mode,
                user_type=user_type,
                image_attachments=image_attachments,
                activity_request_id=getattr(request, "activity_request_id", None),
                activity_user_id=user["id"],
            )
            state = main._create_conversation_from_response(
                user=user,
                workspace=workspace,
                llm_mode=llm_mode,
                user_type=user_type,
                repository_revision=revision,
                question=request.question,
                response=response,
            )
            response["conversation_id"] = state.conversation_id
            response["follow_up_reused"] = False
            response["follow_up_fallback"] = bool(request.follow_up)
            response["token_usage"] = token_usage_payload(token_usage)
            response["answer_user_type"] = user_type
            if not image_attachments:
                main._remember_session_answer(
                    user=user,
                    workspace=workspace,
                    llm_mode=llm_mode,
                    user_type=user_type,
                    repository_revision=revision,
                    question=request.question,
                    response=response,
                )
            if use_repo_cache:
                main._remember_repo_answer(
                    workspace=workspace,
                    user_type=user_type,
                    repository_revision=revision,
                    question=request.question,
                    response=response,
                )
            schedule_answer_token_usage(
                user,
                workspace,
                "repo.ask",
                response,
                repo=repo,
                analytics_context=analytics_context,
            )
            return response
    except LLMCapacityError as error:
        raise HTTPException(
            status_code=503,
            detail=str(error),
            headers={"Retry-After": "5"},
        )
    except RuntimeError as error:
        raise HTTPException(status_code=400, detail=str(error))
    except Exception as error:
        raise HTTPException(status_code=500, detail=f"LLM request failed: {str(error)}")


def answer_compare_request(
    request,
    workspace: str,
    user: dict,
    *,
    repo: Optional[dict] = None,
    left: Optional[dict] = None,
    right: Optional[dict] = None,
    enforce_limit: bool = True,
    analytics_context: Optional[dict] = None,
) -> dict:
    """Run the existing branch-comparison ask flow for any authenticated surface."""
    main = _main()
    if enforce_limit:
        main.enforce_rate_limit(user["id"])
    repo = repo or main._resolve_compare_base_repo(workspace, user)
    left = left or main._resolve_compare_branch(repo, request.left_branch, "Branch A")
    right = right or main._resolve_compare_branch(repo, request.right_branch, "Branch B")
    if left["branch"]["id"] == right["branch"]["id"]:
        raise HTTPException(
            status_code=400,
            detail="Choose two different branches to compare.",
        )
    main.enforce_strict_branch_freshness(left["workspace"])
    main.enforce_strict_branch_freshness(right["workspace"])
    allow_shared = bool(repo["allow_shared_fallback"])
    user_llm = request.user_llm or main.load_user_llm(user["id"])
    llm_mode = (request.llm_mode or "auto").lower()
    user_type = main._effective_answer_user_type(user, request.answer_user_type)
    comparison_workspace = main._comparison_workspace_key(repo, left, right)
    comparison_revision = main._comparison_revision(left, right)
    session_key = str(user.get("_session_key") or "")
    use_session_cache = not (
        request.deep_investigation
        or (llm_mode == "mimo" and not allow_shared)
    )
    if use_session_cache:
        cached_response = main.conversation_store.get_cached_answer(
            session_key=session_key,
            user_id=user["id"],
            workspace=comparison_workspace,
            llm_mode=llm_mode,
            user_type=user_type,
            repository_revision=comparison_revision,
            question=request.question,
        )
        if cached_response:
            response = main._session_cached_answer_response(
                cached_response,
                request.question,
                workspace=None,
            )
            main.record_answer_activity(
                getattr(request, "activity_request_id", None),
                user_id=user["id"],
                workspace=comparison_workspace,
                question=request.question,
                status="answered_from_cache",
                context=response.get("context"),
            )
            state = main._create_conversation_from_response(
                user=user,
                workspace=comparison_workspace,
                llm_mode=llm_mode,
                user_type=user_type,
                repository_revision=comparison_revision,
                question=request.question,
                response=response,
            )
            response["conversation_id"] = state.conversation_id
            response["answer_user_type"] = user_type
            return response

    try:
        with main.llm_admission.slot(), collect_token_usage() as token_usage:
            state = None
            if request.follow_up and request.conversation_id:
                state = main.conversation_store.get(
                    request.conversation_id,
                    user_id=user["id"],
                    session_key=session_key,
                    workspace=comparison_workspace,
                    llm_mode=llm_mode,
                    user_type=user_type,
                    repository_revision=comparison_revision,
                )
            if state and (
                request.deep_investigation
                or main.is_related_follow_up(state, request.question)
            ):
                response = main.answer_compare_follow_up(
                    request.question,
                    state,
                    left,
                    right,
                    user_llm=user_llm,
                    allow_shared_fallback=allow_shared,
                    llm_mode=llm_mode,
                    user_type=user_type,
                    deep_investigation=request.deep_investigation,
                    activity_request_id=getattr(request, "activity_request_id", None),
                    activity_user_id=user["id"],
                )
                main.conversation_store.append(
                    state.conversation_id,
                    question=request.question,
                    answer=response["answer"],
                    context=response.get("context"),
                )
                response["conversation_id"] = state.conversation_id
                response["token_usage"] = token_usage_payload(token_usage)
                response["answer_user_type"] = user_type
                main._remember_session_answer(
                    user=user,
                    workspace=comparison_workspace,
                    llm_mode=llm_mode,
                    user_type=user_type,
                    repository_revision=comparison_revision,
                    question=request.question,
                    response=response,
                )
                schedule_answer_token_usage(
                    user,
                    comparison_workspace,
                    "repo.compare.follow_up",
                    response,
                    repo=repo,
                    analytics_context=analytics_context,
                )
                return response

            response = main.answer_compare(
                request.question,
                left,
                right,
                user_llm=user_llm,
                allow_shared_fallback=allow_shared,
                llm_mode=llm_mode,
                user_type=user_type,
                activity_request_id=getattr(request, "activity_request_id", None),
                activity_user_id=user["id"],
            )
            state = main._create_conversation_from_response(
                user=user,
                workspace=comparison_workspace,
                llm_mode=llm_mode,
                user_type=user_type,
                repository_revision=comparison_revision,
                question=request.question,
                response=response,
            )
            response["conversation_id"] = state.conversation_id
            response["follow_up_reused"] = False
            response["follow_up_fallback"] = bool(request.follow_up)
            response["investigate_deeply_available"] = True
            response["token_usage"] = token_usage_payload(token_usage)
            response["answer_user_type"] = user_type
            main._remember_session_answer(
                user=user,
                workspace=comparison_workspace,
                llm_mode=llm_mode,
                user_type=user_type,
                repository_revision=comparison_revision,
                question=request.question,
                response=response,
            )
            schedule_answer_token_usage(
                user,
                comparison_workspace,
                "repo.compare",
                response,
                repo=repo,
                analytics_context=analytics_context,
            )
            return response
    except LLMCapacityError as error:
        raise HTTPException(
            status_code=503,
            detail=str(error),
            headers={"Retry-After": "5"},
        )
    except RuntimeError as error:
        raise HTTPException(status_code=400, detail=str(error))
    except Exception as error:
        raise HTTPException(status_code=500, detail=f"Comparison failed: {str(error)}")


def published_repos() -> list[dict]:
    return [repo for repo in db.list_repos() if repo["status"] == "published"]


def remote_branch_options(repo: dict, query: str = "", limit: int = 100) -> list[dict]:
    """List remote branches, falling back to already-known branches if Git is slow."""
    needle = (query or "").strip().lower()
    try:
        branches = discover_remote_branches(repo)
    except Exception:
        branches = [
            {
                "name": branch["name"],
                "commit_sha": branch.get("remote_commit_sha") or branch.get("indexed_commit_sha"),
                "is_default": bool(branch.get("is_default")),
            }
            for branch in db.list_repo_branches(repo["id"])
        ]
    if needle:
        branches = [
            branch for branch in branches
            if needle in branch["name"].lower()
        ]
    return branches[:limit]


def approved_branch_options(repo: dict, query: str = "", limit: int = 100) -> list[dict]:
    """Fast branch options for latency-sensitive surfaces such as Slack modals."""
    needle = (query or "").strip().lower()
    branches = [
        {
            "id": branch["id"],
            "name": branch["name"],
            "commit_sha": branch.get("remote_commit_sha") or branch.get("indexed_commit_sha"),
            "is_default": bool(branch.get("is_default")),
            "index_status": branch.get("index_status"),
        }
        for branch in db.list_repo_branches(repo["id"])
    ]
    if needle:
        branches = [
            branch for branch in branches
            if needle in branch["name"].lower()
        ]
    return branches[:limit]


def prepare_repo_branch(repo: dict, branch_name: str, actor: str = "slack") -> dict:
    if repo["status"] != "published":
        raise HTTPException(
            status_code=409,
            detail=f"Repository '{repo['name']}' is not published.",
        )
    try:
        branch = approve_repo_branch(repo, branch_name)
    except ValueError as error:
        raise HTTPException(status_code=400, detail=str(error))
    except RuntimeError as error:
        raise HTTPException(status_code=400, detail=f"Branch discovery failed: {error}")
    accepted = submit_branch_job(branch["id"], sync=True)
    if accepted:
        db.record_audit(actor, "slack_prepare_branch", repo["slug"], branch["name"])
    return db.get_repo_branch(branch["id"])


def prepare_existing_repo_branch(repo: dict, branch_name: str, actor: str = "slack") -> dict:
    """Start sync/index for an already-approved branch without remote discovery."""
    if repo["status"] != "published":
        raise HTTPException(
            status_code=409,
            detail=f"Repository '{repo['name']}' is not published.",
        )
    branch = db.get_repo_branch_by_name(repo["id"], branch_name)
    if not branch:
        raise HTTPException(
            status_code=404,
            detail="Branch is not approved in CodeAtlas yet.",
        )
    accepted = submit_branch_job(branch["id"], sync=True)
    if accepted:
        db.record_audit(actor, "slack_prepare_branch", repo["slug"], branch["name"])
    return db.get_repo_branch(branch["id"])


def wait_for_branch_ready(
    branch_id: int,
    *,
    timeout_seconds: Optional[float] = None,
    poll_seconds: float = 1.0,
) -> dict:
    timeout = (
        float(timeout_seconds)
        if timeout_seconds is not None
        else float(os.environ.get("CODEATLAS_SLACK_BRANCH_WAIT_SECONDS", "900"))
    )
    deadline = time.monotonic() + max(0.0, timeout)
    while True:
        branch = db.get_repo_branch(branch_id)
        if not branch:
            raise HTTPException(status_code=404, detail="Repository branch not found.")
        if (
            branch.get("workspace")
            and branch["index_status"] == "ready"
            and graph_path(branch["workspace"]).exists()
        ):
            return branch
        if branch["index_status"] == "failed":
            detail = branch.get("last_error") or "Branch indexing failed."
            raise HTTPException(status_code=409, detail=detail)
        if not branch_job_running(branch_id) and branch["index_status"] != "indexing":
            submit_branch_job(branch_id, sync=True)
        if time.monotonic() >= deadline:
            raise HTTPException(
                status_code=202,
                detail="Branch is still syncing and indexing.",
            )
        time.sleep(max(0.2, poll_seconds))


def resolve_ready_branch(
    repo: dict,
    branch_name: str,
    *,
    actor: str = "slack",
    wait: bool = True,
) -> dict:
    branch = prepare_repo_branch(repo, branch_name, actor=actor)
    if not wait:
        return branch
    return wait_for_branch_ready(branch["id"])


def resolve_existing_ready_branch(
    repo: dict,
    branch_name: str,
    *,
    actor: str = "slack",
    wait: bool = True,
) -> dict:
    branch = prepare_existing_repo_branch(repo, branch_name, actor=actor)
    if not wait:
        return branch
    return wait_for_branch_ready(branch["id"])
