"""Ask a question across a repo group (several repositories at once).

Mirrors the two-branch comparison flow (`app.main.answer_compare` and
`ask_service.answer_compare_request`): retrieval runs per member repo, then one
agent loop investigates them through a toolbox that routes every call to a
single member. Follow-ups reuse the same conversation thread and re-run the
agent with the thread's earlier evidence.
"""

from __future__ import annotations

import hashlib
import time
from typing import Optional

from fastapi import HTTPException

from . import ask_service
from .agent.tools import GroupRepositoryToolbox, tool_definitions_without_ask_user
from .llm.admission import LLMCapacityError
from .llm.client import (
    PRODUCT_TEAM_QUERY_SUFFIX,
    PRODUCT_TEAM_RESPONSE_INSTRUCTION,
    collect_token_usage,
    generate,
    token_usage_payload,
)
from .repos.groups import resolve_group_members

GROUP_CONTEXT_NODE_LIMIT = 8


def group_workspace_key(group: dict) -> str:
    return f"group:{group['slug']}"


def group_revision(main, targets: list[dict], group: Optional[dict] = None) -> str:
    """Cache identity: changes when any member's index changes, and when the
    admin edits the relationship note, since that is part of the prompt."""
    parts = [
        f"{target['slug']}:{main.repository_revision(target['workspace'])}"
        for target in targets
    ]
    if group is not None:
        note = str(group.get("description") or "").strip()
        parts.append("note:" + hashlib.sha256(note.encode("utf-8")).hexdigest()[:12])
    return "|".join(parts)


def _group_header(group: dict, targets: list[dict]) -> str:
    lines = [
        f"Repository group: {group['name']}.",
        "Repositories: "
        + ", ".join(f"{t['slug']} ({t['name']})" for t in targets)
        + ".",
    ]
    description = str(group.get("description") or "").strip()
    if description:
        lines.append(f"How these repositories relate (from the admin): {description}")
    return "\n".join(lines)


def build_group_context(
    main,
    question: str,
    group: dict,
    targets: list[dict],
    user_type: str,
    retrieval_question: Optional[str] = None,
) -> dict:
    started_at = time.perf_counter()
    lookup_question = retrieval_question or question
    payloads = []
    for target in targets:
        context = main.build_context(
            lookup_question,
            limit=GROUP_CONTEXT_NODE_LIMIT,
            workspace=target["workspace"],
        )
        payloads.append({
            "name": target["name"],
            "slug": target["slug"],
            "workspace": target["workspace"],
            "repository_version": main.repository_version_payload(target["workspace"]),
            "context_nodes": context.get("context_nodes", []),
            "context_relations": context.get("context_relations", []),
            "source_hits": context.get("source_hits", []),
            "llm_context_preview": context.get("llm_context_preview", {}),
        })
    product = user_type == "product_team"
    llm_question = f"{_group_header(group, targets)}\n\nQuestion: {question.rstrip()}"
    if product:
        llm_question = f"{llm_question}\n\n{PRODUCT_TEAM_QUERY_SUFFIX}"
    return {
        "question": question,
        "group_mode": True,
        "group": {"slug": group["slug"], "name": group["name"]},
        "response_style_instruction": PRODUCT_TEAM_RESPONSE_INSTRUCTION if product else "",
        "group_repositories": payloads,
        "retrieval_ms": round((time.perf_counter() - started_at) * 1000, 1),
        "llm_context_preview": {
            "instruction": (
                "Answer using only the evidence from these related repositories. "
                "Keep each repository's findings separate, then explain how they "
                "connect."
            ),
            "question": llm_question,
            "repositories": [
                {
                    "name": payload["name"],
                    "slug": payload["slug"],
                    "repository_version": payload["repository_version"],
                    "evidence": payload["llm_context_preview"],
                }
                for payload in payloads
            ],
        },
    }


def answer_group(
    main,
    question: str,
    group: dict,
    targets: list[dict],
    user_llm: dict = None,
    allow_shared_fallback: bool = True,
    llm_mode: str = None,
    user_type: str = "dev_team",
    activity_request_id: str = None,
    activity_user_id: int = None,
    conversation_state=None,
) -> dict:
    started_at = time.perf_counter()
    workspace_key = group_workspace_key(group)

    def report(status: str) -> None:
        if activity_request_id and activity_user_id is not None:
            main.update_answer_activity(
                activity_request_id,
                user_id=activity_user_id,
                workspace=workspace_key,
                question=question,
                status=status,
            )

    report("understanding_query")
    retrieval_question = None
    agent_context = ""
    if conversation_state is not None:
        retrieval_question = main.follow_up_retrieval_query(question, conversation_state)
        agent_context = main.compact_follow_up_evidence(conversation_state)
    context = build_group_context(
        main, question, group, targets, user_type,
        retrieval_question=retrieval_question,
    )
    report("generating_answer")
    toolbox = GroupRepositoryToolbox(targets)
    if conversation_state is not None:
        toolbox.tool_definitions = tool_definitions_without_ask_user(
            toolbox.tool_definitions
        )
    toolbox.response_style_instruction = context.get("response_style_instruction", "")
    generation_started_at = time.perf_counter()
    result = generate(
        context,
        user_llm=user_llm,
        allow_shared_fallback=allow_shared_fallback,
        llm_mode=llm_mode,
        question=context["llm_context_preview"]["question"],
        toolbox=toolbox,
        agent_context=agent_context,
    )
    response = {
        "question": question,
        "answer": result["answer"],
        "provider_used": result["provider_used"],
        "retrieval_mode": result.get("retrieval_mode", "one_shot"),
        "agent_trace": result.get("agent_trace", []),
        "agent_rounds": result.get("rounds"),
        "agent_tool_calls": result.get("tool_calls", 0),
        "needs_clarification": bool(result.get("needs_clarification")),
        **(
            {"agent_fallback_reason": result["agent_fallback_reason"]}
            if result.get("agent_fallback_reason")
            else {}
        ),
        "context": context,
        "group": context["group"],
        "group_repositories": [
            {
                "name": payload["name"],
                "slug": payload["slug"],
                "repository_version": payload["repository_version"],
            }
            for payload in context["group_repositories"]
        ],
    }
    response["timings_ms"] = {
        "retrieval": context["retrieval_ms"],
        "generation": round((time.perf_counter() - generation_started_at) * 1000, 1),
        "total": round((time.perf_counter() - started_at) * 1000, 1),
    }
    return response


def answer_group_request(
    request,
    group: dict,
    user: dict,
    *,
    enforce_limit: bool = True,
    analytics_context: Optional[dict] = None,
) -> dict:
    """Run the group ask flow for any authenticated surface."""
    main = ask_service._main()
    if enforce_limit:
        main.enforce_rate_limit(user["id"])
    targets = resolve_group_members(group)
    for target in targets:
        main.enforce_strict_branch_freshness(target["workspace"])
    ask_service._record_anonymous_question(
        request, {"slug": group_workspace_key(group), "name": group["name"]}
    )
    # The shared LLM only sees code from repos that allow it, so one repo that
    # opted out keeps the whole group off the shared tier.
    allow_shared = all(bool(t["repo"]["allow_shared_fallback"]) for t in targets)
    user_llm = request.user_llm or main.load_user_llm(user["id"])
    llm_mode = (request.llm_mode or "auto").lower()
    user_type = main._effective_answer_user_type(user, request.answer_user_type)
    workspace = group_workspace_key(group)
    revision = group_revision(main, targets, group)
    session_key = str(user.get("_session_key") or "")
    use_session_cache = not (
        request.deep_investigation
        or (main.is_shared_llm_mode(llm_mode) and not allow_shared)
    )
    cache_conversation_id = (
        str(request.conversation_id or "") if request.follow_up else ""
    )
    if use_session_cache:
        cached = main.conversation_store.get_cached_answer(
            session_key=session_key,
            user_id=user["id"],
            workspace=workspace,
            llm_mode=llm_mode,
            user_type=user_type,
            repository_revision=revision,
            question=request.question,
            conversation_id=cache_conversation_id,
        )
        if cached:
            response = main._session_cached_answer_response(
                cached, request.question, workspace=None
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
            response = answer_group(
                main,
                request.question,
                group,
                targets,
                user_llm=user_llm,
                allow_shared_fallback=allow_shared,
                llm_mode=llm_mode,
                user_type=user_type,
                activity_request_id=getattr(request, "activity_request_id", None),
                activity_user_id=user["id"],
                conversation_state=state,
            )
            if state is not None:
                main.conversation_store.append(
                    state.conversation_id,
                    question=request.question,
                    answer=response["answer"],
                    context=response.get("context"),
                )
                conversation_id = state.conversation_id
                endpoint = "repo.group.follow_up"
            else:
                created = main._create_conversation_from_response(
                    user=user,
                    workspace=workspace,
                    llm_mode=llm_mode,
                    user_type=user_type,
                    repository_revision=revision,
                    question=request.question,
                    response=response,
                )
                conversation_id = created.conversation_id
                endpoint = "repo.group"
            response["conversation_id"] = conversation_id
            response["follow_up_reused"] = False
            response["follow_up_fallback"] = bool(request.follow_up)
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
                conversation_id=conversation_id if state is not None else "",
            )
            ask_service.schedule_answer_token_usage(
                user,
                workspace,
                endpoint,
                response,
                repo=None,
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
        raise HTTPException(status_code=500, detail=f"Group request failed: {str(error)}")
