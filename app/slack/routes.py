"""Slack slash-command and modal integration for CodeAtlas."""

from __future__ import annotations

import hmac
import json
import logging
import os
import time
from concurrent.futures import ThreadPoolExecutor
from hashlib import sha256
from typing import Optional
from urllib.parse import parse_qs

import requests
from fastapi import APIRouter, BackgroundTasks, HTTPException, Request, Response

from .. import ask_service, db

router = APIRouter(prefix="/slack", tags=["slack"])
logger = logging.getLogger(__name__)

ASK_SINGLE = "single_branch"
ASK_COMPARE = "compare_branches"
USER_DEV = "dev_team"
USER_PRODUCT = "product_team"

BLOCK_REPO = "repo"
ACTION_REPO = "repo_select"
BLOCK_ASK_TYPE = "ask_type"
ACTION_ASK_TYPE = "ask_type_select"
BLOCK_BRANCH = "branch"
ACTION_BRANCH = "branch_select"
BLOCK_BASE_BRANCH = "base_branch"
ACTION_BASE_BRANCH = "base_branch_select"
BLOCK_COMPARE_BRANCH = "compare_branch"
ACTION_COMPARE_BRANCH = "compare_branch_select"
BLOCK_USER_TYPE = "user_type"
ACTION_USER_TYPE = "user_type_select"
BLOCK_QUESTION = "question"
ACTION_QUESTION = "question_input"

CALLBACK_ASK = "codeatlas_ask_submit"
CALLBACK_FOLLOW_UP = "codeatlas_follow_up_submit"

ACTION_FOLLOW_UP = "codeatlas_follow_up"
ACTION_DEEP = "codeatlas_investigate_deeply"
ACTION_NEW = "codeatlas_new_question"

_executor = ThreadPoolExecutor(
    max_workers=int(os.environ.get("CODEATLAS_SLACK_MAX_WORKERS", "4")),
    thread_name_prefix="codeatlas-slack",
)


def _env_bool(name: str, default: str = "false") -> bool:
    return os.environ.get(name, default).strip().lower() in {"1", "true", "yes", "on"}


def _csv(name: str) -> set[str]:
    return {
        item.strip()
        for item in os.environ.get(name, "").split(",")
        if item.strip()
    }


def slack_enabled() -> bool:
    return _env_bool("CODEATLAS_SLACK_ENABLED")


def _bot_token() -> str:
    return os.environ.get("CODEATLAS_SLACK_BOT_TOKEN", "").strip()


def _llm_mode() -> str:
    return os.environ.get("CODEATLAS_SLACK_LLM_MODE", "auto").strip().lower() or "auto"


def _truncate(value: str, limit: int = 75) -> str:
    value = str(value or "")
    if len(value) <= limit:
        return value
    return value[: max(0, limit - 1)] + "…"


def _plain(text: str, emoji: bool = True) -> dict:
    return {"type": "plain_text", "text": _truncate(text, 3000), "emoji": emoji}


def _mrkdwn(text: str) -> dict:
    return {"type": "mrkdwn", "text": _truncate(text, 3000)}


def _option(label: str, value: str, description: str = None) -> dict:
    item = {
        "text": _plain(label),
        "value": str(value)[:2000],
    }
    if description:
        item["description"] = _plain(description)
    return item


def _private_metadata(values: dict) -> str:
    compact = {
        key: value
        for key, value in values.items()
        if value not in (None, "", [], {})
    }
    return json.dumps(compact, separators=(",", ":"))[:3000]


def _load_metadata(value: str) -> dict:
    try:
        loaded = json.loads(value or "{}")
    except json.JSONDecodeError:
        return {}
    return loaded if isinstance(loaded, dict) else {}


def _slack_user_id(*sources: dict) -> Optional[str]:
    for source in sources:
        if not isinstance(source, dict):
            continue
        user_id = source.get("slack_user_id") or source.get("user_id")
        if user_id:
            return user_id
    return None


def _form_value(form: dict, key: str) -> str:
    values = form.get(key) or [""]
    return values[0]


def _parse_form(body: bytes) -> dict:
    return parse_qs(body.decode("utf-8"), keep_blank_values=True)


def verify_slack_request(headers, body: bytes) -> None:
    secret = os.environ.get("CODEATLAS_SLACK_SIGNING_SECRET", "").strip()
    if not secret:
        raise HTTPException(status_code=503, detail="Slack signing secret is not configured.")
    timestamp = headers.get("x-slack-request-timestamp")
    signature = headers.get("x-slack-signature")
    if not timestamp or not signature:
        raise HTTPException(status_code=401, detail="Missing Slack signature.")
    try:
        timestamp_value = int(timestamp)
    except ValueError:
        raise HTTPException(status_code=401, detail="Invalid Slack timestamp.")
    if abs(time.time() - timestamp_value) > 60 * 5:
        raise HTTPException(status_code=401, detail="Stale Slack request.")
    base = f"v0:{timestamp}:{body.decode('utf-8')}".encode("utf-8")
    expected = "v0=" + hmac.new(secret.encode("utf-8"), base, sha256).hexdigest()
    if not hmac.compare_digest(expected, signature):
        raise HTTPException(status_code=401, detail="Invalid Slack signature.")


def _authorize_slack_workspace(payload: dict) -> None:
    team = payload.get("team") or {}
    enterprise = payload.get("enterprise") or {}
    team_id = payload.get("team_id") or team.get("id")
    enterprise_id = payload.get("enterprise_id") or enterprise.get("id")
    allowed_teams = _csv("CODEATLAS_SLACK_ALLOWED_TEAM_IDS")
    allowed_enterprises = _csv("CODEATLAS_SLACK_ALLOWED_ENTERPRISE_IDS")
    if allowed_teams and team_id not in allowed_teams:
        raise HTTPException(status_code=403, detail="Slack workspace is not allowed.")
    if allowed_enterprises and enterprise_id not in allowed_enterprises:
        raise HTTPException(status_code=403, detail="Slack enterprise is not allowed.")


def _slack_api(method: str, payload: dict) -> dict:
    token = _bot_token()
    if not token:
        raise RuntimeError("Slack bot token is not configured.")
    response = requests.post(
        f"https://slack.com/api/{method}",
        headers={
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json; charset=utf-8",
        },
        json=payload,
        timeout=10,
    )
    try:
        data = response.json()
    except ValueError as exc:
        raise RuntimeError(f"Slack API returned non-JSON response: {response.text[:200]}") from exc
    if not data.get("ok"):
        raise RuntimeError(f"Slack API {method} failed: {data.get('error', 'unknown_error')}")
    return data


def _post_ephemeral(channel_id: str, user_id: str, text: str, blocks: list[dict] = None) -> None:
    payload = {
        "channel": channel_id,
        "user": user_id,
        "text": text,
    }
    if blocks:
        payload["blocks"] = blocks
    _slack_api("chat.postEphemeral", payload)


def _post_response_url(response_url: str, text: str, blocks: list[dict] = None) -> None:
    if not response_url:
        raise RuntimeError("Slack response_url is not available.")
    payload = {
        "response_type": "ephemeral",
        "replace_original": False,
        "text": text,
    }
    if blocks:
        payload["blocks"] = blocks
    response = requests.post(response_url, json=payload, timeout=10)
    if response.status_code >= 400:
        raise RuntimeError(
            f"Slack response_url failed with HTTP {response.status_code}: "
            f"{response.text[:200]}"
        )


def _send_user_message(values: dict, text: str, blocks: list[dict] = None) -> None:
    response_url = values.get("response_url")
    if response_url:
        try:
            _post_response_url(response_url, text, blocks)
            return
        except Exception:
            pass
    slack_user_id = _slack_user_id(values)
    if not values.get("channel_id") or not slack_user_id:
        raise RuntimeError("Slack channel_id and user_id are required to send a user message.")
    _post_ephemeral(
        values["channel_id"],
        slack_user_id,
        text,
        blocks,
    )


def _repo_options() -> list[dict]:
    repos = ask_service.published_repos()[:100]
    return [
        _option(repo["name"], repo["slug"], repo.get("slug"))
        for repo in repos
    ]


def _matching_option(options: list[dict], value: Optional[str]) -> Optional[dict]:
    if not value:
        return None
    for option in options:
        if option.get("value") == value:
            return option
    return None


def _repo_option(repo_slug: Optional[str], repo_options: list[dict]) -> Optional[dict]:
    return _matching_option(repo_options, repo_slug)


def _repo_by_slug(slug: str) -> Optional[dict]:
    return db.get_repo_by_slug(slug or "")


def _selected_option(value: Optional[str], label: str = None) -> Optional[dict]:
    if not value:
        return None
    return _option(label or value, value)


def _ask_type_option(value: str) -> dict:
    if value == ASK_COMPARE:
        return _option("Compare 2 branch answer", ASK_COMPARE)
    return _option("Single branch answer", ASK_SINGLE)


def _user_type_option(value: str) -> dict:
    if value == USER_PRODUCT:
        return _option("Product team", USER_PRODUCT)
    return _option("Dev team", USER_DEV)


def _branch_status_text(metadata: dict) -> str:
    statuses = []
    for label, key in (
        ("Branch", "branch_status"),
        ("Base branch", "base_branch_status"),
        ("Compare branch", "compare_branch_status"),
    ):
        status = metadata.get(key)
        if status:
            statuses.append(f"*{label}:* {status}")
    return "\n".join(statuses)


def _branch_options(repo_slug: Optional[str]) -> list[dict]:
    repo = _repo_by_slug(repo_slug)
    if not repo:
        return [_option("Select repository first", "__select_repo__")]
    branches = ask_service.approved_branch_options(repo, limit=100)
    if not branches:
        return [_option("No branches found", "__no_branches__")]
    return [
        _option(
            branch["name"],
            branch["name"],
            (branch.get("commit_sha") or "")[:12],
        )
        for branch in branches
    ]


def _branch_static_select(
    action_id: str,
    placeholder: str,
    repo_slug: Optional[str],
    initial: str = None,
) -> dict:
    options = _branch_options(repo_slug)
    element = {
        "type": "static_select",
        "action_id": action_id,
        "placeholder": _plain(placeholder),
        "options": options,
    }
    initial_option = _matching_option(options, initial)
    if initial_option:
        element["initial_option"] = initial_option
    return element


def build_ask_view(metadata: dict) -> dict:
    ask_type = metadata.get("ask_type") or ASK_SINGLE
    user_type = metadata.get("user_type") or USER_DEV
    repo_slug = metadata.get("repo_slug")
    repo_options = _repo_options()
    repo_initial = _repo_option(repo_slug, repo_options)
    blocks = [
        {
            "type": "input",
            "block_id": BLOCK_REPO,
            "dispatch_action": True,
            "label": _plain("Repository"),
            "element": {
                "type": "static_select",
                "action_id": ACTION_REPO,
                "placeholder": _plain("Select repository"),
                "options": repo_options or [_option("No published repositories", "__none__")],
                **({"initial_option": repo_initial} if repo_initial else {}),
            },
        },
        {
            "type": "input",
            "block_id": BLOCK_ASK_TYPE,
            "dispatch_action": True,
            "label": _plain("Ask type"),
            "element": {
                "type": "static_select",
                "action_id": ACTION_ASK_TYPE,
                "options": [
                    _ask_type_option(ASK_SINGLE),
                    _ask_type_option(ASK_COMPARE),
                ],
                "initial_option": _ask_type_option(ask_type),
            },
        },
    ]

    if ask_type == ASK_COMPARE:
        blocks.extend([
            {
                "type": "input",
                "block_id": BLOCK_BASE_BRANCH,
                "dispatch_action": True,
                "label": _plain("Base branch"),
                "element": _branch_static_select(
                    ACTION_BASE_BRANCH,
                    "Select base branch",
                    repo_slug,
                    metadata.get("base_branch"),
                ),
            },
            {
                "type": "input",
                "block_id": BLOCK_COMPARE_BRANCH,
                "dispatch_action": True,
                "label": _plain("Compare branch"),
                "element": _branch_static_select(
                    ACTION_COMPARE_BRANCH,
                    "Select compare branch",
                    repo_slug,
                    metadata.get("compare_branch"),
                ),
            },
        ])
    else:
        blocks.append({
            "type": "input",
            "block_id": BLOCK_BRANCH,
            "dispatch_action": True,
            "label": _plain("Branch"),
            "element": _branch_static_select(
                ACTION_BRANCH,
                "Select branch",
                repo_slug,
                metadata.get("branch"),
            ),
        })

    status_text = _branch_status_text(metadata)
    if status_text:
        blocks.append({
            "type": "context",
            "elements": [_mrkdwn(status_text)],
        })

    blocks.extend([
        {
            "type": "input",
            "block_id": BLOCK_USER_TYPE,
            "label": _plain("User type"),
            "element": {
                "type": "static_select",
                "action_id": ACTION_USER_TYPE,
                "options": [
                    _user_type_option(USER_DEV),
                    _user_type_option(USER_PRODUCT),
                ],
                "initial_option": _user_type_option(user_type),
            },
        },
        {
            "type": "input",
            "block_id": BLOCK_QUESTION,
            "label": _plain("Question"),
            "element": {
                "type": "plain_text_input",
                "action_id": ACTION_QUESTION,
                "multiline": True,
                **(
                    {"initial_value": str(metadata.get("question"))[:3000]}
                    if metadata.get("question")
                    else {}
                ),
            },
        },
    ])
    return {
        "type": "modal",
        "callback_id": CALLBACK_ASK,
        "title": _plain("CodeAtlas"),
        "submit": _plain("Submit"),
        "close": _plain("Cancel"),
        "private_metadata": _private_metadata(metadata),
        "blocks": blocks,
    }


def build_follow_up_view(metadata: dict) -> dict:
    context = metadata.get("topic_label") or "Current CodeAtlas topic"
    return {
        "type": "modal",
        "callback_id": CALLBACK_FOLLOW_UP,
        "title": _plain("Ask Follow-Up"),
        "submit": _plain("Submit"),
        "close": _plain("Cancel"),
        "private_metadata": _private_metadata(metadata),
        "blocks": [
            {
                "type": "context",
                "elements": [_mrkdwn(context)],
            },
            {
                "type": "input",
                "block_id": BLOCK_QUESTION,
                "label": _plain("Follow-up question"),
                "element": {
                    "type": "plain_text_input",
                    "action_id": ACTION_QUESTION,
                    "multiline": True,
                },
            },
        ],
    }


def _state_value(state: dict, block_id: str, action_id: str) -> Optional[str]:
    action = ((state.get("values") or {}).get(block_id) or {}).get(action_id) or {}
    if "selected_option" in action:
        option = action.get("selected_option") or {}
        return option.get("value")
    return action.get("value")


def _collect_view_values(payload: dict) -> dict:
    view = payload.get("view") or {}
    metadata = _load_metadata(view.get("private_metadata") or "")
    state = view.get("state") or {}
    repo_slug = _state_value(state, BLOCK_REPO, ACTION_REPO) or metadata.get("repo_slug")
    repo = _repo_by_slug(repo_slug)
    payload_user_id = (payload.get("user") or {}).get("id")
    slack_user_id = _slack_user_id(metadata) or payload_user_id
    values = {
        **metadata,
        "repo_slug": repo_slug,
        "repo_name": repo["name"] if repo else metadata.get("repo_name"),
        "ask_type": _state_value(state, BLOCK_ASK_TYPE, ACTION_ASK_TYPE)
        or metadata.get("ask_type")
        or ASK_SINGLE,
        "branch": _state_value(state, BLOCK_BRANCH, ACTION_BRANCH) or metadata.get("branch"),
        "base_branch": _state_value(state, BLOCK_BASE_BRANCH, ACTION_BASE_BRANCH)
        or metadata.get("base_branch"),
        "compare_branch": _state_value(state, BLOCK_COMPARE_BRANCH, ACTION_COMPARE_BRANCH)
        or metadata.get("compare_branch"),
        "user_type": _state_value(state, BLOCK_USER_TYPE, ACTION_USER_TYPE)
        or metadata.get("user_type")
        or USER_DEV,
        "question": (_state_value(state, BLOCK_QUESTION, ACTION_QUESTION) or "").strip(),
    }
    if slack_user_id:
        values["slack_user_id"] = slack_user_id
        values["user_id"] = values.get("user_id") or slack_user_id
    return values


def _validate_ask_values(values: dict) -> dict:
    errors = {}
    if not values.get("repo_slug") or values.get("repo_slug") == "__none__":
        errors[BLOCK_REPO] = "Select a repository."
    if values.get("ask_type") == ASK_COMPARE:
        if not values.get("base_branch"):
            errors[BLOCK_BASE_BRANCH] = "Select the base branch."
        if not values.get("compare_branch"):
            errors[BLOCK_COMPARE_BRANCH] = "Select the compare branch."
        if values.get("base_branch") and values.get("base_branch") == values.get("compare_branch"):
            errors[BLOCK_COMPARE_BRANCH] = "Choose a different compare branch."
    elif not values.get("branch"):
        errors[BLOCK_BRANCH] = "Select a branch."
    if (values.get("branch") or "").startswith("__"):
        errors[BLOCK_BRANCH] = "Select a branch."
    if (values.get("base_branch") or "").startswith("__"):
        errors[BLOCK_BASE_BRANCH] = "Select the base branch."
    if (values.get("compare_branch") or "").startswith("__"):
        errors[BLOCK_COMPARE_BRANCH] = "Select the compare branch."
    if values.get("user_type") not in {USER_DEV, USER_PRODUCT}:
        errors[BLOCK_USER_TYPE] = "Select a valid user type."
    if not values.get("question"):
        errors[BLOCK_QUESTION] = "Enter a question."
    return errors


def _topic_label(values: dict) -> str:
    repo = values.get("repo_name") or values.get("repo_slug") or "repository"
    if values.get("ask_type") == ASK_COMPARE:
        return (
            f"*{repo}* · `{values.get('base_branch')}` vs "
            f"`{values.get('compare_branch')}` · {values.get('user_type')}"
        )
    return f"*{repo}* · `{values.get('branch')}` · {values.get('user_type')}"


def _answer_text_blocks(response: dict, topic: dict) -> list[dict]:
    answer = str(response.get("answer") or "No answer was returned.").strip()
    header = _topic_label(topic)
    blocks = [{"type": "section", "text": _mrkdwn(header)}]
    chunks = [answer[index:index + 2900] for index in range(0, len(answer), 2900)] or [answer]
    for chunk in chunks[:8]:
        blocks.append({"type": "section", "text": _mrkdwn(chunk)})
    mode = response.get("retrieval_mode")
    if mode:
        blocks.append({"type": "context", "elements": [_mrkdwn(f"Retrieval: `{mode}`")]})
    if response.get("investigate_deeply_available", True):
        value = _private_metadata(topic)
        blocks.append({
            "type": "actions",
            "elements": [
                {
                    "type": "button",
                    "text": _plain("Ask follow-up"),
                    "action_id": ACTION_FOLLOW_UP,
                    "value": value,
                },
                {
                    "type": "button",
                    "text": _plain("Investigate deeply"),
                    "action_id": ACTION_DEEP,
                    "value": value,
                },
                {
                    "type": "button",
                    "text": _plain("New question"),
                    "action_id": ACTION_NEW,
                    "value": value,
                },
            ],
        })
    return blocks


def _http_detail(exc: Exception) -> str:
    if isinstance(exc, HTTPException):
        return str(exc.detail)
    return str(exc)


def _answer_topic_payload(values: dict, response: dict, branch_context: dict) -> dict:
    topic = {
        **values,
        **branch_context,
        "conversation_id": response.get("conversation_id"),
        "question": response.get("question") or values.get("question"),
    }
    topic["topic_label"] = _topic_label(topic)
    return topic


def _run_single_answer(values: dict, *, follow_up: bool = False, deep: bool = False) -> None:
    from .. import main

    slack_user = values["slack_user_id"]
    repo = _repo_by_slug(values.get("repo_slug"))
    if not repo:
        raise HTTPException(status_code=404, detail="Repository not found.")
    actor = ask_service.slack_actor_user(values["team_id"], slack_user, values.get("user_type"))
    workspace = values.get("branch_workspace")
    branch_context = {}
    if not workspace:
        _send_user_message(
            values,
            "Preparing branch",
            [{"type": "section", "text": _mrkdwn("Syncing and indexing the selected branch...")}],
        )
        branch = ask_service.resolve_existing_ready_branch(
            repo,
            values["branch"],
            actor=f"slack:{values['team_id']}:{slack_user}",
        )
        workspace = branch["workspace"]
        branch_context = {
            "branch_id": branch["id"],
            "branch": branch["name"],
            "branch_workspace": workspace,
        }
    request = main.AskRequest(
        question=values["question"],
        llm_mode=_llm_mode(),
        conversation_id=values.get("conversation_id") if follow_up else None,
        follow_up=follow_up,
        deep_investigation=deep,
        answer_user_type=values.get("user_type") or USER_DEV,
    )
    _send_user_message(
        values,
        "Generating answer",
        [{"type": "section", "text": _mrkdwn("Searching repository context and generating the answer...")}],
    )
    response = ask_service.answer_single_request(request, workspace, actor)
    topic = _answer_topic_payload(values, response, branch_context)
    _send_user_message(
        topic,
        "CodeAtlas answer",
        _answer_text_blocks(response, topic),
    )


def _run_compare_answer(values: dict, *, follow_up: bool = False, deep: bool = False) -> None:
    from .. import main

    slack_user = values["slack_user_id"]
    repo = _repo_by_slug(values.get("repo_slug"))
    if not repo:
        raise HTTPException(status_code=404, detail="Repository not found.")
    actor = ask_service.slack_actor_user(values["team_id"], slack_user, values.get("user_type"))
    branch_context = {}
    base_workspace = values.get("base_branch_workspace")
    compare_workspace = values.get("compare_branch_workspace")
    if not base_workspace or not compare_workspace:
        _send_user_message(
            values,
            "Preparing comparison",
            [{"type": "section", "text": _mrkdwn("Syncing and indexing the selected branches...")}],
        )
        base = ask_service.resolve_existing_ready_branch(
            repo,
            values["base_branch"],
            actor=f"slack:{values['team_id']}:{slack_user}",
        )
        compare = ask_service.resolve_existing_ready_branch(
            repo,
            values["compare_branch"],
            actor=f"slack:{values['team_id']}:{slack_user}",
        )
    else:
        base = {
            "id": values.get("base_branch_id"),
            "name": values.get("base_branch"),
            "workspace": base_workspace,
        }
        compare = {
            "id": values.get("compare_branch_id"),
            "name": values.get("compare_branch"),
            "workspace": compare_workspace,
        }
    left = {"repo": repo, "branch": base, "workspace": base["workspace"]}
    right = {"repo": repo, "branch": compare, "workspace": compare["workspace"]}
    branch_context.update({
        "base_branch_id": base["id"],
        "base_branch": base["name"],
        "base_branch_workspace": base["workspace"],
        "compare_branch_id": compare["id"],
        "compare_branch": compare["name"],
        "compare_branch_workspace": compare["workspace"],
    })
    request = main.CompareRequest(
        question=values["question"],
        left_branch=base["id"],
        right_branch=compare["id"],
        llm_mode=_llm_mode(),
        conversation_id=values.get("conversation_id") if follow_up else None,
        follow_up=follow_up,
        deep_investigation=deep,
        answer_user_type=values.get("user_type") or USER_DEV,
    )
    _send_user_message(
        values,
        "Generating comparison",
        [{"type": "section", "text": _mrkdwn("Searching both branches and generating the comparison...")}],
    )
    response = ask_service.answer_compare_request(
        request,
        repo["workspace"],
        actor,
        repo=repo,
        left=left,
        right=right,
    )
    topic = _answer_topic_payload(values, response, branch_context)
    _send_user_message(
        topic,
        "CodeAtlas comparison",
        _answer_text_blocks(response, topic),
    )


def _run_answer_job(values: dict, *, follow_up: bool = False, deep: bool = False) -> None:
    try:
        if values.get("ask_type") == ASK_COMPARE:
            _run_compare_answer(values, follow_up=follow_up, deep=deep)
        else:
            _run_single_answer(values, follow_up=follow_up, deep=deep)
    except Exception as exc:
        try:
            _send_user_message(
                values,
                "CodeAtlas could not answer",
                [{
                    "type": "section",
                    "text": _mrkdwn(f"I couldn't complete that request.\n\nReason: {_http_detail(exc)}"),
                }],
            )
        except Exception:
            pass


def _start_answer_job(values: dict, *, follow_up: bool = False, deep: bool = False) -> None:
    _executor.submit(_run_answer_job, values, follow_up=follow_up, deep=deep)


def _open_ask_modal(metadata: dict, trigger_id: str) -> None:
    try:
        _slack_api("views.open", {
            "trigger_id": trigger_id,
            "view": build_ask_view(metadata),
        })
    except Exception as exc:
        logger.exception("Failed to open Slack ask modal")
        try:
            _send_user_message(
                metadata,
                "CodeAtlas could not open",
                [{
                    "type": "section",
                    "text": _mrkdwn(f"I couldn't open the CodeAtlas modal.\n\nReason: {_http_detail(exc)}"),
                }],
            )
        except Exception:
            logger.exception("Failed to notify Slack user about modal-open failure")


def _start_modal_open_job(metadata: dict, trigger_id: str) -> None:
    _executor.submit(_open_ask_modal, metadata, trigger_id)


def _prepare_selected_branch(metadata: dict, action_id: str, branch_name: str) -> None:
    repo = _repo_by_slug(metadata.get("repo_slug"))
    if not repo or not branch_name or branch_name.startswith("__"):
        return
    try:
        branch = ask_service.prepare_existing_repo_branch(
            repo,
            branch_name,
            actor=f"slack:{metadata.get('team_id')}:{metadata.get('slack_user_id')}",
        )
        status = (
            "Ready"
            if branch.get("workspace") and branch.get("index_status") == "ready"
            else "Syncing and indexing..."
        )
    except Exception as exc:
        status = f"Could not prepare branch: {_http_detail(exc)}"
    if action_id == ACTION_BASE_BRANCH:
        metadata["base_branch_status"] = status
    elif action_id == ACTION_COMPARE_BRANCH:
        metadata["compare_branch_status"] = status
    else:
        metadata["branch_status"] = status


def _handle_block_actions(payload: dict) -> dict:
    actions = payload.get("actions") or []
    if not actions:
        return {}
    action = actions[0]
    action_id = action.get("action_id")
    if action_id == ACTION_NEW:
        metadata = _load_metadata(action.get("value") or "")
        metadata["question"] = ""
        _slack_api("views.open", {
            "trigger_id": payload["trigger_id"],
            "view": build_ask_view(metadata),
        })
        return {}
    if action_id == ACTION_FOLLOW_UP:
        metadata = _load_metadata(action.get("value") or "")
        _slack_api("views.open", {
            "trigger_id": payload["trigger_id"],
            "view": build_follow_up_view(metadata),
        })
        return {}
    if action_id == ACTION_DEEP:
        metadata = _load_metadata(action.get("value") or "")
        _send_user_message(
            metadata,
            "Investigating deeply",
            [{"type": "section", "text": _mrkdwn("Running a deeper repository investigation...")}],
        )
        _start_answer_job(metadata, follow_up=True, deep=True)
        return {}

    view = payload.get("view") or {}
    metadata = _collect_view_values(payload)
    if action_id == ACTION_REPO:
        metadata.pop("branch", None)
        metadata.pop("base_branch", None)
        metadata.pop("compare_branch", None)
        metadata.pop("branch_status", None)
        metadata.pop("base_branch_status", None)
        metadata.pop("compare_branch_status", None)
    if action_id == ACTION_ASK_TYPE:
        metadata.pop("branch_status", None)
        metadata.pop("base_branch_status", None)
        metadata.pop("compare_branch_status", None)
    if action_id in {ACTION_BRANCH, ACTION_BASE_BRANCH, ACTION_COMPARE_BRANCH}:
        selected = (action.get("selected_option") or {}).get("value")
        _prepare_selected_branch(metadata, action_id, selected)
    _slack_api("views.update", {
        "view_id": view.get("id"),
        "hash": view.get("hash"),
        "view": build_ask_view(metadata),
    })
    return {}


def _handle_block_suggestion(payload: dict) -> dict:
    metadata = _collect_view_values(payload)
    repo = _repo_by_slug(metadata.get("repo_slug"))
    if not repo:
        return {"options": []}
    query = payload.get("value") or ""
    branches = ask_service.approved_branch_options(repo, query=query, limit=100)
    return {
        "options": [
            _option(
                branch["name"],
                branch["name"],
                (branch.get("commit_sha") or "")[:12],
            )
            for branch in branches
        ]
    }


def _handle_view_submission(payload: dict) -> dict:
    view = payload.get("view") or {}
    if view.get("callback_id") == CALLBACK_FOLLOW_UP:
        values = _load_metadata(view.get("private_metadata") or "")
        state = view.get("state") or {}
        question = (_state_value(state, BLOCK_QUESTION, ACTION_QUESTION) or "").strip()
        if not question:
            return {
                "response_action": "errors",
                "errors": {BLOCK_QUESTION: "Enter a follow-up question."},
            }
        values["question"] = question
        _start_answer_job(values, follow_up=True, deep=False)
        return {}

    values = _collect_view_values(payload)
    values["team_id"] = values.get("team_id") or (payload.get("team") or {}).get("id")
    slack_user_id = _slack_user_id(values) or (payload.get("user") or {}).get("id")
    if slack_user_id:
        values["slack_user_id"] = slack_user_id
        values["user_id"] = values.get("user_id") or slack_user_id
    errors = _validate_ask_values(values)
    if errors:
        return {"response_action": "errors", "errors": errors}
    _start_answer_job(values)
    return {}


@router.post("/commands")
async def slash_command(request: Request, background_tasks: BackgroundTasks):
    if not slack_enabled():
        raise HTTPException(status_code=404, detail="Slack integration is not enabled.")
    body = await request.body()
    verify_slack_request(request.headers, body)
    form = _parse_form(body)
    team_id = _form_value(form, "team_id")
    enterprise_id = _form_value(form, "enterprise_id")
    payload = {
        "team_id": team_id,
        "enterprise_id": enterprise_id,
    }
    _authorize_slack_workspace(payload)
    channel_id = _form_value(form, "channel_id")
    slack_user = _form_value(form, "user_id")
    metadata = {
        "team_id": team_id,
        "enterprise_id": enterprise_id,
        "channel_id": channel_id,
        "user_id": slack_user,
        "slack_user_id": slack_user,
        "response_url": _form_value(form, "response_url"),
        "question": _form_value(form, "text").strip(),
        "ask_type": ASK_SINGLE,
        "user_type": USER_DEV,
    }
    logger.info("Accepted Slack slash command for team=%s channel=%s user=%s", team_id, channel_id, slack_user)
    background_tasks.add_task(_open_ask_modal, metadata, _form_value(form, "trigger_id"))
    return Response(status_code=200)


@router.post("/interactions")
async def interactions(request: Request):
    if not slack_enabled():
        raise HTTPException(status_code=404, detail="Slack integration is not enabled.")
    body = await request.body()
    verify_slack_request(request.headers, body)
    form = _parse_form(body)
    payload_raw = _form_value(form, "payload")
    try:
        payload = json.loads(payload_raw)
    except json.JSONDecodeError:
        raise HTTPException(status_code=400, detail="Invalid Slack payload.")
    _authorize_slack_workspace(payload)
    payload_type = payload.get("type")
    if payload_type == "block_suggestion":
        return _handle_block_suggestion(payload)
    if payload_type == "block_actions":
        return _handle_block_actions(payload)
    if payload_type == "view_submission":
        return _handle_view_submission(payload)
    return {}
