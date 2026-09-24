"""Slack slash-command and modal integration for CodeAtlas."""

from __future__ import annotations

import hmac
import json
import logging
import os
import re
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from hashlib import sha256
from typing import Optional
from urllib.parse import parse_qs

import requests
from fastapi import APIRouter, HTTPException, Request, Response

from .. import ask_service, db
from ..config import default_shared_llm_id, shared_llm_mode

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
RELAY_SECRET_HEADER = "x-codeatlas-relay-secret"

_executor = ThreadPoolExecutor(
    max_workers=int(os.environ.get("CODEATLAS_SLACK_MAX_WORKERS", "4")),
    thread_name_prefix="codeatlas-slack",
)

class _TTLCache:
    """Thread-safe key/value store with lazy TTL eviction.

    Shared by the dedupe and pending-question caches below instead of each
    hand-rolling its own dict + lock + sweep-and-evict loop.
    """

    def __init__(self, ttl_seconds: float):
        self._ttl = ttl_seconds
        self._store: dict[str, tuple[object, float]] = {}
        self._lock = threading.Lock()

    def _evict_locked(self, now: float) -> None:
        for key, (_, seen_at) in list(self._store.items()):
            if now - seen_at > self._ttl:
                self._store.pop(key, None)

    def seen(self, key: Optional[str]) -> bool:
        """Record `key` as seen now; return True if it was already seen."""
        if not key:
            return False
        now = time.time()
        with self._lock:
            self._evict_locked(now)
            if key in self._store:
                return True
            self._store[key] = (None, now)
            return False

    def set(self, key: Optional[str], value: object) -> None:
        if not key:
            return
        now = time.time()
        with self._lock:
            self._evict_locked(now)
            self._store[key] = (value, now)

    def pop(self, key: Optional[str]) -> Optional[object]:
        if not key:
            return None
        with self._lock:
            entry = self._store.pop(key, None)
        if not entry:
            return None
        value, seen_at = entry
        if time.time() - seen_at > self._ttl:
            return None
        return value

    def clear(self) -> None:
        with self._lock:
            self._store.clear()


_EVENT_DEDUPE_TTL_SECONDS = 600

# Slack retries an Events API delivery it didn't get a fast ack for, which
# would otherwise post the same answer twice.
_SEEN_EVENT_IDS = _TTLCache(_EVENT_DEDUPE_TTL_SECONDS)

# A DM mention can fire both app_mention and message.im for the same
# physical message; this stops it from being answered twice.
_SEEN_MESSAGE_KEYS = _TTLCache(_EVENT_DEDUPE_TTL_SECONDS)

# When repo inference can't tell which repo a question is about, we ask the
# user to name one; this remembers that original question, scoped to the
# specific thread it was asked in, so a bare repo-name reply *in that
# thread* resumes it instead of being answered as a new, standalone
# one-word question. Thread-scoping (rather than just channel+user) also
# means two concurrent ambiguous questions in the same channel/DM can't
# clobber each other's pending entry.
_PENDING_REPO_QUESTIONS = _TTLCache(_EVENT_DEDUPE_TTL_SECONDS)


def _already_processed_event(event_id: Optional[str]) -> bool:
    return _SEEN_EVENT_IDS.seen(event_id)


def _already_answered_message(channel: Optional[str], ts: Optional[str]) -> bool:
    if not channel or not ts:
        return False
    return _SEEN_MESSAGE_KEYS.seen(f"{channel}:{ts}")


def _remember_pending_question(
    channel: Optional[str], thread_ts: Optional[str], user: Optional[str], question: str
) -> None:
    if not channel or not thread_ts or not user or not question:
        return
    _PENDING_REPO_QUESTIONS.set(f"{channel}:{thread_ts}:{user}", question)


def _take_pending_question(
    channel: Optional[str], thread_ts: Optional[str], user: Optional[str]
) -> Optional[str]:
    if not channel or not thread_ts or not user:
        return None
    return _PENDING_REPO_QUESTIONS.pop(f"{channel}:{thread_ts}:{user}")


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
    """Slack always answers with the default shared LLM.

    Slack actors are synthetic identities that never hold a personal key, so
    "auto" already resolved to the shared tier; naming the default explicitly
    makes that a guarantee rather than a side effect of the tier order. An
    operator can still pin a specific one with CODEATLAS_SLACK_LLM_MODE until
    per-request model selection exists in Slack."""
    configured = os.environ.get("CODEATLAS_SLACK_LLM_MODE", "").strip().lower()
    if configured and configured != "auto":
        return configured
    default_id = default_shared_llm_id()
    return shared_llm_mode(default_id) if default_id else "auto"


def _relay_secret() -> str:
    return os.environ.get("CODEATLAS_SLACK_RELAY_SECRET", "").strip()


def _valid_relay_request(headers) -> bool:
    secret = _relay_secret()
    provided = headers.get(RELAY_SECRET_HEADER, "").strip()
    return bool(secret and provided and hmac.compare_digest(provided, secret))


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


_MENTION_RE = re.compile(r"<@[A-Z0-9]+>")


def _strip_mention(text: str) -> str:
    """Drop @CodeAtlas mention token(s), wherever in the message they are,
    leaving the question ("@codeatlas, how do I..." mentions mid-sentence,
    not just at the start)."""
    stripped = _MENTION_RE.sub("", str(text or ""))
    return re.sub(r"\s+", " ", stripped).strip()


def verify_slack_request(headers, body: bytes) -> None:
    if _valid_relay_request(headers):
        return
    if _env_bool("CODEATLAS_SLACK_REQUIRE_RELAY_SECRET"):
        if not _relay_secret():
            raise HTTPException(status_code=503, detail="Slack relay secret is not configured.")
        raise HTTPException(status_code=401, detail="Invalid Slack relay secret.")

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


def _post_channel_message(channel_id: str, thread_ts: str, text: str, blocks: list[dict] = None) -> None:
    payload = {
        "channel": channel_id,
        "text": text,
        "thread_ts": thread_ts,
    }
    if blocks:
        payload["blocks"] = blocks
    _slack_api("chat.postMessage", payload)


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
    # Mention-originated flows have no response_url and reply in-thread,
    # visible to the channel, instead of ephemerally to just the asker.
    thread_ts = values.get("thread_ts")
    channel_id = values.get("channel_id")
    if thread_ts and channel_id:
        _post_channel_message(channel_id, thread_ts, text, blocks)
        return
    slack_user_id = _slack_user_id(values)
    if not channel_id or not slack_user_id:
        raise RuntimeError("Slack channel_id and user_id are required to send a user message.")
    _post_ephemeral(
        channel_id,
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


def _infer_repo_from_text(text: str, repos: list[dict]) -> Optional[dict]:
    """Deterministic repo match: an explicit name/slug mention, or the only
    published repo. Ambiguous or unmatched text returns None so the caller
    asks the user instead of guessing (semantic classification is Phase B).

    Takes `repos` rather than fetching it, so a caller that already has the
    published-repo list (e.g. to build a "which repository?" prompt on a
    miss) doesn't have to fetch it twice.
    """
    if not repos:
        return None
    lowered = text.lower()
    matches = []
    for repo in repos:
        for candidate in {repo["slug"].strip().lower(), repo["name"].strip().lower()}:
            # (?<!\w)/(?!\w) rather than \b: \b requires a word/non-word
            # transition at the edge itself, so it never matches a
            # candidate that starts or ends with punctuation (e.g. a repo
            # named "Payments API (EU)") even when that candidate appears
            # verbatim in the text.
            if candidate and re.search(rf"(?<!\w){re.escape(candidate)}(?!\w)", lowered):
                matches.append(repo)
                break
    if len(matches) == 1:
        return matches[0]
    if not matches and len(repos) == 1:
        return repos[0]
    return None


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


_MD_BOLD_RE = re.compile(r"\*\*([^*\n]+)\*\*")
_MD_HEADING_RE = re.compile(r"^\s{0,3}(#{1,6})\s+(.*\S)\s*$")
_MD_RULE_RE = re.compile(r"^\s*(?:-{3,}|\*{3,}|_{3,})\s*$")
_MD_BULLET_RE = re.compile(r"^(\s{0,8})[-*+]\s+(.*)$")
_MD_LINK_RE = re.compile(r"\[([^\]\n]+)\]\((https?://[^)\s]+)\)")
_INLINE_CODE_SPLIT_RE = re.compile(r"(`[^`\n]+`)")


def _mrkdwn_inline(value: str) -> str:
    """Inline conversions, skipping inline-code spans so their contents stay
    exactly as written."""
    parts = _INLINE_CODE_SPLIT_RE.split(str(value or ""))
    converted = []
    for part in parts:
        if len(part) > 1 and part.startswith("`") and part.endswith("`"):
            converted.append(part)
            continue
        # Slack bold is a single asterisk; `__` is left alone because it
        # collides with dunder names in a codebase tool.
        part = _MD_BOLD_RE.sub(r"*\1*", part)
        part = _MD_LINK_RE.sub(r"<\2|\1>", part)
        converted.append(part)
    return "".join(converted)


def markdown_to_mrkdwn(text: str) -> str:
    """Translate the Markdown the model writes into Slack's mrkdwn.

    Slack has no headings and uses single asterisks for bold, so an answer
    posted verbatim shows literal ## and ** to the reader. Fenced code is
    passed through untouched."""
    lines = str(text or "").replace("\r\n", "\n").replace("\r", "\n").split("\n")
    out = []
    in_code = False
    for line in lines:
        if line.strip().startswith("```"):
            in_code = not in_code
            out.append(line)
            continue
        if in_code:
            out.append(line)
            continue
        heading = _MD_HEADING_RE.match(line)
        if heading:
            # No heading levels in mrkdwn; bold is the closest equivalent.
            out.append(f"*{_mrkdwn_inline(heading.group(2))}*")
            continue
        if _MD_RULE_RE.match(line):
            # A literal --- reads as noise inside a Slack section.
            out.append("")
            continue
        bullet = _MD_BULLET_RE.match(line)
        if bullet:
            out.append(f"{bullet.group(1)}\u2022 {_mrkdwn_inline(bullet.group(2))}")
            continue
        out.append(_mrkdwn_inline(line))
    return "\n".join(out)


def _mrkdwn_chunks(text: str, limit: int = 2900) -> list[str]:
    """Split on line boundaries so a chunk never cuts through markup."""
    chunks = []
    current = ""
    for line in str(text or "").split("\n"):
        candidate = f"{current}\n{line}" if current else line
        if len(candidate) > limit and current:
            chunks.append(current)
            current = line
        else:
            current = candidate
        while len(current) > limit:
            chunks.append(current[:limit])
            current = current[limit:]
    if current:
        chunks.append(current)
    return chunks or [""]


def _answer_text_blocks(response: dict, topic: dict) -> list[dict]:
    answer = str(response.get("answer") or "No answer was returned.").strip()
    question = str(response.get("question") or topic.get("question") or "").strip()
    header = _topic_label(topic)
    blocks = [{"type": "section", "text": _mrkdwn(header)}]
    if question:
        blocks.append({
            "type": "section",
            "text": _mrkdwn(f"*Question asked:*\n{question}"),
        })
    chunks = _mrkdwn_chunks(markdown_to_mrkdwn(answer))
    for chunk in chunks[:8]:
        blocks.append({"type": "section", "text": _mrkdwn(chunk)})
    context_elements = []
    mode = response.get("retrieval_mode")
    if mode:
        context_elements.append(_mrkdwn(f"Retrieval: `{mode}`"))
    provider = response.get("provider_used")
    if provider:
        context_elements.append(_mrkdwn(f"Model: `{provider}`"))
    if context_elements:
        blocks.append({"type": "context", "elements": context_elements})
    value = _private_metadata(topic)
    actions = [
        {
            "type": "button",
            "text": _plain("Ask follow-up"),
            "action_id": ACTION_FOLLOW_UP,
            "value": value,
        },
    ]
    if response.get("investigate_deeply_available", True):
        actions.append({
            "type": "button",
            "text": _plain("Investigate deeply"),
            "action_id": ACTION_DEEP,
            "value": value,
        })
    actions.append({
        "type": "button",
        "text": _plain("New question"),
        "action_id": ACTION_NEW,
        "value": value,
    })
    blocks.append({
        "type": "actions",
        "elements": actions,
    })
    return blocks


def _http_detail(exc: Exception) -> str:
    if isinstance(exc, HTTPException):
        return str(exc.detail)
    return str(exc)


def _answer_failure_detail(exc: Exception) -> str:
    detail = _http_detail(exc)
    lowered = detail.lower()
    if (
        "no llm provider succeeded" in lowered
        and "shared:" in lowered
        and (
            "invalid api key" in lowered
            or "invalid_key" in lowered
            or "401" in lowered
            or "quota" in lowered
            or "rate limit" in lowered
            or "insufficient" in lowered
            or "exhaust" in lowered
        )
    ):
        return (
            "CodeAtlas could not generate an answer because the shared LLM quota "
            "is unavailable. Please contact an admin."
        )
    return detail


def _answer_topic_payload(values: dict, response: dict, branch_context: dict) -> dict:
    topic = {
        **values,
        **branch_context,
        "conversation_id": response.get("conversation_id"),
        "question": response.get("question") or values.get("question"),
    }
    topic["topic_label"] = _topic_label(topic)
    return topic


def _slack_analytics_context(values: dict, branch: str = None) -> dict:
    return {
        "source": "slack",
        "slack_user_id": _slack_user_id(values),
        "slack_team_id": values.get("team_id"),
        "slack_channel_id": values.get("channel_id"),
        "ask_type": values.get("ask_type") or ASK_SINGLE,
        "branch": branch or values.get("branch"),
    }


def _current_branch(repo: dict, branch_name: str) -> Optional[dict]:
    if not repo or not branch_name:
        return None
    try:
        return db.get_repo_branch_by_name(repo["id"], branch_name)
    except Exception:
        return None


def _default_branch_name(repo: dict) -> str:
    """The repo's current default branch, for flows that skip branch picking.

    Checks already-known branches first: a repo that's already been used
    has its default recorded in the DB, so it doesn't need to pay a live
    git round-trip (remote_branch_options) on every single question — only
    a repo with no branch approved yet needs that network discovery call.
    """
    existing = db.list_repo_branches(repo["id"])
    for branch in existing:
        if branch.get("is_default"):
            return branch["name"]
    try:
        branches = ask_service.remote_branch_options(repo, limit=200)
    except Exception:
        branches = []
    for branch in branches:
        if branch.get("is_default"):
            return branch["name"]
    if branches:
        return branches[0]["name"]
    if existing:
        return existing[0]["name"]
    raise HTTPException(status_code=404, detail="No branch is available for this repository yet.")


def _branch_is_ready(branch: Optional[dict]) -> bool:
    return bool(
        branch
        and branch.get("workspace")
        and branch.get("index_status") == "ready"
    )


def _single_branch_preparation_notice(branch: Optional[dict]) -> tuple[str, bool]:
    status = (branch or {}).get("index_status")
    freshness = (branch or {}).get("freshness_status")
    if status == "indexing":
        return (
            "Branch is still being prepared. I'll notify you when the answer is ready...",
            False,
        )
    if status == "failed":
        return (
            "Re-indexing the selected branch because the previous index failed or is stale...",
            False,
        )
    if status == "never_indexed" or (branch and not branch.get("workspace")):
        return ("Indexing the selected branch for the first time...", False)
    if freshness in {"behind", "diverged"}:
        return (
            "Updating the selected branch index because new commits were found...",
            False,
        )
    if freshness == "checking":
        return ("Syncing the selected branch because new commits may be available...", False)
    if _branch_is_ready(branch):
        return (
            "Branch is ready. Searching repository context and generating the answer...",
            True,
        )
    return ("Preparing the selected branch and checking index status...", False)


def _compare_branch_preparation_notice(
    branches: list[Optional[dict]],
) -> tuple[str, bool]:
    known = [branch for branch in branches if branch]
    statuses = {(branch or {}).get("index_status") for branch in branches}
    freshnesses = {(branch or {}).get("freshness_status") for branch in branches}
    if "indexing" in statuses:
        return (
            "One or both selected branches are still being prepared. "
            "I'll notify you when the comparison is ready...",
            False,
        )
    if "failed" in statuses:
        return (
            "Re-indexing one or both selected branches because a previous index failed "
            "or is stale...",
            False,
        )
    if "never_indexed" in statuses or any(
        branch and not branch.get("workspace") for branch in branches
    ):
        return ("Indexing one or both selected branches before comparison...", False)
    if freshnesses & {"behind", "diverged"}:
        return (
            "Updating one or both selected branch indexes because new commits were found...",
            False,
        )
    if "checking" in freshnesses:
        return (
            "Syncing the selected branches because new commits may be available...",
            False,
        )
    if known and all(_branch_is_ready(branch) for branch in known):
        return (
            "Branches are ready. Searching both branches and generating the comparison...",
            True,
        )
    return ("Preparing the selected branches and checking index status...", False)


def _run_single_answer(values: dict, *, follow_up: bool = False, deep: bool = False) -> None:
    from .. import main

    slack_user = values["slack_user_id"]
    repo = _repo_by_slug(values.get("repo_slug"))
    if not repo:
        raise HTTPException(status_code=404, detail="Repository not found.")
    actor = ask_service.slack_actor_user(values["team_id"], slack_user, values.get("user_type"))
    workspace = values.get("branch_workspace")
    branch_context = {}
    generating_announced = False
    if not workspace:
        notice, generating_announced = _single_branch_preparation_notice(
            _current_branch(repo, values.get("branch"))
        )
        _send_user_message(
            values,
            "Preparing branch",
            [{"type": "section", "text": _mrkdwn(notice)}],
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
        feedback_id=f"feedback-{uuid.uuid4()}",
        llm_mode=_llm_mode(),
        conversation_id=values.get("conversation_id") if follow_up else None,
        follow_up=follow_up,
        deep_investigation=deep,
        answer_user_type=values.get("user_type") or USER_DEV,
    )
    if not generating_announced:
        _send_user_message(
            values,
            "Generating answer",
            [{"type": "section", "text": _mrkdwn("Searching repository context and generating the answer...")}],
        )
    response = ask_service.answer_single_request(
        request,
        workspace,
        actor,
        analytics_context=_slack_analytics_context(
            values,
            branch_context.get("branch") or values.get("branch"),
        ),
    )
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
    generating_announced = False
    if not base_workspace or not compare_workspace:
        notice, generating_announced = _compare_branch_preparation_notice([
            _current_branch(repo, values.get("base_branch")),
            _current_branch(repo, values.get("compare_branch")),
        ])
        _send_user_message(
            values,
            "Preparing comparison",
            [{"type": "section", "text": _mrkdwn(notice)}],
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
        feedback_id=f"feedback-{uuid.uuid4()}",
        left_branch=base["id"],
        right_branch=compare["id"],
        llm_mode=_llm_mode(),
        conversation_id=values.get("conversation_id") if follow_up else None,
        follow_up=follow_up,
        deep_investigation=deep,
        answer_user_type=values.get("user_type") or USER_DEV,
    )
    if not generating_announced:
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
        analytics_context=_slack_analytics_context(
            values,
            f"{base['name']}..{compare['name']}",
        ),
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
                    "text": _mrkdwn(f"I couldn't complete that request.\n\nReason: {_answer_failure_detail(exc)}"),
                }],
            )
        except Exception:
            pass


def _start_answer_job(values: dict, *, follow_up: bool = False, deep: bool = False) -> None:
    _executor.submit(_run_answer_job, values, follow_up=follow_up, deep=deep)


def _run_mention_job(payload: dict, event: dict) -> None:
    """Handle an @codeatlas channel mention or a DM message: infer the
    repo/branch deterministically (Phase A) and answer with the same
    pipeline the modal flow uses."""
    team_id = payload.get("team_id") or (payload.get("team") or {}).get("id")
    channel_id = event.get("channel")
    slack_user = event.get("user")
    thread_ts = event.get("thread_ts") or event.get("ts")
    question = _strip_mention(event.get("text") or "")
    values = {
        "team_id": team_id,
        "channel_id": channel_id,
        "user_id": slack_user,
        "slack_user_id": slack_user,
        "thread_ts": thread_ts,
        "question": question,
        "ask_type": ASK_SINGLE,
        "user_type": USER_PRODUCT,
    }
    if not channel_id or not slack_user:
        logger.warning("Ignoring Slack event missing channel or user.")
        return
    # Everything below (repo inference, branch prep) previously had no
    # safety net: an exception here would vanish in the executor thread
    # with no log and no reply, and since the event is already marked
    # handled before this job runs, even Slack's own retry couldn't help.
    try:
        if not question:
            _send_user_message(
                values,
                "CodeAtlas needs a question",
                [{
                    "type": "section",
                    "text": _mrkdwn(
                        "Ask me something after the mention, e.g. "
                        "`@CodeAtlas how do I go online in the app?`"
                    ),
                }],
            )
            return
        repos = ask_service.published_repos()
        repo = _infer_repo_from_text(question, repos)
        if not repo:
            if not repos:
                _send_user_message(
                    values,
                    "No repositories available",
                    [{
                        "type": "section",
                        "text": _mrkdwn("No repositories are published yet. Ask an admin to publish one."),
                    }],
                )
                return
            _remember_pending_question(channel_id, thread_ts, slack_user, question)
            names = ", ".join(f"`{item['name']}`" for item in repos[:10])
            _send_user_message(
                values,
                "Which repository?",
                [{
                    "type": "section",
                    "text": _mrkdwn(
                        f"I couldn't tell which repository you mean. Mention it by name, e.g. {names}."
                    ),
                }],
            )
            return
        # A bare repo-name reply (e.g. just "sortbuddy") to our own "which
        # repository?" prompt, in the same thread, answers that prompt
        # rather than being treated as a new one-word question.
        if question.strip().lower() in {repo["slug"].lower(), repo["name"].lower()}:
            pending_question = _take_pending_question(channel_id, thread_ts, slack_user)
            if pending_question:
                question = pending_question
                values["question"] = question
        values["repo_slug"] = repo["slug"]
        values["repo_name"] = repo["name"]
        branch_name = _default_branch_name(repo)
        # Only kick off a sync/index job the first time this repo's default
        # branch is used; once it's approved, the shared answer pipeline
        # below already re-checks freshness itself, so doing it again here
        # too just races that check and flickers a stale "still preparing"
        # notice on every later question.
        if not db.get_repo_branch_by_name(repo["id"], branch_name):
            try:
                ask_service.prepare_repo_branch(
                    repo,
                    branch_name,
                    actor=f"slack:{team_id}:{slack_user}",
                )
            except Exception:
                # Two near-simultaneous first-time questions about the same
                # repo can both reach here before either finishes; if the
                # other one already won and created the branch, that's a
                # completed setup, not a failure.
                if not db.get_repo_branch_by_name(repo["id"], branch_name):
                    raise
        values["branch"] = branch_name
    except Exception as exc:
        logger.exception("CodeAtlas could not prepare a repository/branch for a Slack question.")
        try:
            _send_user_message(
                values,
                "CodeAtlas could not answer",
                [{
                    "type": "section",
                    "text": _mrkdwn(f"I couldn't prepare that repository.\n\nReason: {_http_detail(exc)}"),
                }],
            )
        except Exception:
            pass
        return
    _run_answer_job(values, follow_up=False, deep=False)


def _start_mention_job(payload: dict, event: dict) -> None:
    _executor.submit(_run_mention_job, payload, event)


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
        notice, ready = _single_branch_preparation_notice(branch)
        status = "Ready" if ready else notice
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
async def slash_command(request: Request):
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
    _open_ask_modal(metadata, _form_value(form, "trigger_id"))
    return Response(status_code=200)


@router.post("/events")
async def slack_events(request: Request):
    if not slack_enabled():
        raise HTTPException(status_code=404, detail="Slack integration is not enabled.")
    body = await request.body()
    verify_slack_request(request.headers, body)
    try:
        payload = json.loads(body.decode("utf-8"))
    except json.JSONDecodeError:
        raise HTTPException(status_code=400, detail="Invalid Slack payload.")
    if payload.get("type") == "url_verification":
        return {"challenge": payload.get("challenge", "")}
    _authorize_slack_workspace(payload)
    if payload.get("type") != "event_callback":
        return Response(status_code=200)
    if _already_processed_event(payload.get("event_id")):
        return Response(status_code=200)
    event = payload.get("event") or {}
    event_type = event.get("type")
    is_channel_mention = event_type == "app_mention" and not event.get("bot_id")
    # A DM's messages are always directed at CodeAtlas, so no @mention is
    # required there; message.im only fires for actual 1:1/group DMs, and
    # subtype is set for edits/deletes/joins rather than a new message to
    # answer.
    is_dm_message = (
        event_type == "message"
        and event.get("channel_type") == "im"
        and not event.get("subtype")
        and not event.get("bot_id")
    )
    if is_channel_mention or is_dm_message:
        # A DM mention can trigger both app_mention and message.im for the
        # same message; only answer it once.
        if _already_answered_message(event.get("channel"), event.get("ts")):
            return Response(status_code=200)
        logger.info(
            "Accepted Slack %s for team=%s channel=%s user=%s",
            event_type, payload.get("team_id"), event.get("channel"), event.get("user"),
        )
        _start_mention_job(payload, event)
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
