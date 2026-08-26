import base64
import binascii
import copy
import hashlib
import json
import os
import re
import shutil
import secrets
import subprocess
import time
from collections import OrderedDict, defaultdict
from pathlib import Path
from threading import Lock
from typing import Optional

from fastapi import Depends, FastAPI, HTTPException, Query, Request
from fastapi.responses import FileResponse, PlainTextResponse
from pydantic import BaseModel

from . import ask_service, db
from .agent import ComparisonRepositoryToolbox, RepositoryToolbox
from .config import (
    DEFAULT_WORKSPACE,
    graph_path,
    repo_clone_dir,
    retrieval_config_path,
    source_index_path,
)
from .conversations import ConversationState, conversation_store
from .retrieval.flow_map import (
    TOPICS,
    build_discovered_flow,
    discover_flows,
    find_methods,
    load_graph,
    meta_for,
    pretty_name,
    pretty_method,
)
from .retrieval.graph_insights import repo_summary_dynamic
from .retrieval.source_index import search_source_index
from .retrieval.relation_utils import (
    format_link,
    is_noise_node,
    rank_nodes_for_query,
    readable_name,
    search_nodes,
)
from .retrieval.config_schema import load_retrieval_config, seed_default_retrieval_config
from .llm.client import (
    FollowUpNeedsEvidence,
    PRODUCT_TEAM_QUERY_SUFFIX,
    PRODUCT_TEAM_RESPONSE_INSTRUCTION,
    collect_token_usage,
    generate,
    generate_fast_follow_up,
    token_usage_payload,
)
from .llm.admission import LLMCapacityError, llm_admission
from .auth.routes import router as auth_router, load_user_llm
from .auth.security import hash_password
from .auth.sessions import COOKIE_NAME, COOKIE_SECURE, clear_session_cookie, require_user
from .repos.branch_routes import router as branch_router
from .repos.branches import (
    ensure_legacy_repo_branches,
    start_branch_services,
    stop_branch_services,
)
from .repos.routes import router as repos_router
from .slack.routes import router as slack_router

app = FastAPI(title="CodeAtlas", version="0.2.0")

STATIC_DIR = Path(__file__).resolve().parent / "static"
VISITOR_COOKIE_NAME = "ca_site_visitor"
VISITOR_COOKIE_MAX_AGE = 60 * 60 * 24 * 365

# Multi-tenant routers (auth + admin repo lifecycle).
app.include_router(auth_router)
app.include_router(repos_router)
app.include_router(branch_router)
app.include_router(slack_router)


@app.middleware("http")
async def add_noindex_header(request: Request, call_next):
    response = await call_next(request)
    response.headers.setdefault("X-Robots-Tag", "noindex, nofollow, noarchive")
    return response


@app.on_event("startup")
def _startup() -> None:
    """Create tables, register the seeded demo workspace as a repo, and (if
    configured) seed an admin so the instance is usable on first boot."""
    db.init_db()
    db.seed_default_repo()
    ensure_legacy_repo_branches()
    start_branch_services()
    seed_default_retrieval_config()
    admin_user = os.environ.get("CODEATLAS_ADMIN_USER")
    admin_pass = os.environ.get("CODEATLAS_ADMIN_PASS")
    if admin_user and admin_pass and db.user_count() == 0:
        db.create_user(admin_user, hash_password(admin_pass), role="admin")


@app.on_event("shutdown")
def _shutdown() -> None:
    stop_branch_services()


def workspace_source_root(workspace: str) -> Path:
    """Root of the indexed source tree for a workspace, so node source paths
    resolve to real files for code excerpts. The default workspace honors a
    CODEATLAS_SOURCE_ROOT override (handy for the demo graph)."""
    if workspace == DEFAULT_WORKSPACE:
        override = os.environ.get("CODEATLAS_SOURCE_ROOT")
        if override:
            return Path(override)
    return Path(repo_clone_dir(workspace))


# Per-user sliding-window rate limit for LLM asks. In-process (fine for the
# single-process self-host model); resets on restart.
RATE_LIMIT_PER_MIN = int(os.environ.get("CODEATLAS_RATE_LIMIT_PER_MIN", "20"))
_ask_hits: "dict[int, list]" = defaultdict(list)
ASK_ACTIVITY_TTL_SECONDS = 60 * 10
ASK_ACTIVITY_MAX_RECORDS = 256
ASK_ACTIVITY_NODE_LIMIT = 20
ASK_ACTIVITY_RELATION_LIMIT = 12
ASK_ACTIVITY_SOURCE_FILE_LIMIT = 12
ASK_ACTIVITY_VISIBLE_NODE_LIMIT = 8
ASK_ACTIVITY_VISIBLE_RELATION_LIMIT = 6
ASK_ACTIVITY_VISIBLE_SOURCE_FILE_LIMIT = 6
_ask_activity_lock = Lock()
_ask_activity: dict[str, dict] = {}
_ask_activity_id_pattern = re.compile(r"^[A-Za-z0-9_-]{8,96}$")
_ask_activity_stopwords = {
    "about", "after", "again", "also", "and", "are", "can", "code", "does",
    "explain", "feature", "features", "flow", "for", "from", "functionality",
    "functionalities", "happen", "happens", "how", "into", "more", "screen",
    "show", "shows", "tell", "that", "the", "their", "this", "what", "when",
    "where", "which", "with", "work", "works",
}
_source_file_cache: "dict[str, list[tuple[str, Path]]]" = {}
RETRIEVAL_CONTEXT_CACHE_MAX_RECORDS = max(
    0, int(os.environ.get("CODEATLAS_RETRIEVAL_CONTEXT_CACHE_MAX_RECORDS", "256"))
)
_retrieval_context_cache_lock = Lock()
_retrieval_context_cache: "OrderedDict[tuple, dict]" = OrderedDict()

SOURCE_EXTENSIONS = {
    ".kt", ".java", ".py", ".ts", ".tsx", ".js", ".jsx", ".go", ".rb", ".rs",
    ".cpp", ".cc", ".c", ".h", ".hpp", ".cs", ".php", ".swift", ".scala",
    ".dart", ".vue", ".svelte", ".html", ".css", ".scss", ".xml", ".json",
    ".yaml", ".yml", ".toml", ".gradle", ".md",
}
SOURCE_SKIP_DIRS = {
    ".git", ".gradle", ".idea", ".venv", "venv", "__pycache__", "node_modules",
    "build", "dist", ".next", ".turbo", "coverage", "target", ".dart_tool",
}
MAX_SOURCE_SCAN_FILES = int(os.environ.get("CODEATLAS_SOURCE_SCAN_FILES", "2500"))
MAX_SOURCE_SCAN_BYTES = int(os.environ.get("CODEATLAS_SOURCE_SCAN_BYTES", "240000"))
MAX_SOURCE_SNIPPET_CHARS = int(os.environ.get("CODEATLAS_SOURCE_SNIPPET_CHARS", "1800"))
LLM_PREVIEW_NODE_LIMIT = int(os.environ.get("CODEATLAS_LLM_PREVIEW_NODE_LIMIT", "8"))
LLM_PREVIEW_SOURCE_HITS = int(os.environ.get("CODEATLAS_LLM_PREVIEW_SOURCE_HITS", "8"))
LLM_PREVIEW_SNIPPET_CHARS = int(os.environ.get("CODEATLAS_LLM_PREVIEW_SNIPPET_CHARS", "1100"))
QUERY_IMAGE_MAX_COUNT = max(1, int(os.environ.get("CODEATLAS_QUERY_IMAGE_MAX_COUNT", "3")))
QUERY_IMAGE_MAX_BYTES = max(
    1024,
    int(os.environ.get("CODEATLAS_QUERY_IMAGE_MAX_BYTES", str(5 * 1024 * 1024))),
)
QUERY_IMAGE_ALLOWED_TYPES = {"image/png", "image/jpeg", "image/webp"}
SOURCE_QUERY_STOPWORDS = {
    "app", "application", "codebase", "project", "repo", "repository",
    "user", "users", "happen", "happens", "thing", "things",
}
IDENTIFIER_STOPWORDS = {
    "String", "Boolean", "Integer", "Long", "Double", "Float", "List", "ArrayList",
    "MutableList", "HashMap", "Map", "Set", "Flow", "LiveData", "MutableLiveData",
    "StateFlow", "Context", "Bundle", "View", "TextView", "Button", "ImageView",
    "RecyclerView", "Fragment", "Activity", "Override", "Serializable",
}


def enforce_rate_limit(user_id: int) -> None:
    now = time.monotonic()
    hits = [t for t in _ask_hits[user_id] if now - t < 60]
    if len(hits) >= RATE_LIMIT_PER_MIN:
        raise HTTPException(
            status_code=429,
            detail=f"Rate limit reached ({RATE_LIMIT_PER_MIN}/min). Please wait and retry.",
        )
    hits.append(now)
    _ask_hits[user_id] = hits


def _valid_activity_request_id(value: str = None) -> Optional[str]:
    value = str(value or "").strip()
    if not value or not _ask_activity_id_pattern.fullmatch(value):
        return None
    return value


def _activity_node_type(item: dict) -> str:
    name = str(item.get("name") or "")
    node = str(item.get("node") or "")
    source_file = str(item.get("source_file") or "")
    lowered = f"{name} {node} {source_file}".lower()
    if "/test/" in lowered or "test_" in lowered or name.lower().startswith("test"):
        return "Test"
    if source_file.endswith((".sql", ".ddl")) or node.startswith(("table_", "db_table_")):
        return "Table"
    if any(marker in lowered for marker in [" route ", " endpoint ", "router", "controller"]):
        return "Route"
    if source_file and not name:
        return "File"
    if "." in name or node.startswith(("func_", "method_")):
        return "Function"
    if name[:1].isupper() and "." not in name:
        return "Class"
    return "Symbol"


def _activity_source_label(source_file: str, source_location: str = None) -> str:
    source = str(source_file or "").strip()
    location = str(source_location or "").strip()
    if source and location and location != "?":
        return f"{source} {location}"
    return source or location


def _activity_query_terms(question: str = None, context: dict = None) -> list[str]:
    text = str(question or (context or {}).get("question") or "")
    preview = (context or {}).get("llm_context_preview") or {}
    if not text and isinstance(preview, dict):
        text = str(preview.get("question") or "")
    terms = []
    for token in re.findall(r"[A-Za-z][A-Za-z0-9]+", text.lower()):
        if len(token) <= 2 or token in _ask_activity_stopwords:
            continue
        terms.append(token)
        if token.endswith("ies") and len(token) > 4:
            terms.append(f"{token[:-3]}y")
        elif token.endswith("s") and len(token) > 4:
            terms.append(token[:-1])
    seen = set()
    unique = []
    for term in terms:
        if term not in seen:
            unique.append(term)
            seen.add(term)
    return unique[:10]


def _activity_compact(value: str) -> str:
    return str(value or "").lower().replace("_", "").replace("-", "").replace(".", "").replace("/", "")


def _activity_match_score(item: dict, terms: list[str], *, file_item: bool = False) -> int:
    if not terms:
        return 1
    name = str(item.get("name") or item.get("path") or "")
    node = str(item.get("node") or "")
    source = str(item.get("source_file") or item.get("path") or "")
    haystacks = {
        "name": _activity_compact(name),
        "node": _activity_compact(node),
        "source": _activity_compact(source),
    }
    score = 0
    for term in terms:
        compact = _activity_compact(term)
        if compact and compact in haystacks["name"]:
            score += 8
        if compact and compact in haystacks["node"]:
            score += 5
        if compact and compact in haystacks["source"]:
            score += 4
    if file_item and score > 0:
        score += min(5, int(item.get("score") or 0) // 200)
    return score


def _answer_activity_from_context(context: dict, question: str = None) -> dict:
    context = context or {}
    query_terms = _activity_query_terms(question, context)
    if context.get("comparison_mode"):
        repositories = context.get("comparison_repositories") or []
        nodes = []
        relations = []
        source_hits = []
        for repository in repositories:
            branch_label = repository.get("label") or repository.get("name") or "Branch"
            for node in repository.get("context_nodes") or []:
                item = {**node}
                item["branch_label"] = branch_label
                nodes.append(item)
            for relation in repository.get("context_relations") or []:
                item = {**relation}
                item["branch_label"] = branch_label
                relations.append(item)
            for hit in repository.get("source_hits") or []:
                item = {**hit}
                item["branch_label"] = branch_label
                source_hits.append(item)
    else:
        nodes = list(context.get("context_nodes") or [])
        relations = list(context.get("context_relations") or [])
        source_hits = list(context.get("source_hits") or [])

    scored_nodes = [
        (index, _activity_match_score(item, query_terms), item)
        for index, item in enumerate(nodes[:ASK_ACTIVITY_NODE_LIMIT])
    ]
    visible_nodes = [
        item for index, score, item in sorted(scored_nodes, key=lambda row: (-row[1], row[0]))
        if score > 0
    ][:ASK_ACTIVITY_VISIBLE_NODE_LIMIT]

    activity_nodes = []
    visible_node_names = set()
    visible_node_ids = set()
    visible_source_files = set()
    for item in visible_nodes:
        source = _activity_source_label(item.get("source_file"), item.get("source_location"))
        node_name = item.get("name") or readable_name(str(item.get("node") or ""))
        visible_node_names.add(str(node_name))
        visible_node_ids.add(str(item.get("node") or ""))
        if item.get("source_file"):
            visible_source_files.add(str(item.get("source_file")))
        activity_nodes.append({
            "type": _activity_node_type(item),
            "name": node_name,
            "source": source,
            **({"branch_label": item["branch_label"]} if item.get("branch_label") else {}),
        })

    scored_relations = []
    for index, item in enumerate(relations[:ASK_ACTIVITY_RELATION_LIMIT]):
        relation_score = 0
        source_name = str(item.get("source_name") or readable_name(str(item.get("source") or "")))
        target_name = str(item.get("target_name") or readable_name(str(item.get("target") or "")))
        if (
            source_name in visible_node_names
            or target_name in visible_node_names
            or str(item.get("source") or "") in visible_node_ids
            or str(item.get("target") or "") in visible_node_ids
        ):
            relation_score += 8
        relation_score += _activity_match_score({
            "name": f"{source_name} {target_name} {item.get('relation_label') or item.get('relation') or ''}",
            "source_file": item.get("source_file"),
        }, query_terms)
        scored_relations.append((index, relation_score, item))
    visible_relations = [
        item for index, score, item in sorted(scored_relations, key=lambda row: (-row[1], row[0]))
        if score > 0
    ][:ASK_ACTIVITY_VISIBLE_RELATION_LIMIT]

    activity_relations = []
    for item in visible_relations:
        activity_relations.append({
            "from": item.get("source_name") or readable_name(str(item.get("source") or "")),
            "relation": item.get("relation_label") or item.get("relation") or "related to",
            "to": item.get("target_name") or readable_name(str(item.get("target") or "")),
            "source": _activity_source_label(item.get("source_file"), item.get("source_location")),
            **({"branch_label": item["branch_label"]} if item.get("branch_label") else {}),
        })

    seen_files = set()
    scored_source_hits = []
    for index, hit in enumerate(source_hits[:ASK_ACTIVITY_SOURCE_FILE_LIMIT]):
        score = _activity_match_score(hit, query_terms, file_item=True)
        if str(hit.get("path") or "") in visible_source_files:
            score += 8
        scored_source_hits.append((index, score, hit))
    activity_files = []
    for index, score, hit in sorted(scored_source_hits, key=lambda row: (-row[1], row[0])):
        if score <= 0:
            continue
        path = str(hit.get("path") or "").strip()
        if not path or path in seen_files:
            continue
        seen_files.add(path)
        ranges = [
            f"L{snippet.get('start_line')}-L{snippet.get('end_line')}"
            for snippet in (hit.get("snippets") or [])[:2]
            if snippet.get("start_line") and snippet.get("end_line")
        ]
        activity_files.append({
            "path": path,
            "ranges": ranges,
            **({"branch_label": hit["branch_label"]} if hit.get("branch_label") else {}),
        })
        if len(activity_files) >= ASK_ACTIVITY_VISIBLE_SOURCE_FILE_LIMIT:
            break

    return {
        "candidate_node_count": len(nodes),
        "candidate_relation_count": len(relations),
        "candidate_source_file_count": len({str(hit.get("path") or "") for hit in source_hits if hit.get("path")}),
        "node_count": len(activity_nodes),
        "relation_count": len(activity_relations),
        "source_file_count": len(activity_files),
        "query_terms": query_terms,
        "nodes": activity_nodes,
        "relations": activity_relations,
        "source_files": activity_files,
    }


def _prune_answer_activity(now: float) -> None:
    stale = [
        key for key, value in _ask_activity.items()
        if now - float(value.get("_monotonic_updated_at") or 0) > ASK_ACTIVITY_TTL_SECONDS
    ]
    for key in stale:
        _ask_activity.pop(key, None)
    overflow = len(_ask_activity) - ASK_ACTIVITY_MAX_RECORDS
    if overflow > 0:
        oldest = sorted(
            _ask_activity,
            key=lambda key: _ask_activity[key].get("_monotonic_updated_at") or 0,
        )
        for key in oldest[:overflow]:
            _ask_activity.pop(key, None)


def record_answer_activity(
    request_id: str = None,
    *,
    user_id: int,
    workspace: str = None,
    question: str = None,
    status: str = "generating_answer",
    context: dict = None,
) -> None:
    request_id = _valid_activity_request_id(request_id)
    if not request_id:
        return
    now = time.monotonic()
    context_payload = _answer_activity_from_context(context or {}, question=question)
    activity = {
        "request_id": request_id,
        "user_id": int(user_id),
        "workspace": workspace,
        "question": str(question or "")[:240],
        "status": status,
        "stage": status,
        "updated_at": time.time(),
        "_monotonic_updated_at": now,
        **context_payload,
    }
    with _ask_activity_lock:
        _prune_answer_activity(now)
        _ask_activity[request_id] = activity


def update_answer_activity(
    request_id: str = None,
    *,
    user_id: int,
    workspace: str = None,
    question: str = None,
    status: str = None,
    context: dict = None,
) -> None:
    request_id = _valid_activity_request_id(request_id)
    if not request_id:
        return
    now = time.monotonic()
    update = {
        "request_id": request_id,
        "user_id": int(user_id),
        "updated_at": time.time(),
        "_monotonic_updated_at": now,
    }
    if workspace is not None:
        update["workspace"] = workspace
    if question is not None:
        update["question"] = str(question or "")[:240]
    if status:
        update["status"] = status
        update["stage"] = status
    if context is not None:
        update.update(_answer_activity_from_context(context, question=question))

    with _ask_activity_lock:
        _prune_answer_activity(now)
        current = _ask_activity.get(request_id, {})
        if current and int(current.get("user_id") or 0) != int(user_id):
            return
        merged = {**current, **update}
        merged.setdefault("node_count", 0)
        merged.setdefault("relation_count", 0)
        merged.setdefault("source_file_count", 0)
        merged.setdefault("candidate_node_count", 0)
        merged.setdefault("candidate_relation_count", 0)
        merged.setdefault("candidate_source_file_count", 0)
        merged.setdefault("query_terms", [])
        merged.setdefault("nodes", [])
        merged.setdefault("relations", [])
        merged.setdefault("source_files", [])
        _ask_activity[request_id] = merged


@app.get("/repo/ask-activity/{request_id}")
def answer_activity_endpoint(request_id: str, user: dict = Depends(require_user)):
    request_id = _valid_activity_request_id(request_id)
    if not request_id:
        raise HTTPException(status_code=404, detail="Answer activity was not found.")
    with _ask_activity_lock:
        record = copy.deepcopy(_ask_activity.get(request_id))
    if not record or int(record.get("user_id") or 0) != int(user["id"]):
        raise HTTPException(status_code=404, detail="Answer activity was not found.")
    record.pop("user_id", None)
    record.pop("_monotonic_updated_at", None)
    return record


def authorized_workspace(
    workspace: str = DEFAULT_WORKSPACE,
    branch: Optional[int] = None,
    user: dict = Depends(require_user),
) -> str:
    """Resolve + permission-check the target workspace for a query. Admins reach
    every workspace; users only reach repos granted to them."""
    repo = db.get_repo_by_workspace(workspace)
    authorization_workspace = repo["workspace"] if repo else workspace
    if (
        user["role"] != "admin"
        and not db.user_has_repo(user["id"], authorization_workspace)
    ):
        raise HTTPException(status_code=403, detail="You do not have access to this repository.")
    if branch is not None:
        selected = db.get_repo_branch(branch)
        if not selected or not repo or selected["repo_id"] != repo["id"]:
            raise HTTPException(status_code=404, detail="Repository branch not found.")
        if not selected.get("workspace") or selected["index_status"] not in {
            "ready", "indexing",
        }:
            raise HTTPException(
                status_code=409,
                detail="The selected branch has not been indexed successfully yet.",
            )
        return selected["workspace"]
    if repo and workspace == repo["workspace"]:
        selected = db.get_legacy_repo_branch(repo["id"])
        if selected and selected.get("workspace"):
            return selected["workspace"]
    return workspace


def enforce_strict_branch_freshness(workspace: str) -> None:
    branch = db.get_repo_branch_by_workspace(workspace)
    if (
        branch
        and branch["strict_freshness"]
        and branch["freshness_status"] != "up_to_date"
    ):
        raise HTTPException(
            status_code=409,
            detail=(
                "This branch requires a fresh index. Use Sync & index now "
                "before asking a question."
            ),
        )


def read_source_excerpt(
    source_file: str, source_location: str, source_root: Path,
    max_lines: int = 32, max_chars: int = 1100,
) -> "dict | None":
    """Read the actual code at a node's location so the LLM can reason about
    behavior, not just node names. Captures the enclosing block via brace
    balancing, falling back to a small window, and is bounded in size."""
    if not source_file:
        return None

    match = re.search(r"L(\d+)(?:\s*[-–]\s*L?(\d+))?", source_location or "")
    if not match:
        return None

    start = max(1, int(match.group(1)))
    end = int(match.group(2)) if match.group(2) else None

    path = (source_root / source_file).resolve()
    try:
        path.relative_to(source_root)
    except ValueError:
        return None
    if not path.is_file() or RepositoryToolbox.is_sensitive_path_for_root(
        path, source_root
    ):
        return None

    try:
        lines = path.read_text(errors="replace").splitlines()
    except OSError:
        return None
    if start > len(lines):
        return None

    if end:
        last = min(len(lines), end, start + max_lines - 1)
    else:
        last = start
        depth = 0
        opened = False
        for i in range(start - 1, min(len(lines), start - 1 + max_lines)):
            depth += lines[i].count("{") - lines[i].count("}")
            last = i + 1
            if "{" in lines[i]:
                opened = True
            if opened and depth <= 0:
                break
        if not opened:
            last = min(len(lines), start + 11)

    excerpt = "\n".join(lines[start - 1:last])
    if len(excerpt) > max_chars:
        excerpt = excerpt[:max_chars] + "\n…"

    return {"start_line": start, "end_line": last, "code": excerpt}


def _safe_source_root(workspace: str) -> Path:
    return workspace_source_root(workspace).resolve()


def _iter_source_files(source_root: Path):
    if not source_root.exists():
        return

    cache_key = str(source_root)
    cached = _source_file_cache.get(cache_key)
    if cached is not None:
        for item in cached:
            yield item
        return

    indexed = []
    scanned = 0
    for dirpath, dirnames, filenames in os.walk(source_root):
        dirnames[:] = [
            name for name in dirnames
            if name not in SOURCE_SKIP_DIRS and not name.startswith(".cache")
        ]

        for filename in filenames:
            path = Path(dirpath) / filename
            if path.suffix.lower() not in SOURCE_EXTENSIONS and "." in filename:
                continue

            try:
                rel_path = path.relative_to(source_root).as_posix()
                path = RepositoryToolbox.resolve_path_for_root(source_root, rel_path)
                if path.stat().st_size > MAX_SOURCE_SCAN_BYTES:
                    continue
            except (OSError, ValueError):
                continue

            scanned += 1
            if scanned > MAX_SOURCE_SCAN_FILES:
                _source_file_cache[cache_key] = indexed
                return

            item = (rel_path, path)
            indexed.append(item)
            yield item

    _source_file_cache[cache_key] = indexed


def _line_score(line: str, terms: list[str]) -> float:
    lower = line.lower()
    compacted = lower.replace("_", "").replace("-", "").replace(".", "")
    score = 0.0
    for term in terms:
        if not term:
            continue
        if term in lower:
            score += 4.0
        if term.replace("_", "") in compacted:
            score += 2.0
    return score


def _source_kind_score(rel_path: str) -> float:
    lower = rel_path.lower()
    if "/src/main/" in lower:
        return 24.0
    if "/src/test/" in lower or "/src/androidtest/" in lower:
        return -24.0
    if lower.startswith("docs/") or "/docs/" in lower or lower.endswith(".md"):
        return -14.0
    return 0.0


def _source_snippets(
    path: Path,
    terms: list[str],
    max_snippets: int = 2,
    focus_lines: set[int] = None,
) -> list[dict]:
    try:
        lines = path.read_text(errors="replace").splitlines()
    except OSError:
        return []

    scored = []
    focus_lines = focus_lines or set()
    for index, line in enumerate(lines):
        score = _line_score(line, terms)
        if index + 1 in focus_lines:
            score += 12.0
        if score > 0:
            scored.append((score, index))

    scored.sort(reverse=True)

    if not scored and lines:
        scored = [(1.0, 0)]

    snippets = []
    used_ranges = []

    for score, index in scored[:20]:
        start = max(0, index - 8)
        end = min(len(lines), index + 13)

        if any(not (end < used_start or start > used_end) for used_start, used_end in used_ranges):
            continue

        code = "\n".join(lines[start:end])
        if len(code) > MAX_SOURCE_SNIPPET_CHARS:
            code = code[:MAX_SOURCE_SNIPPET_CHARS] + "\n..."

        snippets.append({
            "start_line": start + 1,
            "end_line": end,
            "code": code,
            "score": round(score, 2),
        })
        used_ranges.append((start, end))

        if len(snippets) >= max_snippets:
            break

    return snippets


def _extract_identifiers_from_text(text: str) -> list[str]:
    candidates = re.findall(
        r"\b[A-Z][A-Za-z0-9]{3,}\b|\b[a-z]+(?:[A-Z][A-Za-z0-9]+)+\b|\b[A-Z][A-Z0-9_]{4,}\b",
        text,
    )
    ranked = {}
    for candidate in candidates:
        if candidate in IDENTIFIER_STOPWORDS:
            continue
        if len(candidate) < 5:
            continue
        if candidate.isupper() and "_" not in candidate:
            continue

        score = 1
        if candidate.endswith(("UseCase", "ViewModel", "Repository", "Fragment", "Interactor", "Handler")):
            score += 8
        if candidate.endswith(("Request", "Response", "Data", "Entity", "State", "Event")):
            score += 4
        if any(word in candidate.lower() for word in ("picked", "pickup", "validation", "qcom", "order")):
            score += 5

        ranked[candidate] = max(ranked.get(candidate, 0), score)

    return [
        item for item, _ in sorted(ranked.items(), key=lambda pair: (-pair[1], pair[0]))
    ]


def _identifier_terms_from_hits(source_hits: list[dict], query_terms: list[str], limit: int = 18) -> list[str]:
    text_parts = []
    for hit in source_hits[:8]:
        text_parts.append(hit.get("path", ""))
        for snippet in hit.get("snippets", [])[:2]:
            text_parts.append(snippet.get("code", ""))

    identifiers = _extract_identifiers_from_text("\n".join(text_parts))
    query_compact = " ".join(query_terms).lower()

    def identifier_score(identifier: str) -> int:
        lower = identifier.lower()
        score = 0
        if any(term and term in lower for term in query_terms):
            score += 6
        if any(term and term in query_compact for term in re.findall(r"[A-Z]?[a-z]+|[A-Z]+", identifier)):
            score += 3
        if identifier.endswith(("UseCase", "ViewModel", "Repository", "Fragment")):
            score += 6
        if any(word in lower for word in ("picked", "pickup", "validation", "qcom", "order")):
            score += 7
        return score

    ranked = sorted(set(identifiers), key=lambda item: (-identifier_score(item), item))
    return ranked[:limit]


def _merge_source_hits(primary: list[dict], secondary: list[dict], limit: int = 12) -> list[dict]:
    by_path = {hit["path"]: hit for hit in primary}
    for hit in secondary:
        existing = by_path.get(hit["path"])
        if not existing or hit["score"] > existing["score"]:
            by_path[hit["path"]] = hit

    hits = list(by_path.values())
    hits.sort(key=lambda item: (-item["score"], -_source_kind_score(item["path"]), item["path"]))
    return hits[:limit]


def _path_source_candidates(source_root: Path, terms: list[str], limit: int = 24) -> list[tuple[float, str, Path]]:
    candidates = []
    compact_terms = [t.lower().replace("_", "").replace("-", "").replace(".", "") for t in terms]

    for rel_path, path in _iter_source_files(source_root) or []:
        path_lower = rel_path.lower()
        path_compact = path_lower.replace("_", "").replace("-", "").replace(".", "")
        score = 0.0
        for term, compact_term in zip(terms, compact_terms):
            if term in path_lower:
                score += 35.0
            if compact_term and compact_term in path_compact:
                score += 18.0
        score += _source_kind_score(rel_path)
        if score:
            candidates.append((score, rel_path, path))

    candidates.sort(key=lambda item: (-item[0], item[1]))
    return candidates[:limit]


def _module_search_roots(path_candidates: list[tuple[float, str, Path]], limit: int = 8) -> list[str]:
    roots = []
    for _, rel_path, _ in path_candidates[:limit]:
        if "/src/" in rel_path:
            root = rel_path.split("/src/", 1)[0]
        else:
            root = str(Path(rel_path).parent)

        if not root or root == "." or root in roots:
            continue
        roots.append(root)

    return roots


def _rg_source_matches(
    source_root: Path,
    terms: list[str],
    timeout: float = 8.0,
    search_paths: list[str] = None,
) -> dict[str, set[int]]:
    if not terms or not shutil.which("rg"):
        return {}

    pattern = "|".join(re.escape(term) for term in terms if term)
    if not pattern:
        return {}

    command = [
        "rg",
        "--json",
        "--ignore-case",
        "--line-number",
        "--max-count",
        "12",
        "--max-filesize",
        f"{MAX_SOURCE_SCAN_BYTES}",
    ]
    for skip_dir in SOURCE_SKIP_DIRS:
        command.extend(["--glob", f"!**/{skip_dir}/**"])
    paths = search_paths or ["."]
    command.extend(["--", pattern, *paths])

    try:
        result = subprocess.run(
            command,
            cwd=str(source_root),
            capture_output=True,
            text=True,
            timeout=timeout,
        )
    except (OSError, subprocess.TimeoutExpired):
        return {}

    matches: dict[str, set[int]] = {}
    for line in result.stdout.splitlines():
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            continue
        if event.get("type") != "match":
            continue

        data = event.get("data") or {}
        rel_path = ((data.get("path") or {}).get("text") or "").lstrip("./")
        line_number = data.get("line_number")
        if not rel_path or not line_number:
            continue

        matches.setdefault(rel_path, set()).add(int(line_number))

    return matches


def _search_source_files(
    source_root: Path,
    terms: list[str],
    limit: int = 8,
    index_path: Path = None,
) -> list[dict]:
    if not terms:
        return []
    if index_path is not None:
        indexed_hits = search_source_index(index_path, terms, limit=limit)
        if indexed_hits is not None:
            return indexed_hits

    hits = []
    compact_terms = [t.lower().replace("_", "").replace("-", "").replace(".", "") for t in terms]
    path_candidates = _path_source_candidates(source_root, terms)
    rg_matches = _rg_source_matches(
        source_root,
        terms,
        search_paths=_module_search_roots(path_candidates),
    )

    candidate_paths: dict[str, tuple[float, Path]] = {
        rel_path: (score, path) for score, rel_path, path in path_candidates
    }
    for rel_path, line_numbers in rg_matches.items():
        path = source_root / rel_path
        existing_score = candidate_paths.get(rel_path, (0.0, path))[0]
        candidate_paths[rel_path] = (existing_score + min(80.0, len(line_numbers) * 8.0), path)

    # If ripgrep is unavailable or finds nothing, fall back to the bounded
    # Python scan so small repos still get content search.
    if not candidate_paths:
        for rel_path, path in _iter_source_files(source_root) or []:
            candidate_paths[rel_path] = (0.0, path)

    for rel_path, (initial_score, path) in candidate_paths.items():
        path_lower = rel_path.lower()
        path_compact = path_lower.replace("_", "").replace("-", "").replace(".", "")
        stem_compact = path.stem.lower().replace("_", "").replace("-", "").replace(".", "")
        score = initial_score

        for term, compact_term in zip(terms, compact_terms):
            if term in path_lower:
                score += 35.0
            if compact_term and compact_term in path_compact:
                score += 18.0
            if compact_term and compact_term == stem_compact:
                score += 420.0

        try:
            path = RepositoryToolbox.resolve_path_for_root(source_root, rel_path)
            text = path.read_text(errors="replace")
        except (OSError, ValueError):
            continue

        text_lower = text.lower()
        text_compact = text_lower.replace("_", "").replace("-", "").replace(".", "")
        for term, compact_term in zip(terms, compact_terms):
            count = text_lower.count(term)
            if count:
                score += min(28.0, count * 3.5)
            if compact_term and compact_term != term:
                compact_count = text_compact.count(compact_term)
                if compact_count:
                    score += min(18.0, compact_count * 2.0)

        score += _source_kind_score(rel_path)

        if score <= 0:
            continue

        snippets = _source_snippets(path, terms, focus_lines=rg_matches.get(rel_path, set()))
        if not snippets:
            continue

        hits.append({
            "path": rel_path,
            "score": round(score + sum(s["score"] for s in snippets), 2),
            "snippets": snippets,
        })

    hits.sort(key=lambda item: (-item["score"], item["path"]))
    return hits[:limit]


def _repo_overview(nodes: list[dict], links: list[dict]) -> dict:
    summary = repo_summary_dynamic_from_loaded(nodes, links)
    return {
        "screens": summary["screens"][:10],
        "viewmodels": summary["viewmodels"][:10],
        "repositories": summary["repositories"][:10],
        "services": summary["managers_and_services"][:10],
    }


def repo_summary_dynamic_from_loaded(nodes: list[dict], links: list[dict]) -> dict:
    from .retrieval.graph_insights import (
        detect_managers_and_services,
        detect_repositories,
        detect_screens,
        detect_viewmodels,
    )

    return {
        "screens": detect_screens(nodes, links),
        "viewmodels": detect_viewmodels(nodes, links),
        "repositories": detect_repositories(nodes, links),
        "managers_and_services": detect_managers_and_services(nodes, links),
    }



@app.get("/")
def root(request: Request):
    """Marketing page. Visiting home always starts a signed-out session."""
    db.delete_session(request.cookies.get(COOKIE_NAME))
    visitor_key = request.cookies.get(VISITOR_COOKIE_NAME) or secrets.token_urlsafe(24)
    db.record_site_visit(visitor_key)
    response = FileResponse(STATIC_DIR / "home.html")
    clear_session_cookie(response)
    response.set_cookie(
        VISITOR_COOKIE_NAME,
        visitor_key,
        max_age=VISITOR_COOKIE_MAX_AGE,
        httponly=True,
        secure=COOKIE_SECURE,
        samesite="lax",
        path="/",
    )
    response.headers["Cache-Control"] = "no-store"
    return response


@app.get("/healthz", include_in_schema=False)
def healthz():
    """Load-balancer readiness check covering the app and SQLite data path."""
    try:
        with db.connect() as connection:
            connection.execute("SELECT 1").fetchone()
    except Exception as error:
        raise HTTPException(status_code=503, detail="Database unavailable.") from error
    return {"status": "ok"}


@app.get("/robots.txt", include_in_schema=False)
def robots_txt():
    return PlainTextResponse(
        "User-agent: *\nDisallow: /\n",
        headers={
            "Cache-Control": "public, max-age=3600",
            "X-Robots-Tag": "noindex, nofollow, noarchive",
        },
    )


@app.get("/app")
def ask_ui():
    """The user Ask UI (current login flow): login → repo picker → ask."""
    response = FileResponse(STATIC_DIR / "index.html")
    response.headers["Cache-Control"] = "no-store"
    return response


@app.get("/admin.html")
def admin_console():
    response = FileResponse(STATIC_DIR / "admin.html")
    response.headers["Cache-Control"] = "no-store"
    return response


@app.get("/admin", include_in_schema=False)
def admin_console_alias():
    response = FileResponse(STATIC_DIR / "admin.html")
    response.headers["Cache-Control"] = "no-store"
    return response


# Map source-file extensions to a display language for the public catalog.
_LANG_BY_EXT = {
    ".kt": "Kotlin", ".java": "Java", ".py": "Python", ".ts": "TypeScript",
    ".tsx": "TypeScript", ".js": "JavaScript", ".jsx": "JavaScript", ".go": "Go",
    ".rb": "Ruby", ".rs": "Rust", ".cpp": "C++", ".cc": "C++", ".c": "C",
    ".cs": "C#", ".php": "PHP", ".swift": "Swift", ".scala": "Scala", ".dart": "Dart",
}
_catalog_cache: "dict[str, tuple]" = {}


def _graph_stats(workspace: str) -> "tuple[int, str | None]":
    """Live (node_count, dominant_language) for a workspace's graph, cached by
    file mtime so the public landing page reflects real indexing without
    re-reading large graphs on every hit."""
    repo = db.get_repo_by_workspace(workspace)
    if repo and workspace == repo["workspace"]:
        branch = db.get_legacy_repo_branch(repo["id"])
        if branch and branch.get("workspace"):
            workspace = branch["workspace"]
    path = graph_path(workspace)
    try:
        mtime = path.stat().st_mtime
    except OSError:
        return 0, None
    cached = _catalog_cache.get(str(path))
    if cached and cached[0] == mtime:
        return cached[1], cached[2]
    try:
        nodes, links = load_graph(path)
    except Exception:
        return 0, None
    counts: "dict[str, int]" = {}
    for link in links:
        ext = os.path.splitext(link.get("source_file") or "")[1].lower()
        lang = _LANG_BY_EXT.get(ext)
        if lang:
            counts[lang] = counts.get(lang, 0) + 1
    language = max(counts, key=counts.get) if counts else None
    _catalog_cache[str(path)] = (mtime, len(nodes), language)
    return len(nodes), language


@app.get("/public/catalog")
def public_catalog():
    """Public, unauthenticated catalog of published repositories with live graph
    stats — drives the landing page. Lists only published repos; never source."""
    repos = []
    total_nodes = 0
    for repo in db.list_repos():
        if repo["status"] != "published":
            continue
        node_count, language = _graph_stats(repo["workspace"])
        total_nodes += node_count
        repos.append({
            "name": repo["name"],
            "slug": repo["slug"],
            "status": repo["status"],
            "language": language,
            "nodes": node_count,
        })
    return {"repos": repos, "totals": {"repos": len(repos), "nodes": total_nodes}}


class AskRequest(BaseModel):
    question: str
    llm_mode: Optional[str] = None
    conversation_id: Optional[str] = None
    follow_up: bool = False
    deep_investigation: bool = False
    answer_user_type: Optional[str] = None
    image_attachments: Optional[list[dict]] = None
    activity_request_id: Optional[str] = None
    # Optional bring-your-own-key creds {provider, base_url, api_key, model}.
    user_llm: Optional[dict] = None


def normalize_image_attachments(raw_attachments) -> list[dict]:
    """Validate transient web query images and return provider-ready base64."""
    if not raw_attachments:
        return []
    if not isinstance(raw_attachments, list):
        raise HTTPException(status_code=400, detail="image_attachments must be a list.")
    if len(raw_attachments) > QUERY_IMAGE_MAX_COUNT:
        raise HTTPException(
            status_code=400,
            detail=f"Upload at most {QUERY_IMAGE_MAX_COUNT} images.",
        )

    normalized = []
    for index, item in enumerate(raw_attachments, start=1):
        if not isinstance(item, dict):
            raise HTTPException(status_code=400, detail="Each image attachment must be an object.")
        mime_type = str(
            item.get("mime_type") or item.get("content_type") or ""
        ).strip().lower()
        data = str(item.get("data") or "").strip()
        data_url = str(item.get("data_url") or "").strip()
        if data_url:
            header, separator, encoded = data_url.partition(",")
            if not separator or not header.lower().startswith("data:"):
                raise HTTPException(status_code=400, detail="Invalid image data URL.")
            media = header[5:].split(";", 1)[0].strip().lower()
            if mime_type and media and mime_type != media:
                raise HTTPException(status_code=400, detail="Image content type mismatch.")
            mime_type = mime_type or media
            data = encoded.strip()
        if mime_type not in QUERY_IMAGE_ALLOWED_TYPES:
            raise HTTPException(
                status_code=400,
                detail="Only PNG, JPEG, and WebP images are supported.",
            )
        if not data:
            raise HTTPException(status_code=400, detail="Image data is required.")
        try:
            decoded = base64.b64decode(data, validate=True)
        except (binascii.Error, ValueError):
            raise HTTPException(status_code=400, detail="Image data must be valid base64.")
        if not decoded:
            raise HTTPException(status_code=400, detail="Image data is empty.")
        if len(decoded) > QUERY_IMAGE_MAX_BYTES:
            max_mb = QUERY_IMAGE_MAX_BYTES / (1024 * 1024)
            raise HTTPException(
                status_code=400,
                detail=f"Each image must be {max_mb:.0f} MB or smaller.",
            )
        name = str(item.get("name") or f"image-{index}").strip()[:120]
        normalized.append({
            "name": name,
            "mime_type": mime_type,
            "data": data,
            "size": len(decoded),
        })
    return normalized


class FlowSummaryRequest(BaseModel):
    llm_mode: Optional[str] = None
    answer_user_type: Optional[str] = None
    user_llm: Optional[dict] = None


class CompareRequest(BaseModel):
    question: str
    left_branch: int
    right_branch: int
    llm_mode: Optional[str] = None
    conversation_id: Optional[str] = None
    follow_up: bool = False
    deep_investigation: bool = False
    answer_user_type: Optional[str] = None
    activity_request_id: Optional[str] = None
    user_llm: Optional[dict] = None


@app.get("/health")
def health():
    return {"status": "ok", "service": "codeatlas-api"}


@app.get("/repo/summary")
def repo_summary(workspace: str = Depends(authorized_workspace)):
    return repo_summary_dynamic(graph_path(workspace))


@app.get("/repo/flows/{topic}")
def flow(topic: str, workspace: str = Depends(authorized_workspace)):
    return _flow(topic, workspace)


@app.get("/repo/flows")
def flows(workspace: str = Depends(authorized_workspace)):
    return {"flows": discover_flows(graph_path(workspace))}


def _flow(topic: str, workspace: str):
    topic = topic.lower()

    discovered = build_discovered_flow(topic, graph_path(workspace))
    if discovered:
        return discovered

    if topic not in TOPICS:
        available = [flow["slug"] for flow in discover_flows(graph_path(workspace))]
        raise HTTPException(status_code=404, detail={"available_flows": available})

    nodes, links = load_graph(graph_path(workspace))
    config = TOPICS[topic]
    node_ids = {str(node.get("id") or node.get("label") or node.get("name") or "") for node in nodes}
    configured_nodes = set(config["screens"] + config["viewmodels"] + config["repositories"])
    if not configured_nodes.intersection(node_ids):
        available = [flow["slug"] for flow in discover_flows(graph_path(workspace))]
        raise HTTPException(status_code=404, detail={"available_flows": available})

    methods = find_methods(nodes, config)

    def node_payload(node_id: str):
        source_file, source_location = meta_for(node_id, links)
        return {
            "name": pretty_name(node_id),
            "node": node_id,
            "source_file": source_file,
            "source_location": source_location,
        }

    def method_payload(node_id: str):
        source_file, source_location = meta_for(node_id, links)
        return {
            "name": pretty_method(node_id),
            "node": node_id,
            "source_file": source_file,
            "source_location": source_location,
        }

    return {
        "topic": topic,
        "title": config["title"],
        "high_level_flow": "Screen → ViewModel → Repository → Data/Persistence"
        if config["viewmodels"]
        else "Screen → Repository/Auth service → Data/Persistence",
        "screens": [node_payload(x) for x in config["screens"]],
        "viewmodels": [node_payload(x) for x in config["viewmodels"]],
        "repositories": [node_payload(x) for x in config["repositories"]],
        "important_methods": [method_payload(x) for x in methods[:30]],
    }


@app.post("/repo/ask")
def ask(request: AskRequest, workspace: str = Depends(authorized_workspace)):
    enforce_strict_branch_freshness(workspace)
    q = request.question.lower()
    available_flows = discover_flows(graph_path(workspace))

    for item in available_flows:
        terms = {item["slug"].replace("-", " "), item["slug"], item["name"].lower()}
        if any(term and term in q for term in terms):
            return _flow(item["slug"], workspace)

    if "habit" in q:
        topic = "habit"
    elif "revision" in q or "spaced" in q:
        topic = "revision"
    elif "login" in q or "auth" in q or "sign" in q:
        topic = "login"
    elif "screen" in q:
        return repo_summary_dynamic(graph_path(workspace))
    else:
        return {
            "answer": "I can answer questions about the detected screens and flows in this repository.",
            "supported_topics": ["screens"] + [item["slug"] for item in available_flows],
        }

    try:
        return _flow(topic, workspace)
    except HTTPException as error:
        if error.status_code != 404:
            raise
        return {
            "answer": "That fixed demo flow was not found in this repository. Use one of the detected flows for this repo.",
            "supported_topics": ["screens"] + [item["slug"] for item in available_flows],
        }


@app.get("/repo/nodes/{node_id}")
def node_details(node_id: str, workspace: str = Depends(authorized_workspace)):
    nodes, links = load_graph(graph_path(workspace))

    matching_node = None
    for node in nodes:
        current_id = str(node.get("id") or node.get("label") or node.get("name") or "")
        if current_id == node_id:
            matching_node = node
            break

    if matching_node is None:
        raise HTTPException(status_code=404, detail=f"Node not found: {node_id}")

    source_file, source_location = meta_for(node_id, links)

    connected_links = []
    for link in links:
        if link.get("source") == node_id or link.get("target") == node_id:
            connected_links.append(format_link(link))

    return {
        "name": readable_name(node_id),
        "node": node_id,
        "source_file": source_file,
        "source_location": source_location,
        "connected_links": connected_links[:50],
    }


@app.get("/repo/search")
def search_repo(
    q: str = Query(..., min_length=1),
    limit: int = 30,
    workspace: str = Depends(authorized_workspace),
):
    nodes, links = load_graph(graph_path(workspace))
    config = load_retrieval_config(workspace)
    stopwords = set(config.stopwords)

    # Tokenize so natural-language phrases ("explain habit flow") match, not just
    # exact node substrings. Fall back to the raw query if nothing survives.
    raw_tokens = re.findall(r"[A-Za-z][A-Za-z0-9]+", q.lower())
    terms = [t for t in raw_tokens if t not in stopwords and len(t) > 2]
    if not terms:
        terms = [t for t in raw_tokens if len(t) > 2] or [q]

    results = rank_nodes_for_query(terms, nodes, links, 80, boosts=config.keyword_boosts)

    components = []
    methods = []
    others = []

    for item in results:
        name = item.get("name", "")
        lower_name = name.lower()
        node = item.get("node", "").lower()

        is_component = any(x in lower_name for x in [
            "screen", "viewmodel", "repository", "manager", "scheduler", "service", "dao", "entity", "config"
        ]) and "." not in name

        is_data_model = any(x in lower_name for x in [
            "document", "entity", "state", "history", "uistate"
        ])

        is_method = "." in name and not is_data_model

        # Skip low-value generated primitive/type nodes.
        if node.startswith("app_src_") and not is_component and not is_method:
            continue

        if is_component:
            components.append(item)
        elif is_method:
            methods.append(item)
        else:
            others.append(item)

    return {
        "query": q,
        "components": components[:12],
        "methods": methods[:12],
        "others": others[:8],
        "results": (components[:12] + methods[:12] + others[:8])[:limit],
    }


@app.get("/repo/context")
def repo_context_endpoint(
    question: str = Query(..., min_length=1),
    limit: int = 12,
    workspace: str = Depends(authorized_workspace),
):
    return build_context(question, limit, workspace)


QUERY_EXACT_SYMBOL = "EXACT_SYMBOL"
QUERY_DEFINITION = "DEFINITION"
QUERY_REFERENCES = "REFERENCES"
QUERY_CALLERS = "CALLERS"
QUERY_CALLEES = "CALLEES"
QUERY_FLOW = "FLOW"
QUERY_CONCEPT = "CONCEPT"
QUERY_DEBUG = "DEBUG"

FAST_QUERY_TYPES = {
    QUERY_EXACT_SYMBOL,
    QUERY_DEFINITION,
    QUERY_REFERENCES,
    QUERY_CALLERS,
    QUERY_CALLEES,
}

QUERY_ROUTER_STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "by", "can", "code", "does",
    "do", "file", "files", "for", "from", "get", "give", "how", "i",
    "in", "is", "it", "me", "of", "on", "or", "please", "show", "tell",
    "the", "this", "to", "use", "used", "uses", "what", "when", "where",
    "which", "who", "why", "work", "works",
}


def query_symbol_candidates(question: str, limit: int = 6) -> list[str]:
    candidates = []
    for value in re.findall(r"`([^`]+)`", question or ""):
        value = value.strip()
        if value:
            candidates.append(value)
    for value in re.findall(
        r"\b[A-Za-z_][A-Za-z0-9_]*(?:\.[A-Za-z_][A-Za-z0-9_]*)?\b",
        question or "",
    ):
        lower = value.lower()
        if lower in QUERY_ROUTER_STOPWORDS or len(value) < 3:
            continue
        if value not in candidates:
            candidates.append(value)
    return candidates[:limit]


def route_query_type(question: str) -> str:
    text = f" {str(question or '').strip().lower()} "
    symbols = query_symbol_candidates(question)
    has_symbol_shape = any(
        "." in item
        or "_" in item
        or any(ch.isupper() for ch in item[1:])
        for item in symbols
    )
    word_count = len(re.findall(r"[A-Za-z][A-Za-z0-9_]*", question or ""))

    if re.search(r"\b(debug|bug|crash|error|exception|failing|failure|fix|issue|wrong)\b", text):
        return QUERY_DEBUG
    if re.search(r"\b(flow|workflow|journey|sequence|end[\s-]?to[\s-]?end|process|steps?)\b", text):
        return QUERY_FLOW
    if re.search(r"\b(who calls|callers? of|called by|where .* called)\b", text):
        return QUERY_CALLERS
    if re.search(r"\b(callees? of|what .* calls?|calls from|dependencies of)\b", text):
        return QUERY_CALLEES
    if re.search(r"\b(references?|usages?|where .* used|used by|using)\b", text):
        return QUERY_REFERENCES
    if re.search(r"\b(definition|defined|define|implementation of|what is|where is)\b", text):
        return QUERY_DEFINITION
    if symbols and (has_symbol_shape or word_count <= 2):
        return QUERY_EXACT_SYMBOL
    return QUERY_CONCEPT


def query_type_budget(query_type: str, requested_limit: int, config) -> dict:
    requested_limit = max(1, int(requested_limit or 12))
    default_limit = max(1, int(getattr(config, "node_limit", requested_limit) or requested_limit))
    budgets = {
        QUERY_EXACT_SYMBOL: {"nodes": 4, "relations": 10, "source": 0, "excerpts": 3},
        QUERY_DEFINITION: {"nodes": 6, "relations": 12, "source": 3, "excerpts": 4},
        QUERY_REFERENCES: {"nodes": 8, "relations": 18, "source": 4, "excerpts": 4},
        QUERY_CALLERS: {"nodes": 8, "relations": 18, "source": 3, "excerpts": 4},
        QUERY_CALLEES: {"nodes": 8, "relations": 18, "source": 3, "excerpts": 4},
        QUERY_FLOW: {"nodes": max(default_limit, requested_limit), "relations": 32, "source": 10, "excerpts": 6},
        QUERY_DEBUG: {"nodes": max(default_limit, requested_limit), "relations": 32, "source": 10, "excerpts": 6},
        QUERY_CONCEPT: {"nodes": max(default_limit, requested_limit), "relations": 24, "source": 8, "excerpts": 6},
    }
    budget = budgets.get(query_type, budgets[QUERY_CONCEPT])
    return {
        "nodes": max(1, budget["nodes"]),
        "relations": max(1, budget["relations"]),
        "source": max(0, budget["source"]),
        "excerpts": max(1, min(budget["excerpts"], int(getattr(config, "excerpt_nodes", budget["excerpts"]) or budget["excerpts"]))),
    }


def fast_context_sufficient(query_type: str, context_nodes: list[dict], context_relations: list[dict]) -> bool:
    nodes_with_source = [
        node for node in context_nodes
        if node.get("source_file")
        and node.get("source_file") != "unknown"
        and node.get("source_location")
        and node.get("source_location") != "?"
    ]
    if query_type in {QUERY_EXACT_SYMBOL, QUERY_DEFINITION}:
        return bool(nodes_with_source)
    if query_type == QUERY_CALLERS:
        return bool(nodes_with_source and context_relations)
    if query_type == QUERY_CALLEES:
        return bool(nodes_with_source and context_relations)
    if query_type == QUERY_REFERENCES:
        return bool(nodes_with_source and (context_relations or len(context_nodes) > 1))
    return False


def retrieval_config_fingerprint(config) -> str:
    try:
        payload = json.dumps(
            config.to_dict(),
            sort_keys=True,
            separators=(",", ":"),
        )
    except Exception:
        payload = repr(config)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:24]


def normalized_retrieval_question(question: str) -> str:
    return " ".join(str(question or "").strip().lower().split())


def retrieval_context_cache_key(
    *,
    workspace: str,
    question: str,
    limit: int,
    query_type: str,
    config,
) -> tuple:
    return (
        workspace,
        repository_revision(workspace),
        retrieval_config_fingerprint(config),
        query_type,
        int(limit or 0),
        normalized_retrieval_question(question),
    )


def get_cached_retrieval_context(cache_key: tuple) -> "dict | None":
    if RETRIEVAL_CONTEXT_CACHE_MAX_RECORDS <= 0:
        return None
    with _retrieval_context_cache_lock:
        cached = _retrieval_context_cache.get(cache_key)
        if cached is None:
            return None
        _retrieval_context_cache.move_to_end(cache_key)
        return copy.deepcopy(cached)


def put_cached_retrieval_context(cache_key: tuple, context: dict) -> None:
    if RETRIEVAL_CONTEXT_CACHE_MAX_RECORDS <= 0:
        return
    with _retrieval_context_cache_lock:
        _retrieval_context_cache[cache_key] = copy.deepcopy(context)
        _retrieval_context_cache.move_to_end(cache_key)
        while len(_retrieval_context_cache) > RETRIEVAL_CONTEXT_CACHE_MAX_RECORDS:
            _retrieval_context_cache.popitem(last=False)


def emit_cached_retrieval_activity(activity_callback, context: dict) -> None:
    if not activity_callback:
        return
    for status in (
        "searching_source",
        "matching_source_files",
        "ranking_graph_nodes",
        "expanding_relations",
        "reading_source",
    ):
        try:
            activity_callback(status, context)
        except Exception:
            return


def build_context(
    question: str,
    limit: int = 12,
    workspace: str = DEFAULT_WORKSPACE,
    activity_callback=None,
):
    import re

    def emit_activity(status: str, partial_context: dict) -> None:
        if not activity_callback:
            return
        try:
            activity_callback(status, partial_context)
        except Exception:
            pass

    def compact_text(value: str) -> str:
        return value.lower().replace("_", "").replace("-", "").replace(".", "")

    # Everything below is driven by the workspace's RetrievalConfig — no repo is
    # special-cased in code. The default workspace is seeded with the demo
    # anchors (config_schema.DEFAULT_DESTINY_CONFIG); other repos start from
    # RetrievalConfig() defaults and are tuned from the admin console.
    config = load_retrieval_config(workspace)
    stopwords = set(config.stopwords)

    raw_tokens = re.findall(r"[A-Za-z][A-Za-z0-9]+", question.lower())
    keywords = [t for t in raw_tokens if t not in stopwords and len(t) > 2]
    expanded = set(keywords)

    # Expand query terms with the workspace's synonym map (domain vocabulary).
    for term in list(expanded):
        for syn in config.synonyms.get(term, []):
            expanded.add(syn)

    raw_token_set = set(raw_tokens)
    if ({"log", "logs"} & raw_token_set and "in" in raw_token_set) or {"signin", "login"} & raw_token_set:
        expanded.discard("log")
        expanded.discard("logs")
        expanded.update({"login", "signin", "auth", "authentication"})

    if "sign" in raw_token_set and "in" in raw_token_set:
        expanded.update({"login", "signin", "auth", "authentication"})

    if "sign" in raw_token_set and "up" in raw_token_set:
        expanded.update({"register", "registration", "auth", "authentication"})

    query_terms = list(expanded)
    expanded_compact = {compact_text(t) for t in expanded}
    source_terms = [
        t for t in query_terms
        if len(t) > 2 and t not in SOURCE_QUERY_STOPWORDS
    ] or [
        t for t in raw_tokens
        if len(t) > 2 and t not in SOURCE_QUERY_STOPWORDS
    ]
    query_type = route_query_type(question)
    budget = query_type_budget(query_type, limit, config)
    cache_key = retrieval_context_cache_key(
        workspace=workspace,
        question=question,
        limit=limit,
        query_type=query_type,
        config=config,
    )
    cached_context = get_cached_retrieval_context(cache_key)
    if cached_context is not None:
        emit_cached_retrieval_activity(activity_callback, cached_context)
        return cached_context

    nodes, links = load_graph(graph_path(workspace))
    source_root = _safe_source_root(workspace)
    node_meta_cache = {}
    link_meta_by_node = {}
    for link in links:
        meta = (
            link.get("source_file", "unknown"),
            link.get("source_location", "?"),
        )
        source = link.get("source")
        target = link.get("target")
        if source:
            link_meta_by_node.setdefault(source, meta)
        if target:
            link_meta_by_node.setdefault(target, meta)

    def node_id_of(node):
        return str(node.get("id") or node.get("label") or node.get("name") or "")

    def meta_for_node(node_id: str):
        if node_id not in node_meta_cache:
            node_meta_cache[node_id] = link_meta_by_node.get(node_id, ("unknown", "?"))
        return node_meta_cache[node_id]

    def payload_for_node(node_id: str, score: int = 0):
        source_file, source_location = meta_for_node(node_id)
        return {
            "name": readable_name(node_id),
            "node": node_id,
            "source_file": source_file,
            "source_location": source_location,
            "score": score,
        }

    def canonical_priority(node_id: str, name: str):
        parts = [compact_text(p) for p in node_id.split("_")]

        # Best: class-level duplicate pattern.
        # Example: ui_habitsscreen_habitsscreen
        # Example: data_habitrepository_habitrepository
        if len(parts) >= 2 and parts[-1] == parts[-2]:
            return 0

        # Good: method-level nodes.
        if "." in name:
            return 1

        # Medium: local model/entity nodes.
        if node_id.startswith("local_"):
            return 2

        # Lower: short alias/reference nodes like habitrepository, habitsviewmodel.
        if "_" not in node_id:
            return 4

        # Lowest: generated type/reference nodes.
        if node_id.startswith("app_src_"):
            return 9

        return 3

    def find_best_node_by_name(target_name: str):
        candidates = []

        for node in nodes:
            node_id = node_id_of(node)
            name = readable_name(node_id)

            if name == target_name:
                candidates.append(payload_for_node(node_id, 1000))

        if not candidates:
            return None

        candidates.sort(key=lambda item: (
            canonical_priority(item["node"], item["name"]),
            item["node"]
        ))

        return candidates[0]

    def attach_source_excerpts(context_nodes: list[dict], excerpt_limit: int) -> None:
        for node in context_nodes[:excerpt_limit]:
            excerpt = read_source_excerpt(
                node.get("source_file", ""), node.get("source_location", ""),
                source_root, max_lines=config.excerpt_max_lines,
                max_chars=config.excerpt_max_chars,
            )
            if excerpt:
                node["source_excerpt"] = excerpt["code"]
                node["excerpt_range"] = f"L{excerpt['start_line']}-L{excerpt['end_line']}"

    def source_hits_from_nodes(context_nodes: list[dict]) -> list[dict]:
        hits_by_path = {}
        for node in context_nodes:
            source_file = node.get("source_file")
            if not source_file or source_file == "unknown":
                continue
            hit = hits_by_path.setdefault(source_file, {
                "path": source_file,
                "score": float(node.get("score") or 0),
                "snippets": [],
            })
            hit["score"] = max(hit["score"], float(node.get("score") or 0))
            if node.get("source_excerpt"):
                match = re.search(r"L(\d+)(?:\s*[-–]\s*L?(\d+))?", node.get("excerpt_range") or "")
                start_line = int(match.group(1)) if match else 1
                end_line = int(match.group(2)) if match and match.group(2) else start_line
                hit["snippets"].append({
                    "start_line": start_line,
                    "end_line": end_line,
                    "code": node["source_excerpt"],
                    "score": round(float(node.get("score") or 1), 2),
                })
        hits = list(hits_by_path.values())
        hits.sort(key=lambda item: (-item["score"], item["path"]))
        return hits[:max(1, LLM_PREVIEW_SOURCE_HITS)]

    def assemble_context(context_nodes: list[dict], context_relations: list[dict], source_hits: list[dict]) -> dict:
        preview_nodes = context_nodes[:LLM_PREVIEW_NODE_LIMIT]
        preview_source_hits = source_hits[:LLM_PREVIEW_SOURCE_HITS]
        return {
            "question": question,
            "query_terms": query_terms,
            "context_nodes": context_nodes,
            "context_relations": context_relations,
            "source_hits": source_hits,
            "llm_context_preview": {
                "instruction": "Answer the user's codebase question using the repo overview, graph context, relations, and source search snippets. Cite source files and line numbers. Prefer evidence from source snippets over names. If the evidence is incomplete, say exactly what could not be verified.",
                "pre_search_instruction": config.pre_search_instruction,
                "question": question,
                "repo_overview": _repo_overview(nodes, links),
                "nodes": [
                    {
                        "name": n["name"],
                        "source": f'{n.get("source_file", "")} {n.get("source_location", "")}',
                        **({"code": n["source_excerpt"][:LLM_PREVIEW_SNIPPET_CHARS]} if n.get("source_excerpt") else {}),
                    }
                    for n in preview_nodes
                ],
                "relations": [
                    {
                        "from": r["source_name"],
                        "relation": r["relation_label"],
                        "to": r["target_name"],
                        "source": f'{r.get("source_file", "")} {r.get("source_location", "")}',
                    }
                    for r in context_relations[:24]
                ],
                "source_search_hits": [
                    {
                        "path": hit["path"],
                        "score": hit["score"],
                        "snippets": [
                            {
                                "range": f"L{snippet['start_line']}-L{snippet['end_line']}",
                                "code": snippet["code"][:LLM_PREVIEW_SNIPPET_CHARS],
                            }
                            for snippet in hit["snippets"][:1]
                        ],
                    }
                    for hit in preview_source_hits
                ],
            },
        }

    def add_context_node(context_nodes: list[dict], seen_names: set, seen_nodes: set, item: dict) -> None:
        if not item or item.get("name") in seen_names or item.get("node") in seen_nodes:
            return
        context_nodes.append(item)
        seen_names.add(item.get("name"))
        seen_nodes.add(item.get("node"))

    def relation_allowed_for_fast_path(formatted: dict, selected_node_ids: set, selected_source_files: set) -> bool:
        source_file = formatted.get("source_file") or ""
        if "/src/test/" in source_file or "/src/androidTest/" in source_file:
            return False
        if formatted.get("context") in {"generic_arg", "return_type", "parameter_type"}:
            return False
        source_selected = formatted.get("source") in selected_node_ids
        target_selected = formatted.get("target") in selected_node_ids
        if query_type == QUERY_CALLERS:
            return target_selected and formatted.get("relation") in {"calls", "references"}
        if query_type == QUERY_CALLEES:
            return source_selected and formatted.get("relation") in {"calls", "references"}
        return source_selected or target_selected or source_file in selected_source_files

    if query_type in FAST_QUERY_TYPES:
        emit_activity("searching_source", {
            "question": question,
            "query_terms": query_terms,
        })
        context_nodes = []
        seen_names = set()
        seen_nodes = set()
        for candidate in query_symbol_candidates(question) + query_terms:
            for item in search_nodes(candidate, nodes, links, limit=budget["nodes"] * 2):
                add_context_node(context_nodes, seen_names, seen_nodes, item)
                if len(context_nodes) >= budget["nodes"]:
                    break
            if len(context_nodes) >= budget["nodes"]:
                break

        emit_activity("ranking_graph_nodes", {
            "question": question,
            "query_terms": query_terms,
            "context_nodes": context_nodes,
            "source_hits": [],
        })
        selected_node_ids = {item["node"] for item in context_nodes}
        selected_source_files = {
            item.get("source_file") for item in context_nodes
            if item.get("source_file") and item.get("source_file") != "unknown"
        }
        context_relations = []
        seen_relations = set()
        for link in links:
            formatted = format_link(link)
            if not relation_allowed_for_fast_path(formatted, selected_node_ids, selected_source_files):
                continue
            relation_key = (
                formatted.get("source"),
                formatted.get("target"),
                formatted.get("relation"),
                formatted.get("source_location"),
            )
            if relation_key in seen_relations:
                continue
            context_relations.append(formatted)
            seen_relations.add(relation_key)
            if len(context_relations) >= budget["relations"]:
                break

        emit_activity("expanding_relations", {
            "question": question,
            "query_terms": query_terms,
            "context_nodes": context_nodes,
            "context_relations": context_relations,
            "source_hits": [],
        })
        if fast_context_sufficient(query_type, context_nodes, context_relations):
            attach_source_excerpts(context_nodes, budget["excerpts"])
            source_hits = source_hits_from_nodes(context_nodes)
            emit_activity("reading_source", {
                "question": question,
                "query_terms": query_terms,
                "context_nodes": context_nodes,
                "context_relations": context_relations,
                "source_hits": source_hits,
            })
            context = assemble_context(context_nodes, context_relations, source_hits)
            put_cached_retrieval_context(cache_key, context)
            return context

    emit_activity("searching_source", {
        "question": question,
        "query_terms": query_terms,
    })
    workspace_source_index = source_index_path(workspace)
    source_hits = _search_source_files(
        source_root,
        source_terms,
        limit=10,
        index_path=workspace_source_index,
    )
    seen_follow_terms = set()
    for _ in range(2):
        follow_terms = [
            term for term in _identifier_terms_from_hits(source_hits, source_terms)
            if term not in seen_follow_terms
        ]
        if not follow_terms:
            break
        seen_follow_terms.update(follow_terms)
        follow_hits = _search_source_files(
            source_root,
            follow_terms,
            limit=14,
            index_path=workspace_source_index,
        )
        source_hits = _merge_source_hits(source_hits, follow_hits, limit=14)
    matched_source_files = {hit["path"] for hit in source_hits}
    emit_activity("matching_source_files", {
        "question": question,
        "query_terms": query_terms,
        "source_hits": source_hits,
    })

    def anchor_matches_query(name: str) -> bool:
        """Seed a preferred anchor only when it's relevant to the question, so
        habit anchors fire on habit questions, login anchors on login ones, and
        none on unrelated questions — driven purely by the query, not by code."""
        nc = compact_text(name)
        return any(term and term in nc for term in expanded_compact)

    preferred_components = [c for c in config.preferred_components if anchor_matches_query(c)]
    preferred_methods = [m for m in config.preferred_methods if anchor_matches_query(m)]

    context_nodes = []
    seen_names = set()
    seen_nodes = set()

    # 1. Deterministically add preferred components.
    for name in preferred_components:
        item = find_best_node_by_name(name)
        if item and item["name"] not in seen_names:
            context_nodes.append(item)
            seen_names.add(item["name"])
            seen_nodes.add(item["node"])

    # 2. Deterministically add preferred methods.
    for name in preferred_methods:
        item = find_best_node_by_name(name)
        if item and item["name"] not in seen_names:
            context_nodes.append(item)
            seen_names.add(item["name"])
            seen_nodes.add(item["node"])

    # 3. Promote graph nodes that live in source files directly matched by the
    #    question. This mirrors a code-agent workflow: find files first, then
    #    pull in their symbols and relations.
    if matched_source_files:
        for node in nodes:
            if len(context_nodes) >= config.node_limit:
                break

            node_id = node_id_of(node)
            if is_noise_node(node_id):
                continue

            source_file, _ = meta_for_node(node_id)

            if source_file not in matched_source_files:
                continue

            item = payload_for_node(node_id, 700)
            if item["name"] in seen_names or item["node"] in seen_nodes:
                continue

            context_nodes.append(item)
            seen_names.add(item["name"])
            seen_nodes.add(item["node"])

    # 4. Fill remaining slots from a rarity-weighted, multi-term relevance rank,
    #    amplified by the workspace's keyword boosts. Specific words outrank
    #    generic ones instead of being crowded out by common-substring noise.
    node_limit = max(config.node_limit, limit)
    ranked = rank_nodes_for_query(
        query_terms, nodes, links, limit=node_limit * 4, boosts=config.keyword_boosts
    )

    for item in ranked:
        if len(context_nodes) >= node_limit:
            break

        source_file = item.get("source_file") or ""
        if "/src/test/" in source_file or "/src/androidTest/" in source_file:
            continue

        if item["name"] in seen_names or item["node"] in seen_nodes:
            continue

        context_nodes.append(item)
        seen_names.add(item["name"])
        seen_nodes.add(item["node"])

    emit_activity("ranking_graph_nodes", {
        "question": question,
        "query_terms": query_terms,
        "context_nodes": context_nodes,
        "source_hits": source_hits,
    })

    selected_names = {item["name"] for item in context_nodes}
    selected_node_ids = {item["node"] for item in context_nodes}

    def is_useful_context_relation(formatted):
        """Generic, repo-agnostic relation filter: drop test files and primitive
        type-reference noise, then keep any relation that touches a node already
        selected into the context (by id or display name)."""
        context = formatted.get("context")
        source_file = formatted.get("source_file") or ""

        if "/src/test/" in source_file or "/src/androidTest/" in source_file:
            return False

        # Drop primitive/type generic noise (parameter/return/generic type refs).
        if context in {"generic_arg", "return_type", "parameter_type"}:
            return False

        return (
            formatted.get("source") in selected_node_ids
            or formatted.get("target") in selected_node_ids
            or formatted.get("source_name") in selected_names
            or formatted.get("target_name") in selected_names
            or formatted.get("source_file") in matched_source_files
        )

    context_relations = []
    seen_relations = set()
    relation_limit = max(config.relation_limit, 48 if source_hits else config.relation_limit)

    for link in links:
        formatted = format_link(link)

        if not is_useful_context_relation(formatted):
            continue

        relation_key = (
            formatted.get("source"),
            formatted.get("target"),
            formatted.get("relation"),
            formatted.get("source_location"),
        )

        if relation_key in seen_relations:
            continue

        context_relations.append(formatted)
        seen_relations.add(relation_key)

        if len(context_relations) >= relation_limit:
            break
    emit_activity("expanding_relations", {
        "question": question,
        "query_terms": query_terms,
        "context_nodes": context_nodes,
        "context_relations": context_relations,
        "source_hits": source_hits,
    })

    # Attach real source code for the most relevant nodes so the LLM can explain
    # actual behavior (e.g. what a feature enforces), not just node names. Kept
    # small (top few nodes, short excerpts) so the prompt stays fast.
    attach_source_excerpts(context_nodes, config.excerpt_nodes)
    emit_activity("reading_source", {
        "question": question,
        "query_terms": query_terms,
        "context_nodes": context_nodes,
        "context_relations": context_relations,
        "source_hits": source_hits,
    })

    context = assemble_context(context_nodes, context_relations, source_hits)
    put_cached_retrieval_context(cache_key, context)
    return context


FOLLOW_UP_EVIDENCE_CHARS = max(
    4000, int(os.environ.get("CODEATLAS_FOLLOW_UP_EVIDENCE_CHARS", "10000"))
)
FOLLOW_UP_TURN_ANSWER_CHARS = max(
    1000, int(os.environ.get("CODEATLAS_FOLLOW_UP_TURN_ANSWER_CHARS", "2500"))
)
FOLLOW_UP_STOPWORDS = {
    "about", "after", "again", "also", "and", "are", "can", "could", "does",
    "explain", "feature", "flow", "for", "from", "handle", "handled", "happen",
    "happens", "have", "how", "into", "more", "process", "should", "than",
    "that", "the", "their", "then", "there", "these", "they", "this", "those",
    "was", "what", "when", "where", "which", "with", "work", "working", "works",
    "would", "your",
}
FOLLOW_UP_REFERENCE_PATTERN = re.compile(
    r"\b(it|its|that|this|those|these|they|them|same|previous|above|earlier|"
    r"continue|next)\b|\b(what|how) about\b|^\s*(and|also|then)\b",
    re.IGNORECASE,
)


def repository_version_payload(workspace: str) -> "dict | None":
    branch = db.get_repo_branch_by_workspace(workspace)
    if not branch:
        return None
    return {
        "repository": branch["repo_name"],
        "repository_slug": branch["repo_slug"],
        "branch_id": branch["id"],
        "branch": branch["name"],
        "commit_sha": branch.get("indexed_commit_sha"),
        "remote_commit_sha": branch.get("remote_commit_sha"),
        "indexed_at": branch.get("indexed_at"),
        "freshness_status": branch["freshness_status"],
        "behind_count": branch.get("behind_count", 0),
        "last_checked_at": branch.get("last_checked_at"),
    }


def repository_revision(workspace: str) -> str:
    """Stable cache identity that changes whenever indexed evidence changes."""
    version = repository_version_payload(workspace)
    parts = []
    if version:
        parts.append("|".join(
            str(version.get(key) or "")
            for key in ("branch_id", "commit_sha", "indexed_at")
        ))
    for label, path in (
        ("graph", graph_path(workspace)),
        ("retrieval", retrieval_config_path(workspace)),
        ("source_index", source_index_path(workspace)),
    ):
        try:
            stat = path.stat()
            parts.append(f"{label}:{stat.st_mtime_ns}:{stat.st_size}")
        except OSError:
            parts.append(f"{label}:missing")
    return "|".join(parts)


def is_related_follow_up(state: ConversationState, question: str) -> bool:
    """Conservative, zero-latency topic check before reusing cached evidence."""
    text = str(question or "").strip().lower()
    if not text or not state.turns:
        return False
    if FOLLOW_UP_REFERENCE_PATTERN.search(text):
        return True

    question_terms = {
        term
        for term in re.findall(r"[a-z0-9_.$/-]{3,}", text)
        if term not in FOLLOW_UP_STOPWORDS
    }
    if not question_terms:
        return False
    previous = state.turns[-1]
    prior_text = (
        f"{previous.get('question', '')} {previous.get('answer', '')}"
    ).lower()
    prior_terms = set(re.findall(r"[a-z0-9_.$/-]{3,}", prior_text))
    return bool(question_terms & prior_terms)


def compact_follow_up_evidence(state: ConversationState) -> str:
    turns = "\n\n".join(
        (
            f"Question: {str(turn.get('question', ''))[:800]}\n"
            f"Answer: {str(turn.get('answer', ''))[:FOLLOW_UP_TURN_ANSWER_CHARS]}"
        )
        for turn in state.turns[-2:]
    )
    preview = (state.context or {}).get("llm_context_preview", {})
    def compact_single_evidence(item: dict) -> dict:
        return {
            "repo_overview": item.get("repo_overview"),
            "nodes": [
                {
                    "name": node.get("name"),
                    "source": node.get("source"),
                    **(
                        {"code": str(node.get("code", ""))[:700]}
                        if node.get("code")
                        else {}
                    ),
                }
                for node in (item.get("nodes") or [])[:6]
            ],
            "relations": (item.get("relations") or [])[:12],
            "source_search_hits": [
                {
                    "path": hit.get("path"),
                    "snippets": [
                        {
                            "range": snippet.get("range"),
                            "code": str(snippet.get("code", ""))[:800],
                        }
                        for snippet in (hit.get("snippets") or [])[:1]
                    ],
                }
                for hit in (item.get("source_search_hits") or [])[:6]
            ],
        }

    if preview.get("branches"):
        compact_preview = {
            "comparison": "two indexed branches of one repository",
            "branches": [
                {
                    "label": branch.get("label"),
                    "name": branch.get("name"),
                    "branch": branch.get("branch"),
                    "branch_id": branch.get("branch_id"),
                    "repo_name": branch.get("repo_name"),
                    "repository_version": branch.get("repository_version"),
                    "evidence": compact_single_evidence(branch.get("evidence") or {}),
                }
                for branch in (preview.get("branches") or [])[:2]
            ],
        }
    else:
        compact_preview = compact_single_evidence(preview)
    evidence = json.dumps(
        compact_preview,
        ensure_ascii=False,
        separators=(",", ":"),
    )
    return (
        f"Recent conversation:\n{turns}\n\n"
        "Verified repository evidence from the same indexed commit:\n"
        f"{evidence[:FOLLOW_UP_EVIDENCE_CHARS]}"
    )


def _answer_response(
    question: str,
    result: dict,
    context: dict,
    workspace: str,
) -> dict:
    return {
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
        "repository_version": repository_version_payload(workspace),
    }


def _zero_token_usage_payload() -> dict:
    payload = token_usage_payload({})
    payload["available"] = True
    return payload


def _session_cached_answer_response(
    cached_response: dict,
    question: str,
    workspace: str = None,
) -> dict:
    response = copy.deepcopy(cached_response or {})
    response["question"] = question
    response["retrieval_mode"] = "session_cache"
    response["session_cache_hit"] = True
    response["follow_up_reused"] = False
    response["follow_up_fallback"] = False
    response["deep_investigation"] = False
    response["investigate_deeply_available"] = True
    if workspace:
        response["repository_version"] = repository_version_payload(workspace)
    response["timings_ms"] = {
        "retrieval": 0.0,
        "generation": 0.0,
        "total": 0.0,
    }
    response["token_usage"] = _zero_token_usage_payload()
    return response


def answer_from_cached_audience_evidence(
    question: str,
    cached_response: dict,
    workspace: str = DEFAULT_WORKSPACE,
    user_llm: dict = None,
    allow_shared_fallback: bool = True,
    llm_mode: str = None,
    user_type: str = "product_team",
    activity_request_id: str = None,
    activity_user_id: int = None,
) -> dict:
    """Render an audience-specific answer from a prior grounded answer/context.

    This is intentionally used for Product-from-Dev reuse only by the request
    orchestrator. If the compact renderer decides the cached evidence is not
    enough, it raises FollowUpNeedsEvidence and the caller runs the normal full
    investigation path.
    """
    started_at = time.perf_counter()
    context = copy.deepcopy((cached_response or {}).get("context") or {})
    preview = context.setdefault("llm_context_preview", {})
    response_style_instruction = (
        PRODUCT_TEAM_RESPONSE_INSTRUCTION
        if user_type == "product_team"
        else ""
    )
    llm_question = (
        f"{question.rstrip()}\n\n{PRODUCT_TEAM_QUERY_SUFFIX}"
        if user_type == "product_team"
        else question
    )
    context["question"] = question
    context["response_style_instruction"] = response_style_instruction
    preview["question"] = llm_question
    if activity_request_id and activity_user_id is not None:
        update_answer_activity(
            activity_request_id,
            user_id=activity_user_id,
            workspace=workspace,
            question=question,
            status="using_conversation_evidence",
            context=context,
        )

    state = ConversationState(
        conversation_id="audience-cache",
        user_id=0,
        session_key="",
        workspace=workspace,
        llm_mode=llm_mode or "auto",
        user_type=str((cached_response or {}).get("answer_user_type") or "dev_team"),
        repository_revision="",
        context=context,
        turns=[
            {
                "question": (cached_response or {}).get("question") or question,
                "answer": (cached_response or {}).get("answer") or "",
            }
        ],
    )
    evidence = compact_follow_up_evidence(state)
    if activity_request_id and activity_user_id is not None:
        update_answer_activity(
            activity_request_id,
            user_id=activity_user_id,
            workspace=workspace,
            question=question,
            status="generating_answer",
            context=context,
        )

    generation_started_at = time.perf_counter()
    result = generate_fast_follow_up(
        context,
        evidence,
        user_llm=user_llm,
        allow_shared_fallback=allow_shared_fallback,
        llm_mode=llm_mode,
        question=llm_question,
    )
    generation_ms = round((time.perf_counter() - generation_started_at) * 1000, 1)
    response = _answer_response(question, result, context, workspace)
    response["retrieval_mode"] = "audience_cache"
    response["audience_cache_hit"] = True
    response["follow_up_reused"] = False
    response["follow_up_fallback"] = False
    response["deep_investigation"] = False
    response["investigate_deeply_available"] = True
    response["timings_ms"] = {
        "retrieval": 0.0,
        "generation": generation_ms,
        "total": round((time.perf_counter() - started_at) * 1000, 1),
    }
    return response


def _remember_session_answer(
    *,
    user: dict,
    workspace: str,
    llm_mode: str,
    user_type: str,
    repository_revision: str,
    question: str,
    response: dict,
) -> None:
    if response.get("session_cache_hit"):
        return
    conversation_store.store_cached_answer(
        session_key=str(user.get("_session_key") or ""),
        user_id=user["id"],
        workspace=workspace,
        llm_mode=llm_mode,
        user_type=user_type,
        repository_revision=repository_revision,
        question=question,
        response=response,
    )


def _request_uses_shared_tier_only(llm_mode: str, user_llm: Optional[dict]) -> bool:
    """True only when this request is guaranteed to be answered by the shared
    ("Mimo") tier: either explicitly requested, or auto mode with no personal
    key on file to try first. Mirrors the tier order in llm.client.generate()."""
    if llm_mode == "mimo":
        return True
    return llm_mode == "auto" and not (user_llm and user_llm.get("api_key"))


def _remember_repo_answer(
    *,
    workspace: str,
    user_type: str,
    repository_revision: str,
    question: str,
    response: dict,
) -> None:
    """Cache a fresh shared-tier answer for reuse across any user asking the
    same question against the same indexed revision. Only the shared tier is
    eligible: a BYOK answer is that user's own paid-for compute, not something
    to hand to a different user without their key."""
    if response.get("session_cache_hit"):
        return
    if not str(response.get("provider_used") or "").startswith("shared:"):
        return
    conversation_store.store_repo_cached_answer(
        workspace=workspace,
        user_type=user_type,
        repository_revision=repository_revision,
        question=question,
        response=response,
    )


def _create_conversation_from_response(
    *,
    user: dict,
    workspace: str,
    llm_mode: str,
    user_type: str,
    repository_revision: str,
    question: str,
    response: dict,
):
    return conversation_store.create(
        user_id=user["id"],
        session_key=str(user.get("_session_key") or ""),
        workspace=workspace,
        llm_mode=llm_mode,
        user_type=user_type,
        repository_revision=repository_revision,
        context=response.get("context") or {},
        question=question,
        answer=response["answer"],
    )


def _resolve_compare_base_repo(workspace: str, user: dict) -> dict:
    value = str(workspace or "").strip()
    if not value:
        raise HTTPException(status_code=400, detail="Repository is required.")
    repo = db.get_repo_by_workspace(value)
    if not repo:
        raise HTTPException(status_code=404, detail="Repository not found.")
    if repo["status"] != "published":
        raise HTTPException(
            status_code=409,
            detail=f"Repository '{repo['name']}' is not published.",
        )
    if user["role"] != "admin" and not db.user_has_repo(user["id"], repo["workspace"]):
        raise HTTPException(
            status_code=403,
            detail="You do not have access to this repository.",
        )
    return repo


def _resolve_compare_branch(repo: dict, branch_id: int, label: str) -> dict:
    branch = db.get_repo_branch(branch_id)
    if not branch or branch["repo_id"] != repo["id"]:
        raise HTTPException(
            status_code=404,
            detail=f"{label} was not found for this repository.",
        )
    workspace = branch.get("workspace")
    if not workspace or branch["index_status"] not in {"ready", "indexing"}:
        raise HTTPException(
            status_code=409,
            detail=f"{label} has not been indexed successfully yet.",
        )
    if not graph_path(workspace).exists():
        raise HTTPException(
            status_code=409,
            detail=f"{label} does not have indexed graph data available.",
        )
    return {
        "repo": repo,
        "branch": branch,
        "workspace": workspace,
    }


def _comparison_branch_payload(label: str, resolved: dict, context: dict) -> dict:
    repo = resolved["repo"]
    branch = resolved.get("branch")
    workspace = resolved["workspace"]
    branch_name = branch.get("name") if branch else None
    return {
        "label": label,
        "name": branch_name or repo["name"],
        "repo_name": repo["name"],
        "slug": repo["slug"],
        "workspace": workspace,
        "branch_id": branch.get("id") if branch else None,
        "branch": branch_name,
        "repository_version": repository_version_payload(workspace),
        "context_nodes": context.get("context_nodes", []),
        "context_relations": context.get("context_relations", []),
        "source_hits": context.get("source_hits", []),
        "llm_context_preview": context.get("llm_context_preview", {}),
    }


def _comparison_workspace_key(repo: dict, left: dict, right: dict) -> str:
    repo_workspace = repo.get("workspace") or left.get("workspace") or repo.get("slug") or repo.get("id")
    return (
        f"compare:{repo_workspace}:"
        f"{left['branch']['id']}:{right['branch']['id']}"
    )


def _comparison_revision(left: dict, right: dict) -> str:
    return (
        f"left:{left['branch']['id']}:{repository_revision(left['workspace'])}"
        f"|right:{right['branch']['id']}:{repository_revision(right['workspace'])}"
    )


def _effective_answer_user_type(user: dict, requested: str = None) -> str:
    """Resolve the answer audience without changing the authenticated user type."""
    authenticated_type = user.get("user_type") or "dev_team"
    requested_type = (requested or authenticated_type or "dev_team").strip()
    if requested_type not in {"dev_team", "product_team"}:
        raise HTTPException(
            status_code=400,
            detail="answer_user_type must be 'dev_team' or 'product_team'.",
        )
    if authenticated_type == "product_team":
        return "product_team"
    return "product_team" if requested_type == "product_team" else "dev_team"


def build_compare_context(
    question: str,
    left: dict,
    right: dict,
    user_type: str,
    activity_callback=None,
) -> dict:
    def emit_activity(status: str, partial_context: dict) -> None:
        if not activity_callback:
            return
        try:
            activity_callback(status, partial_context)
        except Exception:
            pass

    started_at = time.perf_counter()
    left_context = build_context(question, limit=12, workspace=left["workspace"])
    left_payload = _comparison_branch_payload("Branch A", left, left_context)
    emit_activity("ranking_graph_nodes", {
        "question": question,
        "comparison_mode": True,
        "comparison_repositories": [left_payload],
    })
    right_context = build_context(question, limit=12, workspace=right["workspace"])
    right_payload = _comparison_branch_payload("Branch B", right, right_context)
    emit_activity("expanding_relations", {
        "question": question,
        "comparison_mode": True,
        "comparison_repositories": [left_payload, right_payload],
    })
    response_style_instruction = (
        PRODUCT_TEAM_RESPONSE_INSTRUCTION
        if user_type == "product_team"
        else ""
    )
    llm_question = (
        f"{question.rstrip()}\n\n{PRODUCT_TEAM_QUERY_SUFFIX}"
        if user_type == "product_team"
        else question
    )
    return {
        "question": question,
        "comparison_mode": True,
        "response_style_instruction": response_style_instruction,
        "comparison_repositories": [left_payload, right_payload],
        "retrieval_ms": round((time.perf_counter() - started_at) * 1000, 1),
        "llm_context_preview": {
            "instruction": (
                "Compare the two indexed branches using only their evidence. Keep "
                "Branch A and Branch B findings separate before summarizing similarities "
                "and differences."
            ),
            "question": llm_question,
            "branches": [
                {
                    "label": payload["label"],
                    "name": payload["name"],
                    "repo_name": payload["repo_name"],
                    "slug": payload["slug"],
                    "branch": payload["branch"],
                    "branch_id": payload["branch_id"],
                    "repository_version": payload["repository_version"],
                    "evidence": payload["llm_context_preview"],
                }
                for payload in (left_payload, right_payload)
            ],
        },
    }


def answer_compare(
    question: str,
    left: dict,
    right: dict,
    user_llm: dict = None,
    allow_shared_fallback: bool = True,
    llm_mode: str = None,
    user_type: str = "dev_team",
    activity_request_id: str = None,
    activity_user_id: int = None,
) -> dict:
    started_at = time.perf_counter()
    activity_workspace = None
    activity_callback = None
    if activity_request_id and activity_user_id is not None:
        activity_workspace = _comparison_workspace_key(left["repo"], left, right)
        update_answer_activity(
            activity_request_id,
            user_id=activity_user_id,
            workspace=activity_workspace,
            question=question,
            status="understanding_query",
        )

        def activity_callback(status: str, partial_context: dict) -> None:
            update_answer_activity(
                activity_request_id,
                user_id=activity_user_id,
                workspace=activity_workspace,
                question=question,
                status=status,
                context=partial_context,
            )

    context = build_compare_context(
        question,
        left,
        right,
        user_type,
        activity_callback=activity_callback,
    )
    if activity_request_id and activity_user_id is not None:
        update_answer_activity(
            activity_request_id,
            user_id=activity_user_id,
            workspace=activity_workspace,
            question=question,
            status="generating_answer",
            context=context,
        )
    toolbox = ComparisonRepositoryToolbox(
        context["comparison_repositories"][0],
        context["comparison_repositories"][1],
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
    )
    result_mode = result.get("retrieval_mode")
    response = {
        "question": question,
        "answer": result["answer"],
        "provider_used": result["provider_used"],
        "retrieval_mode": (
            "compare_agentic" if result_mode == "agentic" else "compare_one_shot"
        ),
        "agent_trace": result.get("agent_trace", []),
        "agent_rounds": result.get("rounds"),
        "agent_tool_calls": result.get("tool_calls", 0),
        "needs_clarification": bool(result.get("needs_clarification")),
        "context": context,
        "comparison_repositories": context["comparison_repositories"],
        "repository_versions": {
            "left": context["comparison_repositories"][0]["repository_version"],
            "right": context["comparison_repositories"][1]["repository_version"],
        },
    }
    response["timings_ms"] = {
        "retrieval": context["retrieval_ms"],
        "generation": round((time.perf_counter() - generation_started_at) * 1000, 1),
        "total": round((time.perf_counter() - started_at) * 1000, 1),
    }
    return response


def _comparison_repository_versions(context: dict) -> dict:
    repositories = context.get("comparison_repositories") or []
    return {
        "left": repositories[0].get("repository_version") if len(repositories) > 0 else None,
        "right": repositories[1].get("repository_version") if len(repositories) > 1 else None,
    }


def answer_compare_follow_up(
    question: str,
    state: ConversationState,
    left: dict,
    right: dict,
    user_llm: dict = None,
    allow_shared_fallback: bool = True,
    llm_mode: str = None,
    user_type: str = "dev_team",
    deep_investigation: bool = False,
    activity_request_id: str = None,
    activity_user_id: int = None,
) -> dict:
    """Answer a follow-up from branch-comparison evidence or rerun comparison."""
    started_at = time.perf_counter()
    activity_kwargs = (
        {
            "activity_request_id": activity_request_id,
            "activity_user_id": activity_user_id,
        }
        if activity_request_id and activity_user_id is not None
        else {}
    )
    if deep_investigation:
        response = answer_compare(
            question,
            left,
            right,
            user_llm=user_llm,
            allow_shared_fallback=allow_shared_fallback,
            llm_mode=llm_mode,
            user_type=user_type,
            **activity_kwargs,
        )
        response["follow_up_reused"] = False
        response["follow_up_fallback"] = True
        response["deep_investigation"] = True
        response["investigate_deeply_available"] = False
        response["timings_ms"]["total"] = round(
            (time.perf_counter() - started_at) * 1000,
            1,
        )
        return response

    context = copy.deepcopy(state.context or {})
    preview = context.setdefault("llm_context_preview", {})
    response_style_instruction = (
        PRODUCT_TEAM_RESPONSE_INSTRUCTION
        if user_type == "product_team"
        else ""
    )
    llm_question = (
        f"{question.rstrip()}\n\n{PRODUCT_TEAM_QUERY_SUFFIX}"
        if user_type == "product_team"
        else question
    )
    context["question"] = question
    context["response_style_instruction"] = response_style_instruction
    preview["question"] = llm_question
    if activity_request_id and activity_user_id is not None:
        update_answer_activity(
            activity_request_id,
            user_id=activity_user_id,
            workspace=_comparison_workspace_key(left["repo"], left, right),
            question=question,
            status="using_conversation_evidence",
            context=context,
        )
    evidence = compact_follow_up_evidence(state)
    if activity_request_id and activity_user_id is not None:
        update_answer_activity(
            activity_request_id,
            user_id=activity_user_id,
            workspace=_comparison_workspace_key(left["repo"], left, right),
            question=question,
            status="generating_answer",
            context=context,
        )

    fast_started_at = time.perf_counter()
    try:
        result = generate_fast_follow_up(
            context,
            evidence,
            user_llm=user_llm,
            allow_shared_fallback=allow_shared_fallback,
            llm_mode=llm_mode,
            question=llm_question,
        )
    except FollowUpNeedsEvidence:
        fast_gate_ms = round((time.perf_counter() - fast_started_at) * 1000, 1)
        response = answer_compare(
            question,
            left,
            right,
            user_llm=user_llm,
            allow_shared_fallback=allow_shared_fallback,
            llm_mode=llm_mode,
            user_type=user_type,
            **activity_kwargs,
        )
        response["follow_up_reused"] = False
        response["follow_up_fallback"] = True
        response["deep_investigation"] = False
        response["investigate_deeply_available"] = False
        response["timings_ms"]["follow_up_gate"] = fast_gate_ms
        response["timings_ms"]["total"] = round(
            (time.perf_counter() - started_at) * 1000,
            1,
        )
        return response

    generation_ms = round((time.perf_counter() - fast_started_at) * 1000, 1)
    return {
        "question": question,
        "answer": result["answer"],
        "provider_used": result["provider_used"],
        "retrieval_mode": "compare_follow_up_cache",
        "agent_trace": result.get("agent_trace", []),
        "agent_rounds": result.get("rounds"),
        "agent_tool_calls": result.get("tool_calls", 0),
        "context": context,
        "comparison_repositories": context.get("comparison_repositories", []),
        "repository_versions": _comparison_repository_versions(context),
        "follow_up_reused": True,
        "follow_up_fallback": False,
        "deep_investigation": False,
        "investigate_deeply_available": True,
        "timings_ms": {
            "follow_up_generation": generation_ms,
            "total": round((time.perf_counter() - started_at) * 1000, 1),
        },
    }



def answer_question(question: str, workspace: str = DEFAULT_WORKSPACE,
                    user_llm: dict = None, allow_shared_fallback: bool = True,
                    llm_mode: str = None, user_type: str = "dev_team",
                    answer_mode: str = None,
                    image_attachments: list[dict] = None,
                    activity_request_id: str = None,
                    activity_user_id: int = None) -> dict:
    """Build context for a workspace and run the LLM fallback chain. Shared by
    the user ask endpoint and the admin test panel."""
    started_at = time.perf_counter()
    retrieval_started_at = time.perf_counter()
    activity_callback = None
    if activity_request_id and activity_user_id is not None:
        update_answer_activity(
            activity_request_id,
            user_id=activity_user_id,
            workspace=workspace,
            question=question,
            status="understanding_query",
        )

        def activity_callback(status: str, partial_context: dict) -> None:
            update_answer_activity(
                activity_request_id,
                user_id=activity_user_id,
                workspace=workspace,
                question=question,
                status=status,
                context=partial_context,
            )

    context = build_context(
        question,
        limit=16,
        workspace=workspace,
        activity_callback=activity_callback,
    )
    if activity_request_id and activity_user_id is not None:
        update_answer_activity(
            activity_request_id,
            user_id=activity_user_id,
            workspace=workspace,
            question=question,
            status="generating_answer",
            context=context,
        )
    toolbox = RepositoryToolbox(workspace)
    retrieval_ms = round((time.perf_counter() - retrieval_started_at) * 1000, 1)
    response_style_instruction = (
        PRODUCT_TEAM_RESPONSE_INSTRUCTION
        if user_type == "product_team"
        else ""
    )
    llm_question = (
        f"{question.rstrip()}\n\n{PRODUCT_TEAM_QUERY_SUFFIX}"
        if user_type == "product_team"
        else question
    )
    product_flow_summary = (
        user_type == "product_team" and answer_mode == "flow_summary"
    )
    context["response_style_instruction"] = response_style_instruction
    if product_flow_summary:
        context["product_flow_summary"] = True
    context["llm_context_preview"]["question"] = llm_question
    toolbox.response_style_instruction = response_style_instruction
    toolbox.product_flow_summary = product_flow_summary
    generation_started_at = time.perf_counter()
    result = generate(
        context,
        user_llm=user_llm,
        allow_shared_fallback=allow_shared_fallback,
        llm_mode=llm_mode,
        question=llm_question,
        toolbox=toolbox,
        image_attachments=image_attachments,
    )
    generation_ms = round((time.perf_counter() - generation_started_at) * 1000, 1)
    response = _answer_response(question, result, context, workspace)
    response["timings_ms"] = {
        "retrieval": retrieval_ms,
        "generation": generation_ms,
        "total": round((time.perf_counter() - started_at) * 1000, 1),
    }
    return response


def answer_follow_up(
    question: str,
    state: ConversationState,
    workspace: str = DEFAULT_WORKSPACE,
    user_llm: dict = None,
    allow_shared_fallback: bool = True,
    llm_mode: str = None,
    user_type: str = "dev_team",
    deep_investigation: bool = False,
    activity_request_id: str = None,
    activity_user_id: int = None,
) -> dict:
    """Answer from revision-matched evidence, using repository tools only as needed."""
    started_at = time.perf_counter()
    activity_kwargs = (
        {
            "activity_request_id": activity_request_id,
            "activity_user_id": activity_user_id,
        }
        if activity_request_id and activity_user_id is not None
        else {}
    )
    if deep_investigation:
        response = answer_question(
            question,
            workspace=workspace,
            user_llm=user_llm,
            allow_shared_fallback=allow_shared_fallback,
            llm_mode=llm_mode,
            user_type=user_type,
            **activity_kwargs,
        )
        response["follow_up_reused"] = False
        response["follow_up_fallback"] = True
        response["deep_investigation"] = True
        response["investigate_deeply_available"] = False
        response["timings_ms"]["total"] = round(
            (time.perf_counter() - started_at) * 1000,
            1,
        )
        return response

    context = copy.deepcopy(state.context or {})
    preview = context.setdefault("llm_context_preview", {})
    response_style_instruction = (
        PRODUCT_TEAM_RESPONSE_INSTRUCTION
        if user_type == "product_team"
        else ""
    )
    llm_question = (
        f"{question.rstrip()}\n\n{PRODUCT_TEAM_QUERY_SUFFIX}"
        if user_type == "product_team"
        else question
    )
    context["question"] = question
    context["response_style_instruction"] = response_style_instruction
    preview["question"] = llm_question
    if activity_request_id and activity_user_id is not None:
        update_answer_activity(
            activity_request_id,
            user_id=activity_user_id,
            workspace=workspace,
            question=question,
            status="using_conversation_evidence",
            context=context,
        )
    evidence = compact_follow_up_evidence(state)
    if activity_request_id and activity_user_id is not None:
        update_answer_activity(
            activity_request_id,
            user_id=activity_user_id,
            workspace=workspace,
            question=question,
            status="generating_answer",
            context=context,
        )

    fast_started_at = time.perf_counter()
    try:
        result = generate_fast_follow_up(
            context,
            evidence,
            user_llm=user_llm,
            allow_shared_fallback=allow_shared_fallback,
            llm_mode=llm_mode,
            question=llm_question,
        )
    except FollowUpNeedsEvidence:
        fast_gate_ms = round((time.perf_counter() - fast_started_at) * 1000, 1)
        response = answer_question(
            question,
            workspace=workspace,
            user_llm=user_llm,
            allow_shared_fallback=allow_shared_fallback,
            llm_mode=llm_mode,
            user_type=user_type,
            **activity_kwargs,
        )
        response["follow_up_reused"] = False
        response["follow_up_fallback"] = True
        response["deep_investigation"] = False
        response["investigate_deeply_available"] = False
        response["timings_ms"]["follow_up_gate"] = fast_gate_ms
        response["timings_ms"]["total"] = round(
            (time.perf_counter() - started_at) * 1000,
            1,
        )
        return response

    generation_ms = round((time.perf_counter() - fast_started_at) * 1000, 1)
    response = _answer_response(question, result, context, workspace)
    response["follow_up_reused"] = True
    response["follow_up_fallback"] = False
    response["deep_investigation"] = False
    response["investigate_deeply_available"] = True
    response["timings_ms"] = {
        "follow_up_generation": generation_ms,
        "total": round((time.perf_counter() - started_at) * 1000, 1),
    }
    return response


def flow_summary_question(flow_title: str, user_type: str) -> str:
    """Build an audience-aware flow question while keeping the selected internal
    flow name available for repository search."""
    quoted_title = json.dumps((flow_title or "selected flow")[:200])
    if user_type == "product_team":
        return (
            f"Investigate the repository flow identified internally as {quoted_title}. "
            "Provide a brief product-friendly summary of what the flow achieves, "
            "when it starts, its main user-visible steps, relevant alternate or "
            "failure outcomes, and its final result. Use simple, clear language. "
            "Do not repeat the internal flow identifier unless it is a user-facing "
            "feature name. Do not include technical terms, file names, class names, "
            "function or method names, endpoints, code identifiers, source citations, "
            "or implementation details. Do not invent behavior that is not supported "
            "by repository evidence."
        )
    return (
        f"Investigate the repository flow identified as {quoted_title}. Provide a "
        "brief developer-focused summary of its purpose, entry point, main control "
        "and data path, important branches or failure outcomes, and final result. "
        "Mention only the most relevant components, methods, files, and endpoints "
        "with source citations; do not return a raw inventory of graph nodes."
    )


def build_flow_summary_context(
    question: str,
    flow_data: dict,
    workspace: str,
    user_type: str,
) -> dict:
    """Use the selected flow payload directly instead of rediscovering it via
    broad repository search. Flow chips are already graph-scoped; this keeps the
    LLM request small, relevant, and fast."""
    source_root = _safe_source_root(workspace)
    response_style_instruction = (
        PRODUCT_TEAM_RESPONSE_INSTRUCTION
        if user_type == "product_team"
        else ""
    )
    sections = [
        ("entry_points", flow_data.get("entry_points") or flow_data.get("screens") or []),
        ("viewmodels", flow_data.get("viewmodels") or []),
        ("repositories", flow_data.get("repositories") or []),
        ("important_methods", flow_data.get("important_methods") or []),
    ]
    context_nodes = []
    seen_nodes = set()

    for section, items in sections:
        limit = 8 if section == "important_methods" else 12
        for item in items[:limit]:
            node_id = item.get("node")
            if not node_id or node_id in seen_nodes:
                continue
            seen_nodes.add(node_id)
            payload = dict(item)
            payload["section"] = section
            excerpt = read_source_excerpt(
                payload.get("source_file", ""),
                payload.get("source_location", ""),
                source_root,
                max_lines=24,
                max_chars=900,
            )
            if excerpt:
                payload["source_excerpt"] = excerpt["code"]
                payload["excerpt_range"] = (
                    f"L{excerpt['start_line']}-L{excerpt['end_line']}"
                )
            context_nodes.append(payload)

    preview_nodes = context_nodes[:LLM_PREVIEW_NODE_LIMIT]
    return {
        "question": question,
        "query_terms": [flow_data.get("topic") or flow_data.get("title") or "flow"],
        "context_nodes": context_nodes,
        "context_relations": [],
        "source_hits": [],
        "response_style_instruction": response_style_instruction,
        **(
            {"product_flow_summary": True}
            if user_type == "product_team"
            else {}
        ),
        "llm_context_preview": {
            "instruction": (
                "Answer using only the selected flow evidence below. Cite source "
                "files and line numbers for developer-facing claims. If this "
                "flow evidence is incomplete, say what could not be verified."
            ),
            "pre_search_instruction": "",
            "question": question,
            "flow": {
                "title": flow_data.get("title"),
                "topic": flow_data.get("topic"),
                "high_level_flow": flow_data.get("high_level_flow"),
                "sections": {
                    section: [
                        {
                            "name": item.get("name"),
                            "source": (
                                f"{item.get('source_file', '')} "
                                f"{item.get('source_location', '')}"
                            ).strip(),
                        }
                        for item in items
                    ]
                    for section, items in sections
                },
            },
            "nodes": [
                {
                    "name": n["name"],
                    "source": (
                        f'{n.get("source_file", "")} '
                        f'{n.get("source_location", "")}'
                    ).strip(),
                    "section": n.get("section"),
                    **(
                        {"code": n["source_excerpt"][:LLM_PREVIEW_SNIPPET_CHARS]}
                        if n.get("source_excerpt")
                        else {}
                    ),
                }
                for n in preview_nodes
            ],
            "relations": [],
            "source_search_hits": [],
        },
    }


@app.post("/repo/ask-llm")
def ask_llm_endpoint(
    request: AskRequest,
    workspace: str = Depends(authorized_workspace),
    user: dict = Depends(require_user),
):
    return ask_service.answer_single_request(request, workspace, user)


@app.post("/repo/compare")
def compare_repos_endpoint(
    request: CompareRequest,
    workspace: str = Query(DEFAULT_WORKSPACE),
    user: dict = Depends(require_user),
):
    return ask_service.answer_compare_request(request, workspace, user)


@app.post("/repo/flows/{topic}/summary")
def flow_summary_endpoint(
    topic: str,
    request: FlowSummaryRequest,
    workspace: str = Depends(authorized_workspace),
    user: dict = Depends(require_user),
):
    enforce_rate_limit(user["id"])
    enforce_strict_branch_freshness(workspace)
    flow_data = _flow(topic, workspace)
    user_type = _effective_answer_user_type(user, request.answer_user_type)
    question = flow_summary_question(flow_data.get("title") or topic, user_type)
    repo = db.get_repo_by_workspace(workspace)
    allow_shared = bool(repo["allow_shared_fallback"]) if repo else True
    user_llm = request.user_llm or load_user_llm(user["id"])
    try:
        with llm_admission.slot(), collect_token_usage() as token_usage:
            started_at = time.perf_counter()
            retrieval_started_at = time.perf_counter()
            context = build_flow_summary_context(
                question,
                flow_data,
                workspace,
                user_type,
            )
            retrieval_ms = round((time.perf_counter() - retrieval_started_at) * 1000, 1)
            generation_started_at = time.perf_counter()
            generated = generate(
                context,
                user_llm=user_llm,
                allow_shared_fallback=allow_shared,
                llm_mode=request.llm_mode,
            )
            generation_ms = round((time.perf_counter() - generation_started_at) * 1000, 1)
            result = _answer_response(question, generated, context, workspace)
            result["timings_ms"] = {
                "retrieval": retrieval_ms,
                "generation": generation_ms,
                "total": round((time.perf_counter() - started_at) * 1000, 1),
            }
            result["question"] = flow_data.get("title") or topic
            result["flow_topic"] = flow_data.get("topic") or topic
            state = conversation_store.create(
                user_id=user["id"],
                workspace=workspace,
                llm_mode=(request.llm_mode or "auto").lower(),
                user_type=user_type,
                repository_revision=repository_revision(workspace),
                context=result.get("context") or {},
                question=result["question"],
                answer=result["answer"],
            )
            result["conversation_id"] = state.conversation_id
            result["follow_up_reused"] = False
            result["token_usage"] = token_usage_payload(token_usage)
            result["answer_user_type"] = user_type
            ask_service.schedule_answer_token_usage(
                user,
                workspace,
                "repo.flow_summary",
                result,
                repo=repo,
            )
            return result
    except LLMCapacityError as error:
        raise HTTPException(
            status_code=503,
            detail=str(error),
            headers={"Retry-After": "5"},
        )
    except RuntimeError as error:
        raise HTTPException(status_code=400, detail=str(error))
    except Exception as error:
        raise HTTPException(
            status_code=500,
            detail=f"Flow summary request failed: {str(error)}",
        )
