import json
import os
import re
import shutil
import subprocess
import time
from collections import defaultdict
from pathlib import Path
from typing import Optional

from fastapi import Depends, FastAPI, HTTPException, Query
from fastapi.responses import FileResponse
from pydantic import BaseModel

from . import db
from .config import DEFAULT_WORKSPACE, graph_path, repo_clone_dir
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
from .retrieval.relation_utils import (
    format_link,
    is_noise_node,
    rank_nodes_for_query,
    readable_name,
    search_nodes,
)
from .retrieval.config_schema import load_retrieval_config, seed_default_retrieval_config
from .llm.client import generate
from .auth.routes import router as auth_router, load_user_llm
from .auth.security import hash_password
from .auth.sessions import require_user
from .repos.routes import router as repos_router

app = FastAPI(title="CodeAtlas", version="0.2.0")

STATIC_DIR = Path(__file__).resolve().parent / "static"

# Multi-tenant routers (auth + admin repo lifecycle).
app.include_router(auth_router)
app.include_router(repos_router)


@app.on_event("startup")
def _startup() -> None:
    """Create tables, register the seeded demo workspace as a repo, and (if
    configured) seed an admin so the instance is usable on first boot."""
    db.init_db()
    db.seed_default_repo()
    seed_default_retrieval_config()
    admin_user = os.environ.get("CODEATLAS_ADMIN_USER")
    admin_pass = os.environ.get("CODEATLAS_ADMIN_PASS")
    if admin_user and admin_pass and db.user_count() == 0:
        db.create_user(admin_user, hash_password(admin_pass), role="admin")


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
_source_file_cache: "dict[str, list[tuple[str, Path]]]" = {}

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


def authorized_workspace(
    workspace: str = DEFAULT_WORKSPACE, user: dict = Depends(require_user)
) -> str:
    """Resolve + permission-check the target workspace for a query. Admins reach
    every workspace; users only reach repos granted to them."""
    if user["role"] != "admin" and not db.user_has_repo(user["id"], workspace):
        raise HTTPException(status_code=403, detail="You do not have access to this repository.")
    return workspace


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
    if not path.is_file():
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
                if path.stat().st_size > MAX_SOURCE_SCAN_BYTES:
                    continue
                rel_path = path.relative_to(source_root).as_posix()
            except OSError:
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


def _search_source_files(source_root: Path, terms: list[str], limit: int = 8) -> list[dict]:
    if not terms:
        return []

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
            text = path.read_text(errors="replace")
        except OSError:
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
def root():
    """Marketing / landing page."""
    return FileResponse(STATIC_DIR / "home.html")


@app.get("/app")
def ask_ui():
    """The user Ask UI (current login flow): login → repo picker → ask."""
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/admin.html")
def admin_console():
    return FileResponse(STATIC_DIR / "admin.html")


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
    # Optional bring-your-own-key creds {provider, base_url, api_key, model}.
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


def build_context(question: str, limit: int = 12, workspace: str = DEFAULT_WORKSPACE):
    import re

    nodes, links = load_graph(graph_path(workspace))
    source_root = _safe_source_root(workspace)
    node_meta_cache = {}

    def node_id_of(node):
        return str(node.get("id") or node.get("label") or node.get("name") or "")

    def compact_text(value: str) -> str:
        return value.lower().replace("_", "").replace("-", "").replace(".", "")

    def meta_for_node(node_id: str):
        if node_id not in node_meta_cache:
            node_meta_cache[node_id] = meta_for(node_id, links)
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
    source_hits = _search_source_files(source_root, source_terms, limit=10)
    seen_follow_terms = set()
    for _ in range(2):
        follow_terms = [
            term for term in _identifier_terms_from_hits(source_hits, source_terms)
            if term not in seen_follow_terms
        ]
        if not follow_terms:
            break
        seen_follow_terms.update(follow_terms)
        follow_hits = _search_source_files(source_root, follow_terms, limit=14)
        source_hits = _merge_source_hits(source_hits, follow_hits, limit=14)
    matched_source_files = {hit["path"] for hit in source_hits}

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

    # Attach real source code for the most relevant nodes so the LLM can explain
    # actual behavior (e.g. what a feature enforces), not just node names. Kept
    # small (top few nodes, short excerpts) so the prompt stays fast.
    for node in context_nodes[:config.excerpt_nodes]:
        excerpt = read_source_excerpt(
            node.get("source_file", ""), node.get("source_location", ""),
            source_root, max_lines=config.excerpt_max_lines,
            max_chars=config.excerpt_max_chars,
        )
        if excerpt:
            node["source_excerpt"] = excerpt["code"]
            node["excerpt_range"] = f"L{excerpt['start_line']}-L{excerpt['end_line']}"

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



def answer_question(question: str, workspace: str = DEFAULT_WORKSPACE,
                    user_llm: dict = None, allow_shared_fallback: bool = True,
                    llm_mode: str = None) -> dict:
    """Build context for a workspace and run the LLM fallback chain. Shared by
    the user ask endpoint and the admin test panel."""
    context = build_context(question, limit=16, workspace=workspace)
    result = generate(
        context,
        user_llm=user_llm,
        allow_shared_fallback=allow_shared_fallback,
        llm_mode=llm_mode,
    )
    return {
        "question": question,
        "answer": result["answer"],
        "provider_used": result["provider_used"],
        "context": context,
    }


@app.post("/repo/ask-llm")
def ask_llm_endpoint(
    request: AskRequest,
    workspace: str = Depends(authorized_workspace),
    user: dict = Depends(require_user),
):
    enforce_rate_limit(user["id"])
    repo = db.get_repo_by_workspace(workspace)
    allow_shared = bool(repo["allow_shared_fallback"]) if repo else True
    # Tier 1: an explicit per-request key wins; otherwise the user's stored BYOK key.
    user_llm = request.user_llm or load_user_llm(user["id"])
    try:
        return answer_question(
            request.question,
            workspace=workspace,
            user_llm=user_llm,
            allow_shared_fallback=allow_shared,
            llm_mode=request.llm_mode,
        )
    except RuntimeError as error:
        raise HTTPException(status_code=400, detail=str(error))
    except Exception as error:
        raise HTTPException(status_code=500, detail=f"LLM request failed: {str(error)}")
