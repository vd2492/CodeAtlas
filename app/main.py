import os
import re
import time
from collections import defaultdict
from pathlib import Path
from typing import Optional

from fastapi import Depends, FastAPI, HTTPException, Query
from fastapi.responses import FileResponse
from pydantic import BaseModel

from . import db
from .config import DEFAULT_WORKSPACE, graph_path, repo_clone_dir
from .retrieval.flow_map import TOPICS, load_graph, meta_for, pretty_name, pretty_method, find_methods
from .retrieval.graph_insights import repo_summary_dynamic
from .retrieval.relation_utils import readable_name, format_link, search_nodes, rank_nodes_for_query
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


def _flow(topic: str, workspace: str):
    topic = topic.lower()

    if topic not in TOPICS:
        raise HTTPException(status_code=404, detail="Supported topics: habit, revision, login")

    nodes, links = load_graph(graph_path(workspace))
    config = TOPICS[topic]

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
            "answer": "I can currently answer questions about screens, habit flow, revision flow, and login/auth flow.",
            "supported_topics": ["screens", "habit", "revision", "login"],
        }

    return _flow(topic, workspace)


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

    def node_id_of(node):
        return str(node.get("id") or node.get("label") or node.get("name") or "")

    def compact_text(value: str) -> str:
        return value.lower().replace("_", "").replace("-", "").replace(".", "")

    def payload_for_node(node_id: str, score: int = 0):
        source_file, source_location = meta_for(node_id, links)
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

    query_terms = list(expanded)
    expanded_compact = {compact_text(t) for t in expanded}

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

    # 3. Fill remaining slots from a rarity-weighted, multi-term relevance rank,
    #    amplified by the workspace's keyword boosts. Specific words outrank
    #    generic ones instead of being crowded out by common-substring noise.
    node_limit = config.node_limit
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
        )

    context_relations = []
    seen_relations = set()
    relation_limit = config.relation_limit

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
    source_root = workspace_source_root(workspace)
    for node in context_nodes[:config.excerpt_nodes]:
        excerpt = read_source_excerpt(
            node.get("source_file", ""), node.get("source_location", ""),
            source_root, max_lines=config.excerpt_max_lines,
            max_chars=config.excerpt_max_chars,
        )
        if excerpt:
            node["source_excerpt"] = excerpt["code"]
            node["excerpt_range"] = f"L{excerpt['start_line']}-L{excerpt['end_line']}"

    return {
        "question": question,
        "query_terms": query_terms,
        "context_nodes": context_nodes,
        "context_relations": context_relations,
        "llm_context_preview": {
            "instruction": "Answer the user's codebase question using this graph context AND the source code excerpts provided per node. Cite source files and line numbers. Prefer explaining behavior from the code excerpts. Do not guess beyond the provided context.",
            "question": question,
            "nodes": [
                {
                    "name": n["name"],
                    "source": f'{n.get("source_file", "")} {n.get("source_location", "")}',
                    **({"code": n["source_excerpt"]} if n.get("source_excerpt") else {}),
                }
                for n in context_nodes
            ],
            "relations": [
                {
                    "from": r["source_name"],
                    "relation": r["relation_label"],
                    "to": r["target_name"],
                    "source": f'{r.get("source_file", "")} {r.get("source_location", "")}',
                }
                for r in context_relations
            ],
        },
    }



def answer_question(question: str, workspace: str = DEFAULT_WORKSPACE,
                    user_llm: dict = None, allow_shared_fallback: bool = True) -> dict:
    """Build context for a workspace and run the LLM fallback chain. Shared by
    the user ask endpoint and the admin test panel."""
    context = build_context(question, limit=16, workspace=workspace)
    result = generate(context, user_llm=user_llm, allow_shared_fallback=allow_shared_fallback)
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
        )
    except RuntimeError as error:
        raise HTTPException(status_code=400, detail=str(error))
    except Exception as error:
        raise HTTPException(status_code=500, detail=f"LLM request failed: {str(error)}")
