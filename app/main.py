import os
import re
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import FileResponse
from pydantic import BaseModel

from .config import DEFAULT_WORKSPACE, repo_clone_dir
from .retrieval.flow_map import TOPICS, load_graph, meta_for, pretty_name, pretty_method, find_methods
from .retrieval.graph_insights import repo_summary_dynamic
from .retrieval.relation_utils import readable_name, format_link, search_nodes, rank_nodes_for_query
from .llm.client import generate
from .auth.routes import router as auth_router
from .repos.routes import router as repos_router

app = FastAPI(title="CodeAtlas", version="0.2.0")

STATIC_DIR = Path(__file__).resolve().parent / "static"

# Root of the indexed source tree, so node source paths resolve to real files.
# Defaults to the default workspace's cloned repo; override for the demo graph
# (e.g. CODEATLAS_SOURCE_ROOT=/path/to/destiny) to get code excerpts.
SOURCE_ROOT = Path(os.environ.get("CODEATLAS_SOURCE_ROOT", repo_clone_dir(DEFAULT_WORKSPACE)))

# Skeleton routers for the multi-tenant features (Phase 2+). Mounted now so the
# routes exist; handlers are stubs until those phases land.
app.include_router(auth_router)
app.include_router(repos_router)


def read_source_excerpt(source_file: str, source_location: str, max_lines: int = 32) -> "dict | None":
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

    path = (SOURCE_ROOT / source_file).resolve()
    try:
        path.relative_to(SOURCE_ROOT)
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
    if len(excerpt) > 1100:
        excerpt = excerpt[:1100] + "\n…"

    return {"start_line": start, "end_line": last, "code": excerpt}



@app.get("/")
def root():
    return FileResponse(STATIC_DIR / "index.html")


class AskRequest(BaseModel):
    question: str
    # Optional bring-your-own-key creds {provider, base_url, api_key, model}.
    user_llm: Optional[dict] = None


@app.get("/health")
def health():
    return {"status": "ok", "service": "codeatlas-api"}


@app.get("/repo/summary")
def repo_summary():
    return repo_summary_dynamic()


@app.get("/repo/flows/{topic}")
def flow(topic: str):
    topic = topic.lower()

    if topic not in TOPICS:
        raise HTTPException(status_code=404, detail="Supported topics: habit, revision, login")

    nodes, links = load_graph()
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
def ask(request: AskRequest):
    q = request.question.lower()

    if "habit" in q:
        topic = "habit"
    elif "revision" in q or "spaced" in q:
        topic = "revision"
    elif "login" in q or "auth" in q or "sign" in q:
        topic = "login"
    elif "screen" in q:
        return repo_summary()
    else:
        return {
            "answer": "I can currently answer questions about screens, habit flow, revision flow, and login/auth flow.",
            "supported_topics": ["screens", "habit", "revision", "login"],
        }

    return flow(topic)


@app.get("/repo/nodes/{node_id}")
def node_details(node_id: str):
    nodes, links = load_graph()

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


SEARCH_STOPWORDS = {
    "how", "does", "do", "what", "where", "when", "why", "which", "who",
    "is", "are", "the", "a", "an", "work", "works", "working", "use",
    "uses", "used", "using", "tell", "explain", "show", "me", "in",
    "of", "to", "for", "and", "or", "with", "this", "that", "about",
    "flow",
}


@app.get("/repo/search")
def search_repo(q: str = Query(..., min_length=1), limit: int = 30):
    nodes, links = load_graph()

    # Tokenize so natural-language phrases ("explain habit flow") match, not just
    # exact node substrings. Fall back to the raw query if nothing survives.
    raw_tokens = re.findall(r"[A-Za-z][A-Za-z0-9]+", q.lower())
    terms = [t for t in raw_tokens if t not in SEARCH_STOPWORDS and len(t) > 2]
    if not terms:
        terms = [t for t in raw_tokens if len(t) > 2] or [q]

    results = rank_nodes_for_query(terms, nodes, links, 80)

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
def repo_context(question: str = Query(..., min_length=1), limit: int = 12):
    import re

    nodes, links = load_graph()

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

    stopwords = {
        "how", "does", "do", "what", "where", "when", "why", "which", "who",
        "is", "are", "the", "a", "an", "work", "works", "working", "use",
        "uses", "used", "using", "tell", "explain", "show", "me", "in",
        "of", "to", "for", "and", "or", "with", "this", "that"
    }

    raw_tokens = re.findall(r"[A-Za-z][A-Za-z0-9]+", question.lower())
    keywords = [t for t in raw_tokens if t not in stopwords and len(t) > 2]
    expanded = set(keywords)

    if "habit" in expanded or "habits" in expanded:
        expanded.update(["habit", "habits"])

    if "completion" in expanded or "complete" in expanded or "completed" in expanded:
        expanded.update(["completion", "complete", "completed", "state", "today", "toggle", "streak"])

    query_terms = list(expanded)

    is_habit_completion = "habit" in expanded and any(
        x in expanded for x in ["completion", "complete", "completed", "state", "today", "toggle"]
    )

    preferred_components = []
    preferred_methods = []

    if is_habit_completion:
        preferred_components = [
            "HabitsScreen",
            "HomeScreen",
            "HabitsViewModel",
            "HomeViewModel",
            "HabitRepository",
            "HabitDao",
            "HabitEntity",
            "HabitCompletionEntity",
        ]

        preferred_methods = [
            "HomeViewModel.toggleHabit",
            "HabitsViewModel.continueHabitStreak",
            "HabitRepository.getTodayHabitsWithCompletion",
            "HabitRepository.setHabitStateToday",
            "HabitRepository.computeDisplayedHabitStreak",
            "HabitRepository.computeHabitStreak",
            "HabitRepository.currentUserHabitsCollection",
            "HabitRepository.calculateHabitHistory",
            "HabitDao.iscompletedon",
        ]

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

    # 3. Fill remaining slots from a rarity-weighted, multi-term relevance rank.
    #    Specific words (e.g. "strict") outrank generic ones (e.g. "mode"/"app")
    #    instead of being crowded out by common-substring noise and alphabetical
    #    tie-breaks.
    ranked = rank_nodes_for_query(query_terms, nodes, links, limit=limit * 4)

    for item in ranked:
        if len(context_nodes) >= limit:
            break

        source_file = item.get("source_file") or ""
        if "/src/test/" in source_file or "/src/androidTest/" in source_file:
            continue

        if is_habit_completion and "revision" in item["name"].lower():
            continue

        if item["name"] in seen_names or item["node"] in seen_nodes:
            continue

        context_nodes.append(item)
        seen_names.add(item["name"])
        seen_nodes.add(item["node"])

    selected_names = {item["name"] for item in context_nodes}
    selected_node_ids = {item["node"] for item in context_nodes}

    preferred_name_tokens = {compact_text(x) for x in preferred_components + preferred_methods}

    def is_useful_context_relation(formatted):
        source_name = formatted.get("source_name") or ""
        target_name = formatted.get("target_name") or ""
        relation = formatted.get("relation")
        context = formatted.get("context")
        source_file = formatted.get("source_file") or ""

        if "/src/test/" in source_file or "/src/androidTest/" in source_file:
            return False

        combined = compact_text(source_name + " " + target_name)

        if is_habit_completion and "revision" in combined:
            return False

        # Drop primitive/type generic noise.
        if context in {"generic_arg", "return_type", "parameter_type"}:
            if target_name in {"HabitRepository", "HabitsScreen", "HomeScreen", "HabitsViewModel", "HomeViewModel"}:
                return False

        # Keep definitions only for selected preferred methods.
        if relation == "method":
            return target_name in selected_names or compact_text(target_name) in preferred_name_tokens

        # Keep actual calls that involve selected components/methods.
        if relation == "calls":
            if source_name in selected_names or target_name in selected_names:
                return True

            return any(token in combined for token in preferred_name_tokens)

        # Keep screen/viewmodel/repository references.
        if relation == "references":
            important_pairs = [
                ("habitsscreen", "habitsviewmodel"),
                ("homescreen", "homeviewmodel"),
                ("habitsviewmodel", "habitrepository"),
                ("homeviewmodel", "habitrepository"),
                ("habitrepository", "habitdao"),
            ]
            return any(a in combined and b in combined for a, b in important_pairs)

        return False

    context_relations = []
    seen_relations = set()

    for link in links:
        formatted = format_link(link)

        source = formatted.get("source")
        target = formatted.get("target")

        # First preference: relations directly attached to selected nodes.
        attached = source in selected_node_ids or target in selected_node_ids

        if not attached and not is_useful_context_relation(formatted):
            continue

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

        if len(context_relations) >= 24:
            break

    # Attach real source code for the most relevant nodes so the LLM can explain
    # actual behavior (e.g. what a feature enforces), not just node names. Kept
    # small (top few nodes, short excerpts) so the prompt stays fast.
    for node in context_nodes[:6]:
        excerpt = read_source_excerpt(
            node.get("source_file", ""), node.get("source_location", ""), max_lines=22
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



@app.post("/repo/ask-llm")
def ask_llm_endpoint(request: AskRequest):
    try:
        context = repo_context(question=request.question, limit=16)
        result = generate(context, user_llm=request.user_llm)

        return {
            "question": request.question,
            "answer": result["answer"],
            "provider_used": result["provider_used"],
            "context": context,
        }
    except RuntimeError as error:
        raise HTTPException(status_code=400, detail=str(error))
    except Exception as error:
        raise HTTPException(status_code=500, detail=f"LLM request failed: {str(error)}")
