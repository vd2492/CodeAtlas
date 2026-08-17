"""Persisted source retrieval index for query-time evidence lookup.

The index is intentionally lexical today: it stores bounded, non-sensitive
source lines once during repo indexing, then query-time retrieval can rank and
snippet those records without walking and reading the repository again.
"""

from __future__ import annotations

import json
import os
import time
import uuid
from pathlib import Path

from ..agent.tools import RepositoryToolbox

SOURCE_INDEX_VERSION = 1
SOURCE_INDEX_EXTENSIONS = {
    ".kt", ".java", ".py", ".ts", ".tsx", ".js", ".jsx", ".go", ".rb", ".rs",
    ".cpp", ".cc", ".c", ".h", ".hpp", ".cs", ".php", ".swift", ".scala",
    ".dart", ".vue", ".svelte", ".html", ".css", ".scss", ".xml", ".json",
    ".yaml", ".yml", ".toml", ".gradle", ".md",
}
SOURCE_INDEX_SKIP_DIRS = {
    ".git", ".gradle", ".idea", ".venv", "venv", "__pycache__", "node_modules",
    "build", "dist", ".next", ".turbo", "coverage", "target", ".dart_tool",
}
SOURCE_INDEX_MAX_FILES = int(os.environ.get("CODEATLAS_SOURCE_INDEX_FILES", "2500"))
SOURCE_INDEX_MAX_FILE_BYTES = int(os.environ.get("CODEATLAS_SOURCE_INDEX_FILE_BYTES", "240000"))
SOURCE_INDEX_MAX_SNIPPET_CHARS = int(os.environ.get("CODEATLAS_SOURCE_INDEX_SNIPPET_CHARS", "1800"))


def _compact(value: str) -> str:
    return value.lower().replace("_", "").replace("-", "").replace(".", "")


def _source_kind_score(rel_path: str) -> float:
    lower = rel_path.lower()
    if "/src/main/" in lower:
        return 24.0
    if "/src/test/" in lower or "/src/androidtest/" in lower:
        return -24.0
    if lower.startswith("docs/") or "/docs/" in lower or lower.endswith(".md"):
        return -14.0
    return 0.0


def _line_score(line: str, terms: list[str]) -> float:
    lower = line.lower()
    compacted = _compact(lower)
    score = 0.0
    for term in terms:
        if not term:
            continue
        if term in lower:
            score += 4.0
        if term.replace("_", "") in compacted:
            score += 2.0
    return score


def _source_snippets_from_lines(
    lines: list[str],
    terms: list[str],
    max_snippets: int = 2,
) -> list[dict]:
    scored = []
    for index, line in enumerate(lines):
        score = _line_score(line, terms)
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
        if len(code) > SOURCE_INDEX_MAX_SNIPPET_CHARS:
            code = code[:SOURCE_INDEX_MAX_SNIPPET_CHARS] + "\n..."
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


def _iter_indexable_files(source_root: Path):
    source_root = source_root.resolve()
    if not source_root.exists():
        return
    scanned = 0
    for dirpath, dirnames, filenames in os.walk(source_root):
        dirnames[:] = [
            name for name in dirnames
            if name not in SOURCE_INDEX_SKIP_DIRS and not name.startswith(".cache")
        ]
        for filename in sorted(filenames):
            path = Path(dirpath) / filename
            if path.suffix.lower() not in SOURCE_INDEX_EXTENSIONS and "." in filename:
                continue
            try:
                rel_path = path.relative_to(source_root).as_posix()
                resolved = RepositoryToolbox.resolve_path_for_root(source_root, rel_path)
                if resolved.stat().st_size > SOURCE_INDEX_MAX_FILE_BYTES:
                    continue
            except (OSError, ValueError):
                continue
            scanned += 1
            if scanned > SOURCE_INDEX_MAX_FILES:
                return
            yield rel_path, resolved


def build_source_index(source_root: Path, index_path: Path) -> dict:
    source_root = source_root.resolve()
    files = []
    for rel_path, path in _iter_indexable_files(source_root) or []:
        try:
            stat = path.stat()
            lines = path.read_text(errors="replace").splitlines()
        except OSError:
            continue
        files.append({
            "path": rel_path,
            "size": stat.st_size,
            "mtime_ns": stat.st_mtime_ns,
            "kind_score": _source_kind_score(rel_path),
            "lower_path": rel_path.lower(),
            "compact_path": _compact(rel_path),
            "stem_compact": _compact(path.stem),
            "lines": lines,
        })

    payload = {
        "version": SOURCE_INDEX_VERSION,
        "generated_at": time.time(),
        "file_count": len(files),
        "files": files,
    }
    index_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = index_path.with_name(f".{index_path.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp")
    try:
        temporary_path.write_text(json.dumps(payload, separators=(",", ":")))
        temporary_path.replace(index_path)
    finally:
        temporary_path.unlink(missing_ok=True)
    return payload


def load_source_index(index_path: Path) -> dict | None:
    try:
        payload = json.loads(index_path.read_text())
    except (OSError, json.JSONDecodeError):
        return None
    if payload.get("version") != SOURCE_INDEX_VERSION:
        return None
    if not isinstance(payload.get("files"), list):
        return None
    return payload


def search_source_index(index_path: Path, terms: list[str], limit: int = 8) -> list[dict] | None:
    if not terms:
        return []
    payload = load_source_index(index_path)
    if payload is None:
        return None

    compact_terms = [_compact(term) for term in terms]
    hits = []
    for item in payload.get("files") or []:
        lines = item.get("lines") or []
        if not isinstance(lines, list):
            continue
        path_lower = str(item.get("lower_path") or "").lower()
        path_compact = str(item.get("compact_path") or "")
        stem_compact = str(item.get("stem_compact") or "")
        text_lower = "\n".join(str(line) for line in lines).lower()
        text_compact = _compact(text_lower)
        score = float(item.get("kind_score") or 0.0)

        for term, compact_term in zip(terms, compact_terms):
            if term in path_lower:
                score += 35.0
            if compact_term and compact_term in path_compact:
                score += 18.0
            if compact_term and compact_term == stem_compact:
                score += 420.0
            count = text_lower.count(term)
            if count:
                score += min(28.0, count * 3.5)
            if compact_term and compact_term != term:
                compact_count = text_compact.count(compact_term)
                if compact_count:
                    score += min(18.0, compact_count * 2.0)

        if score <= 0:
            continue
        snippets = _source_snippets_from_lines(lines, terms)
        if not snippets:
            continue
        hits.append({
            "path": item.get("path"),
            "score": round(score + sum(snippet["score"] for snippet in snippets), 2),
            "snippets": snippets,
        })

    hits.sort(key=lambda row: (-row["score"], row["path"]))
    return hits[:limit]
