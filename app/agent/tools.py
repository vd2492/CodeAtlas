"""Workspace-scoped, read-only tools exposed to tool-calling LLMs.

The model can search and inspect repository data, but it cannot execute code,
write files, or read outside the authorized workspace.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
from collections import defaultdict
from pathlib import Path
from typing import Any

from ..config import DEFAULT_WORKSPACE, graph_path, repo_clone_dir
from ..retrieval.config_schema import load_retrieval_config
from ..retrieval.flow_map import load_graph
from ..retrieval.relation_utils import (
    format_link,
    is_noise_node,
    rank_nodes_for_query,
    readable_name,
)


MAX_READ_LINES = int(os.environ.get("CODEATLAS_AGENT_READ_LINES", "240"))
MAX_READ_CHARS = int(os.environ.get("CODEATLAS_AGENT_READ_CHARS", "30000"))
MAX_READ_FILE_BYTES = int(os.environ.get("CODEATLAS_AGENT_READ_FILE_BYTES", "2000000"))
MAX_SEARCH_FILES = int(os.environ.get("CODEATLAS_AGENT_SEARCH_FILES", "3000"))
MAX_SEARCH_FILE_BYTES = int(os.environ.get("CODEATLAS_AGENT_SEARCH_FILE_BYTES", "300000"))
MAX_SEARCH_RESULT_CHARS = int(os.environ.get("CODEATLAS_AGENT_TOOL_RESULT_CHARS", "45000"))

SOURCE_EXTENSIONS = {
    ".c", ".cc", ".cpp", ".cs", ".css", ".dart", ".go", ".gradle", ".h",
    ".hpp", ".html", ".java", ".js", ".json", ".jsx", ".kt", ".kts", ".md",
    ".php", ".py", ".rb", ".rs", ".scala", ".scss", ".svelte", ".swift",
    ".toml", ".ts", ".tsx", ".vue", ".xml", ".yaml", ".yml",
}
SKIP_DIRECTORIES = {
    ".git", ".gradle", ".idea", ".next", ".turbo", ".venv", "__pycache__",
    "build", "coverage", "dist", "node_modules", "target", "venv",
}
SENSITIVE_FILE_NAMES = {
    ".env", ".npmrc", ".pypirc", "credentials.json", "google-services.json",
    "googleservice-info.plist", "id_dsa", "id_ecdsa", "id_ed25519", "id_rsa",
}
SENSITIVE_SUFFIXES = {".jks", ".keystore", ".key", ".p12", ".pem", ".pfx"}


def _object_schema(properties: dict, required: list[str] = None) -> dict:
    return {
        "type": "object",
        "properties": properties,
        "required": required or [],
        "additionalProperties": False,
    }


TOOL_DEFINITIONS = [
    {
        "name": "search_code",
        "description": (
            "Search source text, file paths, and the structural graph. Use this "
            "first for concepts, features, error text, class names, or fuzzy questions."
        ),
        "parameters": _object_schema(
            {
                "query": {
                    "type": "string",
                    "description": "Words, identifier, phrase, or error text to search for.",
                },
                "limit": {
                    "type": "integer",
                    "description": "Maximum source files and graph nodes to return (1-12).",
                },
            },
            ["query"],
        ),
    },
    {
        "name": "read_file",
        "description": (
            "Read an exact source file with stable line numbers. Use after search "
            "to verify behavior and gather evidence for citations."
        ),
        "parameters": _object_schema(
            {
                "path": {
                    "type": "string",
                    "description": "Repository-relative file path returned by another tool.",
                },
                "start_line": {
                    "type": "integer",
                    "description": "First 1-based line to read. Defaults to 1.",
                },
                "end_line": {
                    "type": "integer",
                    "description": "Last 1-based line to read. Bounded by the server.",
                },
            },
            ["path"],
        ),
    },
    {
        "name": "list_directory",
        "description": (
            "List files and folders at a repository-relative path. Use to learn "
            "the project layout or find nearby files."
        ),
        "parameters": _object_schema(
            {
                "path": {
                    "type": "string",
                    "description": "Repository-relative directory path; use an empty string for root.",
                },
            }
        ),
    },
    {
        "name": "find_definition",
        "description": (
            "Find graph nodes that define or represent a symbol and return their "
            "source paths and locations."
        ),
        "parameters": _object_schema(
            {
                "symbol": {
                    "type": "string",
                    "description": "Class, method, function, service, or other symbol name.",
                },
            },
            ["symbol"],
        ),
    },
    {
        "name": "find_references",
        "description": (
            "Find structural graph relations connected to a symbol, including "
            "calls, references, containment, inheritance, and implementation."
        ),
        "parameters": _object_schema(
            {
                "symbol": {
                    "type": "string",
                    "description": "Symbol whose incoming and outgoing relations should be inspected.",
                },
                "limit": {
                    "type": "integer",
                    "description": "Maximum relations to return (1-50).",
                },
            },
            ["symbol"],
        ),
    },
    {
        "name": "get_callers",
        "description": "Find methods or components that call the requested symbol.",
        "parameters": _object_schema(
            {
                "symbol": {
                    "type": "string",
                    "description": "Called method, function, class, or graph symbol.",
                },
                "limit": {
                    "type": "integer",
                    "description": "Maximum callers to return (1-50).",
                },
            },
            ["symbol"],
        ),
    },
]


class RepositoryToolbox:
    """Execute bounded read-only tools against one authorized workspace."""

    def __init__(self, workspace: str):
        self.workspace = workspace
        self.source_root = self._source_root(workspace)
        self.nodes, self.links = load_graph(graph_path(workspace))
        self.config = load_retrieval_config(workspace)
        self.trace: list[dict] = []

        self.node_by_id = {
            self._node_id(node): node for node in self.nodes if self._node_id(node)
        }
        self.incoming: dict[str, list[dict]] = defaultdict(list)
        self.outgoing: dict[str, list[dict]] = defaultdict(list)
        for link in self.links:
            source = str(link.get("source") or "")
            target = str(link.get("target") or "")
            if source:
                self.outgoing[source].append(link)
            if target:
                self.incoming[target].append(link)

    @staticmethod
    def _source_root(workspace: str) -> Path:
        if workspace == DEFAULT_WORKSPACE:
            override = os.environ.get("CODEATLAS_SOURCE_ROOT")
            if override:
                return Path(override).expanduser().resolve()
        return repo_clone_dir(workspace).resolve()

    @staticmethod
    def _node_id(node: dict) -> str:
        return str(node.get("id") or node.get("label") or node.get("name") or "")

    @staticmethod
    def _bounded_int(value: Any, default: int, minimum: int, maximum: int) -> int:
        try:
            parsed = int(value)
        except (TypeError, ValueError):
            parsed = default
        return min(maximum, max(minimum, parsed))

    def _resolve_path(self, relative_path: str, require_directory: bool = False) -> Path:
        clean = str(relative_path or "").strip().lstrip("/")
        candidate = (self.source_root / clean).resolve()
        try:
            candidate.relative_to(self.source_root)
        except ValueError as exc:
            raise ValueError("path is outside the authorized repository") from exc
        if require_directory and not candidate.is_dir():
            raise ValueError(f"directory not found: {clean or '.'}")
        if not require_directory and not candidate.is_file():
            raise ValueError(f"file not found: {clean}")
        if not require_directory and self._is_sensitive_path(candidate):
            raise ValueError("reading likely credential/secret files is not allowed")
        return candidate

    def _is_sensitive_path(self, path: Path) -> bool:
        try:
            relative = path.relative_to(self.source_root)
        except ValueError:
            return True
        lower_name = path.name.lower()
        if lower_name in SENSITIVE_FILE_NAMES or path.suffix.lower() in SENSITIVE_SUFFIXES:
            return True
        if lower_name.startswith(".env.") and lower_name not in {
            ".env.example", ".env.sample", ".env.template",
        }:
            return True
        return any(part == ".git" for part in relative.parts)

    def call(self, name: str, arguments: dict | None) -> str:
        """Validate, execute, bound, and JSON-encode one model-requested tool call."""
        args = arguments if isinstance(arguments, dict) else {}
        methods = {
            "search_code": self.search_code,
            "read_file": self.read_file,
            "list_directory": self.list_directory,
            "find_definition": self.find_definition,
            "find_references": self.find_references,
            "get_callers": self.get_callers,
        }
        method = methods.get(name)
        if method is None:
            result = {"ok": False, "error": f"unknown tool: {name}"}
        else:
            try:
                result = method(**args)
                result["ok"] = True
            except (TypeError, ValueError, OSError) as exc:
                result = {"ok": False, "error": str(exc)}

        encoded = json.dumps(result, ensure_ascii=False)
        if len(encoded) > MAX_SEARCH_RESULT_CHARS:
            result = {
                "ok": result.get("ok", False),
                "truncated": True,
                "message": "Tool result exceeded the server limit. Narrow the query or line range.",
                "preview": encoded[:MAX_SEARCH_RESULT_CHARS],
            }
            encoded = json.dumps(result, ensure_ascii=False)

        self.trace.append({
            "tool": name,
            "arguments": args,
            "result": self._trace_summary(result),
        })
        return encoded

    @staticmethod
    def _trace_summary(result: dict) -> dict:
        summary = {"ok": bool(result.get("ok"))}
        for key in (
            "path", "start_line", "end_line", "total_lines", "entry_count",
            "source_hit_count", "graph_hit_count", "definition_count",
            "reference_count", "caller_count",
        ):
            if key in result:
                summary[key] = result[key]
        if result.get("error"):
            summary["error"] = result["error"]
        return summary

    def _iter_source_files(self):
        if not self.source_root.is_dir():
            return
        scanned = 0
        for dirpath, dirnames, filenames in os.walk(self.source_root):
            dirnames[:] = sorted(
                name for name in dirnames
                if name not in SKIP_DIRECTORIES and not name.startswith(".cache")
            )
            for filename in sorted(filenames):
                path = Path(dirpath) / filename
                if path.suffix.lower() not in SOURCE_EXTENSIONS and "." in filename:
                    continue
                try:
                    resolved = path.resolve()
                    resolved.relative_to(self.source_root)
                    if not resolved.is_file() or resolved.stat().st_size > MAX_SEARCH_FILE_BYTES:
                        continue
                    relative = path.relative_to(self.source_root).as_posix()
                except (OSError, ValueError):
                    continue
                if self._is_sensitive_path(resolved):
                    continue
                yield relative, resolved
                scanned += 1
                if scanned >= MAX_SEARCH_FILES:
                    return

    @staticmethod
    def _query_terms(query: str) -> list[str]:
        terms = re.findall(r"[A-Za-z0-9_.$:/-]+", query)
        seen = set()
        result = []
        for term in terms:
            normalized = term.strip("._-/$:").lower()
            if len(normalized) < 2 or normalized in seen:
                continue
            seen.add(normalized)
            result.append(normalized)
        return result[:16]

    @staticmethod
    def _line_numbered(lines: list[str], start_index: int) -> str:
        return "\n".join(
            f"L{start_index + offset}: {line}" for offset, line in enumerate(lines)
        )

    def _source_matches_with_rg(self, terms: list[str]) -> dict[str, set[int]]:
        if not terms or not shutil.which("rg") or not self.source_root.is_dir():
            return {}
        pattern = "|".join(re.escape(term) for term in terms)
        command = [
            "rg", "--json", "--ignore-case", "--line-number", "--max-count", "20",
            "--max-filesize", str(MAX_SEARCH_FILE_BYTES),
        ]
        for skipped in sorted(SKIP_DIRECTORIES):
            command.extend(["--glob", f"!**/{skipped}/**"])
        command.extend(["--", pattern, "."])
        try:
            response = subprocess.run(
                command,
                cwd=str(self.source_root),
                capture_output=True,
                text=True,
                timeout=10,
            )
        except (OSError, subprocess.TimeoutExpired):
            return {}

        matches: dict[str, set[int]] = defaultdict(set)
        for raw_line in response.stdout.splitlines():
            try:
                event = json.loads(raw_line)
            except json.JSONDecodeError:
                continue
            if event.get("type") != "match":
                continue
            data = event.get("data") or {}
            relative = str((data.get("path") or {}).get("text") or "").lstrip("./")
            line_number = data.get("line_number")
            if relative and line_number:
                matches[relative].add(int(line_number))
        return dict(matches)

    def _source_matches_fallback(self, terms: list[str]) -> dict[str, set[int]]:
        matches: dict[str, set[int]] = defaultdict(set)
        for relative, path in self._iter_source_files() or []:
            try:
                lines = path.read_text(errors="replace").splitlines()
            except OSError:
                continue
            for index, line in enumerate(lines):
                lower = line.lower()
                if any(term in lower for term in terms):
                    matches[relative].add(index + 1)
                    if len(matches[relative]) >= 20:
                        break
        return dict(matches)

    def _source_search(self, terms: list[str], limit: int) -> list[dict]:
        matches = self._source_matches_with_rg(terms)
        path_scores: dict[str, float] = {}

        for relative, line_numbers in matches.items():
            path_scores[relative] = min(80.0, len(line_numbers) * 8.0)

        for relative, _ in self._iter_source_files() or []:
            lower = relative.lower()
            compact = re.sub(r"[^a-z0-9]", "", lower)
            score = path_scores.get(relative, 0.0)
            for term in terms:
                if term in lower:
                    score += 28.0
                compact_term = re.sub(r"[^a-z0-9]", "", term)
                if compact_term and compact_term in compact:
                    score += 12.0
            if "/src/main/" in lower:
                score += 10.0
            if "/src/test/" in lower or "/src/androidtest/" in lower:
                score -= 18.0
            if score > 0:
                path_scores[relative] = score

        if not matches:
            matches = self._source_matches_fallback(terms)
            for relative, line_numbers in matches.items():
                path_scores[relative] = path_scores.get(relative, 0.0) + min(
                    80.0, len(line_numbers) * 8.0
                )

        ranked_paths = sorted(path_scores, key=lambda path: (-path_scores[path], path))
        hits = []
        for relative in ranked_paths[:limit]:
            try:
                path = self._resolve_path(relative)
                lines = path.read_text(errors="replace").splitlines()
            except (OSError, ValueError):
                continue

            focus = sorted(matches.get(relative, set()))[:2] or [1]
            snippets = []
            used = []
            for line_number in focus:
                start = max(1, line_number - 5)
                end = min(len(lines), line_number + 8)
                if any(start <= old_end and end >= old_start for old_start, old_end in used):
                    continue
                snippets.append({
                    "start_line": start,
                    "end_line": end,
                    "code": self._line_numbered(lines[start - 1:end], start),
                })
                used.append((start, end))

            hits.append({
                "path": relative,
                "score": round(path_scores[relative], 2),
                "snippets": snippets,
            })
        return hits

    def _expanded_terms(self, query: str) -> list[str]:
        terms = self._query_terms(query)
        stopwords = set(self.config.stopwords)
        expanded = [term for term in terms if term not in stopwords]
        for term in list(expanded):
            expanded.extend(self.config.synonyms.get(term, []))
        return list(dict.fromkeys(expanded or terms))[:24]

    def search_code(self, query: str, limit: int = 8) -> dict:
        if not isinstance(query, str) or not query.strip():
            raise ValueError("query is required")
        limit = self._bounded_int(limit, 8, 1, 12)
        terms = self._expanded_terms(query)
        source_hits = self._source_search(terms, limit)
        graph_hits = rank_nodes_for_query(
            terms,
            self.nodes,
            self.links,
            limit=limit,
            boosts=self.config.keyword_boosts,
        )
        return {
            "query": query,
            "terms": terms,
            "source_hit_count": len(source_hits),
            "graph_hit_count": len(graph_hits),
            "source_hits": source_hits,
            "graph_hits": graph_hits,
        }

    def read_file(
        self,
        path: str,
        start_line: int = 1,
        end_line: int | None = None,
    ) -> dict:
        resolved = self._resolve_path(path)
        if resolved.stat().st_size > MAX_READ_FILE_BYTES:
            raise ValueError(
                f"file exceeds the {MAX_READ_FILE_BYTES}-byte read limit: {path}"
            )
        try:
            lines = resolved.read_text(errors="replace").splitlines()
        except OSError as exc:
            raise ValueError(f"could not read file: {path}") from exc

        total = len(lines)
        if total == 0:
            return {
                "path": resolved.relative_to(self.source_root).as_posix(),
                "start_line": 1,
                "end_line": 0,
                "total_lines": 0,
                "truncated": False,
                "content": "",
            }
        start = self._bounded_int(start_line, 1, 1, max(1, total))
        requested_end = end_line if end_line is not None else start + MAX_READ_LINES - 1
        end = self._bounded_int(requested_end, start, start, max(start, total))
        end = min(end, start + MAX_READ_LINES - 1)
        selected = lines[start - 1:end]
        content = self._line_numbered(selected, start)
        if len(content) > MAX_READ_CHARS:
            content = content[:MAX_READ_CHARS] + "\n[truncated by character limit]"

        return {
            "path": resolved.relative_to(self.source_root).as_posix(),
            "start_line": start,
            "end_line": end,
            "total_lines": total,
            "truncated": end < total or len(content) >= MAX_READ_CHARS,
            "content": content,
        }

    def list_directory(self, path: str = "") -> dict:
        resolved = self._resolve_path(path, require_directory=True)
        entries = []
        for child in sorted(resolved.iterdir(), key=lambda item: (not item.is_dir(), item.name.lower())):
            if child.name in SKIP_DIRECTORIES or child.name.startswith(".cache"):
                continue
            try:
                child.resolve().relative_to(self.source_root)
            except ValueError:
                continue
            entries.append({
                "name": child.name,
                "type": "directory" if child.is_dir() else "file",
                "path": child.relative_to(self.source_root).as_posix(),
            })
            if len(entries) >= 100:
                break
        return {
            "path": resolved.relative_to(self.source_root).as_posix() or ".",
            "entry_count": len(entries),
            "entries": entries,
        }

    def _matching_node_ids(self, symbol: str, limit: int = 10) -> list[str]:
        terms = self._query_terms(symbol)
        ranked = rank_nodes_for_query(terms or [symbol], self.nodes, self.links, limit=limit)
        return [item["node"] for item in ranked]

    def find_definition(self, symbol: str) -> dict:
        if not isinstance(symbol, str) or not symbol.strip():
            raise ValueError("symbol is required")
        matches = rank_nodes_for_query(
            self._query_terms(symbol) or [symbol],
            self.nodes,
            self.links,
            limit=12,
        )
        definitions = []
        for match in matches:
            node = self.node_by_id.get(match["node"], {})
            definitions.append({
                "name": match["name"],
                "node": match["node"],
                "source_file": node.get("source_file") or match.get("source_file"),
                "source_location": node.get("source_location") or match.get("source_location"),
                "score": match.get("score"),
            })
        return {
            "symbol": symbol,
            "definition_count": len(definitions),
            "definitions": definitions,
        }

    def find_references(self, symbol: str, limit: int = 30) -> dict:
        if not isinstance(symbol, str) or not symbol.strip():
            raise ValueError("symbol is required")
        limit = self._bounded_int(limit, 30, 1, 50)
        node_ids = self._matching_node_ids(symbol)
        relations = []
        seen = set()
        for node_id in node_ids:
            for link in self.incoming.get(node_id, []) + self.outgoing.get(node_id, []):
                key = (
                    link.get("source"), link.get("target"), link.get("relation"),
                    link.get("source_file"), link.get("source_location"),
                )
                if key in seen:
                    continue
                seen.add(key)
                formatted = format_link(link)
                if formatted.get("context") in {"parameter_type", "return_type", "generic_arg"}:
                    continue
                relations.append(formatted)
                if len(relations) >= limit:
                    break
            if len(relations) >= limit:
                break
        return {
            "symbol": symbol,
            "matched_nodes": node_ids,
            "reference_count": len(relations),
            "references": relations,
        }

    def get_callers(self, symbol: str, limit: int = 30) -> dict:
        if not isinstance(symbol, str) or not symbol.strip():
            raise ValueError("symbol is required")
        limit = self._bounded_int(limit, 30, 1, 50)
        node_ids = self._matching_node_ids(symbol)
        callers = []
        seen = set()
        for node_id in node_ids:
            for link in self.incoming.get(node_id, []):
                if link.get("relation") != "calls":
                    continue
                source_id = str(link.get("source") or "")
                key = (source_id, link.get("source_file"), link.get("source_location"))
                if key in seen:
                    continue
                seen.add(key)
                source_node = self.node_by_id.get(source_id, {})
                callers.append({
                    "name": readable_name(source_id),
                    "node": source_id,
                    "source_file": link.get("source_file") or source_node.get("source_file"),
                    "source_location": link.get("source_location") or source_node.get("source_location"),
                    "calls": readable_name(node_id),
                })
                if len(callers) >= limit:
                    break
            if len(callers) >= limit:
                break
        return {
            "symbol": symbol,
            "matched_nodes": node_ids,
            "caller_count": len(callers),
            "callers": callers,
        }
