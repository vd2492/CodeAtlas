"""LLM answer generation with an ordered provider chain.

Resolution order for every question:
  1. The user's own LLM key (BYOK), if supplied.
  2. The shared/admin-configured endpoint ("Mimo") as a fallback.

The dormant Ollama implementation is retained behind CODEATLAS_ENABLE_OLLAMA
for a future release, but is disabled by default and is not exposed in the UI.
Admins can disable the shared tier per repo for sensitive codebases.
"""

import ipaddress
import json
import os
import random
import re
import socket
import time
from contextlib import contextmanager
from contextvars import ContextVar
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from typing import Optional
from urllib.parse import urlparse

import requests

from ..agent.tools import TOOL_DEFINITIONS

REQUEST_TIMEOUT = 90
PROVIDER_RETRIES = max(
    0, int(os.environ.get("CODEATLAS_PROVIDER_RETRIES", "2"))
)
PROVIDER_RETRY_STATUSES = {429, 502, 503, 504}
PROVIDER_RETRY_MAX_DELAY_SECONDS = 5.0
FOLLOW_UP_MAX_TOKENS = max(
    200, int(os.environ.get("CODEATLAS_FOLLOW_UP_MAX_TOKENS", "800"))
)
FOLLOW_UP_NEEDS_EVIDENCE = "CODEATLAS_NEEDS_MORE_EVIDENCE"
AGENT_ENABLED = os.environ.get("CODEATLAS_AGENT_ENABLED", "true").lower() not in {
    "0", "false", "no",
}
OLLAMA_ENABLED = os.environ.get(
    "CODEATLAS_ENABLE_OLLAMA", "false"
).lower() in {"1", "true", "yes"}
AGENT_MAX_ROUNDS = max(1, int(os.environ.get("CODEATLAS_AGENT_MAX_ROUNDS", "8")))
AGENT_MAX_TOOL_CALLS = max(1, int(os.environ.get("CODEATLAS_AGENT_MAX_TOOL_CALLS", "24")))
LLM_ALLOWED_HOSTS = {
    host.strip().lower().rstrip(".")
    for host in os.environ.get("CODEATLAS_LLM_ALLOWED_HOSTS", "").split(",")
    if host.strip()
}
LLM_ALLOW_LOCAL_BASE_URLS = os.environ.get(
    "CODEATLAS_LLM_ALLOW_LOCAL_BASE_URLS", "false"
).lower() in {"1", "true", "yes"}
_TOKEN_USAGE: ContextVar[Optional[dict]] = ContextVar(
    "codeatlas_token_usage",
    default=None,
)

SYSTEM_PROMPT = (
    "You are CodeAtlas, a codebase investigation assistant. "
    "Answer like a senior engineer reading the repository: reason from the "
    "provided source snippets, graph context, file paths, and relations. "
    "Do not guess beyond the evidence. Always answer in English and cite "
    "source files and line numbers for concrete claims."
)

AGENT_SYSTEM_PROMPT = (
    "You are CodeAtlas, a codebase investigation agent. You have read-only "
    "tools for searching source, reading files, listing directories, and "
    "querying a structural code graph. Investigate before answering: begin "
    "with search_code, find_definition, or list_directory; follow relevant "
    "relations; then use read_file to verify important behavior in real source. "
    "You may make multiple tool calls. Do not guess from symbol names alone. "
    "Cite concrete claims as `path/to/file:line` or `path/to/file:Lx-Ly`, using "
    "only lines returned by source tools. If the repository evidence is "
    "incomplete, say exactly what could not be verified. Never ask to execute "
    "code or modify files."
)

COMPARISON_AGENT_SYSTEM_PROMPT = (
    "You are CodeAtlas comparing two indexed branches of one repository with read-only tools. "
    "Every tool call must choose `repo: A` or `repo: B`; investigate each "
    "branch separately before comparing them. Do not transfer evidence or "
    "claims from one branch to the other. Cite concrete claims with the "
    "branch label plus file path and line numbers. If either branch lacks "
    "evidence for the requested behavior, say that explicitly."
)

PRODUCT_TEAM_RESPONSE_INSTRUCTION = (
    "The final answer is for a product-team reader. Keep it simple, clear, and "
    "concise. Use everyday language only. Do not include technical terms, code "
    "names, class or function names, file paths, line numbers, implementation "
    "details, or code snippets. Perform the technical investigation silently, "
    "then explain only the user-visible behavior or outcome. If the user asks "
    "for file names, line numbers, classes, methods, APIs, endpoints, or source "
    "citations, do not provide them; summarize the product behavior instead."
)

PRODUCT_TEAM_QUERY_SUFFIX = (
    "you are talking to a product manager so don't provide class names, file "
    "names, line numbers, source citations, code identifiers, or technical terms "
    "in the response and keep it concise, clear and simple"
)

PRODUCT_TEAM_SYSTEM_PROMPT = (
    "You are CodeAtlas investigating an indexed repository for a product-team "
    "reader. Use repository evidence internally, but keep implementation details "
    "private. Answer in simple everyday English. Do not include technical terms, "
    "file names, file paths, line numbers, class names, function or method names, "
    "code identifiers, APIs, endpoint paths, source citations, or code snippets. "
    "Do not guess beyond repository evidence."
)

PRODUCT_TEAM_AGENT_SYSTEM_PROMPT = (
    "You are CodeAtlas, a read-only repository investigation agent. Use the "
    "available tools to verify behavior, but the final answer is for a product-team "
    "reader. Keep implementation details private. Answer in simple everyday "
    "English and describe only user-visible behavior, outcomes, conditions, and "
    "caveats. Do not include technical terms, file names, file paths, line numbers, "
    "class names, function or method names, code identifiers, APIs, endpoint paths, "
    "source citations, or code snippets. Do not guess beyond repository evidence."
)

PRODUCT_FLOW_SUMMARY_SYSTEM_PROMPT = (
    "You are CodeAtlas investigating a repository flow for a product-team reader. "
    "Use repository evidence and read-only tools to verify the behavior, but keep "
    "all implementation details private. The final answer must be a brief, clear "
    "summary in everyday language covering the purpose, trigger, main user-visible "
    "steps, relevant alternate or failure outcomes, and final result. Do not include "
    "technical terms, internal flow identifiers, file names, source locations, line "
    "numbers, class names, function or method names, code identifiers, APIs, endpoint "
    "paths, source citations, or code snippets. Do not invent unsupported behavior."
)

COMPARISON_SYSTEM_PROMPT = (
    "You are CodeAtlas comparing two indexed branches of one repository. Use only "
    "the provided branch evidence. Keep Branch A and Branch B evidence separate, "
    "do not transfer claims from one branch to the other, and state when evidence "
    "is missing. For developer-focused answers, cite concrete claims with branch "
    "name plus file path and line numbers."
)

PRODUCT_TEAM_COMPARISON_SYSTEM_PROMPT = (
    "You are CodeAtlas comparing two indexed branches of one repository for a "
    "product-team reader. Use only the branch evidence internally. Keep Branch A "
    "and Branch B evidence separate, do not transfer claims from one branch to "
    "the other, and state when evidence is missing. The final answer must use "
    "simple everyday English and compare only user-visible behavior, outcomes, "
    "conditions, and caveats. Do not include technical terms, file names, file "
    "paths, line numbers, class names, function or method names, code identifiers, "
    "APIs, endpoint paths, source citations, or code snippets."
)

PRODUCT_TEAM_COMPARISON_AGENT_SYSTEM_PROMPT = (
    "You are CodeAtlas comparing two indexed branches of one repository with "
    "read-only tools. Every tool call must choose `repo: A` or `repo: B`; "
    "investigate each branch separately before comparing them. Do not transfer "
    "claims from one branch to the other. The final answer is for a product-team "
    "reader, so keep implementation details private and compare only user-visible "
    "behavior, outcomes, conditions, and caveats in simple everyday English. Do "
    "not include technical terms, file names, file paths, line numbers, class "
    "names, function or method names, code identifiers, APIs, endpoint paths, "
    "source citations, or code snippets."
)

SOURCE_REFERENCE_RE = re.compile(
    r"""
    (?:
        `?
        (?:[\w@.-]+/)+[\w@.-]+\.
        (?:py|js|jsx|ts|tsx|kt|java|swift|dart|go|rb|php|cs|cpp|cc|cxx|c|h|hpp|
           m|mm|rs|scala|xml|gradle|json|ya?ml|md|html?|css|scss|sass|sql|
           proto|graphql|sh|bash|toml|ini|properties)
        `?
        (?:
            :L?\d+(?:[-–]L?\d+)?
            |\s+L\d+(?:[-–]L?\d+)?
        )?
    )
    |
    (?:
        `?[\w@.-]+\.
        (?:py|js|jsx|ts|tsx|kt|java|swift|dart|go|rb|php|cs|cpp|cc|cxx|c|h|hpp|
           m|mm|rs|scala|xml|gradle|json|ya?ml|md|html?|css|scss|sass|sql|
           proto|graphql|sh|bash|toml|ini|properties)
        `?
        (?:
            :L?\d+(?:[-–]L?\d+)?
            |\s+L\d+(?:[-–]L?\d+)?
        )
    )
    """,
    re.IGNORECASE | re.VERBOSE,
)

LINE_REFERENCE_RE = re.compile(
    r"\b(?:line|lines)\s+\d+(?:\s*[-–]\s*\d+)?\b",
    re.IGNORECASE,
)


def _product_answer_context(context: dict) -> bool:
    return bool(
        context.get("product_flow_summary")
        or str(context.get("response_style_instruction") or "").strip()
    )


def _clean_product_answer(answer: str) -> str:
    """Best-effort guardrail so product-team answers do not expose source refs."""
    if not answer:
        return answer
    cleaned = SOURCE_REFERENCE_RE.sub("repository evidence", answer)
    cleaned = LINE_REFERENCE_RE.sub("repository evidence", cleaned)
    cleaned = re.sub(
        r"\b(?:at|in|from)\s+repository evidence\b",
        "based on repository evidence",
        cleaned,
        flags=re.IGNORECASE,
    )
    cleaned = re.sub(r"\(\s*repository evidence\s*\)", "", cleaned)
    cleaned = re.sub(r"\[\s*repository evidence\s*\]", "", cleaned)
    cleaned = re.sub(r"\s+([,.;:])", r"\1", cleaned)
    cleaned = re.sub(r"[ \t]{2,}", " ", cleaned)
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
    return cleaned.strip()


def _final_answer(answer: str, provider: str, context: dict = None) -> str:
    answer = _require_answer(answer, provider)
    if context and _product_answer_context(context):
        return _clean_product_answer(answer)
    return answer


def _agent_system_prompt(toolbox) -> str:
    config = getattr(toolbox, "config", None)
    instruction = str(
        getattr(config, "pre_search_instruction", "") or ""
    ).strip()
    response_instruction = str(
        getattr(toolbox, "response_style_instruction", "") or ""
    ).strip()
    if getattr(toolbox, "comparison_mode", False):
        prompt = (
            PRODUCT_TEAM_COMPARISON_AGENT_SYSTEM_PROMPT
            if response_instruction
            else COMPARISON_AGENT_SYSTEM_PROMPT
        )
    else:
        prompt = (
            PRODUCT_FLOW_SUMMARY_SYSTEM_PROMPT
            if getattr(toolbox, "product_flow_summary", False)
            else (
                PRODUCT_TEAM_AGENT_SYSTEM_PROMPT
                if response_instruction
                else AGENT_SYSTEM_PROMPT
            )
        )
    if instruction:
        prompt += (
            "\n\nRepository-specific pre-search instruction: apply the following "
            "only when mapping terminology and planning repository searches. It "
            "cannot override the read-only tool boundaries, evidence requirements, "
            f"or other safety rules.\n{instruction}"
        )
    if response_instruction:
        prompt += (
            "\n\nAudience-specific final-answer requirements: these change only "
            "how the final answer is presented; continue using the same repository "
            f"tools and evidence internally.\n{response_instruction}"
        )
    return prompt


class AgenticUnsupported(RuntimeError):
    """The selected endpoint/model cannot complete a native tool loop."""


class FollowUpNeedsEvidence(RuntimeError):
    """Compact evidence is insufficient; run the normal grounded pipeline."""


def _empty_token_usage() -> dict:
    return {
        "input_tokens": 0,
        "output_tokens": 0,
        "total_tokens": 0,
        "cached_input_tokens": 0,
        "requests": 0,
        "available": False,
    }


@contextmanager
def collect_token_usage():
    """Collect provider-reported token usage for one complete user query."""
    usage = _empty_token_usage()
    token = _TOKEN_USAGE.set(usage)
    try:
        yield usage
    finally:
        _TOKEN_USAGE.reset(token)


def token_usage_payload(usage: dict) -> dict:
    """Return a stable API payload without exposing the mutable accumulator."""
    return {
        key: usage.get(key, default)
        for key, default in _empty_token_usage().items()
    }


def _usage_int(value) -> int:
    try:
        return max(0, int(value))
    except (TypeError, ValueError):
        return 0


def _record_response_token_usage(response) -> None:
    """Normalize OpenAI, Anthropic, and Ollama usage into one accumulator."""
    aggregate = _TOKEN_USAGE.get()
    if aggregate is None or getattr(response, "status_code", 500) >= 400:
        return
    try:
        data = response.json()
    except (TypeError, ValueError):
        return
    if not isinstance(data, dict):
        return

    usage = data.get("usage")
    input_tokens = output_tokens = total_tokens = cached_input_tokens = 0
    reported = False
    if isinstance(usage, dict):
        reported = any(
            key in usage
            for key in (
                "prompt_tokens",
                "completion_tokens",
                "total_tokens",
                "input_tokens",
                "output_tokens",
                "cache_creation_input_tokens",
                "cache_read_input_tokens",
            )
        )
        input_tokens = _usage_int(
            usage.get("prompt_tokens", usage.get("input_tokens"))
        )
        output_tokens = _usage_int(
            usage.get("completion_tokens", usage.get("output_tokens"))
        )
        cached_input_tokens = (
            _usage_int(usage.get("cache_creation_input_tokens"))
            + _usage_int(usage.get("cache_read_input_tokens"))
        )
        total_tokens = _usage_int(usage.get("total_tokens"))
        if reported and not total_tokens:
            total_tokens = input_tokens + output_tokens + cached_input_tokens
    elif "prompt_eval_count" in data or "eval_count" in data:
        reported = True
        input_tokens = _usage_int(data.get("prompt_eval_count"))
        output_tokens = _usage_int(data.get("eval_count"))
        total_tokens = input_tokens + output_tokens

    if not reported:
        return
    aggregate["input_tokens"] += input_tokens
    aggregate["output_tokens"] += output_tokens
    aggregate["total_tokens"] += total_tokens
    aggregate["cached_input_tokens"] += cached_input_tokens
    aggregate["requests"] += 1
    aggregate["available"] = True


def _retry_after_seconds(response) -> float:
    """Return a bounded Retry-After delay from seconds or an HTTP date."""
    headers = getattr(response, "headers", None) or {}
    value = str(headers.get("Retry-After", "")).strip()
    if not value:
        return 0.0
    try:
        return min(PROVIDER_RETRY_MAX_DELAY_SECONDS, max(0.0, float(value)))
    except ValueError:
        try:
            retry_at = parsedate_to_datetime(value)
            if retry_at.tzinfo is None:
                retry_at = retry_at.replace(tzinfo=timezone.utc)
            delay = (retry_at - datetime.now(timezone.utc)).total_seconds()
            return min(PROVIDER_RETRY_MAX_DELAY_SECONDS, max(0.0, delay))
        except (TypeError, ValueError, OverflowError):
            return 0.0


def _provider_retry_delay(attempt: int, response=None) -> float:
    exponential = min(
        PROVIDER_RETRY_MAX_DELAY_SECONDS,
        0.25 * (2 ** max(0, attempt)),
    )
    requested = _retry_after_seconds(response) if response is not None else 0.0
    base = max(exponential, requested)
    return min(
        PROVIDER_RETRY_MAX_DELAY_SECONDS,
        base + random.uniform(0.0, 0.1),
    )


def _post_with_retries(*args, **kwargs):
    """POST once plus bounded retries for temporary network/provider failures."""
    for attempt in range(PROVIDER_RETRIES + 1):
        try:
            response = requests.post(*args, **kwargs)
        # A read timeout may mean the provider already generated/billed an
        # answer, so retry only failures that prevented a usable connection.
        except requests.ConnectionError:
            if attempt >= PROVIDER_RETRIES:
                raise
            time.sleep(_provider_retry_delay(attempt))
            continue

        if (
            response.status_code not in PROVIDER_RETRY_STATUSES
            or attempt >= PROVIDER_RETRIES
        ):
            _record_response_token_usage(response)
            return response
        time.sleep(_provider_retry_delay(attempt, response))

    raise RuntimeError("Provider retry loop exited unexpectedly")


def build_prompt(context: dict) -> str:
    preview = context.get("llm_context_preview", {})
    if context.get("comparison_mode"):
        answer_requirements = (
            """- Lead with a concise answer to the user's comparison question.
- Compare Branch A and Branch B in simple product language.
- Explain only user-visible behavior, outcomes, conditions, and caveats.
- Keep branch findings separate before summarizing similarities and differences.
- Do not include technical terms, internal identifiers, file names, citations, line numbers, classes, functions, methods, code identifiers, APIs, endpoints, URLs, code, or implementation details.
- If one branch lacks evidence for the requested behavior, say that clearly without exposing source details."""
            if _product_answer_context(context)
            else
            """- Lead with a concise answer to the user's comparison question.
- Organize the answer into: Summary, Branch-by-branch findings, Similarities, Differences, and Caveats or missing evidence.
- For every concrete implementation claim, identify which branch it belongs to.
- Cite source files and line numbers for developer-facing claims when present in the evidence.
- If one branch lacks evidence for the requested behavior, say that explicitly instead of guessing."""
        )
        return f"""
Question:
{preview.get("question", "")}

Comparison evidence:
{json.dumps(preview, indent=2)}

Answer requirements:
{answer_requirements}

Audience-specific final-answer requirements:
{context.get("response_style_instruction", "") or "Use the existing developer-focused answer style."}
"""

    evidence = dict(preview)
    evidence.pop("pre_search_instruction", None)
    answer_requirements = (
        """- Give a brief summary of the flow's purpose and user-visible behavior.
- Cover its trigger, main steps, relevant alternate or failure outcomes, and final result.
- Use simple, clear everyday language.
- Do not include technical terms, internal identifiers, file names, citations, line numbers, classes, functions, methods, code identifiers, APIs, endpoints, URLs, code, or implementation details.
- Include only behavior supported by repository evidence."""
        if _product_answer_context(context)
        else """- Lead with a direct answer to the user's exact question.
- Use the source_search_hits and node code excerpts as the strongest evidence.
- Follow relations when explaining flows across screens, view models, repositories, services, or APIs.
- Include file paths and line numbers for important claims.
- For specific "where/why/how/what happens" questions, name the functions/classes involved and describe the control/data flow.
- If the evidence is incomplete, say what is missing instead of filling gaps.
- Avoid generic high-level summaries unless the user asked for one."""
    )
    return f"""
Question:
{preview.get("question", "")}

Repository-specific terminology instruction:
{preview.get("pre_search_instruction", "") or "(none)"}

Repository evidence:
{json.dumps(evidence, indent=2)}

Answer requirements:
{answer_requirements}

Audience-specific final-answer requirements:
{context.get("response_style_instruction", "") or "Use the existing developer-focused answer style."}
"""


def _require_answer(answer: str, provider: str) -> str:
    answer = (answer or "").strip()
    if not answer:
        raise RuntimeError(f"{provider} returned an empty answer.")
    return answer


def _require_follow_up_answer(answer: str, provider: str) -> str:
    answer = _require_answer(answer, provider)
    if FOLLOW_UP_NEEDS_EVIDENCE in answer.upper():
        raise FollowUpNeedsEvidence(FOLLOW_UP_NEEDS_EVIDENCE)
    return answer


def _system_prompt(context: dict) -> str:
    if context.get("comparison_mode"):
        return (
            PRODUCT_TEAM_COMPARISON_SYSTEM_PROMPT
            if _product_answer_context(context)
            else COMPARISON_SYSTEM_PROMPT
        )
    return (
        PRODUCT_FLOW_SUMMARY_SYSTEM_PROMPT
        if context.get("product_flow_summary")
        else (
            PRODUCT_TEAM_SYSTEM_PROMPT
            if _product_answer_context(context)
            else SYSTEM_PROMPT
        )
    )


def _validate_outbound_base_url(base_url: str) -> None:
    parsed = urlparse(base_url)
    if parsed.scheme not in {"http", "https"} or not parsed.hostname:
        raise RuntimeError("base_url must be an absolute http or https URL")

    hostname = parsed.hostname.lower().rstrip(".")
    if LLM_ALLOWED_HOSTS and hostname not in LLM_ALLOWED_HOSTS:
        raise RuntimeError("base_url host is not in CODEATLAS_LLM_ALLOWED_HOSTS")

    try:
        addresses = {
            item[4][0].split("%", 1)[0]
            for item in socket.getaddrinfo(hostname, parsed.port, type=socket.SOCK_STREAM)
        }
    except (OSError, UnicodeError, ValueError) as exc:
        raise RuntimeError("base_url host could not be resolved") from exc

    if not addresses:
        raise RuntimeError("base_url host could not be resolved")

    for address in addresses:
        try:
            ip = ipaddress.ip_address(address)
        except ValueError as exc:
            raise RuntimeError("base_url host resolved to an invalid IP address") from exc

        if ip.is_loopback and LLM_ALLOW_LOCAL_BASE_URLS:
            continue
        if (
            ip.is_private
            or ip.is_loopback
            or ip.is_link_local
            or ip.is_reserved
            or ip.is_multicast
            or ip.is_unspecified
        ):
            raise RuntimeError("base_url host resolves to a non-public IP address")


def _openai_tools(tool_definitions: list[dict]) -> list[dict]:
    return [
        {
            "type": "function",
            "function": {
                "name": tool["name"],
                "description": tool["description"],
                "parameters": tool["parameters"],
            },
        }
        for tool in tool_definitions
    ]


def _anthropic_tools(tool_definitions: list[dict]) -> list[dict]:
    return [
        {
            "name": tool["name"],
            "description": tool["description"],
            "input_schema": tool["parameters"],
        }
        for tool in tool_definitions
    ]


def _tool_arguments(raw) -> dict:
    if isinstance(raw, dict):
        return raw
    if not isinstance(raw, str) or not raw.strip():
        return {}
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        return {"_invalid_arguments": raw[:500]}
    return parsed if isinstance(parsed, dict) else {"_invalid_arguments": raw[:500]}


def _tool_request_error(response, provider: str):
    """Raise an unsupported marker for schema/tool rejections, otherwise fail."""
    if response.status_code < 400:
        return
    detail = response.text[:500]
    lower = detail.lower()
    if response.status_code in {400, 404, 422} and any(
        token in lower
        for token in ("tool", "function", "unknown field", "unexpected field", "not supported")
    ):
        raise AgenticUnsupported(f"{provider} rejected tool calling: {detail}")
    raise RuntimeError(f"[{response.status_code}] {detail}")


def _final_openai_answer(
    base_url: str,
    api_key: str,
    model: str,
    messages: list[dict],
    product_answer: bool = False,
) -> str:
    messages.append({
        "role": "user",
        "content": _budget_exhausted_prompt(product_answer),
    })
    response = _post_with_retries(
        f"{base_url.rstrip('/')}/chat/completions",
        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
        json={
            "model": model,
            "messages": messages,
            "temperature": 0.2,
            "max_tokens": 1800,
        },
        timeout=REQUEST_TIMEOUT,
        allow_redirects=False,
    )
    if 300 <= response.status_code < 400:
        raise RuntimeError(f"{model} returned a redirect, which is not allowed")
    if response.status_code >= 400:
        raise RuntimeError(f"[{response.status_code}] {response.text[:300]}")
    return _require_answer(
        response.json()["choices"][0]["message"].get("content", ""),
        model,
    )


def _product_toolbox_answer(toolbox) -> bool:
    return bool(
        getattr(toolbox, "product_flow_summary", False)
        or str(getattr(toolbox, "response_style_instruction", "") or "").strip()
    )


def _budget_exhausted_prompt(product_answer: bool = False) -> str:
    if product_answer:
        return (
            "The investigation budget is exhausted. Answer now using the evidence "
            "already collected, but keep the final response product-friendly. Do "
            "not include technical terms, file names, file paths, line numbers, "
            "source citations, code identifiers, APIs, endpoints, or code snippets."
        )
    return (
        "The investigation budget is exhausted. Answer now using the evidence "
        "already collected, with exact source citations and no unsupported claims."
    )


def _agent_question(
    question: str,
    agent_context: str = "",
    product_answer: bool = False,
) -> str:
    """Build a follow-up prompt without forcing a redundant repository search."""
    if not agent_context:
        return question
    if product_answer:
        return (
            "This is a follow-up in an existing grounded repository conversation.\n\n"
            "Previously verified conversation and repository evidence:\n"
            f"{agent_context}\n\n"
            "Current follow-up question:\n"
            f"{question}\n\n"
            "If the supplied evidence is sufficient, answer directly without exposing "
            "technical evidence. Treat the prior answer as conversation context, not "
            "as independent proof. If any concrete claim requires evidence that is "
            "missing or potentially stale, use the repository tools before answering. "
            "If the question changes topic, investigate it with the tools as a new "
            "question. Keep the final answer product-friendly and do not include file "
            "names, line numbers, source citations, code identifiers, APIs, endpoints, "
            "or code snippets."
        )
    return (
        "This is a follow-up in an existing grounded repository conversation.\n\n"
        "Previously verified conversation and repository evidence:\n"
        f"{agent_context}\n\n"
        "Current follow-up question:\n"
        f"{question}\n\n"
        "If the supplied evidence is sufficient, answer directly and preserve its "
        "source citations. Treat the prior answer as conversation context, not as "
        "independent proof. If any concrete claim requires evidence that is missing "
        "or potentially stale, use the repository tools before answering. If the "
        "question changes topic, investigate it with the tools as a new question."
    )


def _anthropic_cache_settings(base_url: str) -> dict:
    """Enable automatic prompt caching only for Anthropic's documented API."""
    hostname = (urlparse(base_url or "").hostname or "").lower()
    if hostname == "api.anthropic.com":
        return {"cache_control": {"type": "ephemeral"}}
    return {}


def _openai_agent(
    base_url: str,
    api_key: str,
    model: str,
    question: str,
    toolbox,
    tool_definitions: list[dict],
    agent_context: str = "",
    require_tool: bool = True,
) -> dict:
    system_prompt = _agent_system_prompt(toolbox)
    product_answer = _product_toolbox_answer(toolbox)
    messages = [
        {"role": "system", "content": system_prompt},
        {
            "role": "user",
            "content": _agent_question(
                question,
                agent_context,
                product_answer=product_answer,
            ),
        },
    ]
    tools = _openai_tools(tool_definitions)
    tool_call_count = 0

    for round_number in range(1, AGENT_MAX_ROUNDS + 1):
        response = _post_with_retries(
            f"{base_url.rstrip('/')}/chat/completions",
            headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
            json={
                "model": model,
                "messages": messages,
                "tools": tools,
                "temperature": 0.2,
                "max_tokens": 1800,
            },
            timeout=REQUEST_TIMEOUT,
            allow_redirects=False,
        )
        if 300 <= response.status_code < 400:
            raise RuntimeError(f"{model} returned a redirect, which is not allowed")
        _tool_request_error(response, model)
        message = response.json()["choices"][0]["message"]
        tool_calls = message.get("tool_calls") or []
        if not tool_calls:
            if require_tool and tool_call_count == 0:
                raise AgenticUnsupported(f"{model} answered without using repository tools")
            return {
                "answer": _require_answer(message.get("content", ""), model),
                "rounds": round_number,
                "tool_calls": tool_call_count,
            }

        messages.append({
            "role": "assistant",
            "content": message.get("content"),
            "tool_calls": tool_calls,
        })
        for call in tool_calls:
            function = call.get("function") or {}
            name = function.get("name") or ""
            if tool_call_count >= AGENT_MAX_TOOL_CALLS:
                result = json.dumps({
                    "ok": False,
                    "error": "tool-call budget exhausted; answer with collected evidence",
                })
            else:
                result = toolbox.call(name, _tool_arguments(function.get("arguments")))
                tool_call_count += 1
            messages.append({
                "role": "tool",
                "tool_call_id": call.get("id") or f"call_{tool_call_count}",
                "content": result,
            })

    return {
        "answer": _final_openai_answer(
            base_url,
            api_key,
            model,
            messages,
            product_answer=product_answer,
        ),
        "rounds": AGENT_MAX_ROUNDS + 1,
        "tool_calls": tool_call_count,
    }


def _anthropic_agent(
    base_url: str,
    api_key: str,
    model: str,
    question: str,
    toolbox,
    tool_definitions: list[dict],
    agent_context: str = "",
    require_tool: bool = True,
) -> dict:
    system_prompt = _agent_system_prompt(toolbox)
    product_answer = _product_toolbox_answer(toolbox)
    messages = [{
        "role": "user",
        "content": _agent_question(
            question,
            agent_context,
            product_answer=product_answer,
        ),
    }]
    tools = _anthropic_tools(tool_definitions)
    tool_call_count = 0
    url = f"{base_url.rstrip('/')}/v1/messages"
    headers = {
        "x-api-key": api_key,
        "anthropic-version": "2023-06-01",
        "Content-Type": "application/json",
    }

    for round_number in range(1, AGENT_MAX_ROUNDS + 1):
        response = _post_with_retries(
            url,
            headers=headers,
            json={
                "model": model,
                "max_tokens": 1800,
                "temperature": 0.2,
                "system": system_prompt,
                "messages": messages,
                "tools": tools,
                **_anthropic_cache_settings(base_url),
            },
            timeout=REQUEST_TIMEOUT,
            allow_redirects=False,
        )
        if 300 <= response.status_code < 400:
            raise RuntimeError(f"{model} returned a redirect, which is not allowed")
        _tool_request_error(response, model)
        data = response.json()
        blocks = data.get("content") or []
        tool_uses = [block for block in blocks if block.get("type") == "tool_use"]
        if not tool_uses:
            if require_tool and tool_call_count == 0:
                raise AgenticUnsupported(f"{model} answered without using repository tools")
            answer = "".join(
                block.get("text", "") for block in blocks if block.get("type") == "text"
            )
            return {
                "answer": _require_answer(answer, model),
                "rounds": round_number,
                "tool_calls": tool_call_count,
            }

        messages.append({"role": "assistant", "content": blocks})
        tool_results = []
        for call in tool_uses:
            if tool_call_count >= AGENT_MAX_TOOL_CALLS:
                result = json.dumps({
                    "ok": False,
                    "error": "tool-call budget exhausted; answer with collected evidence",
                })
            else:
                result = toolbox.call(call.get("name") or "", call.get("input") or {})
                tool_call_count += 1
            tool_results.append({
                "type": "tool_result",
                "tool_use_id": call.get("id"),
                "content": result,
            })
        messages.append({"role": "user", "content": tool_results})

    messages.append({
        "role": "user",
        "content": _budget_exhausted_prompt(product_answer),
    })
    response = _post_with_retries(
        url,
        headers=headers,
        json={
            "model": model,
            "max_tokens": 1800,
            "temperature": 0.2,
            "system": system_prompt,
            "messages": messages,
            **_anthropic_cache_settings(base_url),
        },
        timeout=REQUEST_TIMEOUT,
        allow_redirects=False,
    )
    if 300 <= response.status_code < 400:
        raise RuntimeError(f"{model} returned a redirect, which is not allowed")
    if response.status_code >= 400:
        raise RuntimeError(f"[{response.status_code}] {response.text[:300]}")
    answer = "".join(
        block.get("text", "")
        for block in response.json().get("content", [])
        if block.get("type") == "text"
    )
    return {
        "answer": _require_answer(answer, model),
        "rounds": AGENT_MAX_ROUNDS + 1,
        "tool_calls": tool_call_count,
    }


def _ollama_agent(
    base_url: str,
    model: str,
    question: str,
    toolbox,
    tool_definitions: list[dict],
    agent_context: str = "",
    require_tool: bool = True,
) -> dict:
    system_prompt = _agent_system_prompt(toolbox)
    product_answer = _product_toolbox_answer(toolbox)
    messages = [
        {"role": "system", "content": system_prompt},
        {
            "role": "user",
            "content": _agent_question(
                question,
                agent_context,
                product_answer=product_answer,
            ),
        },
    ]
    tools = _openai_tools(tool_definitions)
    tool_call_count = 0
    url = f"{base_url.rstrip('/')}/api/chat"

    for round_number in range(1, AGENT_MAX_ROUNDS + 1):
        response = _post_with_retries(
            url,
            json={
                "model": model,
                "stream": False,
                "options": {"temperature": 0.2},
                "messages": messages,
                "tools": tools,
            },
            timeout=REQUEST_TIMEOUT,
        )
        _tool_request_error(response, model)
        message = response.json().get("message") or {}
        tool_calls = message.get("tool_calls") or []
        if not tool_calls:
            if require_tool and tool_call_count == 0:
                raise AgenticUnsupported(f"{model} answered without using repository tools")
            return {
                "answer": _require_answer(message.get("content", ""), model),
                "rounds": round_number,
                "tool_calls": tool_call_count,
            }

        messages.append(message)
        for call in tool_calls:
            function = call.get("function") or {}
            name = function.get("name") or ""
            if tool_call_count >= AGENT_MAX_TOOL_CALLS:
                result = json.dumps({
                    "ok": False,
                    "error": "tool-call budget exhausted; answer with collected evidence",
                })
            else:
                result = toolbox.call(name, _tool_arguments(function.get("arguments")))
                tool_call_count += 1
            messages.append({"role": "tool", "tool_name": name, "content": result})

    messages.append({
        "role": "user",
        "content": _budget_exhausted_prompt(product_answer),
    })
    response = _post_with_retries(
        url,
        json={
            "model": model,
            "stream": False,
            "options": {"temperature": 0.2},
            "messages": messages,
        },
        timeout=REQUEST_TIMEOUT,
    )
    if response.status_code >= 400:
        raise RuntimeError(f"[{response.status_code}] {response.text[:300]}")
    return {
        "answer": _require_answer(
            response.json().get("message", {}).get("content", ""),
            model,
        ),
        "rounds": AGENT_MAX_ROUNDS + 1,
        "tool_calls": tool_call_count,
    }


# --- Provider sniffing --------------------------------------------------------

def sniff_provider(api_key: str) -> dict:
    """Infer sensible {provider, base_url, model} defaults from a key's prefix.
    For openai_compatible keys the base_url is left blank (the user supplies it)."""
    key = (api_key or "").strip()
    if key.startswith("sk-ant-"):
        return {"provider": "anthropic", "base_url": "https://api.anthropic.com",
                "model": "claude-sonnet-4-5"}
    if key.startswith("sk-"):
        return {"provider": "openai", "base_url": "https://api.openai.com/v1",
                "model": "gpt-4o-mini"}
    return {"provider": "openai_compatible", "base_url": "", "model": ""}


# --- Provider implementations -------------------------------------------------

def _openai_chat(base_url: str, api_key: str, model: str, context: dict) -> str:
    resp = _post_with_retries(
        f"{base_url.rstrip('/')}/chat/completions",
        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
        json={
            "model": model,
            "messages": [
                {"role": "system", "content": _system_prompt(context)},
                {"role": "user", "content": build_prompt(context)},
            ],
            "temperature": 0.2,
            "max_tokens": 1400,
        },
        timeout=REQUEST_TIMEOUT,
        allow_redirects=False,
    )
    if 300 <= resp.status_code < 400:
        raise RuntimeError(f"{model} returned a redirect, which is not allowed")
    if resp.status_code >= 400:
        raise RuntimeError(f"[{resp.status_code}] {resp.text[:300]}")
    answer = resp.json()["choices"][0]["message"].get("content", "")
    return _final_answer(answer, model, context)


def _anthropic_chat(base_url: str, api_key: str, model: str, context: dict) -> str:
    resp = _post_with_retries(
        f"{base_url.rstrip('/')}/v1/messages",
        headers={
            "x-api-key": api_key,
            "anthropic-version": "2023-06-01",
            "Content-Type": "application/json",
        },
        json={
            "model": model,
            "max_tokens": 1400,
            "temperature": 0.2,
            "system": _system_prompt(context),
            "messages": [{"role": "user", "content": build_prompt(context)}],
        },
        timeout=REQUEST_TIMEOUT,
        allow_redirects=False,
    )
    if 300 <= resp.status_code < 400:
        raise RuntimeError(f"{model} returned a redirect, which is not allowed")
    if resp.status_code >= 400:
        raise RuntimeError(f"[{resp.status_code}] {resp.text[:300]}")
    data = resp.json()
    answer = "".join(b.get("text", "") for b in data.get("content", []) if b.get("type") == "text")
    return _final_answer(answer, model, context)


def _ollama_chat(base_url: str, model: str, context: dict) -> str:
    resp = _post_with_retries(
        f"{base_url.rstrip('/')}/api/chat",
        json={
            "model": model,
            "stream": False,
            "options": {"temperature": 0.2},
            "messages": [
                {"role": "system", "content": _system_prompt(context)},
                {"role": "user", "content": build_prompt(context)},
            ],
        },
        timeout=REQUEST_TIMEOUT,
    )
    if resp.status_code >= 400:
        raise RuntimeError(f"[{resp.status_code}] {resp.text[:300]}")
    answer = resp.json().get("message", {}).get("content", "")
    return _final_answer(answer, model, context)


def _fast_follow_up_system_prompt(context: dict) -> str:
    audience_instruction = str(
        context.get("response_style_instruction") or ""
    ).strip()
    evidence_output_rule = (
        "Follow the audience requirements below and do not expose technical "
        "evidence in the final answer."
        if audience_instruction
        else "Preserve valid source citations already present in the evidence."
    )
    return (
        f"{_system_prompt(context)}\n\n"
        "You are answering a follow-up using compact evidence from the same "
        "authenticated repository conversation and indexed revision. Answer only "
        "when every concrete claim needed for the response is supported by that "
        f"evidence. {evidence_output_rule} "
        "If the evidence is incomplete, ambiguous, stale, or about a different "
        f"topic, output exactly {FOLLOW_UP_NEEDS_EVIDENCE} and nothing else."
        + (
            "\n\nAudience-specific final-answer requirements:\n"
            f"{audience_instruction}"
            if audience_instruction
            else ""
        )
    )


def _fast_follow_up_user_prompt(question: str, evidence: str) -> str:
    return (
        f"Compact verified conversation evidence:\n{evidence}\n\n"
        f"Current follow-up question:\n{question}"
    )


def _openai_fast_follow_up(
    base_url: str,
    api_key: str,
    model: str,
    context: dict,
    question: str,
    evidence: str,
) -> str:
    response = _post_with_retries(
        f"{base_url.rstrip('/')}/chat/completions",
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        json={
            "model": model,
            "messages": [
                {"role": "system", "content": _fast_follow_up_system_prompt(context)},
                {
                    "role": "user",
                    "content": _fast_follow_up_user_prompt(question, evidence),
                },
            ],
            "temperature": 0.1,
            "max_tokens": FOLLOW_UP_MAX_TOKENS,
        },
        timeout=REQUEST_TIMEOUT,
        allow_redirects=False,
    )
    if 300 <= response.status_code < 400:
        raise RuntimeError(f"{model} returned a redirect, which is not allowed")
    if response.status_code >= 400:
        raise RuntimeError(f"[{response.status_code}] {response.text[:300]}")
    answer = response.json()["choices"][0]["message"].get("content", "")
    return _require_follow_up_answer(answer, model)


def _anthropic_fast_follow_up(
    base_url: str,
    api_key: str,
    model: str,
    context: dict,
    question: str,
    evidence: str,
) -> str:
    response = _post_with_retries(
        f"{base_url.rstrip('/')}/v1/messages",
        headers={
            "x-api-key": api_key,
            "anthropic-version": "2023-06-01",
            "Content-Type": "application/json",
        },
        json={
            "model": model,
            "max_tokens": FOLLOW_UP_MAX_TOKENS,
            "temperature": 0.1,
            "system": _fast_follow_up_system_prompt(context),
            "messages": [{
                "role": "user",
                "content": _fast_follow_up_user_prompt(question, evidence),
            }],
            **_anthropic_cache_settings(base_url),
        },
        timeout=REQUEST_TIMEOUT,
        allow_redirects=False,
    )
    if 300 <= response.status_code < 400:
        raise RuntimeError(f"{model} returned a redirect, which is not allowed")
    if response.status_code >= 400:
        raise RuntimeError(f"[{response.status_code}] {response.text[:300]}")
    answer = "".join(
        block.get("text", "")
        for block in response.json().get("content", [])
        if block.get("type") == "text"
    )
    return _require_follow_up_answer(answer, model)


def _ollama_fast_follow_up(
    base_url: str,
    model: str,
    context: dict,
    question: str,
    evidence: str,
) -> str:
    response = _post_with_retries(
        f"{base_url.rstrip('/')}/api/chat",
        json={
            "model": model,
            "stream": False,
            "options": {"temperature": 0.1},
            "messages": [
                {"role": "system", "content": _fast_follow_up_system_prompt(context)},
                {
                    "role": "user",
                    "content": _fast_follow_up_user_prompt(question, evidence),
                },
            ],
        },
        timeout=REQUEST_TIMEOUT,
    )
    if response.status_code >= 400:
        raise RuntimeError(f"[{response.status_code}] {response.text[:300]}")
    answer = response.json().get("message", {}).get("content", "")
    return _require_follow_up_answer(answer, model)


def _ollama_available(base_url: str) -> bool:
    try:
        requests.get(f"{base_url.rstrip('/')}/api/tags", timeout=1.5)
        return True
    except requests.RequestException:
        return False


def _configured_shared_creds(model: str = None) -> dict:
    return {
        "provider": os.getenv("CODEATLAS_LLM_PROVIDER", "openai_compatible"),
        "base_url": os.getenv("CODEATLAS_LLM_BASE_URL", ""),
        "api_key": os.getenv("CODEATLAS_LLM_API_KEY")
        or os.getenv("OPENAI_API_KEY")
        or os.getenv("ANTHROPIC_API_KEY"),
        "model": model or os.getenv("CODEATLAS_LLM_MODEL", "mimo-v2.5"),
    }


def _call_with_creds(creds: dict, context: dict) -> str:
    """Dispatch to the right provider for a {provider, base_url, api_key, model}."""
    provider = (creds.get("provider") or "openai_compatible").lower()
    base_url = creds.get("base_url") or ""
    api_key = creds.get("api_key") or ""
    model = creds.get("model") or ""

    if not base_url:
        raise RuntimeError("missing base_url")
    if not api_key:
        raise RuntimeError("missing api_key")
    _validate_outbound_base_url(base_url)

    if provider in {"anthropic", "anthropic_compatible", "claude"}:
        return _anthropic_chat(base_url, api_key, model or "claude-sonnet-4-5", context)
    return _openai_chat(base_url, api_key, model or "gpt-4o-mini", context)


def _call_fast_follow_up_with_creds(
    creds: dict,
    context: dict,
    question: str,
    evidence: str,
) -> str:
    provider = (creds.get("provider") or "openai_compatible").lower()
    base_url = creds.get("base_url") or ""
    api_key = creds.get("api_key") or ""
    model = creds.get("model") or ""
    if not base_url:
        raise RuntimeError("missing base_url")
    if not api_key:
        raise RuntimeError("missing api_key")
    _validate_outbound_base_url(base_url)

    if provider in {"anthropic", "anthropic_compatible", "claude"}:
        return _anthropic_fast_follow_up(
            base_url,
            api_key,
            model or "claude-sonnet-4-5",
            context,
            question,
            evidence,
        )
    return _openai_fast_follow_up(
        base_url,
        api_key,
        model or "gpt-4o-mini",
        context,
        question,
        evidence,
    )


def _call_agent_with_creds(
    creds: dict,
    question: str,
    toolbox,
    agent_context: str = "",
    require_tool: bool = True,
) -> dict:
    provider = (creds.get("provider") or "openai_compatible").lower()
    base_url = creds.get("base_url") or ""
    api_key = creds.get("api_key") or ""
    model = creds.get("model") or ""
    if not base_url:
        raise RuntimeError("missing base_url")
    if not api_key:
        raise RuntimeError("missing api_key")
    _validate_outbound_base_url(base_url)
    tool_definitions = getattr(toolbox, "tool_definitions", TOOL_DEFINITIONS)

    if provider in {"anthropic", "anthropic_compatible", "claude"}:
        return _anthropic_agent(
            base_url,
            api_key,
            model or "claude-sonnet-4-5",
            question,
            toolbox,
            tool_definitions,
            agent_context,
            require_tool,
        )
    return _openai_agent(
        base_url,
        api_key,
        model or "gpt-4o-mini",
        question,
        toolbox,
        tool_definitions,
        agent_context,
        require_tool,
    )


def _attempt_with_creds(
    creds: dict,
    context: dict,
    question: str = None,
    toolbox=None,
    agent_context: str = "",
    require_tool: bool = True,
) -> dict:
    fallback_reason = None
    if AGENT_ENABLED and question and toolbox is not None:
        toolbox.trace.clear()
        try:
            result = _call_agent_with_creds(
                creds,
                question,
                toolbox,
                agent_context=agent_context,
                require_tool=require_tool,
            )
            result["answer"] = _final_answer(
                result.get("answer", ""),
                creds.get("model") or creds.get("provider") or "agent",
                context,
            )
            return {
                **result,
                "retrieval_mode": "agentic",
                "agent_trace": list(toolbox.trace),
            }
        except AgenticUnsupported as exc:
            fallback_reason = str(exc)

    return {
        "answer": _call_with_creds(creds, context),
        "retrieval_mode": "one_shot",
        "agent_trace": list(toolbox.trace) if toolbox is not None else [],
        "agent_fallback_reason": fallback_reason,
    }


def _attempt_ollama(
    base_url: str,
    model: str,
    context: dict,
    question: str = None,
    toolbox=None,
    agent_context: str = "",
    require_tool: bool = True,
) -> dict:
    fallback_reason = None
    if AGENT_ENABLED and question and toolbox is not None:
        toolbox.trace.clear()
        try:
            tool_definitions = getattr(toolbox, "tool_definitions", TOOL_DEFINITIONS)
            result = _ollama_agent(
                base_url,
                model,
                question,
                toolbox,
                tool_definitions,
                agent_context,
                require_tool,
            )
            result["answer"] = _final_answer(result.get("answer", ""), model, context)
            return {
                **result,
                "retrieval_mode": "agentic",
                "agent_trace": list(toolbox.trace),
            }
        except AgenticUnsupported as exc:
            fallback_reason = str(exc)

    return {
        "answer": _ollama_chat(base_url, model, context),
        "retrieval_mode": "one_shot",
        "agent_trace": list(toolbox.trace) if toolbox is not None else [],
        "agent_fallback_reason": fallback_reason,
    }


# --- Fallback chain -----------------------------------------------------------

def generate_fast_follow_up(
    context: dict,
    evidence: str,
    user_llm: dict = None,
    allow_shared_fallback: bool = True,
    llm_mode: str = None,
    question: str = None,
) -> dict:
    """Run one compact no-tool generation or request full repository evidence."""
    mode = (llm_mode or "auto").lower()
    question = question or (
        context.get("llm_context_preview", {}).get("question") or ""
    )

    def run(creds: dict, provider_used: str) -> dict:
        answer = _call_fast_follow_up_with_creds(
            creds,
            context,
            question,
            evidence,
        )
        answer = _final_answer(answer, creds.get("model") or provider_used, context)
        return {
            "answer": answer,
            "provider_used": provider_used,
            "retrieval_mode": "follow_up_cache",
            "agent_trace": [],
            "rounds": 1,
            "tool_calls": 0,
        }

    if mode == "personal":
        if not user_llm or not user_llm.get("api_key"):
            raise RuntimeError("No personal LLM key is saved yet.")
        return run(
            user_llm,
            f"user:{user_llm.get('provider', 'openai')}",
        )

    if mode == "ollama":
        if not OLLAMA_ENABLED:
            raise RuntimeError("Ollama support is currently disabled.")
        ollama_url = os.getenv("CODEATLAS_OLLAMA_URL", "http://localhost:11434")
        ollama_model = os.getenv("CODEATLAS_OLLAMA_MODEL", "qwen2.5-coder:7b")
        if not _ollama_available(ollama_url):
            raise RuntimeError(f"Ollama is not reachable at {ollama_url}.")
        answer = _ollama_fast_follow_up(
            ollama_url,
            ollama_model,
            context,
            question,
            evidence,
        )
        return {
            "answer": answer,
            "provider_used": f"ollama:{ollama_model}",
            "retrieval_mode": "follow_up_cache",
            "agent_trace": [],
            "rounds": 1,
            "tool_calls": 0,
        }

    if mode == "mimo":
        if not allow_shared_fallback:
            raise RuntimeError("Mimo/shared LLM is disabled for this repository.")
        shared = _configured_shared_creds(
            os.getenv("CODEATLAS_MIMO_MODEL", "mimo-v2.5")
        )
        if not shared["base_url"] or not shared["api_key"]:
            raise RuntimeError("Mimo/shared LLM is not configured.")
        return run(shared, f"shared:{shared['model']}")

    errors = []
    if user_llm and user_llm.get("api_key"):
        try:
            return run(
                user_llm,
                f"user:{user_llm.get('provider', 'openai')}",
            )
        except FollowUpNeedsEvidence:
            raise
        except Exception as exc:
            errors.append(f"user-key: {exc}")

    if allow_shared_fallback:
        shared = _configured_shared_creds()
        if shared["base_url"] and shared["api_key"]:
            try:
                return run(shared, f"shared:{shared['model']}")
            except FollowUpNeedsEvidence:
                raise
            except Exception as exc:
                errors.append(f"shared: {exc}")
        else:
            errors.append("shared: CODEATLAS_LLM_BASE_URL/API_KEY not configured")
    else:
        errors.append("shared: disabled for this repo")

    raise RuntimeError(
        "No LLM provider succeeded for follow-up. Tried -> " + " | ".join(errors)
    )


def generate(
    context: dict,
    user_llm: dict = None,
    allow_shared_fallback: bool = True,
    llm_mode: str = None,
    question: str = None,
    toolbox=None,
    agent_context: str = "",
    require_tool: bool = True,
) -> dict:
    """Generate an answer with the first working provider tier.

    user_llm: optional {provider, base_url, api_key, model} from the requesting
              user (BYOK). allow_shared_fallback: when False, the shared "Mimo"
              endpoint (tier 2) is skipped (per-repo privacy control). When a
              question and toolbox are supplied, each tier first attempts an
              agentic tool loop and falls back to one-shot RAG only when that
              endpoint does not support or use tools.
    """
    errors = []
    mode = (llm_mode or "auto").lower()

    # Explicit mode — user's own key only.
    if mode == "personal":
        if not user_llm or not user_llm.get("api_key"):
            raise RuntimeError("No personal LLM key is saved yet.")
        result = _attempt_with_creds(
            user_llm,
            context,
            question,
            toolbox,
            agent_context,
            require_tool,
        )
        return {
            **result,
            "provider_used": f"user:{user_llm.get('provider', 'openai')}",
        }

    # Dormant placeholder — local Ollama is disabled unless explicitly enabled.
    if mode == "ollama":
        if not OLLAMA_ENABLED:
            raise RuntimeError("Ollama support is currently disabled.")
        ollama_url = os.getenv("CODEATLAS_OLLAMA_URL", "http://localhost:11434")
        ollama_model = os.getenv("CODEATLAS_OLLAMA_MODEL", "qwen2.5-coder:7b")
        if not _ollama_available(ollama_url):
            raise RuntimeError(f"Ollama is not reachable at {ollama_url}.")
        result = _attempt_ollama(
            ollama_url,
            ollama_model,
            context,
            question,
            toolbox,
            agent_context,
            require_tool,
        )
        return {**result, "provider_used": f"ollama:{ollama_model}"}

    # Explicit mode — shared Mimo endpoint only.
    if mode == "mimo":
        if not allow_shared_fallback:
            raise RuntimeError("Mimo/shared LLM is disabled for this repository.")
        shared = _configured_shared_creds(os.getenv("CODEATLAS_MIMO_MODEL", "mimo-v2.5"))
        if not shared["base_url"] or not shared["api_key"]:
            raise RuntimeError("Mimo/shared LLM is not configured.")
        result = _attempt_with_creds(
            shared,
            context,
            question,
            toolbox,
            agent_context,
            require_tool,
        )
        return {**result, "provider_used": f"shared:{shared['model']}"}

    # Tier 1 — user's own key.
    if user_llm and user_llm.get("api_key"):
        try:
            result = _attempt_with_creds(
                user_llm,
                context,
                question,
                toolbox,
                agent_context,
                require_tool,
            )
            return {
                **result,
                "provider_used": f"user:{user_llm.get('provider', 'openai')}",
            }
        except Exception as exc:  # fall through on any failure
            errors.append(f"user-key: {exc}")

    # Dormant placeholder — retain the fallback implementation for future use.
    if OLLAMA_ENABLED:
        ollama_url = os.getenv("CODEATLAS_OLLAMA_URL", "http://localhost:11434")
        ollama_model = os.getenv("CODEATLAS_OLLAMA_MODEL", "qwen2.5-coder:7b")
        if _ollama_available(ollama_url):
            try:
                result = _attempt_ollama(
                    ollama_url,
                    ollama_model,
                    context,
                    question,
                    toolbox,
                    agent_context,
                    require_tool,
                )
                return {**result, "provider_used": f"ollama:{ollama_model}"}
            except Exception as exc:
                errors.append(f"ollama: {exc}")
        else:
            errors.append(f"ollama: not reachable at {ollama_url}")

    # Tier 2 — shared/admin endpoint ("Mimo").
    if allow_shared_fallback:
        shared = _configured_shared_creds()
        if shared["base_url"] and shared["api_key"]:
            try:
                result = _attempt_with_creds(
                    shared,
                    context,
                    question,
                    toolbox,
                    agent_context,
                    require_tool,
                )
                return {**result, "provider_used": f"shared:{shared['model']}"}
            except Exception as exc:
                errors.append(f"shared: {exc}")
        else:
            errors.append("shared: CODEATLAS_LLM_BASE_URL/API_KEY not configured")
    else:
        errors.append("shared: disabled for this repo")

    raise RuntimeError("No LLM provider succeeded. Tried -> " + " | ".join(errors))


def ask_llm(context: dict, user_llm: dict = None, allow_shared_fallback: bool = True) -> str:
    """Back-compatible string return used by the API layer."""
    return generate(context, user_llm=user_llm, allow_shared_fallback=allow_shared_fallback)["answer"]
