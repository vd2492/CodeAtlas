"""LLM answer generation with an ordered fallback chain.

Resolution order for every question:
  1. The user's own LLM key (BYOK), if supplied.
  2. A locally running Ollama model (free, private), if reachable.
  3. The shared/admin-configured endpoint ("Mimo") as a last resort.

Each tier falls through to the next on absence OR failure (bad key, rate
limit, timeout). Admins can disable tier 3 per repo for sensitive codebases so
private code is never sent to the shared endpoint.
"""

import json
import os

import requests

from ..agent.tools import TOOL_DEFINITIONS

REQUEST_TIMEOUT = 90
AGENT_ENABLED = os.environ.get("CODEATLAS_AGENT_ENABLED", "true").lower() not in {
    "0", "false", "no",
}
AGENT_MAX_ROUNDS = max(1, int(os.environ.get("CODEATLAS_AGENT_MAX_ROUNDS", "8")))
AGENT_MAX_TOOL_CALLS = max(1, int(os.environ.get("CODEATLAS_AGENT_MAX_TOOL_CALLS", "24")))

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


class AgenticUnsupported(RuntimeError):
    """The selected endpoint/model cannot complete a native tool loop."""


def build_prompt(context: dict) -> str:
    preview = context.get("llm_context_preview", {})
    return f"""
Question:
{preview.get("question", "")}

Repository evidence:
{json.dumps(preview, indent=2)}

Answer requirements:
- Lead with a direct answer to the user's exact question.
- Use the source_search_hits and node code excerpts as the strongest evidence.
- Follow relations when explaining flows across screens, view models, repositories, services, or APIs.
- Include file paths and line numbers for important claims.
- For specific "where/why/how/what happens" questions, name the functions/classes involved and describe the control/data flow.
- If the evidence is incomplete, say what is missing instead of filling gaps.
- Avoid generic high-level summaries unless the user asked for one.
"""


def _require_answer(answer: str, provider: str) -> str:
    answer = (answer or "").strip()
    if not answer:
        raise RuntimeError(f"{provider} returned an empty answer.")
    return answer


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
) -> str:
    messages.append({
        "role": "user",
        "content": (
            "The investigation budget is exhausted. Answer now using the evidence "
            "already collected, with exact source citations and no unsupported claims."
        ),
    })
    response = requests.post(
        f"{base_url.rstrip('/')}/chat/completions",
        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
        json={
            "model": model,
            "messages": messages,
            "temperature": 0.2,
            "max_tokens": 1800,
        },
        timeout=REQUEST_TIMEOUT,
    )
    if response.status_code >= 400:
        raise RuntimeError(f"[{response.status_code}] {response.text[:300]}")
    return _require_answer(
        response.json()["choices"][0]["message"].get("content", ""),
        model,
    )


def _openai_agent(
    base_url: str,
    api_key: str,
    model: str,
    question: str,
    toolbox,
    tool_definitions: list[dict],
) -> dict:
    messages = [
        {"role": "system", "content": AGENT_SYSTEM_PROMPT},
        {"role": "user", "content": question},
    ]
    tools = _openai_tools(tool_definitions)
    tool_call_count = 0

    for round_number in range(1, AGENT_MAX_ROUNDS + 1):
        response = requests.post(
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
        )
        _tool_request_error(response, model)
        message = response.json()["choices"][0]["message"]
        tool_calls = message.get("tool_calls") or []
        if not tool_calls:
            if tool_call_count == 0:
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
        "answer": _final_openai_answer(base_url, api_key, model, messages),
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
) -> dict:
    messages = [{"role": "user", "content": question}]
    tools = _anthropic_tools(tool_definitions)
    tool_call_count = 0
    url = f"{base_url.rstrip('/')}/v1/messages"
    headers = {
        "x-api-key": api_key,
        "anthropic-version": "2023-06-01",
        "Content-Type": "application/json",
    }

    for round_number in range(1, AGENT_MAX_ROUNDS + 1):
        response = requests.post(
            url,
            headers=headers,
            json={
                "model": model,
                "max_tokens": 1800,
                "temperature": 0.2,
                "system": AGENT_SYSTEM_PROMPT,
                "messages": messages,
                "tools": tools,
            },
            timeout=REQUEST_TIMEOUT,
        )
        _tool_request_error(response, model)
        data = response.json()
        blocks = data.get("content") or []
        tool_uses = [block for block in blocks if block.get("type") == "tool_use"]
        if not tool_uses:
            if tool_call_count == 0:
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
        "content": (
            "The investigation budget is exhausted. Answer now using the evidence "
            "already collected, with exact source citations and no unsupported claims."
        ),
    })
    response = requests.post(
        url,
        headers=headers,
        json={
            "model": model,
            "max_tokens": 1800,
            "temperature": 0.2,
            "system": AGENT_SYSTEM_PROMPT,
            "messages": messages,
        },
        timeout=REQUEST_TIMEOUT,
    )
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
) -> dict:
    messages = [
        {"role": "system", "content": AGENT_SYSTEM_PROMPT},
        {"role": "user", "content": question},
    ]
    tools = _openai_tools(tool_definitions)
    tool_call_count = 0
    url = f"{base_url.rstrip('/')}/api/chat"

    for round_number in range(1, AGENT_MAX_ROUNDS + 1):
        response = requests.post(
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
            if tool_call_count == 0:
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
        "content": (
            "The investigation budget is exhausted. Answer now using the evidence "
            "already collected, with exact source citations and no unsupported claims."
        ),
    })
    response = requests.post(
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
    resp = requests.post(
        f"{base_url.rstrip('/')}/chat/completions",
        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
        json={
            "model": model,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": build_prompt(context)},
            ],
            "temperature": 0.2,
            "max_tokens": 1400,
        },
        timeout=REQUEST_TIMEOUT,
    )
    if resp.status_code >= 400:
        raise RuntimeError(f"[{resp.status_code}] {resp.text[:300]}")
    answer = resp.json()["choices"][0]["message"].get("content", "")
    return _require_answer(answer, model)


def _anthropic_chat(base_url: str, api_key: str, model: str, context: dict) -> str:
    resp = requests.post(
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
            "system": SYSTEM_PROMPT,
            "messages": [{"role": "user", "content": build_prompt(context)}],
        },
        timeout=REQUEST_TIMEOUT,
    )
    if resp.status_code >= 400:
        raise RuntimeError(f"[{resp.status_code}] {resp.text[:300]}")
    data = resp.json()
    answer = "".join(b.get("text", "") for b in data.get("content", []) if b.get("type") == "text")
    return _require_answer(answer, model)


def _ollama_chat(base_url: str, model: str, context: dict) -> str:
    resp = requests.post(
        f"{base_url.rstrip('/')}/api/chat",
        json={
            "model": model,
            "stream": False,
            "options": {"temperature": 0.2},
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": build_prompt(context)},
            ],
        },
        timeout=REQUEST_TIMEOUT,
    )
    if resp.status_code >= 400:
        raise RuntimeError(f"[{resp.status_code}] {resp.text[:300]}")
    answer = resp.json().get("message", {}).get("content", "")
    return _require_answer(answer, model)


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

    if provider in {"anthropic", "anthropic_compatible", "claude"}:
        return _anthropic_chat(base_url, api_key, model or "claude-sonnet-4-5", context)
    return _openai_chat(base_url, api_key, model or "gpt-4o-mini", context)


def _call_agent_with_creds(creds: dict, question: str, toolbox) -> dict:
    provider = (creds.get("provider") or "openai_compatible").lower()
    base_url = creds.get("base_url") or ""
    api_key = creds.get("api_key") or ""
    model = creds.get("model") or ""
    if not base_url:
        raise RuntimeError("missing base_url")
    if not api_key:
        raise RuntimeError("missing api_key")

    if provider in {"anthropic", "anthropic_compatible", "claude"}:
        return _anthropic_agent(
            base_url,
            api_key,
            model or "claude-sonnet-4-5",
            question,
            toolbox,
            TOOL_DEFINITIONS,
        )
    return _openai_agent(
        base_url,
        api_key,
        model or "gpt-4o-mini",
        question,
        toolbox,
        TOOL_DEFINITIONS,
    )


def _attempt_with_creds(creds: dict, context: dict, question: str = None, toolbox=None) -> dict:
    fallback_reason = None
    if AGENT_ENABLED and question and toolbox is not None:
        toolbox.trace.clear()
        try:
            result = _call_agent_with_creds(creds, question, toolbox)
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
) -> dict:
    fallback_reason = None
    if AGENT_ENABLED and question and toolbox is not None:
        toolbox.trace.clear()
        try:
            result = _ollama_agent(
                base_url,
                model,
                question,
                toolbox,
                TOOL_DEFINITIONS,
            )
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

def generate(
    context: dict,
    user_llm: dict = None,
    allow_shared_fallback: bool = True,
    llm_mode: str = None,
    question: str = None,
    toolbox=None,
) -> dict:
    """Generate an answer with the first working provider tier.

    user_llm: optional {provider, base_url, api_key, model} from the requesting
              user (BYOK). allow_shared_fallback: when False, the shared "Mimo"
              endpoint (tier 3) is skipped (per-repo privacy control). When a
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
        result = _attempt_with_creds(user_llm, context, question, toolbox)
        return {
            **result,
            "provider_used": f"user:{user_llm.get('provider', 'openai')}",
        }

    # Explicit mode — local Ollama only.
    if mode == "ollama":
        ollama_url = os.getenv("CODEATLAS_OLLAMA_URL", "http://localhost:11434")
        ollama_model = os.getenv("CODEATLAS_OLLAMA_MODEL", "qwen2.5-coder:7b")
        if not _ollama_available(ollama_url):
            raise RuntimeError(f"Ollama is not reachable at {ollama_url}.")
        result = _attempt_ollama(
            ollama_url, ollama_model, context, question, toolbox
        )
        return {**result, "provider_used": f"ollama:{ollama_model}"}

    # Explicit mode — shared Mimo endpoint only.
    if mode == "mimo":
        if not allow_shared_fallback:
            raise RuntimeError("Mimo/shared LLM is disabled for this repository.")
        shared = _configured_shared_creds(os.getenv("CODEATLAS_MIMO_MODEL", "mimo-v2.5"))
        if not shared["base_url"] or not shared["api_key"]:
            raise RuntimeError("Mimo/shared LLM is not configured.")
        result = _attempt_with_creds(shared, context, question, toolbox)
        return {**result, "provider_used": f"shared:{shared['model']}"}

    # Tier 1 — user's own key.
    if user_llm and user_llm.get("api_key"):
        try:
            result = _attempt_with_creds(user_llm, context, question, toolbox)
            return {
                **result,
                "provider_used": f"user:{user_llm.get('provider', 'openai')}",
            }
        except Exception as exc:  # fall through on any failure
            errors.append(f"user-key: {exc}")

    # Tier 2 — local Ollama.
    ollama_url = os.getenv("CODEATLAS_OLLAMA_URL", "http://localhost:11434")
    ollama_model = os.getenv("CODEATLAS_OLLAMA_MODEL", "qwen2.5-coder:7b")
    if _ollama_available(ollama_url):
        try:
            result = _attempt_ollama(
                ollama_url, ollama_model, context, question, toolbox
            )
            return {**result, "provider_used": f"ollama:{ollama_model}"}
        except Exception as exc:
            errors.append(f"ollama: {exc}")
    else:
        errors.append(f"ollama: not reachable at {ollama_url}")

    # Tier 3 — shared/admin endpoint ("Mimo").
    if allow_shared_fallback:
        shared = _configured_shared_creds()
        if shared["base_url"] and shared["api_key"]:
            try:
                result = _attempt_with_creds(shared, context, question, toolbox)
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
