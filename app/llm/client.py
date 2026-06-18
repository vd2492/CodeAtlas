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

REQUEST_TIMEOUT = 90

SYSTEM_PROMPT = (
    "You are CodeAtlas, a codebase investigation assistant. "
    "Answer like a senior engineer reading the repository: reason from the "
    "provided source snippets, graph context, file paths, and relations. "
    "Do not guess beyond the evidence. Always answer in English and cite "
    "source files and line numbers for concrete claims."
)


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


# --- Fallback chain -----------------------------------------------------------

def generate(
    context: dict,
    user_llm: dict = None,
    allow_shared_fallback: bool = True,
    llm_mode: str = None,
) -> dict:
    """Return {"answer", "provider_used"} using the first working tier.

    user_llm: optional {provider, base_url, api_key, model} from the requesting
              user (BYOK). allow_shared_fallback: when False, the shared "Mimo"
              endpoint (tier 3) is skipped (per-repo privacy control).
    """
    errors = []
    mode = (llm_mode or "auto").lower()

    # Explicit mode — user's own key only.
    if mode == "personal":
        if not user_llm or not user_llm.get("api_key"):
            raise RuntimeError("No personal LLM key is saved yet.")
        answer = _call_with_creds(user_llm, context)
        return {"answer": answer, "provider_used": f"user:{user_llm.get('provider', 'openai')}"}

    # Explicit mode — local Ollama only.
    if mode == "ollama":
        ollama_url = os.getenv("CODEATLAS_OLLAMA_URL", "http://localhost:11434")
        ollama_model = os.getenv("CODEATLAS_OLLAMA_MODEL", "qwen2.5-coder:7b")
        if not _ollama_available(ollama_url):
            raise RuntimeError(f"Ollama is not reachable at {ollama_url}.")
        answer = _ollama_chat(ollama_url, ollama_model, context)
        return {"answer": answer, "provider_used": f"ollama:{ollama_model}"}

    # Explicit mode — shared Mimo endpoint only.
    if mode == "mimo":
        if not allow_shared_fallback:
            raise RuntimeError("Mimo/shared LLM is disabled for this repository.")
        shared = _configured_shared_creds(os.getenv("CODEATLAS_MIMO_MODEL", "mimo-v2.5"))
        if not shared["base_url"] or not shared["api_key"]:
            raise RuntimeError("Mimo/shared LLM is not configured.")
        answer = _call_with_creds(shared, context)
        return {"answer": answer, "provider_used": f"shared:{shared['model']}"}

    # Tier 1 — user's own key.
    if user_llm and user_llm.get("api_key"):
        try:
            answer = _call_with_creds(user_llm, context)
            return {"answer": answer, "provider_used": f"user:{user_llm.get('provider', 'openai')}"}
        except Exception as exc:  # fall through on any failure
            errors.append(f"user-key: {exc}")

    # Tier 2 — local Ollama.
    ollama_url = os.getenv("CODEATLAS_OLLAMA_URL", "http://localhost:11434")
    ollama_model = os.getenv("CODEATLAS_OLLAMA_MODEL", "qwen2.5-coder:7b")
    if _ollama_available(ollama_url):
        try:
            answer = _ollama_chat(ollama_url, ollama_model, context)
            return {"answer": answer, "provider_used": f"ollama:{ollama_model}"}
        except Exception as exc:
            errors.append(f"ollama: {exc}")
    else:
        errors.append(f"ollama: not reachable at {ollama_url}")

    # Tier 3 — shared/admin endpoint ("Mimo").
    if allow_shared_fallback:
        shared = _configured_shared_creds()
        if shared["base_url"] and shared["api_key"]:
            try:
                answer = _call_with_creds(shared, context)
                return {"answer": answer, "provider_used": f"shared:{shared['model']}"}
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
