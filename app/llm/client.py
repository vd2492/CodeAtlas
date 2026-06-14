"""LLM answer generation with an ordered fallback chain.

Resolution order for every question:
  1. The user's own LLM key (BYOK), if supplied.
  2. A locally running Ollama model (free, private), if reachable.
  3. The shared/admin-configured endpoint ("Kimi") as a last resort.

Each tier falls through to the next on absence OR failure (bad key, rate
limit, timeout). Admins can disable tier 3 per repo for sensitive codebases so
private code is never sent to the shared endpoint.
"""

import json
import os

import requests

REQUEST_TIMEOUT = 90

SYSTEM_PROMPT = (
    "You are CodeAtlas, a codebase explanation assistant. "
    "Your only source of truth is the provided graph context and code excerpts. "
    "Do not guess beyond the context. Always answer in English. "
    "Always include source files and line numbers when explaining. "
    "Be concise and focused; avoid restating every node."
)


def build_prompt(context: dict) -> str:
    preview = context.get("llm_context_preview", {})
    return f"""
Question:
{preview.get("question", "")}

Graph context:
{json.dumps(preview, indent=2)}

Answer requirements:
- Explain in simple PM/QA/dev-friendly English.
- Be concise: aim for under ~300 words. Lead with a direct answer, then a few key details.
- Use only the provided graph context and code excerpts.
- Mention source files and line numbers.
- Do not list every node; focus on what answers the question.
- If the context is not enough, say what is missing.
- Do not invent files, functions, or behavior.
"""


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
    return resp.json()["choices"][0]["message"]["content"]


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
    return "".join(b.get("text", "") for b in data.get("content", []) if b.get("type") == "text")


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
    return resp.json().get("message", {}).get("content", "")


def _ollama_available(base_url: str) -> bool:
    try:
        requests.get(f"{base_url.rstrip('/')}/api/tags", timeout=1.5)
        return True
    except requests.RequestException:
        return False


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

def generate(context: dict, user_llm: dict = None, allow_shared_fallback: bool = True) -> dict:
    """Return {"answer", "provider_used"} using the first working tier.

    user_llm: optional {provider, base_url, api_key, model} from the requesting
              user (BYOK). allow_shared_fallback: when False, the shared "Kimi"
              endpoint (tier 3) is skipped (per-repo privacy control).
    """
    errors = []

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

    # Tier 3 — shared/admin endpoint ("Kimi").
    if allow_shared_fallback:
        shared = {
            "provider": os.getenv("CODEATLAS_LLM_PROVIDER", "openai_compatible"),
            "base_url": os.getenv("CODEATLAS_LLM_BASE_URL", ""),
            "api_key": os.getenv("CODEATLAS_LLM_API_KEY")
            or os.getenv("OPENAI_API_KEY")
            or os.getenv("ANTHROPIC_API_KEY"),
            "model": os.getenv("CODEATLAS_LLM_MODEL", "kimi-2.5"),
        }
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
