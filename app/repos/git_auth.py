"""Secure server-side Git authentication for supported private providers.

Central read-only credentials are injected by deployment as environment
variables. They are exposed to Git only through GIT_ASKPASS and only for the
explicitly supported hosts. Tokens are never placed in clone URLs, command
arguments, SQLite rows, or audit log details.
"""

from __future__ import annotations

import os
import re
import subprocess
from pathlib import Path
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

SENSITIVE_URL_QUERY_KEYS = {
    "access_token",
    "api_key",
    "apikey",
    "auth",
    "authorization",
    "oauth_token",
    "password",
    "passwd",
    "private_token",
    "token",
}
SECRET_ENV_KEYS = {
    "GH_TOKEN",
    "GITHUB_TOKEN",
    "CODEATLAS_BITBUCKET_API_TOKEN",
    "CODEATLAS_BITBUCKET_TOKEN",
    "BITBUCKET_TOKEN",
}
GITHUB_HOST = "github.com"
BITBUCKET_HOST = "bitbucket.org"
_ASKPASS_PATH = Path(__file__).with_name("git_askpass.py")
_CREDENTIAL_URL_RE = re.compile(r"(https?://)[^/@\s]+@", re.IGNORECASE)


class GitAuthError(ValueError):
    """Raised when a clone URL would violate the credential safety policy."""


def _normal_host(value: str | None) -> str:
    return (value or "").strip().lower().rstrip(".")


def _configured_secret_values() -> list[str]:
    values = []
    for key in SECRET_ENV_KEYS:
        value = os.environ.get(key)
        if value:
            values.append(value)
    return values


def _base_git_env() -> dict[str, str]:
    env = {
        key: value
        for key, value in os.environ.items()
        if key not in SECRET_ENV_KEYS
        and not key.startswith("CODEATLAS_GIT_ASKPASS_")
    }
    env["GIT_TERMINAL_PROMPT"] = "0"
    env["GCM_INTERACTIVE"] = "never"
    env["GH_PROMPT_DISABLED"] = "1"
    # Disable inherited credential helpers so only CodeAtlas' explicit
    # provider-scoped GIT_ASKPASS path can supply HTTPS credentials.
    env["GIT_CONFIG_COUNT"] = "1"
    env["GIT_CONFIG_KEY_0"] = "credential.helper"
    env["GIT_CONFIG_VALUE_0"] = ""
    return env


def sanitize_url_for_storage(source_url: str) -> str:
    """Remove URL credentials and redact sensitive query values."""
    if "://" not in source_url:
        return source_url
    parsed = urlsplit(source_url)
    sanitized_netloc = parsed.netloc.rsplit("@", 1)[-1]
    query_pairs = parse_qsl(parsed.query, keep_blank_values=True)
    sanitized_query_pairs = [
        (key, "[redacted]" if key.lower() in SENSITIVE_URL_QUERY_KEYS else value)
        for key, value in query_pairs
    ]
    if (
        sanitized_netloc == parsed.netloc
        and sanitized_query_pairs == query_pairs
    ):
        return source_url
    return urlunsplit((
        parsed.scheme,
        sanitized_netloc,
        parsed.path,
        urlencode(sanitized_query_pairs),
        parsed.fragment,
    ))


def sanitize_git_error(value: str) -> str:
    """Redact credentials and configured Git tokens from user-visible errors."""
    redacted = _CREDENTIAL_URL_RE.sub(r"\1[redacted]@", (value or "").strip())
    for secret in _configured_secret_values():
        if len(secret) >= 4:
            redacted = redacted.replace(secret, "[redacted]")
    return redacted[:1000]


def _http_url_parts(source_url: str):
    if "://" not in source_url:
        return None
    parsed = urlsplit(source_url)
    if parsed.scheme.lower() not in {"http", "https"}:
        return None
    return parsed


def validate_clone_url(source_url: str, method: str | None = None) -> None:
    """Reject URLs that would persist or expose credentials.

    SSH-style URLs such as git@github.com:org/repo.git remain valid. GitHub CLI
    shorthand such as owner/repo remains valid for the existing gh clone method.
    """
    url = (source_url or "").strip()
    if not url:
        raise GitAuthError("source_url is required.")
    parsed = _http_url_parts(url)
    if not parsed:
        return
    if parsed.username or parsed.password or "@" in parsed.netloc:
        raise GitAuthError(
            "Source URLs must not include embedded credentials. "
            "Use the server-side GitHub/Bitbucket credentials configured for CodeAtlas."
        )
    sensitive_keys = [
        key
        for key, _ in parse_qsl(parsed.query, keep_blank_values=True)
        if key.lower() in SENSITIVE_URL_QUERY_KEYS
    ]
    if sensitive_keys:
        raise GitAuthError(
            "Source URLs must not include access tokens or secret query parameters."
        )
    host = _normal_host(parsed.hostname)
    if host in {GITHUB_HOST, BITBUCKET_HOST} and parsed.scheme.lower() != "https":
        raise GitAuthError("GitHub and Bitbucket clone URLs must use HTTPS or SSH.")


def provider_for_url(source_url: str | None) -> str | None:
    parsed = _http_url_parts(source_url or "")
    if not parsed or parsed.scheme.lower() != "https":
        return None
    host = _normal_host(parsed.hostname)
    if host == GITHUB_HOST:
        return "github"
    if host == BITBUCKET_HOST:
        return "bitbucket"
    return None


def git_env_for_url(source_url: str | None) -> dict[str, str]:
    """Build a sanitized Git environment for a remote URL."""
    env = _base_git_env()
    provider = provider_for_url(source_url)
    username = ""
    token = ""
    if provider == "github":
        username = "x-access-token"
        token = os.environ.get("GH_TOKEN") or os.environ.get("GITHUB_TOKEN") or ""
    elif provider == "bitbucket":
        username = "x-bitbucket-api-token-auth"
        token = (
            os.environ.get("CODEATLAS_BITBUCKET_API_TOKEN")
            or os.environ.get("CODEATLAS_BITBUCKET_TOKEN")
            or os.environ.get("BITBUCKET_TOKEN")
            or ""
        )
    if username and token:
        env["GIT_ASKPASS"] = str(_ASKPASS_PATH)
        env["CODEATLAS_GIT_ASKPASS_USERNAME"] = username
        env["CODEATLAS_GIT_ASKPASS_TOKEN"] = token
    return env


def git_env_for_gh_cli() -> dict[str, str]:
    """Build a non-interactive environment for the existing gh clone method."""
    env = _base_git_env()
    for key in ("GH_TOKEN", "GITHUB_TOKEN"):
        value = os.environ.get(key)
        if value:
            env[key] = value
    return env


def origin_url_for_repo(repo: Path) -> str | None:
    try:
        result = subprocess.run(
            ["git", "config", "--get", "remote.origin.url"],
            cwd=str(repo),
            capture_output=True,
            text=True,
            timeout=30,
            env=_base_git_env(),
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    if result.returncode != 0:
        return None
    return result.stdout.strip() or None


def git_env_for_repo(repo: Path) -> dict[str, str]:
    origin_url = origin_url_for_repo(repo)
    if origin_url:
        validate_clone_url(origin_url)
    return git_env_for_url(origin_url)
