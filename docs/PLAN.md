# CodeAtlas — Build Plan

Evolving the single-repo CodeAtlas tool into a multi-tenant, self-hostable
codebase intelligence platform. Everything is free/open-source to run; the only
optional cost is the shared LLM tier.

## Architecture at a glance

- **FastAPI** backend, **SQLite** for users/repos/access, **vanilla JS** UI.
- **Per-workspace data** under `data/workspaces/<workspace>/`:
  `repo/` (clone), `graph/graph.json` (index), `retrieval_config.json` (tuning).
- **graphify** for structural indexing (no LLM needed).
- **LLM fallback chain**: user key → Ollama → shared endpoint.
- **Agentic retrieval**: tool-capable models iteratively search/read source and
  graph data; unsupported models retain the one-shot RAG fallback.
- **Safe tuning**: retrieval is config-driven (`RetrievalConfig`); admins never
  edit or run backend code from the browser.

## Phases

### Phase 1 — Foundation + skeleton ✅ (this pass)
- Standalone repo, working query/answer engine moved into a package.
- LLM fallback chain (`app/llm/client.py`), with `provider_used` surfaced.
- SQLite schema (`app/db.py`), per-repo `RetrievalConfig` schema + load/save.
- Clone/index helpers (`app/repos/`), stub auth/admin routers.
- Default workspace seeded with a sample graph so it runs out of the box.

### Phase 2 — Auth & repo lifecycle ✅
- Real login/sessions (DB-backed `sessions` table + HttpOnly `ca_session` cookie);
  first-run bootstrap (first user = admin), optional `CODEATLAS_ADMIN_USER/PASS` seed.
- Admin: add repo (https/ssh/gh) → clone → index → publish; `status` transitions
  `new→cloned→indexed→published`. Admin "test retrieval & answers" panel runs the
  LLM against a specific workspace before exposing it.
- Wired `app/repos/routes.py` + `app/auth/routes.py` to the DB and clone/index helpers.
- Admin console UI (`static/admin.html`); Ask UI login-gated with a repo picker.
- **Access control pulled forward** (chosen during the build): every `/repo/*`
  query is scoped to a `workspace` and permission-checked (admins bypass; users
  need an explicit grant). Per-repo `allow_shared_fallback` enforced in the ask
  path. Phase 4 now only owns the remaining polish: revoke UI, audit log.

### Phase 3 — Config-driven retrieval ✅
- `build_context` is now driven entirely by the workspace's `RetrievalConfig`:
  stopwords, synonyms, keyword boosts (threaded into `rank_nodes_for_query`),
  query-relevant preferred components/methods, node/relation limits, and excerpt
  sizes all apply **per workspace**. The relation filter is now generic (drop
  test files + primitive-type noise, keep relations touching selected nodes).
- Destiny-specific anchors migrated out of code into the **default workspace's**
  seeded config (`config_schema.DEFAULT_DESTINY_CONFIG` +
  `seed_default_retrieval_config()`, written on first boot since `data/` is
  gitignored). New repos start from `RetrievalConfig()` defaults, tunable from
  the admin console.
- Admin "test retrieval & answers" panel (built in Phase 2) drives edit→re-run→
  compare. Verified: demo answers unchanged (login is actually better now),
  editing synonyms/boosts changes the grounded context set, and no destiny
  anchors leak into other repos.
- Note: the legacy `/repo/flows/{topic}` + `/repo/ask` demo endpoints still use
  `flow_map.TOPICS` (destiny-only flow explorer); they're separate from the
  config-driven answer path and untouched.

### Phase 4 — Access control & publishing ✅
- Publish + grant + per-query scoping + `allow_shared_fallback` enforcement
  landed in Phase 2. Phase 4 added the remaining **management surface**:
  - **Revoke** access (`POST /admin/repos/{slug}/revoke`) and **Members** view
    (`GET .../members`) — wired to the existing `db.revoke_access` /
    `list_repo_members`. Revokes take effect on the user's next page load.
  - **Delete repo** (`DELETE /admin/repos/{slug}`): removes the DB row (grants
    cascade) and the workspace dir (`cloning.remove_workspace`); the seeded
    `default` repo is protected. (Replaces the manual DB+rm step.)
  - **Privacy toggle** (`PATCH .../privacy`): flip `allow_shared_fallback` from
    the UI; verified that disabling it blocks the shared LLM tier.
  - **Audit log**: append-only `audit_log` table + `record_audit`/`list_audit`,
    recorded on login/bootstrap/create-user/add/index/publish/grant/revoke/
    privacy/delete, surfaced read-only at `GET /auth/admin/audit` and in the
    console.
- Admin console (`static/admin.html`) extended with Members/Privacy/Delete row
  actions and an Audit log section. User Ask UI unchanged (scoping already live).

### Phase 5 — BYOK & polish ✅
- **BYOK**: per-user LLM key encrypted at rest with **Fernet**
  (`app/auth/crypto.py`; secret from `CODEATLAS_SECRET_KEY` env or generated to
  `data/secret.key`, gitignored, mode 0600) and stored in `users.llm_creds`.
  Used as LLM tier 1 for that user's questions; `/repo/ask-llm` loads + decrypts
  it (request-body `user_llm` still works as an override).
- **Provider sniffing** (`client.sniff_provider`): `sk-ant-*`→anthropic,
  `sk-*`→openai, else openai_compatible (base_url required). Key-management panel
  in the Ask UI (`GET/PUT/DELETE /auth/me/llm`); `GET` returns a hint, never the
  key. The answer view shows which tier responded (`user:` / `ollama:` /
  `shared:`).
- **Rate limits**: per-user sliding window on `/repo/ask-llm`
  (`CODEATLAS_RATE_LIMIT_PER_MIN`, default 20) → HTTP 429 over-limit.
- **User management**: admin `DELETE /auth/admin/users/{username}` (refuses self
  and the last admin) + Users table with delete in the console.
- Multi-workspace graph loading already landed in Phase 2/3; audit log in Phase 4.
- Verified: tier-1 BYOK answers as `user:*`, key is ciphertext at rest and never
  returned, clearing falls back to `shared:`, rate limit trips at the cap, and
  user deletion guards hold.

### Phase 6 — Agentic retrieval MVP ✅
- Added workspace-scoped, read-only tools for source search, ranged file reads,
  directory listing, graph definitions/references, and caller discovery.
- Added native multi-turn tool loops for OpenAI-compatible Chat Completions,
  Anthropic Messages, and Ollama Chat. Each loop has round/tool/output budgets.
- Provider behavior is backward compatible: a model that rejects or skips tools
  uses the existing compact-context answer path on the same provider.
- `/repo/ask-llm` now reports `retrieval_mode`, tool counts, and a compact
  `agent_trace`; the Ask UI shows the investigation steps.
- Hardened reads with workspace containment, symlink resolution, secret-file
  blocking, source-size limits, and no write/execute tools.
- Optimized graph node ranking to use source metadata already stored on nodes
  instead of rescanning the complete relation list for each candidate.
- Added unit coverage for all three provider protocols, fallback behavior,
  hybrid search, graph traversal, bounded reads, secret blocking, and path
  traversal.

## Safe retrieval config (admin-tunable, no code execution)

`app/retrieval/config_schema.py :: RetrievalConfig`

| Field | Purpose |
|---|---|
| `stopwords` | words ignored in question keyword matching |
| `synonyms` | term → expansions (domain vocabulary) |
| `keyword_boosts` | term → score multiplier |
| `preferred_components` / `preferred_methods` | deterministic anchors for key questions |
| `node_limit` / `relation_limit` | context size |
| `excerpt_nodes` / `excerpt_max_lines` / `excerpt_max_chars` | source-code excerpt budget |
| `allow_shared_fallback` | privacy: forbid the shared LLM tier for this repo |

## Security notes
- No browser-driven code execution — tuning is data only.
- Agent tools are read-only, workspace-contained, and budget-limited.
- Private repos: prefer user-key or Ollama; disable shared fallback per repo.
- Secrets via env/`.env` (gitignored); BYOK creds encrypted at rest (Phase 5).
