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
- **Safe tuning**: retrieval is config-driven (`RetrievalConfig`); admins never
  edit or run backend code from the browser.

## Phases

### Phase 1 — Foundation + skeleton ✅ (this pass)
- Standalone repo, working query/answer engine moved into a package.
- LLM fallback chain (`app/llm/client.py`), with `provider_used` surfaced.
- SQLite schema (`app/db.py`), per-repo `RetrievalConfig` schema + load/save.
- Clone/index helpers (`app/repos/`), stub auth/admin routers.
- Default workspace seeded with a sample graph so it runs out of the box.

### Phase 2 — Auth & repo lifecycle
- Real login/sessions; admin bootstrap (first user = admin).
- Admin: add repo (https/ssh/gh) → clone → index → workspace `status` transitions.
- Wire `app/repos/routes.py` + `app/auth/routes.py` to the DB and helpers.
- Admin console UI (`static/admin.html`).

### Phase 3 — Config-driven retrieval
- Thread `RetrievalConfig` through the context builder so stopwords, synonyms,
  keyword boosts, preferred components/methods, node/relation limits, and
  excerpt sizes apply **per workspace**.
- Admin "test retrieval & answers" panel: edit config → re-run → compare.
- Migrate the current hardcoded (destiny-specific) anchors into the default
  workspace's config.

### Phase 4 — Access control & publishing
- Publish a workspace; grant/revoke per-user access (`repo_access`).
- Users see only authorized repos; every query is scoped + permission-checked.
- Per-repo `allow_shared_fallback` enforced in the LLM chain.

### Phase 5 — BYOK & polish
- Per-user encrypted LLM creds (tier 1) stored in `users.llm_creds`.
- Provider sniffing from key prefix; UI to manage keys.
- Audit log, rate limits, multi-workspace graph loading.

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
- Private repos: prefer user-key or Ollama; disable shared fallback per repo.
- Secrets via env/`.env` (gitignored); BYOK creds encrypted at rest (Phase 5).
