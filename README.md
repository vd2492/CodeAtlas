# CodeAtlas

Self-hostable codebase intelligence. Host private repositories, index them into
a graph, safely tune retrieval per repo, control who can access each one, and
let PMs / QAs / developers / stakeholders ask grounded questions about the code
without direct repo access.

> Status: **Phase 1 — foundation + skeleton.** The single-repo query/answer
> engine works today against a default workspace. Multi-tenant admin/user
> features are scaffolded (routes + schema + config) and built out in phases.
> See [docs/PLAN.md](docs/PLAN.md).

## Quickstart

```bash
./run.sh                      # creates .venv, installs deps, starts on :8000
```

Open http://localhost:8000 for the Ask UI. A **default workspace** is seeded
with a sample graph so the tool runs immediately.

To get LLM answers, configure at least one tier (see `.env.example`):
- **Your own key** — passed per request (BYOK), used first.
- **Ollama** — `ollama serve` + a code model (e.g. `qwen2.5-coder:7b`), free & private.
- **Shared "Kimi"** — `CODEATLAS_LLM_*` env, used only as a last resort.

To get **code excerpts** in answers for the seeded demo graph, point the source
root at the original repo: `CODEATLAS_SOURCE_ROOT=/path/to/destiny`.

## LLM fallback chain

Every question resolves through `app/llm/client.py` in order:

1. **User key** (BYOK) → 2. **Ollama** (local) → 3. **Shared endpoint** ("Kimi").

Each tier falls through on absence *or* failure. The shared tier can be disabled
per repo (`allow_shared_fallback`) so private code is never sent off-box.

## Layout

```
app/
  main.py            FastAPI app + current query/answer endpoints
  config.py          paths & workspaces
  db.py              SQLite: users, repos, repo_access
  llm/client.py      key → Ollama → Kimi fallback chain
  retrieval/         ranker, context builder, flow maps, graph insights
    config_schema.py per-repo safe retrieval config (stopwords, synonyms, ...)
  repos/             clone (https/ssh/gh) + graphify indexing + admin routes (stub)
  auth/              password hashing + auth routes (stub)
  static/            user Ask UI + admin console (placeholder)
data/                gitignored: sqlite db, cloned repos, per-workspace graphs+config
```

## Roles (target)

- **Admin** — adds repos, indexes them, tests retrieval/answers, tunes the safe
  config, publishes, and grants user access.
- **User** — logs in, picks an authorized repo, asks questions.
