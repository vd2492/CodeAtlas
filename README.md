# CodeAtlas

**Self-hostable codebase intelligence.** Host private repositories, index them
into a queryable graph, safely tune retrieval per repo, control who can access
each one, and let PMs, QAs, developers, and stakeholders ask grounded questions
about the code — with answers tailored to their audience — without giving them
direct repository access.

Everything runs on your own box. Private code never has to leave it.

---

## What it does

- **Index any repo into a graph.** An admin clones a repository (HTTPS, SSH, or
  the GitHub CLI) and indexes it into a structural graph of files, symbols, and
  relations — no LLM needed for indexing.
- **Ask grounded questions.** Users ask in natural language ("How does login
  work?", "Which files are involved in this feature?") and the selected model
  iteratively searches the graph, follows symbols, and reads real source before
  answering. Dev-team users receive the existing technical response with
  file/line references; product-team users receive a concise, plain-language
  explanation without class names or technical terminology.
- **Per-repo, config-driven tuning.** Admins improve retrieval quality with
  safe, data-only knobs (stopwords, synonyms, keyword boosts, preferred
  components/methods, context/excerpt sizes, and a pre-search terminology
  instruction). Configs can be loaded, validated, reset to defaults, and saved
  per repository. No code is ever executed from the browser.
- **Audience-aware answers.** Admins assign each account a user type:
  **Dev team** preserves detailed engineering answers, while **Product team**
  automatically asks the model for simple, clear, concise answers without
  internal implementation details. User type can be changed from the existing
  user-edit flow.
- **Access control.** Users log in and only see repositories an admin has
  explicitly granted them; every query is permission-checked.
- **Branch-aware, freshness-tracked answers.** Admins approve remote branches,
  CodeAtlas indexes each branch in an isolated Git worktree, and users can select
  the exact indexed branch and commit before asking. Freshness checks and
  **Sync & index now** keep the active graph aligned with the remote branch.
- **Bring your own LLM key (BYOK).** Each user can store their own LLM key
  (encrypted at rest) to be used as their first-choice model.

## Quickstart

```bash
./run.sh        # creates a virtualenv, installs deps, starts the server on :8000
```

- **Landing page:** http://localhost:8000/
- **Ask UI:** http://localhost:8000/app
- **Admin console:** http://localhost:8000/admin.html

On first run the admin console walks you through creating the first admin
account. A **default demo workspace** is seeded so the tool works immediately.

### Docker Compose

The production image includes Git, SSH, the GitHub CLI, and the pinned
`graphify` indexer. Start it with:

```bash
cp .env.example .env
docker compose up -d --build
docker compose ps
```

By default, Compose publishes port `8000` and mounts `./data` at `/app/data`.
Set `CODEATLAS_COOKIE_SECURE=false` in `.env` only when testing over local
plain HTTP; keep it `true` behind the production HTTPS load balancer.
For a VM with a separately mounted persistent disk, set this in `.env`:

```bash
CODEATLAS_DATA_PATH=/srv/codeatlas/data
CODEATLAS_BIND_ADDRESS=0.0.0.0
CODEATLAS_PORT=8000
```

The image runs as UID/GID `10001`, so prepare the VM data mount before starting:

```bash
sudo install -d -o 10001 -g 10001 /srv/codeatlas/data
sudo chown -R 10001:10001 /srv/codeatlas/data
sudo chmod 750 /srv/codeatlas/data
```

For private repositories, inject `GH_TOKEN` for the `gh` clone method or add a
Compose override that mounts a read-only deploy-key directory at
`/home/codeatlas/.ssh`. Never bake repository credentials into the image.

The container intentionally runs one Uvicorn worker because branch polling is
in-process and the application uses SQLite. When placing it behind a GCP
internal HTTPS Application Load Balancer, use `/healthz` for the health check,
allow backend port `8000` only from the proxy-only subnet and health-check
ranges, and set the backend timeout high enough for synchronous indexing.

If startup fails with `sqlite3.OperationalError: unable to open database file`,
the mounted host directory is not writable by the container. Confirm that
`CODEATLAS_DATA_PATH` points to an existing directory, apply the ownership
commands above, and recreate the service:

```bash
docker compose down
docker compose up -d --build
docker compose logs -f app
```

## Configuration

All configuration is via environment variables — copy `.env.example` to `.env`
and fill in what you need (the `.env` file is gitignored). Nothing is required
to boot; the relevant groups are:

- **Shared LLM tier** — an OpenAI-compatible or Anthropic-compatible endpoint
  used as a fallback for answering questions.
- **Reserved Ollama integration** — retained behind a disabled feature flag for
  a future release; it is not exposed to users.
- **Paths / source root** — optional overrides for data directory and the source
  tree used to pull code excerpts into answers.
- **Branch synchronization** — worker count, freshness polling, user-triggered
  sync cooldown, and old-version retention.

See `.env.example` for the full list of keys and inline notes.

### LLM fallback chain

Every question resolves through `app/llm/client.py` in order:

1. **User key (BYOK)** → 2. **Shared endpoint**

Each tier falls through on absence *or* failure. The shared tier can be disabled
per repository (`allow_shared_fallback`), so sensitive code is never sent to a
shared endpoint. The dormant Ollama code is disabled by default with
`CODEATLAS_ENABLE_OLLAMA=false` and has no user-facing control.

### Agentic retrieval

Tool-capable models receive six read-only repository tools:

- `search_code`, `read_file`, and `list_directory`
- `find_definition`, `find_references`, and `get_callers`

The model can search, inspect the result, follow a relation, and read additional
source over several rounds. Tools are workspace-scoped, path traversal and
likely secret files are blocked, and all reads have line/byte limits. If an
endpoint does not support tool calling—or the selected model answers without
using a tool—CodeAtlas automatically uses the original one-shot context path for
that provider.

The API response reports `retrieval_mode` (`agentic` or `one_shot`) and a compact
`agent_trace`; the Ask UI displays this investigation under Grounded Evidence.

### Fast grounded follow-ups

The Ask UI sends an opaque conversation ID instead of replaying previous answers
from the browser. For a related follow-up, the server reuses the prior verified
evidence when the authenticated user, repository, indexed branch revision,
audience type, and LLM mode still match. The model may answer directly from that
evidence or invoke the normal read-only repository tools when more verification
is needed. Unrelated questions, expired state, repository reindexing, and mode
changes automatically take the full retrieval path.

Conversation state is an in-process TTL cache and contains no API credentials.
It is an optional acceleration layer: losing it on restart only makes the next
question use full retrieval. Configure its lifetime and bounds with the
`CODEATLAS_CONVERSATION_*` variables in `.env.example`. Official Anthropic API
requests also enable its documented ephemeral prompt cache; compatible
third-party endpoints are left unchanged.

## How a repository goes live

`Clone → Approve branches → Index → Test → Tune → Publish → Grant access`

An admin clones a repo, approves and indexes its remote branches, tests retrieval
and answer quality, tunes the per-repo config until answers are good, publishes
the workspace, and grants access to selected users. Users then log in, pick an
authorized repo and indexed branch, inspect its commit/freshness metadata, and
ask away.

## Architecture

- **Backend:** FastAPI (Python), **SQLite** for users / repos / access /
  sessions / audit log.
- **Frontend:** vanilla HTML/CSS/JS — a marketing landing page, a user Ask UI,
  and an admin console (dark/light themed).
- **Indexing:** a structural graph extractor (no LLM required).
- **Retrieval:** agentic, read-only repository investigation for tool-capable
  models—searching code, reading bounded file excerpts, and following
  definitions, references, callers, and graph relationships. A per-workspace
  `RetrievalConfig` seeds the investigation, with automatic fallback to
  one-shot keyword + graph ranking when agentic tool use is unavailable.

```
app/
  agent/tools.py     workspace-scoped source + graph tools for the LLM
  main.py            FastAPI app + query/answer endpoints, startup wiring
  config.py          paths & per-workspace layout
  db.py              SQLite: users, repos, repo_access, sessions, audit_log
  auth/              sessions, password hashing, BYOK key encryption, auth routes
  repos/             clone, branch worktrees, freshness jobs, indexing, lifecycle routes
  retrieval/         ranker, context builder, per-repo RetrievalConfig
  llm/client.py      agent loops + BYOK → shared fallback; dormant Ollama hook
  static/            landing page, user Ask UI, admin console
data/                gitignored: sqlite db, cloned repos, per-workspace graphs/config, secret key
docs/PLAN.md         build plan / phase history
```

## Roles and user types

- **Admin** — clone & index repos, test and tune retrieval, publish, manage
  approved branches and sync settings, manage users and per-repo access, toggle
  the shared-LLM privacy setting, and review an audit log of privileged actions.
- **User** — log in with admin-provided credentials, see only authorized repos,
  select an indexed branch, optionally request an authorized refresh, optionally
  set their own LLM key, and ask grounded questions.

Role controls permissions. User type controls only how final LLM answers are
presented:

- **Dev team** (the default, including existing accounts) — keeps the existing
  technical answer style and source references.
- **Product team** — keeps the same repository selection, retrieval, tools,
  evidence gathering, permissions, and provider fallback chain, but presents the
  final answer in concise everyday language without class names or technical
  terms.

## Security & privacy

- Self-hosted; private code stays on your infrastructure.
- Cookie-based sessions; passwords hashed with PBKDF2.
- Per-user LLM keys are encrypted at rest.
- Retrieval tuning is **configuration data only** — never browser-driven code
  execution.
- Per-repo control over whether the shared LLM tier may be used.
- Secrets live in environment / `.env` (gitignored); the local encryption key
  and database are never committed.
