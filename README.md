# CodeAtlas

**Self-hostable codebase intelligence.** Host private repositories, index them
into a queryable graph, safely tune retrieval per repo, control who can access
each one, and let PMs, QAs, developers, and stakeholders ask grounded questions
about the code — with answers that cite real files and lines — without giving
them direct repository access.

Everything runs on your own box. Private code never has to leave it.

---

## What it does

- **Index any repo into a graph.** An admin clones a repository (HTTPS, SSH, or
  the GitHub CLI) and indexes it into a structural graph of files, symbols, and
  relations — no LLM needed for indexing.
- **Ask grounded questions.** Users ask in natural language ("How does login
  work?", "Which files are involved in this feature?") and the selected model
  iteratively searches the graph, follows symbols, and reads real source before
  answering with file/line references.
- **Per-repo, config-driven tuning.** Admins improve retrieval quality with
  safe, data-only knobs (stopwords, synonyms, keyword boosts, preferred
  components/methods, context/excerpt sizes). No code is ever executed from the
  browser.
- **Access control.** Users log in and only see repositories an admin has
  explicitly granted them; every query is permission-checked.
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

## Configuration

All configuration is via environment variables — copy `.env.example` to `.env`
and fill in what you need (the `.env` file is gitignored). Nothing is required
to boot; the relevant groups are:

- **Shared LLM tier** — an OpenAI-compatible or Anthropic-compatible endpoint
  used as a fallback for answering questions.
- **Local Ollama** — point at a local Ollama instance + code model for a free,
  fully private tier.
- **Paths / source root** — optional overrides for data directory and the source
  tree used to pull code excerpts into answers.

See `.env.example` for the full list of keys and inline notes.

### LLM fallback chain

Every question resolves through `app/llm/client.py` in order:

1. **User key (BYOK)** → 2. **Local Ollama** → 3. **Shared endpoint**

Each tier falls through on absence *or* failure. The shared tier can be disabled
per repository (`allow_shared_fallback`), so sensitive code is never sent to a
shared endpoint.

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

## How a repository goes live

`Clone → Index → Test → Tune → Publish → Grant access`

An admin clones and indexes a repo, tests retrieval and answer quality against
it, tunes the per-repo config until answers are good, publishes the workspace,
and grants access to selected users. Users then log in, pick an authorized repo,
and ask away.

## Architecture

- **Backend:** FastAPI (Python), **SQLite** for users / repos / access /
  sessions / audit log.
- **Frontend:** vanilla HTML/CSS/JS — a marketing landing page, a user Ask UI,
  and an admin console (dark/light themed).
- **Indexing:** a structural graph extractor (no LLM required).
- **Retrieval:** keyword + graph ranking driven by a per-workspace
  `RetrievalConfig`; builds a compact context of nodes, relations, and source
  excerpts that is handed to the LLM.

```
app/
  agent/tools.py     workspace-scoped source + graph tools for the LLM
  main.py            FastAPI app + query/answer endpoints, startup wiring
  config.py          paths & per-workspace layout
  db.py              SQLite: users, repos, repo_access, sessions, audit_log
  auth/              sessions, password hashing, BYOK key encryption, auth routes
  repos/             clone (https/ssh/gh), indexing, admin lifecycle routes
  retrieval/         ranker, context builder, per-repo RetrievalConfig
  llm/client.py      agent loops + BYOK → Ollama → shared fallback chain
  static/            landing page, user Ask UI, admin console
data/                gitignored: sqlite db, cloned repos, per-workspace graphs/config, secret key
docs/PLAN.md         build plan / phase history
```

## Roles

- **Admin** — clone & index repos, test and tune retrieval, publish, manage
  users and per-repo access, toggle the shared-LLM privacy setting, and review
  an audit log of privileged actions.
- **User** — log in with admin-provided credentials, see only authorized repos,
  optionally set their own LLM key, and ask grounded questions.

## Security & privacy

- Self-hosted; private code stays on your infrastructure.
- Cookie-based sessions; passwords hashed with PBKDF2.
- Per-user LLM keys are encrypted at rest.
- Retrieval tuning is **configuration data only** — never browser-driven code
  execution.
- Per-repo control over whether the shared LLM tier may be used.
- Secrets live in environment / `.env` (gitignored); the local encryption key
  and database are never committed.
