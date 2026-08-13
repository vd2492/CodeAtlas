# CodeAtlas

**Self-hostable codebase intelligence.** Host private repositories, index them
into a queryable graph, safely tune retrieval per repo, control who can access
each one, and let PMs, QAs, developers, and stakeholders ask grounded questions
about the code, with answers tailored to their audience, without giving them
direct repository access.

Everything runs on your own box. Private code never has to leave it.

---

## Table of Contents

1. [Functionality, Features, and Tech Stack](#1-functionality-features-and-tech-stack)
   - [What CodeAtlas Does](#what-codeatlas-does)
   - [Authentication and Access Model](#authentication-and-access-model)
   - [Repository Lifecycle](#repository-lifecycle)
   - [LLM and Retrieval Behavior](#llm-and-retrieval-behavior)
   - [Architecture and Tech Stack](#architecture-and-tech-stack)
   - [Roles and User Types](#roles-and-user-types)
   - [Security and Privacy](#security-and-privacy)
2. [Setup Instructions and Application Usage](#2-setup-instructions-and-application-usage)
   - [Local Setup](#local-setup)
   - [Docker and VM Setup](#docker-and-vm-setup)
   - [Environment Configuration](#environment-configuration)
   - [Google SSO Setup](#google-sso-setup)
   - [Slack Integration Setup](#slack-integration-setup)
   - [Admin Workflow](#admin-workflow)
   - [User Workflow](#user-workflow)
   - [Slack User Workflow](#slack-user-workflow)
   - [Production and Staging Notes](#production-and-staging-notes)
   - [Troubleshooting](#troubleshooting)

---

## 1. Functionality, Features, and Tech Stack

### What CodeAtlas Does

- **Index any repo into a graph.** An admin clones a repository using HTTPS,
  SSH, or the GitHub CLI and indexes it into a structural graph of files,
  symbols, and relations. Indexing does not require an LLM.
- **Ask grounded questions.** Users ask in natural language, such as "How does
  login work?" or "Which files are involved in this feature?" The selected model
  searches the graph, follows symbols, and reads real source before answering.
- **Audience-aware answers.** Dev-team users receive technical answers with
  file and line references. Product-team users receive concise, plain-language
  answers without class names or internal implementation terminology.
- **Per-repo retrieval tuning.** Admins improve retrieval quality with safe,
  data-only controls such as stopwords, synonyms, keyword boosts, preferred
  components or methods, context sizes, excerpt sizes, and pre-search
  terminology instructions. No code is ever executed from the browser.
- **Access control.** Users only see repositories an admin has explicitly
  granted them. Every query is permission-checked.
- **Branch-aware answers.** Admins approve remote branches. CodeAtlas indexes
  each branch in an isolated Git worktree, and users select the exact indexed
  branch and commit before asking.
- **Freshness tracking.** Freshness checks and **Sync & index now** keep the
  active graph aligned with the remote branch.
- **Bring your own LLM key.** Each user can store their own LLM key, encrypted
  at rest, to be used as their first-choice model.
- **Web image questions.** In the web Ask UI, users can attach or paste PNG,
  JPEG, or WebP screenshots/images to a new single-branch question. Images are
  transient request context only; they are not indexed or stored.
- **Slack ask surface.** A Slack workspace can use `/codeatlas` to ask the same
  grounded text questions from Slack against published repositories and approved
  branches.

### Authentication and Access Model

CodeAtlas supports a Google-first migration path:

- **Password mode** keeps the existing username/password login only.
- **Mixed mode** keeps username/password login and adds Google sign-in.
- **Google mode** uses Google sign-in only and disables credential user
  creation, password bootstrap, and password updates.

Admins can manage users in three practical ways:

- Create a normal credential user with username, password, role, and user type
  while running in password or mixed mode.
- Grant access for Google/Gmail login by entering an email address, role, user
  type, and repository grants.
- Edit an existing user to associate or update a Gmail ID so that user can sign
  in with Google without losing existing repository access.

Google users are provisioned before login by default. A first Google sign-in
links the Google identity to the provisioned email, then future logins use that
Google identity. For a fresh Google-only deployment, set
`CODEATLAS_GOOGLE_BOOTSTRAP_ADMIN_EMAILS` to allow one or more Gmail IDs to
create the first admin through Google sign-in.

### Repository Lifecycle

`Clone -> Approve branches -> Index -> Test -> Tune -> Publish -> Grant access`

An admin clones a repo, approves and indexes its remote branches, tests retrieval
and answer quality, tunes the per-repo config until answers are good, publishes
the workspace, and grants access to selected users. Users then log in, pick an
authorized repo and indexed branch, inspect its commit and freshness metadata,
and ask questions.

### LLM and Retrieval Behavior

Every question resolves through `app/llm/client.py` in order:

1. **User key (BYOK)**
2. **Shared endpoint**

Each tier falls through on absence or failure. The shared tier can be disabled
per repository with `allow_shared_fallback`, so sensitive code is never sent to a
shared endpoint. The dormant Ollama code is disabled by default with
`CODEATLAS_ENABLE_OLLAMA=false` and has no user-facing control.

Tool-capable models receive seven read-only repository tools:

- `search_code`, `read_file`, and `list_directory`
- `find_definition`, `find_references`, and `get_callers`
- `ask_user` to pause and ask a short clarifying question instead of guessing

The model can search, inspect the result, follow a relation, and read additional
source over several rounds. Tools are workspace-scoped, path traversal and
likely secret files are blocked, and all reads have line and byte limits. If an
endpoint does not support tool calling, or the selected model answers without
using a tool, CodeAtlas automatically uses the original one-shot context path
for that provider.

When a search or definition result turns up multiple similarly ranked matches in
unrelated parts of the repository, the model can call `ask_user` instead of
silently picking one. That question becomes the answer for the turn. Reply in
the follow-up question field so the investigation resumes with the right
context instead of starting over. Product-team answers keep the clarifying
question in plain language.

A query in progress can be cancelled from either the main ask field or the
follow-up field.

The API response reports `retrieval_mode` (`agentic` or `one_shot`) and a compact
`agent_trace`. The Ask UI displays this investigation under Grounded Evidence.

#### Fast Grounded Follow-Ups

The Ask UI sends an opaque conversation ID instead of replaying previous answers
from the browser. For a related follow-up, the server reuses the prior verified
evidence when the authenticated user, repository, indexed branch revision,
audience type, and LLM mode still match.

A related question first uses a compact, tool-free request with a smaller output
budget. The model must request more evidence instead of guessing when that
compact context is insufficient. CodeAtlas then automatically runs the normal
full retrieval and read-only agent pipeline.

Cached follow-up answers also offer **Investigate deeply**, which reruns that
same question through the full repository investigation on demand.

If a user asks the same question again in the same authenticated session,
repository, branch revision, audience type, and LLM mode, CodeAtlas returns the
previous successful answer from a bounded in-memory session cache with zero new
LLM tokens. Those repeated-question answers are clearly labeled and also offer
**Investigate deeply** to bypass the cache and refresh the answer.

A second, broader cache sits behind the per-session cache. A fresh
non-follow-up question answered by the shared LLM tier is cached per repository,
indexed revision, and audience type, not per user or session. If ten people on a
team ask the same question, only the first pays for a full investigation; the
rest are served instantly with zero new tokens.

Deep investigations and BYOK-answered questions are never stored or served
through the repo-scoped cache, since a personal key's answer is that user's own
paid-for compute. This repo-scoped cache defaults to a 6-hour lifetime and up to
1,000 entries. See `CODEATLAS_REPO_ANSWER_CACHE_*` in `.env.example`.

Conversation state is an in-process TTL cache and contains no API credentials.
It is an optional acceleration layer: losing it on restart only makes the next
question use full retrieval. Configure its lifetime and bounds with the
`CODEATLAS_CONVERSATION_*` variables in `.env.example`. Official Anthropic API
requests also enable its documented ephemeral prompt cache; compatible
third-party endpoints are left unchanged. Responses include `timings_ms`, and
the Ask UI shows total server time beside the retrieval mode.

### Architecture and Tech Stack

- **Backend:** FastAPI and Python.
- **Database:** SQLite for users, repositories, repo access, sessions, and audit
  logs.
- **Frontend:** vanilla HTML, CSS, and JavaScript for the landing page, Ask UI,
  and admin console.
- **Indexing:** structural graph extraction through the pinned `graphify`
  indexer, with no LLM required for indexing.
- **Retrieval:** agentic, read-only repository investigation for tool-capable
  models, with fallback to one-shot keyword and graph ranking.
- **Authentication:** cookie sessions, password login, and optional Google
  sign-in.
- **Integrations:** optional Slack slash command and modal flow.
- **Deployment:** local virtualenv, Docker Compose, or VM/container deployment.

```
app/
  ask_service.py     shared ask orchestration for web and Slack surfaces
  agent/tools.py     workspace-scoped source + graph tools for the LLM
  main.py            FastAPI app + query/answer endpoints, startup wiring
  config.py          paths & per-workspace layout
  db.py              SQLite: users, repos, repo_access, sessions, audit_log
  auth/              sessions, password hashing, Google auth, BYOK encryption
  repos/             clone, branch worktrees, freshness jobs, indexing, routes
  retrieval/         ranker, context builder, per-repo RetrievalConfig
  llm/client.py      agent loops + BYOK -> shared fallback; dormant Ollama hook
  slack/             /codeatlas slash command, modal, and Slack answer delivery
  static/            landing page, user Ask UI, admin console
data/                gitignored: sqlite db, cloned repos, graphs, secret key
docs/PLAN.md         build plan / phase history
```

### Roles and User Types

- **Admin** can clone and index repos, test and tune retrieval, publish
  repositories, manage approved branches and sync settings, manage users and
  per-repo access, toggle the shared-LLM privacy setting, and review the audit
  log.
- **User** can log in with admin-provided credentials or a provisioned Google
  account, see only authorized repos, select an indexed branch, optionally
  request an authorized refresh, optionally set a personal LLM key, and ask
  grounded questions.

Role controls permissions. User type controls only how final LLM answers are
presented:

- **Dev team** keeps the technical answer style and source references.
- **Product team** keeps the same repository selection, retrieval, tools,
  evidence gathering, permissions, and provider fallback chain, but presents the
  final answer in concise everyday language without class names or technical
  terms.

### Security and Privacy

- Self-hosted; private code stays on your infrastructure.
- Cookie-based sessions; passwords are hashed with PBKDF2.
- Per-user LLM keys are encrypted at rest.
- Retrieval tuning is configuration data only, never browser-driven code
  execution.
- Per-repo control decides whether the shared LLM tier may be used.
- Slack requests are verified with the Slack signing secret.
- Google sign-in can be restricted by domain and requires admin provisioning by
  default.
- Secrets live in environment variables or `.env`, which is gitignored. The
  local encryption key and database are never committed.

---

## 2. Setup Instructions and Application Usage

### Local Setup

Start the local server:

```bash
./run.sh
```

The script creates a virtualenv, installs dependencies, and starts the server on
port `8000`.

- **Landing page:** http://localhost:8000/
- **Ask UI:** http://localhost:8000/app
- **Admin console:** http://localhost:8000/admin.html

On first run, the admin console walks you through creating the first admin
account. A default demo workspace is seeded so the tool works immediately.

### Docker and VM Setup

The production image includes Git, SSH, the GitHub CLI, and the pinned
`graphify` indexer.

```bash
cp .env.example .env
docker compose up -d --build
docker compose ps
```

By default, Compose publishes port `8000` and mounts `./data` at `/app/data`.
Set `CODEATLAS_COOKIE_SECURE=false` in `.env` only when testing over local plain
HTTP. Keep it `true` behind the production HTTPS load balancer.

For a VM with a separately mounted persistent disk, set:

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

For private GitHub and Bitbucket HTTPS repositories, inject centrally managed
read-only credentials into the container:

```bash
GH_TOKEN=<read-only GitHub token>
CODEATLAS_BITBUCKET_API_TOKEN=<read-only Bitbucket token>
```

CodeAtlas supplies these credentials to Git through `GIT_ASKPASS` only for
`github.com` and `bitbucket.org`. Credentials are never stored in clone URLs,
SQLite, audit logs, or command arguments. SSH deploy keys are still supported
through a read-only `/home/codeatlas/.ssh` mount when that fits your deployment
better. Never bake repository credentials into the image.

The container intentionally runs one Uvicorn worker because branch polling is
in-process and the application uses SQLite. When placing it behind a GCP
internal HTTPS Application Load Balancer, use `/healthz` for the health check,
allow backend port `8000` only from the proxy-only subnet and health-check
ranges, and set the backend timeout high enough for synchronous indexing.

Long-running answer requests are protected by a bounded, process-local FIFO
queue. The Compose defaults allow 12 active LLM pipelines and 20 queued
requests, with a 15-second queue timeout. Excess traffic receives a retryable
`503` response while login, admin, and health-check traffic remains responsive.
Temporary provider `429`, `502`, `503`, and `504` responses are retried twice.

Tune these only after load-testing the target VM:

```bash
CODEATLAS_MAX_CONCURRENT_LLM_REQUESTS=12
CODEATLAS_MAX_QUEUED_LLM_REQUESTS=20
CODEATLAS_LLM_QUEUE_TIMEOUT_SECONDS=15
CODEATLAS_PROVIDER_RETRIES=2
```

### Environment Configuration

All configuration is via environment variables. Copy `.env.example` to `.env`
and fill in what you need. The `.env` file is gitignored.

Main configuration groups:

- **Shared LLM tier:** OpenAI-compatible or Anthropic-compatible endpoint used
  as fallback for answering questions.
- **Reserved Ollama integration:** retained behind a disabled feature flag for a
  future release; it is not exposed to users.
- **Private Git credentials:** optional read-only GitHub or Bitbucket secrets
  used for private HTTPS clone, branch discovery, fetch, freshness checks, and
  sync.
- **Paths and source root:** optional overrides for data directory and the source
  tree used to pull code excerpts into answers.
- **Sessions and login throttling:** auth mode, Google client ID, cookie
  settings, session lifetime, and login rate limits.
- **Branch synchronization:** worker count, freshness polling, user-triggered
  sync cooldown, and old-version retention.
- **Web image attachments:** optional count and size limits for transient image
  context on new single-branch web questions.
- **Slack org ask surface:** optional `/codeatlas` slash command integration.

See `.env.example` for the full list of keys and inline notes.

### Google SSO Setup

Use Google SSO when admins should grant access by Google email/Gmail ID instead
of only creating username/password users.

1. In Google Cloud Console, create or select the project for CodeAtlas auth.
2. Configure the OAuth consent screen for your organization.
3. Create an OAuth Client ID of type **Web application**.
4. Add authorized JavaScript origins for each host that serves CodeAtlas:

```text
http://localhost:8000
https://codeatlas.example.com
```

This implementation uses Google Identity Services to receive an ID token in the
browser and posts it to `/auth/google`; it does not require a separate redirect
callback URL for normal operation.

Set the server environment:

```bash
CODEATLAS_AUTH_MODE=mixed
CODEATLAS_GOOGLE_CLIENT_ID=<Google OAuth web client ID>
CODEATLAS_GOOGLE_ALLOWED_DOMAINS=example.com
CODEATLAS_GOOGLE_BOOTSTRAP_ADMIN_EMAILS=admin@example.com
CODEATLAS_GOOGLE_AUTO_CREATE=false
```

Use `mixed` during migration so existing username/password users continue to
work while Google login is introduced. Associate every existing user with a
Gmail ID, verify at least one admin can sign in with Google, then switch to
`CODEATLAS_AUTH_MODE=google`. In Google mode, admins can still map existing
users to Gmail IDs and adjust roles, user type, and repository grants, but
credential creation and password updates are disabled. Keep
`CODEATLAS_GOOGLE_AUTO_CREATE=false` so admins must grant or associate Google
emails before users can enter the application.

After changing Google auth settings, restart the CodeAtlas service.

### Slack Integration Setup

CodeAtlas can expose the same answer engine used by the browser UI through a
Slack slash command.

In Slack, `/codeatlas` opens a modal with:

- Repository
- Ask type: single branch answer, or compare 2 branch answer
- Branch, or base branch and compare branch
- User type
- Question

Only published repositories are listed. Branch dropdowns list approved branches.
When a user selects a branch, CodeAtlas starts sync/index in the background. The
submitted answer waits for the selected branch to become ready before answering
with the same retrieval, cache, follow-up, and **Investigate deeply** behavior
as the web Ask UI.

Create a Slack app for the target workspace, add the `/codeatlas` slash command,
enable interactivity, and install the app with these bot scopes:

```text
commands
chat:write
```

For local testing through ngrok, start CodeAtlas locally and expose port `8000`:

```bash
ngrok http 8000
```

Use the ngrok HTTPS host with these paths in the Slack app:

```text
Slash command Request URL: https://<ngrok-host>/slack/commands
Interactivity Request URL: https://<ngrok-host>/slack/interactions
```

For staging, configure Slack with:

```text
Slash command Request URL: https://codeatlas.example.com/slack/commands
Interactivity Request URL: https://codeatlas.example.com/slack/interactions
```

Set these on the VM/container environment:

```bash
CODEATLAS_SLACK_ENABLED=true
CODEATLAS_SLACK_SIGNING_SECRET=<Slack app signing secret>
CODEATLAS_SLACK_BOT_TOKEN=<Bot User OAuth Token, xoxb-...>
CODEATLAS_SLACK_ALLOWED_TEAM_IDS=<Slack workspace ID, T...>
CODEATLAS_SLACK_LLM_MODE=auto
CODEATLAS_SLACK_BRANCH_WAIT_SECONDS=900
```

`CODEATLAS_SLACK_ALLOWED_TEAM_IDS` is a workspace allowlist, not a person or
channel allowlist. Keeping it set still lets anyone in that Slack workspace use
the command in DMs, groups, or channels where Slack allows the command, while
blocking requests from external workspaces. Leave it blank only for deliberately
open internal testing.

Slack does not use per-user BYOK keys. It sends questions through the shared LLM
tier configured for the server, so keep the shared fallback and repo privacy
settings aligned with your production policy.

After changing Slack scopes, reinstall the Slack app to the workspace. After
changing CodeAtlas environment variables, restart the CodeAtlas service.

### Admin Workflow

1. Log in to `/admin.html`.
2. On first run, create the first admin account with the enabled auth mode.
3. Clone the required repository.
4. Approve the remote branches that should be available.
5. Run **Sync & index now** for the required branches.
6. Test answer quality from the admin tools.
7. Tune retrieval configuration when needed.
8. Publish the repository.
9. Create or grant users access:
   - Use **Grant Access for Gmail Login** to provision a Google email, choose
     role, choose user type, and grant one or more repositories.
   - Use **Create User Using Credentials** only while running password or mixed
     mode.
   - Use **Edit user** to update username, associated Gmail ID, and user type;
     password updates are available only outside Google-only mode.
10. Review repo access and audit logs as needed.

### User Workflow

1. Open `/app`.
2. Sign in with username/password or Google, depending on what the admin has
   provisioned and what `CODEATLAS_AUTH_MODE` allows.
3. Select an authorized repository.
4. Select an indexed branch and review commit/freshness metadata.
5. Ask a question, optionally attaching or pasting PNG, JPEG, or WebP images in
   the main web query field.
6. Ask follow-up questions in the same topic when needed.
7. Use **Investigate deeply** when the cached or fast follow-up answer needs a
   full repository investigation.
8. Optionally save a personal LLM key if BYOK is enabled for their workflow.

### Slack User Workflow

1. Type `/codeatlas` in Slack.
2. Select a published repository.
3. Select ask type:
   - **Single branch answer** for one branch.
   - **Compare 2 branch answer** for branch-to-branch comparison.
4. Select the required branch, or base and compare branches.
5. Select user type.
6. Enter the question and submit.
7. Receive an ephemeral Slack answer.
8. Use **Ask follow-up**, **Investigate deeply**, or **New question** from the
   Slack answer actions.

### Production and Staging Notes

For staging or production, ask DevOps to:

1. Pull the latest `main`.
2. Set the required `.env` values on the VM/container environment.
3. Make sure the persistent data directory exists and is writable by UID/GID
   `10001`.
4. Restart the CodeAtlas service.
5. Confirm `/healthz` returns healthy.
6. Update Google authorized JavaScript origins if the host changed.
7. Replace ngrok Slack URLs with staging or production Slack callback URLs.
8. Reinstall the Slack app if scopes changed.

Do not send or commit `.env` files. Share secrets through the team's approved
secret-management path.

### Troubleshooting

- **Google `invalid_client`:** confirm `CODEATLAS_GOOGLE_CLIENT_ID` is the Web
  OAuth client ID and the CodeAtlas host is listed in authorized JavaScript
  origins.
- **Google button missing:** confirm `CODEATLAS_AUTH_MODE=mixed` or `google`,
  `CODEATLAS_GOOGLE_CLIENT_ID` is set, and the service was restarted.
- **Google login denied:** confirm the admin provisioned or associated the same
  email address and that `CODEATLAS_GOOGLE_ALLOWED_DOMAINS` allows the domain.
- **Slack `/codeatlas` returns 404:** confirm the VM has the latest code,
  `CODEATLAS_SLACK_ENABLED=true`, the service was restarted, and Slack points to
  `/slack/commands`.
- **Slack modal action times out:** confirm the interactivity URL points to
  `/slack/interactions` and the public HTTPS URL reaches the VM.
- **Slack branch list is empty:** publish the repository and approve/index the
  required branches from the admin console first.
- **Sync returns 429:** the branch sync cooldown is active. Wait for the
  displayed retry time before triggering another manual sync.
- **Git reports a stale lock file:** make sure no Git process is running for
  that repo. If the process already crashed, remove the stale lock file from the
  affected repository worktree and run sync again.
- **SQLite cannot open the database:** confirm `CODEATLAS_DATA_PATH` points to an
  existing directory writable by UID/GID `10001`, then recreate the service:

```bash
docker compose down
docker compose up -d --build
docker compose logs -f app
```
