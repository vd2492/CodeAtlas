#!/usr/bin/env bash
# Dev launcher. Creates a venv on first run, installs deps, starts the server.
set -e
cd "$(dirname "$0")"

if [ ! -d .venv ]; then
  python3 -m venv .venv
  ./.venv/bin/pip install -q --upgrade pip
  ./.venv/bin/pip install -q -r requirements.txt
fi

[ -f .env ] && set -a && . ./.env && set +a

exec ./.venv/bin/uvicorn app.main:app --reload --reload-dir app --port 8000
