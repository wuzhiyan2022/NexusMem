#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"

cd "${REPO_ROOT}"

export LINEARRAG_EMBEDDING_MODEL="${LINEARRAG_EMBEDDING_MODEL:-model/all-mpnet-base-v2}"
export LINEARRAG_SPACY_MODEL="${LINEARRAG_SPACY_MODEL:-en_core_web_trf}"
export LINEARRAG_STORAGE_DIR="${LINEARRAG_STORAGE_DIR:-import_api}"
export LINEARRAG_DEFAULT_TOP_K="${LINEARRAG_DEFAULT_TOP_K:-100}"

HOST="${LINEARRAG_HOST:-0.0.0.0}"
PORT="${LINEARRAG_PORT:-8000}"

if [[ -z "${LINEARRAG_API_TOKEN:-${MEMORY_API_KEY:-}}" ]]; then
  echo "WARNING: LINEARRAG_API_TOKEN is not set; /add and /search will accept unauthenticated requests." >&2
fi

exec uvicorn api_server:app --host "${HOST}" --port "${PORT}"
