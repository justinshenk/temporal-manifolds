#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

log() {
  printf '[run-conversational-selected-acts] %s\n' "$*"
}

if ! command -v uv >/dev/null 2>&1; then
  printf 'Missing uv. Install it first: https://docs.astral.sh/uv/getting-started/installation/\n' >&2
  exit 1
fi

cd "${REPO_ROOT}"

if [[ "${SKIP_UV_SYNC:-0}" != "1" ]]; then
  log "Syncing Python dependencies from uv.lock"
  uv sync --locked
fi

if command -v gcloud >/dev/null 2>&1; then
  if ! gcloud auth application-default print-access-token >/dev/null 2>&1; then
    log "Starting Application Default Credentials login"
    gcloud auth application-default login --no-launch-browser
  fi
else
  log "gcloud is unavailable; using credentials configured through the environment"
fi

log "Caching conversational layer_out/21 activations at token -1 to selected_acts/"
exec uv run python "${REPO_ROOT}/scripts/cache_conversational_selected_acts.py" "$@"
