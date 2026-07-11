#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DEFAULT_SCENARIO_PATH="${REPO_ROOT}/configs/activation_caching/conversational_default.yaml"
SCENARIO_PATH="${SCENARIO_PATH:-${DEFAULT_SCENARIO_PATH}}"

log() {
  printf '[run-activation-caching-scenario] %s\n' "$*"
}

require_file() {
  local path description
  path="$1"
  description="$2"
  if [[ ! -f "${path}" ]]; then
    printf 'Missing %s: %s\n' "${description}" "${path}" >&2
    exit 1
  fi
}

prepare_python_env() {
  if ! command -v uv >/dev/null 2>&1; then
    printf 'Missing uv. Install it first: https://docs.astral.sh/uv/getting-started/installation/\n' >&2
    exit 1
  fi

  if [[ "${SKIP_UV_SYNC:-0}" == "1" ]]; then
    log "Skipping uv sync because SKIP_UV_SYNC=1"
    return
  fi

  log "Syncing Python dependencies from uv.lock"
  uv sync --locked
}

run_scenario() {
  log "Running ${SCENARIO_PATH#${REPO_ROOT}/}"
  uv run temporal-manifolds-workflow "${SCENARIO_PATH}"
}

main() {
  cd "${REPO_ROOT}"
  require_file "${SCENARIO_PATH}" "activation-caching scenario config"
  require_file "${REPO_ROOT}/.env" ".env file containing GCP_PROJECT_ID and GCS_BUCKET_NAME"
  prepare_python_env
  run_scenario
}

main "$@"
