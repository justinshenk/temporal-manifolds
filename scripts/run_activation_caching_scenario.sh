#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DEFAULT_SCENARIO_PATH="${REPO_ROOT}/configs/activation_caching/conversational_all.yaml"
SCENARIO_PATH="${SCENARIO_PATH:-${DEFAULT_SCENARIO_PATH}}"
ACTIVATION_CACHING_PYTHON_SCRIPT="${ACTIVATION_CACHING_PYTHON_SCRIPT:-}"
GCLOUD_INSTALL_DIR="${GCLOUD_INSTALL_DIR:-${HOME}/google-cloud-sdk}"
GCLOUD_INSTALL_TMPDIR=""

log() {
  printf '[run-activation-caching-scenario] %s\n' "$*"
}

cleanup_gcloud_install() {
  if [[ -n "${GCLOUD_INSTALL_TMPDIR}" && -d "${GCLOUD_INSTALL_TMPDIR}" ]]; then
    rm -rf "${GCLOUD_INSTALL_TMPDIR}"
  fi
}

env_value() {
  local key value
  key="$1"

  if [[ -n "${!key:-}" ]]; then
    printf '%s\n' "${!key}"
    return
  fi

  if [[ ! -f "${REPO_ROOT}/.env" ]]; then
    return
  fi

  value="$(
    grep -E "^[[:space:]]*${key}=" "${REPO_ROOT}/.env" \
      | tail -n 1 \
      | sed -E 's/^[^=]*=//; s/^["'"'"']//; s/["'"'"']$//'
  )"
  printf '%s\n' "${value}"
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

install_gcloud_cli() {
  if command -v gcloud >/dev/null 2>&1; then
    log "gcloud CLI already installed: $(command -v gcloud)"
    return
  fi

  if [[ -x "${GCLOUD_INSTALL_DIR}/bin/gcloud" ]]; then
    export PATH="${GCLOUD_INSTALL_DIR}/bin:${PATH}"
    log "Using existing gcloud CLI at ${GCLOUD_INSTALL_DIR}/bin/gcloud"
    return
  fi

  if [[ -e "${GCLOUD_INSTALL_DIR}" ]]; then
    printf 'Cannot install gcloud CLI: %s already exists but does not contain bin/gcloud\n' "${GCLOUD_INSTALL_DIR}" >&2
    exit 1
  fi

  local os arch package
  os="$(uname -s | tr '[:upper:]' '[:lower:]')"
  arch="$(uname -m)"

  case "${os}:${arch}" in
    linux:x86_64|linux:amd64) package="google-cloud-cli-linux-x86_64.tar.gz" ;;
    linux:aarch64|linux:arm64) package="google-cloud-cli-linux-arm.tar.gz" ;;
    darwin:x86_64|darwin:amd64) package="google-cloud-cli-darwin-x86_64.tar.gz" ;;
    darwin:aarch64|darwin:arm64) package="google-cloud-cli-darwin-arm.tar.gz" ;;
    *)
      printf 'Unsupported OS/architecture for automatic gcloud install: %s/%s\n' "${os}" "${arch}" >&2
      exit 1
      ;;
  esac

  GCLOUD_INSTALL_TMPDIR="$(mktemp -d)"
  trap cleanup_gcloud_install EXIT

  log "Installing gcloud CLI to ${GCLOUD_INSTALL_DIR}"
  curl -fsSL "https://dl.google.com/dl/cloudsdk/channels/rapid/downloads/${package}" \
    -o "${GCLOUD_INSTALL_TMPDIR}/${package}"
  tar -xzf "${GCLOUD_INSTALL_TMPDIR}/${package}" -C "${GCLOUD_INSTALL_TMPDIR}"
  "${GCLOUD_INSTALL_TMPDIR}/google-cloud-sdk/install.sh" --quiet --path-update=false --usage-reporting=false
  mv "${GCLOUD_INSTALL_TMPDIR}/google-cloud-sdk" "${GCLOUD_INSTALL_DIR}"
  export PATH="${GCLOUD_INSTALL_DIR}/bin:${PATH}"
}

login_gcloud_no_browser() {
  if gcloud auth application-default print-access-token >/dev/null 2>&1; then
    log "gcloud Application Default Credentials already available; skipping login"
  else
    log "Starting gcloud Application Default Credentials login without launching a browser"
    gcloud auth application-default login --no-launch-browser
  fi

  local project_id
  project_id="$(env_value GCP_PROJECT_ID)"
  if [[ -n "${project_id}" ]]; then
    log "Setting gcloud project to ${project_id}"
    gcloud config set project "${project_id}" >/dev/null

    if [[ "${SET_ADC_QUOTA_PROJECT:-0}" == "1" ]]; then
      log "Setting ADC quota project to ${project_id}"
      if ! gcloud auth application-default set-quota-project "${project_id}" >/dev/null; then
        log "Could not set ADC quota project. Continuing because ADC credentials are present."
      fi
    fi
  fi
}

prepare_env_file() {
  if [[ ! -f "${REPO_ROOT}/.env.example" ]]; then
    printf 'Missing .env.example at %s\n' "${REPO_ROOT}/.env.example" >&2
    exit 1
  fi

  if [[ -f "${REPO_ROOT}/.env" && "${FORCE_ENV_COPY:-0}" != "1" ]]; then
    log ".env already exists; leaving it unchanged. Set FORCE_ENV_COPY=1 to overwrite it."
    return
  fi

  cp "${REPO_ROOT}/.env.example" "${REPO_ROOT}/.env"
  log "Copied .env.example to .env"
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
  if [[ -n "${ACTIVATION_CACHING_PYTHON_SCRIPT}" ]]; then
    log "Running ${ACTIVATION_CACHING_PYTHON_SCRIPT#${REPO_ROOT}/}"
    uv run python "${ACTIVATION_CACHING_PYTHON_SCRIPT}" "$@"
    return
  fi

  log "Running ${SCENARIO_PATH#${REPO_ROOT}/}"
  uv run temporal-manifolds-workflow "${SCENARIO_PATH}"
}

main() {
  cd "${REPO_ROOT}"
  if [[ -n "${ACTIVATION_CACHING_PYTHON_SCRIPT}" ]]; then
    require_file "${ACTIVATION_CACHING_PYTHON_SCRIPT}" "activation-caching Python script"
  else
    require_file "${SCENARIO_PATH}" "activation-caching scenario config"
  fi
  prepare_env_file
  install_gcloud_cli
  prepare_python_env
  login_gcloud_no_browser
  run_scenario "$@"
}

main "$@"
