#!/usr/bin/env bash
set -euo pipefail

SCRIPT_REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
REPO_URL="${REPO_URL:-https://github.com/justinshenk/temporal-manifolds.git}"
REPO_BRANCH="${REPO_BRANCH:-dev}"
if [[ -d "${SCRIPT_REPO_ROOT}/.git" ]]; then
  REPO_CLONE_DIR="${REPO_CLONE_DIR:-${SCRIPT_REPO_ROOT}}"
else
  REPO_CLONE_DIR="${REPO_CLONE_DIR:-${HOME}/temporal-manifolds}"
fi
REPO_ROOT="${REPO_CLONE_DIR}"
SCENARIO_DIR="${REPO_ROOT}/configs/scenarios"
GCLOUD_INSTALL_DIR="${GCLOUD_INSTALL_DIR:-${HOME}/google-cloud-sdk}"
GCLOUD_INSTALL_TMPDIR=""

if [[ -z "${MPLBACKEND:-}" || "${MPLBACKEND}" == "module://matplotlib_inline.backend_inline" ]]; then
  export MPLBACKEND=Agg
fi

log() {
  printf '[run-eap-scenarios] %s\n' "$*"
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

clone_dev_branch() {
  if [[ -d "${REPO_CLONE_DIR}/.git" ]]; then
    REPO_ROOT="$(cd "${REPO_CLONE_DIR}" && pwd)"
    SCENARIO_DIR="${REPO_ROOT}/configs/scenarios"
    log "Using repository at ${REPO_ROOT}"
    return
  fi

  if [[ -e "${REPO_CLONE_DIR}" ]]; then
    printf 'Cannot clone repository: %s already exists but is not a git checkout\n' "${REPO_CLONE_DIR}" >&2
    exit 1
  fi

  log "Cloning ${REPO_URL} branch ${REPO_BRANCH} into ${REPO_CLONE_DIR}"
  git clone --branch "${REPO_BRANCH}" --single-branch "${REPO_URL}" "${REPO_CLONE_DIR}"
  REPO_ROOT="$(cd "${REPO_CLONE_DIR}" && pwd)"
  SCENARIO_DIR="${REPO_ROOT}/configs/scenarios"
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

  cd "${REPO_ROOT}"

  log "Syncing Python dependencies from uv.lock"
  uv sync --locked

  log "Checking PyTorch installation"
  uv run python -c 'import torch; print(f"torch {torch.__version__}")'
}

scenario_workflow() {
  local scenario
  scenario="$1"
  grep -E '^[[:space:]]*workflow:[[:space:]]*"?eap"?[[:space:]]*$' "${scenario}" >/dev/null
}

run_scenarios() {
  if [[ ! -d "${SCENARIO_DIR}" ]]; then
    printf 'Missing scenario directory: %s\n' "${SCENARIO_DIR}" >&2
    exit 1
  fi

  cd "${REPO_ROOT}"

  local scenario found=0
  shopt -s nullglob
  for scenario in "${SCENARIO_DIR}"/*.yaml "${SCENARIO_DIR}"/*.yml; do
    if ! scenario_workflow "${scenario}"; then
      log "Skipping non-EAP scenario ${scenario#${REPO_ROOT}/}"
      continue
    fi

    found=1
    log "Running EAP scenario ${scenario#${REPO_ROOT}/}"
    uv run temporal-manifolds-workflow "${scenario}"
  done
  shopt -u nullglob

  if [[ "${found}" -eq 0 ]]; then
    printf 'No EAP scenario YAML files found in %s\n' "${SCENARIO_DIR}" >&2
    exit 1
  fi
}

main() {
  clone_dev_branch
  install_gcloud_cli
  prepare_env_file
  prepare_python_env
  login_gcloud_no_browser
  run_scenarios
}

main "$@"
