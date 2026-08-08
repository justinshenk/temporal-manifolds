#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
GCLOUD_INSTALL_DIR="${GCLOUD_INSTALL_DIR:-${HOME}/google-cloud-sdk}"
GCLOUD_INSTALL_TMPDIR=""

log() {
  printf '[run-conversational-selected-acts] %s\n' "$*"
}

cleanup_gcloud_install() {
  if [[ -n "${GCLOUD_INSTALL_TMPDIR}" && -d "${GCLOUD_INSTALL_TMPDIR}" ]]; then
    rm -rf -- "${GCLOUD_INSTALL_TMPDIR}"
  fi
}

env_value() {
  local key value
  key="$1"
  value="$(
    grep -E "^[[:space:]]*${key}=" "${REPO_ROOT}/.env" \
      | tail -n 1 \
      | sed -E 's/^[^=]*=//; s/^[[:space:]]*//; s/[[:space:]]*$//; s/^["'"'"']//; s/["'"'"']$//'
  )"
  printf '%s\n' "${value}"
}

load_gcp_config() {
  if [[ ! -f "${REPO_ROOT}/.env" ]]; then
    printf 'Missing .env at %s\n' "${REPO_ROOT}/.env" >&2
    exit 1
  fi

  export GCP_PROJECT_ID="$(env_value GCP_PROJECT_ID)"
  export GCS_BUCKET_NAME="$(env_value GCS_BUCKET_NAME)"
  if [[ -z "${GCP_PROJECT_ID}" ]]; then
    printf 'GCP_PROJECT_ID must be set in %s\n' "${REPO_ROOT}/.env" >&2
    exit 1
  fi
  if [[ -z "${GCS_BUCKET_NAME}" ]]; then
    printf 'GCS_BUCKET_NAME must be set in %s\n' "${REPO_ROOT}/.env" >&2
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
    printf 'Cannot install gcloud CLI: %s exists but does not contain bin/gcloud\n' \
      "${GCLOUD_INSTALL_DIR}" >&2
    exit 1
  fi
  if ! command -v curl >/dev/null 2>&1 || ! command -v tar >/dev/null 2>&1; then
    printf 'Installing gcloud requires curl and tar.\n' >&2
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
      printf 'Unsupported OS/architecture for gcloud installation: %s/%s\n' \
        "${os}" "${arch}" >&2
      exit 1
      ;;
  esac

  GCLOUD_INSTALL_TMPDIR="$(mktemp -d)"
  trap cleanup_gcloud_install EXIT
  log "Installing gcloud CLI to ${GCLOUD_INSTALL_DIR}"
  curl -fsSL "https://dl.google.com/dl/cloudsdk/channels/rapid/downloads/${package}" \
    -o "${GCLOUD_INSTALL_TMPDIR}/${package}"
  tar -xzf "${GCLOUD_INSTALL_TMPDIR}/${package}" -C "${GCLOUD_INSTALL_TMPDIR}"
  "${GCLOUD_INSTALL_TMPDIR}/google-cloud-sdk/install.sh" \
    --quiet --path-update=false --usage-reporting=false
  mkdir -p "$(dirname "${GCLOUD_INSTALL_DIR}")"
  mv "${GCLOUD_INSTALL_TMPDIR}/google-cloud-sdk" "${GCLOUD_INSTALL_DIR}"
  export PATH="${GCLOUD_INSTALL_DIR}/bin:${PATH}"
  cleanup_gcloud_install
  trap - EXIT
}

authenticate_gcloud() {
  if gcloud auth application-default print-access-token >/dev/null 2>&1; then
    log "Application Default Credentials already available"
  else
    log "Starting Application Default Credentials login"
    gcloud auth application-default login --no-launch-browser
  fi
  gcloud config set project "${GCP_PROJECT_ID}" >/dev/null
  if ! gcloud auth application-default set-quota-project "${GCP_PROJECT_ID}" \
    >/dev/null 2>&1; then
    log "Could not set the ADC quota project; continuing with the available credentials"
  fi
  log "Using GCP project=${GCP_PROJECT_ID} bucket=${GCS_BUCKET_NAME}"
}

if ! command -v uv >/dev/null 2>&1; then
  printf 'Missing uv. Install it first: https://docs.astral.sh/uv/getting-started/installation/\n' >&2
  exit 1
fi

cd "${REPO_ROOT}"
load_gcp_config
install_gcloud_cli
authenticate_gcloud

if [[ "${SKIP_UV_SYNC:-0}" != "1" ]]; then
  log "Syncing Python dependencies from uv.lock"
  uv sync --locked
fi

log "Caching conversational layer_out/21 activations at token -1 to selected_acts/"
exec uv run python "${REPO_ROOT}/scripts/cache_conversational_selected_acts.py" "$@"
