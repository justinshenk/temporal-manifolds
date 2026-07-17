#!/usr/bin/env bash
set -euo pipefail

PROJECT_ID="temporal-interp-exp"
BUCKET_NAME="temporal-research-bucket"

REMOTE_PATH="${1:-}"
LOCAL_DESTINATION="${2:-.}"

if [[ -z "${REMOTE_PATH}" ]]; then
  read -r -p "Path in gs://${BUCKET_NAME}/ to download: " REMOTE_PATH
fi

if [[ -z "${REMOTE_PATH}" ]]; then
  printf 'A bucket path is required.\n' >&2
  exit 1
fi

if gcloud auth print-access-token >/dev/null 2>&1; then
  printf 'Existing gcloud authentication found; skipping login.\n'
else
  gcloud auth login --no-launch-browser
fi

gcloud config set project "${PROJECT_ID}"
gcloud storage cp --recursive \
  "gs://${BUCKET_NAME}/${REMOTE_PATH#/}" \
  "${LOCAL_DESTINATION}"
