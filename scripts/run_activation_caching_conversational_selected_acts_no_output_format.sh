#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export ACTIVATION_CACHING_PYTHON_SCRIPT="${REPO_ROOT}/scripts/cache_conversational_selected_acts.py"

exec bash "${REPO_ROOT}/scripts/run_activation_caching_scenario.sh" \
  --remove-output-format-constraints \
  --gcs-prefix NOF_selected_acts \
  --output-dir "${REPO_ROOT}/results/selected_acts_no_output_format" \
  "$@"
