#!/usr/bin/env bash
set -euo pipefail

# Cache the fixed-contract activation for every dataset registered in
# temporal_manifolds.dataset.generate.DATASETS. Extra arguments can restrict or
# configure the run, for example: --dataset abstract --no-save-to-gcp.

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export ACTIVATION_CACHING_PYTHON_SCRIPT="${REPO_ROOT}/scripts/cache_all_selected_acts.py"

exec bash "${REPO_ROOT}/scripts/run_activation_caching_scenario.sh" \
  --output-root "${REPO_ROOT}/results" \
  "$@"
