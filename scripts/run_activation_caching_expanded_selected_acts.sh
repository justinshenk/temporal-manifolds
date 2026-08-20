#!/usr/bin/env bash
set -euo pipefail

# Caches layer_out/17..35 at prompt tokens -2 and -1 for every dataset and uploads
# each dataset to its own GCS directory under an `expanded_` prefix.

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export ACTIVATION_CACHING_PYTHON_SCRIPT="${REPO_ROOT}/scripts/cache_expanded_selected_acts.py"

exec bash "${REPO_ROOT}/scripts/run_activation_caching_scenario.sh" \
  --first-layer 17 \
  --last-layer 35 \
  --output-root "${REPO_ROOT}/results" \
  "$@"
