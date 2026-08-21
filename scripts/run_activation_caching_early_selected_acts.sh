#!/usr/bin/env bash
set -euo pipefail

# Caches layer_out/0..16 at prompt tokens -2 and -1 for every dataset and uploads each
# dataset under an `early_` GCS prefix, so these batches stay separate from the
# `expanded_` (layers 17..35) caches.

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export ACTIVATION_CACHING_PYTHON_SCRIPT="${REPO_ROOT}/scripts/cache_expanded_selected_acts.py"

exec bash "${REPO_ROOT}/scripts/run_activation_caching_scenario.sh" \
  --first-layer 0 \
  --last-layer 16 \
  --prefix-tag early_ \
  --output-root "${REPO_ROOT}/results" \
  "$@"
