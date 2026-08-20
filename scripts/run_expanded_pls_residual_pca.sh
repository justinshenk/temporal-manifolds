#!/usr/bin/env bash
set -euo pipefail

# Downloads each expanded_* activation folder, fits 6 PLS components against
# log10_time_horizon_months on layer_out/21 (both cached positions concatenated),
# fits 3 PCA components on the PLS residuals, writes one CSV per folder, and
# deletes the local copy before moving on.

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export ACTIVATION_CACHING_PYTHON_SCRIPT="${REPO_ROOT}/scripts/fit_expanded_pls_residual_pca.py"

exec bash "${REPO_ROOT}/scripts/run_activation_caching_scenario.sh" \
  --data-root "${REPO_ROOT}/data/expanded_pls" \
  --output-dir "${REPO_ROOT}/results/expanded_pls" \
  "$@"
