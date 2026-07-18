#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export SCENARIO_PATH="${SCENARIO_PATH:-${REPO_ROOT}/configs/activation_caching/conversational_after_assistant_residual_stream.yaml}"

exec bash "${REPO_ROOT}/scripts/run_activation_caching_scenario.sh" "$@"
