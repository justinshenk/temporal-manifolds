from __future__ import annotations

from pathlib import Path

from temporal_manifolds.workflows.eap_common import read_scenario_gcs_prefix
from temporal_manifolds.workflows.eap_ig_workflow import (
    WORKFLOW_DEFINITION,
    WorkflowConfig,
    parse_args,
)
from temporal_manifolds.workflows.runner import (
    load_scenario,
    scenario_args_to_argv,
    workflow_lookup,
)


SCENARIO_PATH = (
    Path.cwd() / "configs" / "scenarios" / "eap_ig_precomputed_top_100.yaml"
)


def test_precomputed_eap_ig_scenario_uses_distinct_artifact_prefix() -> None:
    spec = workflow_lookup()["eap-ig"]
    scenario = load_scenario(SCENARIO_PATH)
    args = parse_args(scenario_args_to_argv(spec, scenario))
    config = WorkflowConfig.from_args(WORKFLOW_DEFINITION, args)

    assert args.modules == ["eap-ig", "top-components", "node-selection"]
    assert args.selection_limit == 100
    assert args.save_to_gcp is True
    assert config.selected_nodes_dir == (
        Path.cwd() / "data" / "selected_nodes" / "eap_ig_selected_100"
    )
    assert read_scenario_gcs_prefix(SCENARIO_PATH) == "eap-ig"
    assert (
        read_scenario_gcs_prefix(
            SCENARIO_PATH,
            key="artifact_gcs_prefix",
            fallback_key="gcs_prefix",
        )
        == "eap-ig-selected-100"
    )
