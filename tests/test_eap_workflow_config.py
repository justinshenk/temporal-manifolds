from __future__ import annotations

from pathlib import Path

from temporal_manifolds.workflows.eap_ig_workflow import (
    ARTIFACT_LABEL,
    WORKFLOW_DEFINITION,
    WorkflowConfig,
    parse_args,
)
from temporal_manifolds.workflows.runner import args_mapping_to_argv, workflow_lookup
from temporal_manifolds.workflows.runner import (
    load_scenario,
    scenario_args_to_argv,
)


def test_eap_scenario_can_disable_completeness() -> None:
    spec = workflow_lookup()["eap-ig"]

    argv = args_mapping_to_argv(
        spec,
        {
            "modules": ["top-components"],
            "compute-completeness": False,
        },
    )

    assert "--no-compute-completeness" in argv
    args = parse_args(argv)
    assert args.compute_completeness is False


def test_eap_ig_selected_nodes_path_uses_eap_ig_label() -> None:
    args = parse_args(
        [
            "--selected-nodes-dir",
            "data/selected_nodes",
        ]
    )
    config = WorkflowConfig.from_args(WORKFLOW_DEFINITION, args)

    assert ARTIFACT_LABEL == "eap_ig"
    assert (
        config.selected_nodes_path
        == Path.cwd() / "data" / "selected_nodes" / "final_500_eap_ig.pkl"
    )


def test_eap_ig_scenario_resumes_gcp_scores_and_uploads_100_node_selection() -> None:
    spec = workflow_lookup()["eap-ig"]
    scenario = load_scenario(
        Path.cwd() / "configs" / "scenarios" / "eap_ig_top_components.yaml"
    )
    args = parse_args(scenario_args_to_argv(spec, scenario))

    assert args.modules == ["eap-ig", "top-components", "node-selection"]
    assert args.selection_limit == 100
    assert args.save_to_gcp is True
