from __future__ import annotations

from pathlib import Path

from temporal_manifolds.workflows.eap_ig_workflow import (
    ARTIFACT_LABEL,
    WORKFLOW_DEFINITION,
    WorkflowConfig,
    parse_args,
)
from temporal_manifolds.workflows.runner import args_mapping_to_argv, workflow_lookup


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
