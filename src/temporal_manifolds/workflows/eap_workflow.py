"""Single-command workflow for the Q&A vanilla EAP analysis pipeline."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

from temporal_manifolds.workflows.eap_common import (
    EAPWorkflowDefinition,
    GradientSide,
    REPO_ROOT,
    WorkflowConfig,
    build_parser as _build_parser,
    discover_configs,
    main_for_definition,
    parse_args as _parse_args,
    resolve_repo_path,
    run_attribution_stage,
    run_python_script,
    run_workflow,
)

__all__ = [
    "ARTIFACT_LABEL",
    "DEFAULT_COMPLETENESS_FIGURES_DIR",
    "DEFAULT_EAP_CONFIG_DIR",
    "DEFAULT_EAP_RESULTS_DIR",
    "DEFAULT_SELECTED_NODES_DIR",
    "DEFAULT_TOP_COMPONENTS_DIR",
    "STAGE_ORDER",
    "WORKFLOW_DEFINITION",
    "WorkflowConfig",
    "build_parser",
    "discover_eap_configs",
    "main",
    "parse_args",
    "resolve_repo_path",
    "run_eap_stage",
    "run_python_script",
    "run_workflow",
]

DEFAULT_EAP_CONFIG_DIR = REPO_ROOT / "configs" / "eap_ig"
DEFAULT_EAP_RESULTS_DIR = REPO_ROOT / "results" / "eap"
DEFAULT_TOP_COMPONENTS_DIR = REPO_ROOT / "data" / "top_n_nodes_eap"
DEFAULT_SELECTED_NODES_DIR = REPO_ROOT / "data" / "selected_nodes_eap"
DEFAULT_COMPLETENESS_FIGURES_DIR = REPO_ROOT / "results" / "figures" / "eap_completeness"
ARTIFACT_LABEL = "eap"

WORKFLOW_DEFINITION = EAPWorkflowDefinition(
    workflow_name="eap",
    description_name="vanilla EAP",
    attribution_stage="eap",
    config_option="--eap-config-dir",
    results_option="--eap-results-dir",
    default_config_dir=DEFAULT_EAP_CONFIG_DIR,
    default_results_dir=DEFAULT_EAP_RESULTS_DIR,
    default_top_components_dir=DEFAULT_TOP_COMPONENTS_DIR,
    default_selected_nodes_dir=DEFAULT_SELECTED_NODES_DIR,
    default_completeness_figures_dir=DEFAULT_COMPLETENESS_FIGURES_DIR,
    artifact_label=ARTIFACT_LABEL,
    method="eap",
    supports_compute_gradient_at=True,
)
STAGE_ORDER = WORKFLOW_DEFINITION.stage_order


def discover_eap_configs(config_dir: Path) -> list[Path]:
    """Return the Q&A YAML configs in stable filename order."""
    return discover_configs(config_dir)


def run_eap_stage(
    *,
    config_dir: Path,
    results_dir: Path,
    compute_gradient_at: GradientSide,
    save_to_gcp: bool,
) -> list[Path]:
    """Run the Q&A vanilla EAP script once per config file."""
    return run_attribution_stage(
        definition=WORKFLOW_DEFINITION,
        config_dir=config_dir,
        results_dir=results_dir,
        compute_gradient_at=compute_gradient_at,
        save_to_gcp=save_to_gcp,
    )


def build_parser():
    """Build the CLI parser for the workflow entrypoint."""
    return _build_parser(WORKFLOW_DEFINITION)


def parse_args(argv: Sequence[str] | None = None):
    """Parse CLI arguments for the workflow."""
    return _parse_args(WORKFLOW_DEFINITION, argv)


def main(argv: Sequence[str] | None = None) -> None:
    """Run the workflow with the requested modules."""
    main_for_definition(WORKFLOW_DEFINITION, argv)


if __name__ == "__main__":
    main()
