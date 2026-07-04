"""Shared workflow machinery for Q&A EAP-family attribution pipelines."""

from __future__ import annotations

import argparse
import subprocess
import sys
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import yaml

from temporal_manifolds.utils.eap_ig_artifacts import (
    build_top_components,
    resolve_selected_nodes_path,
    write_selected_nodes,
)
from temporal_manifolds.utils.gcs_upload import upload_files_to_gcs

GradientSide = Literal["clean", "corrupted"]

REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_SCENARIO_DIR = REPO_ROOT / "configs" / "scenarios"
EAP_INPUTS_SCRIPT = (
    REPO_ROOT / "src" / "temporal_manifolds" / "eap_ig" / "eap_ig_inputs_QandA.py"
)
TOP_COMPONENTS_STAGE = "top-components"
NODE_SELECTION_STAGE = "node-selection"


@dataclass(frozen=True)
class EAPWorkflowDefinition:
    """Static differences between EAP-family workflows."""

    workflow_name: str
    description_name: str
    attribution_stage: str
    config_option: str
    results_option: str
    default_config_dir: Path
    default_results_dir: Path
    default_top_components_dir: Path
    default_selected_nodes_dir: Path
    default_completeness_figures_dir: Path
    artifact_label: str | None = None
    method: str | None = None
    supports_compute_gradient_at: bool = False
    default_compute_gradient_at: GradientSide = "clean"

    @property
    def stage_order(self) -> tuple[str, ...]:
        """Return stages in fixed execution order for this workflow."""
        return (self.attribution_stage, TOP_COMPONENTS_STAGE, NODE_SELECTION_STAGE)


@dataclass(frozen=True)
class WorkflowConfig:
    """Resolved runtime configuration for a Q&A EAP-family workflow."""

    definition: EAPWorkflowDefinition
    attribution_config_dir: Path
    attribution_results_dir: Path
    top_components_dir: Path
    selected_nodes_dir: Path
    completeness_figures_dir: Path
    top_n: int
    selection_limit: int
    compute_gradient_at: GradientSide
    compute_completeness: bool
    save_to_gcp: bool
    scenario_config: Path | None
    modules: tuple[str, ...]

    @classmethod
    def from_args(
        cls,
        definition: EAPWorkflowDefinition,
        args: argparse.Namespace,
    ) -> "WorkflowConfig":
        """Build a resolved workflow config from parsed CLI args."""
        return cls(
            definition=definition,
            attribution_config_dir=resolve_repo_path(args.attribution_config_dir),
            attribution_results_dir=resolve_repo_path(args.attribution_results_dir),
            top_components_dir=resolve_repo_path(args.top_components_dir),
            selected_nodes_dir=resolve_repo_path(args.selected_nodes_dir),
            completeness_figures_dir=resolve_repo_path(args.completeness_figures_dir),
            top_n=args.top_n,
            selection_limit=args.selection_limit,
            compute_gradient_at=args.compute_gradient_at,
            compute_completeness=args.compute_completeness,
            save_to_gcp=args.save_to_gcp,
            scenario_config=(
                resolve_repo_path(args.scenario_config)
                if args.scenario_config is not None
                else None
            ),
            modules=tuple(args.modules),
        )

    @property
    def selected_nodes_path(self) -> Path:
        """Return the selected-node artifact path for this workflow run."""
        return resolve_selected_nodes_path(
            self.selected_nodes_dir,
            self.top_n,
            artifact_label=self.definition.artifact_label,
        )

    def includes_stage(self, stage: str) -> bool:
        """Return whether the requested stage was explicitly selected."""
        return stage in self.modules


def resolve_repo_path(path: str | Path) -> Path:
    """Resolve repository-relative paths from CLI args and config files."""
    candidate = Path(path).expanduser()
    if candidate.is_absolute():
        return candidate
    return REPO_ROOT / candidate


def run_python_script(script_path: Path, args: Sequence[str]) -> None:
    """Run a repository script with the current Python interpreter."""
    command = [sys.executable, str(script_path), *args]
    subprocess.run(command, cwd=REPO_ROOT, check=True)


def discover_configs(config_dir: Path) -> list[Path]:
    """Return Q&A YAML configs in stable filename order."""
    configs = sorted(config_dir.glob("*.yaml"), key=lambda path: path.name.lower())
    if not configs:
        raise FileNotFoundError(f"No YAML configs found in {config_dir}")
    return configs


def default_scenario_config_path(
    definition: EAPWorkflowDefinition,
    compute_gradient_at: GradientSide,
) -> Path:
    """Return the scenario YAML used as the sole GCS-prefix source."""
    if definition.workflow_name == "eap-ig":
        return DEFAULT_SCENARIO_DIR / "eap_ig_top_components.yaml"
    if definition.workflow_name == "eap":
        return DEFAULT_SCENARIO_DIR / f"eap_top_components_{compute_gradient_at}.yaml"
    raise ValueError(f"No default scenario config for workflow: {definition.workflow_name}")


def read_scenario_gcs_prefix(scenario_config: Path) -> str:
    """Read the GCS prefix from a scenario YAML file under configs/scenarios."""
    scenario_config = scenario_config.resolve()
    scenario_root = DEFAULT_SCENARIO_DIR.resolve()
    try:
        scenario_config.relative_to(scenario_root)
    except ValueError as exc:
        raise ValueError(
            f"GCS prefix must be read from a YAML file in {scenario_root}: {scenario_config}"
        ) from exc

    with scenario_config.open("r", encoding="utf-8") as f:
        scenario = yaml.safe_load(f) or {}
    if not isinstance(scenario, dict):
        raise TypeError(f"Scenario config must be a mapping: {scenario_config}")

    raw_prefix = scenario.get("gcs_prefix", scenario.get("gcs-prefix", ""))
    return "" if raw_prefix is None else str(raw_prefix)


def run_attribution_stage(
    *,
    definition: EAPWorkflowDefinition,
    config_dir: Path,
    results_dir: Path,
    compute_gradient_at: GradientSide,
    save_to_gcp: bool,
    gcs_prefix: str = "",
) -> list[Path]:
    """Run the Q&A attribution script once per config file."""
    config_paths = discover_configs(config_dir)
    for config_path in config_paths:
        args = ["--config", str(config_path), "--results-root", str(results_dir)]
        if definition.method is not None:
            args.extend(["--method", definition.method])
        if definition.supports_compute_gradient_at:
            args.extend(["--compute-gradient-at", compute_gradient_at])
        args.extend(["--gcs-prefix", gcs_prefix])
        args.append("--save-to-gcp" if save_to_gcp else "--no-save-to-gcp")
        run_python_script(EAP_INPUTS_SCRIPT, args)
    return config_paths


def build_parser(definition: EAPWorkflowDefinition) -> argparse.ArgumentParser:
    """Build the CLI parser for an EAP-family workflow entrypoint."""
    parser = argparse.ArgumentParser(
        description=(
            f"Run the Q&A {definition.description_name} workflow end-to-end "
            "from the command line."
        ),
    )

    path_group = parser.add_argument_group("paths")
    path_group.add_argument(
        definition.config_option,
        dest="attribution_config_dir",
        type=Path,
        default=definition.default_config_dir,
        help=f"Directory containing the {definition.description_name} Q&A YAML configs.",
    )
    path_group.add_argument(
        definition.results_option,
        dest="attribution_results_dir",
        type=Path,
        default=definition.default_results_dir,
        help=(
            f"Root directory holding the {definition.description_name} NPZ outputs "
            "grouped by case."
        ),
    )
    path_group.add_argument(
        "--top-components-dir",
        type=Path,
        default=definition.default_top_components_dir,
        help="Directory where top-component artifacts should be written.",
    )
    path_group.add_argument(
        "--selected-nodes-dir",
        type=Path,
        default=definition.default_selected_nodes_dir,
        help="Directory where selected-node artifacts should be written.",
    )
    path_group.add_argument(
        "--completeness-figures-dir",
        type=Path,
        default=definition.default_completeness_figures_dir,
        help="Directory where completeness figures should be written.",
    )

    selection_group = parser.add_argument_group("selection")
    selection_group.add_argument(
        "--top-n",
        type=int,
        default=500,
        help="Number of top components to keep per horizon before node selection.",
    )
    selection_group.add_argument(
        "--selection-limit",
        type=int,
        default=300,
        help="How many top components per horizon/sign to consider during node selection.",
    )
    if definition.supports_compute_gradient_at:
        selection_group.add_argument(
            "--compue-gradient-at",
            "--compute-gradient-at",
            dest="compute_gradient_at",
            choices=("clean", "corrupted"),
            default=definition.default_compute_gradient_at,
            help=f"Prompt side where {definition.description_name} computes gradients.",
        )
    else:
        parser.set_defaults(compute_gradient_at=definition.default_compute_gradient_at)

    execution_group = parser.add_argument_group("execution")
    execution_group.add_argument(
        "--compute-completeness",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Compute and save completeness figures during the top-components stage.",
    )
    execution_group.add_argument(
        "--save-to-gcp",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            f"Upload generated {definition.description_name} artifacts to the "
            "configured GCS bucket."
        ),
    )
    execution_group.add_argument(
        "--scenario-config",
        type=Path,
        default=None,
        help=argparse.SUPPRESS,
    )
    execution_group.add_argument(
        "--modules",
        nargs="+",
        choices=definition.stage_order,
        default=definition.stage_order,
        help="Workflow modules to run. Modules execute in the workflow's fixed order.",
    )

    return parser


def parse_args(
    definition: EAPWorkflowDefinition,
    argv: Sequence[str] | None = None,
) -> argparse.Namespace:
    """Parse CLI arguments for an EAP-family workflow."""
    return build_parser(definition).parse_args(argv)


def run_workflow(config: WorkflowConfig) -> None:
    """Execute the requested EAP-family workflow stages."""
    definition = config.definition
    scenario_config = config.scenario_config or default_scenario_config_path(
        definition,
        config.compute_gradient_at,
    )
    gcs_prefix = read_scenario_gcs_prefix(scenario_config) if config.save_to_gcp else ""

    if config.includes_stage(definition.attribution_stage):
        run_attribution_stage(
            definition=definition,
            config_dir=config.attribution_config_dir,
            results_dir=config.attribution_results_dir,
            compute_gradient_at=config.compute_gradient_at,
            save_to_gcp=config.save_to_gcp,
            gcs_prefix=gcs_prefix,
        )

    if config.includes_stage(TOP_COMPONENTS_STAGE):
        top_component_pickles, completeness_figures = build_top_components(
            results_dir=config.attribution_results_dir,
            top_components_dir=config.top_components_dir,
            completeness_figures_dir=config.completeness_figures_dir,
            top_n=config.top_n,
            compute_completeness=config.compute_completeness,
        )
        top_component_artifacts = [
            artifact_path
            for pickle_path in top_component_pickles.values()
            for artifact_path in (pickle_path, pickle_path.with_suffix(".json"))
        ]
        top_component_artifacts.extend(completeness_figures.values())
        upload_files_to_gcs(
            top_component_artifacts,
            enabled=config.save_to_gcp,
            prefix=gcs_prefix,
            upload_root=REPO_ROOT,
        )

    if config.includes_stage(NODE_SELECTION_STAGE):
        selected_node_artifacts = write_selected_nodes(
            top_components_dir=config.top_components_dir,
            selected_nodes_dir=config.selected_nodes_dir,
            top_n=config.top_n,
            selection_limit=config.selection_limit,
            artifact_label=definition.artifact_label,
        )
        upload_files_to_gcs(
            selected_node_artifacts,
            enabled=config.save_to_gcp,
            prefix=gcs_prefix,
            upload_root=REPO_ROOT,
        )


def main_for_definition(
    definition: EAPWorkflowDefinition,
    argv: Sequence[str] | None = None,
) -> None:
    """Run an EAP-family workflow with the requested modules."""
    args = parse_args(definition, argv)
    run_workflow(WorkflowConfig.from_args(definition, args))
