"""Dispatch named temporal-manifolds workflows from scenario configs."""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from dataclasses import dataclass
from importlib import import_module
from pathlib import Path
from typing import Callable

import yaml

WorkflowMain = Callable[[Sequence[str] | None], None]
REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_SCENARIO_DIR = REPO_ROOT / "configs" / "scenarios"


@dataclass(frozen=True)
class WorkflowSpec:
    """CLI metadata for a runnable workflow module."""

    name: str
    module: str
    description: str
    aliases: tuple[str, ...] = ()


WORKFLOWS: tuple[WorkflowSpec, ...] = (
    WorkflowSpec(
        name="eap-ig",
        module="temporal_manifolds.workflows.eap_ig_workflow",
        description="Run Q&A EAP-IG attribution, top-component selection, and node selection.",
        aliases=("eap_ig",),
    ),
    WorkflowSpec(
        name="eap",
        module="temporal_manifolds.workflows.eap_workflow",
        description="Run Q&A vanilla EAP attribution, top-component selection, and node selection.",
        aliases=("vanilla-eap", "vanilla_eap"),
    ),
    WorkflowSpec(
        name="activation-caching",
        module="temporal_manifolds.workflows.activation_caching_workflow",
        description="Generate activation prompts, completions, and selected-node activation caches.",
        aliases=("activation_caching", "activations"),
    ),
)


def workflow_lookup() -> dict[str, WorkflowSpec]:
    """Return workflow specs keyed by canonical names and aliases."""
    lookup: dict[str, WorkflowSpec] = {}
    for spec in WORKFLOWS:
        lookup[spec.name] = spec
        for alias in spec.aliases:
            lookup[alias] = spec
    return lookup


def format_workflow_list() -> str:
    """Return a human-readable list of available workflows."""
    lines = ["available workflows:"]
    for spec in WORKFLOWS:
        alias_text = f" aliases: {', '.join(spec.aliases)}" if spec.aliases else ""
        lines.append(f"  {spec.name:<18} {spec.description}{alias_text}")
    return "\n".join(lines)


def build_parser() -> argparse.ArgumentParser:
    """Build the lightweight dispatcher parser."""
    parser = argparse.ArgumentParser(
        description="Run one temporal-manifolds workflow from a scenario YAML config.",
        epilog=format_workflow_list(),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "scenario",
        type=Path,
        help=(
            "Scenario YAML path, or a scenario name resolved under configs/scenarios. "
            "Names may omit the .yaml suffix."
        ),
    )
    return parser


def load_workflow_main(spec: WorkflowSpec) -> WorkflowMain:
    """Import the selected workflow module and return its main function."""
    module = import_module(spec.module)
    main = getattr(module, "main", None)
    if not callable(main):
        raise TypeError(f"Workflow module {spec.module!r} does not define callable main(argv)")
    return main


def resolve_scenario_path(scenario: Path) -> Path:
    """Resolve a CLI scenario argument to a YAML file."""
    scenario = scenario.expanduser()
    if scenario.is_absolute() or scenario.parent != Path("."):
        return scenario

    if scenario.suffix:
        return DEFAULT_SCENARIO_DIR / scenario
    return DEFAULT_SCENARIO_DIR / f"{scenario}.yaml"


def load_scenario(path: Path) -> dict:
    """Load and validate a workflow scenario config."""
    if not path.exists():
        raise FileNotFoundError(f"Scenario config not found: {path}")

    with path.open("r", encoding="utf-8") as f:
        scenario = yaml.safe_load(f)

    if not isinstance(scenario, dict):
        raise TypeError(f"Scenario config must be a mapping: {path}")
    if "workflow" not in scenario:
        raise KeyError(f"Scenario config must include a 'workflow' key: {path}")
    return scenario


def option_strings_for_workflow(spec: WorkflowSpec) -> set[str]:
    """Return option strings accepted by a workflow parser when available."""
    module = import_module(spec.module)
    build_parser = getattr(module, "build_parser", None)
    if not callable(build_parser):
        return set()

    parser = build_parser()
    return {
        option_string
        for action in parser._actions
        for option_string in action.option_strings
    }


def workflow_uses_config_only(spec: WorkflowSpec) -> bool:
    """Return whether a workflow accepts only its scenario YAML path."""
    return bool(getattr(import_module(spec.module), "CONFIG_ONLY", False))


def normalize_option_name(name: str) -> str:
    """Convert a YAML mapping key into a CLI option name."""
    return f"--{name.replace('_', '-')}"


def args_mapping_to_argv(spec: WorkflowSpec, args: dict) -> list[str]:
    """Convert a YAML args mapping into workflow CLI argv."""
    option_strings = option_strings_for_workflow(spec)
    argv: list[str] = []

    for key, value in args.items():
        option = normalize_option_name(str(key))
        if value is None:
            continue
        if isinstance(value, bool):
            if value:
                argv.append(option)
                continue

            negative_option = f"--no-{option[2:]}"
            if negative_option in option_strings:
                argv.append(negative_option)
            continue

        argv.append(option)
        if isinstance(value, list):
            argv.extend(str(item) for item in value)
        else:
            argv.append(str(value))

    return argv


def scenario_args_to_argv(spec: WorkflowSpec, scenario: dict) -> list[str]:
    """Return workflow-specific argv from a scenario config."""
    args = scenario.get("args", [])
    if args is None:
        return []
    if isinstance(args, list):
        return [str(arg) for arg in args]
    if isinstance(args, dict):
        return args_mapping_to_argv(spec, args)
    raise TypeError("Scenario 'args' must be a list, mapping, or null")


def main(argv: Sequence[str] | None = None) -> None:
    """Dispatch to the workflow selected by a scenario config."""
    argv = sys.argv[1:] if argv is None else list(argv)
    parser = build_parser()
    args = parser.parse_args(argv)
    scenario = load_scenario(resolve_scenario_path(args.scenario))
    workflow_name = str(scenario["workflow"])
    lookup = workflow_lookup()
    if workflow_name not in lookup:
        raise ValueError(
            f"Unknown workflow {workflow_name!r}. Expected one of: {', '.join(sorted(lookup))}"
        )

    spec = lookup[workflow_name]
    if workflow_uses_config_only(spec):
        workflow_args = ["--scenario-config", str(resolve_scenario_path(args.scenario))]
    else:
        workflow_args = scenario_args_to_argv(spec, scenario)
        if "--scenario-config" in option_strings_for_workflow(spec):
            workflow_args.extend(["--scenario-config", str(resolve_scenario_path(args.scenario))])
    load_workflow_main(spec)(workflow_args)


if __name__ == "__main__":
    main()
