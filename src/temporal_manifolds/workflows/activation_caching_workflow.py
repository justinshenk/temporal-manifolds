"""Single-command workflow for generating and caching temporal activations."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Sequence, TypeAlias

from temporal_manifolds.activations.dataset_gen import DATASETS, generate_task_dataset
from temporal_manifolds.activations.extract_activations import POSITION_SELECTION_POLICIES

StageName: TypeAlias = Literal[
    "dataset",
    "completions",
    "activations",
]
PositionSelectionPolicy: TypeAlias = Literal["default", "all"]

REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_PROMPT_RECORDS_PATH = REPO_ROOT / "data" / "model_completions" / "activation_prompts.json"
DEFAULT_COMPLETIONS_PATH = REPO_ROOT / "data" / "model_completions" / "completions_256.jsonl"
DEFAULT_SELECTED_NODES_PATH = REPO_ROOT / "data" / "selected_nodes" / "final_500_eap_ig.pkl"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "results" / "feature_geometry_new_activations"
DEFAULT_MODEL_NAME = "Qwen/Qwen3-4B-Instruct-2507"

STAGE_ORDER: tuple[StageName, ...] = (
    "dataset",
    "completions",
    "activations",
)
STAGE_TO_INDEX = {stage: index for index, stage in enumerate(STAGE_ORDER)}


@dataclass(frozen=True)
class WorkflowConfig:
    """Resolved runtime configuration for activation caching."""

    dataset: str
    prompt_records_path: Path
    completions_path: Path
    nodes_path: Path
    output_dir: Path
    model_name: str
    batch_size: int
    max_new_tokens: int
    temperature: float
    top_k: int
    randomize_template: bool
    dtype: str | None
    device: str | None
    attn_type: str
    position_selection_policy: PositionSelectionPolicy
    max_samples: int | None
    overwrite: bool
    save_to_gcp: bool
    gcp_project_id: str | None
    gcs_bucket_name: str | None
    gcs_prefix: str | None
    start_at: StageName
    stop_after: StageName

    @classmethod
    def from_args(cls, args: argparse.Namespace) -> "WorkflowConfig":
        """Build a resolved workflow config from parsed CLI args."""
        return cls(
            dataset=args.dataset,
            prompt_records_path=resolve_repo_path(args.prompt_records_path),
            completions_path=resolve_repo_path(args.completions_path),
            nodes_path=resolve_repo_path(args.nodes_path),
            output_dir=resolve_repo_path(args.output_dir),
            model_name=args.model_name,
            batch_size=args.batch_size,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_k=args.top_k,
            randomize_template=args.randomize_template,
            dtype=args.dtype,
            device=args.device,
            attn_type=args.attn_type,
            position_selection_policy=args.position_selection_policy,
            max_samples=args.max_samples,
            overwrite=args.overwrite,
            save_to_gcp=args.save_to_gcp,
            gcp_project_id=args.gcp_project_id,
            gcs_bucket_name=args.gcs_bucket_name,
            gcs_prefix=args.gcs_prefix,
            start_at=args.start_at,
            stop_after=args.stop_after,
        )

    def includes_stage(self, stage: StageName) -> bool:
        """Return whether the requested stage lies inside the stage window."""
        return is_stage_requested(stage, self.start_at, self.stop_after)


def resolve_repo_path(path: str | Path) -> Path:
    """Resolve repository-relative paths from CLI args."""
    candidate = Path(path).expanduser()
    if candidate.is_absolute():
        return candidate
    return REPO_ROOT / candidate


def validate_stage_window(start_at: StageName, stop_after: StageName) -> None:
    """Validate that the requested stage window is well ordered."""
    if STAGE_TO_INDEX[start_at] > STAGE_TO_INDEX[stop_after]:
        raise ValueError(f"start-at={start_at!r} must come before stop-after={stop_after!r}")


def is_stage_requested(
    stage: StageName,
    start_at: StageName,
    stop_after: StageName,
) -> bool:
    """Return whether ``stage`` lies inside the requested stage window."""
    start_index = STAGE_TO_INDEX[start_at]
    stop_index = STAGE_TO_INDEX[stop_after]
    stage_index = STAGE_TO_INDEX[stage]
    return start_index <= stage_index <= stop_index


def run_dataset_stage(
    *,
    dataset: str,
    output_path: Path,
    randomize_template: bool,
) -> Path:
    """Generate prompt records for the requested activation dataset."""
    generate_task_dataset(
        output_path=output_path,
        randomize_template=randomize_template,
        dataset=dataset,
    )
    return output_path


def run_completions_stage(
    *,
    prompt_records_path: Path,
    completions_path: Path,
    model_name: str,
    batch_size: int,
    max_new_tokens: int,
    temperature: float,
    top_k: int,
    dtype: str | None,
    device: str | None,
    attn_type: str,
) -> Path:
    """Generate completions from prompt records."""
    from temporal_manifolds.activations.generate_completions import write_completions

    write_completions(
        output_path=completions_path,
        input_path=prompt_records_path,
        model_name=model_name,
        batch_size=batch_size,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_k=top_k,
        dtype=dtype,
        device=device,
        attn_type=attn_type,
    )
    return completions_path


def run_activations_stage(
    *,
    completions_path: Path,
    nodes_path: Path,
    output_dir: Path,
    model_name: str,
    dtype: str | None,
    device: str | None,
    attn_type: str,
    position_selection_policy: PositionSelectionPolicy,
    max_samples: int | None,
    overwrite: bool,
    save_to_gcp: bool,
    gcp_project_id: str | None,
    gcs_bucket_name: str | None,
    gcs_prefix: str | None,
) -> Path:
    """Cache selected-node activations from generated completions."""
    from temporal_manifolds.activations.extract_activations import cache_completion_activations

    cache_completion_activations(
        input_path=completions_path,
        model_name=model_name,
        nodes_path=nodes_path,
        output_dir=output_dir,
        dtype=dtype,
        device=device,
        attn_type=attn_type,
        position_selection_policy=position_selection_policy,
        max_samples=max_samples,
        overwrite=overwrite,
        save_to_gcp=save_to_gcp,
        gcp_project_id=gcp_project_id,
        gcs_bucket_name=gcs_bucket_name,
        gcs_prefix=gcs_prefix,
    )
    return output_dir


def build_parser() -> argparse.ArgumentParser:
    """Build the CLI parser for the workflow entrypoint."""
    parser = argparse.ArgumentParser(
        description="Run the activation-caching workflow end-to-end.",
    )

    data_group = parser.add_argument_group("data")
    data_group.add_argument(
        "--dataset",
        choices=sorted(DATASETS),
        default="conversational",
        help="Activation dataset configuration to generate.",
    )
    data_group.add_argument(
        "--prompt-records-path",
        type=Path,
        default=DEFAULT_PROMPT_RECORDS_PATH,
        help="JSON prompt-record artifact written by the dataset stage.",
    )
    data_group.add_argument(
        "--completions-path",
        type=Path,
        default=DEFAULT_COMPLETIONS_PATH,
        help="JSONL completion artifact written by the completions stage.",
    )
    data_group.add_argument(
        "--nodes-path",
        type=Path,
        default=DEFAULT_SELECTED_NODES_PATH,
        help="Pickle file containing selected EAP-IG nodes to cache.",
    )
    data_group.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory where activation cache .pt files should be written.",
    )

    model_group = parser.add_argument_group("model")
    model_group.add_argument("--model-name", default=DEFAULT_MODEL_NAME)
    model_group.add_argument("--dtype", default=None)
    model_group.add_argument("--device", default=None)
    model_group.add_argument("--attn-type", default="sdpa")

    completion_group = parser.add_argument_group("completions")
    completion_group.add_argument("--batch-size", type=int, default=128)
    completion_group.add_argument("--max-new-tokens", type=int, default=256)
    completion_group.add_argument("--temperature", type=float, default=0.7)
    completion_group.add_argument("--top-k", type=int, default=20)
    completion_group.add_argument(
        "--randomize-template",
        action="store_true",
        help="Choose one prompt template per task/time sample in the dataset stage.",
    )

    activation_group = parser.add_argument_group("activations")
    activation_group.add_argument(
        "--position-selection-policy",
        choices=POSITION_SELECTION_POLICIES,
        default="default",
        help="Policy used to select completion token positions for activation caching.",
    )
    activation_group.add_argument("--max-samples", type=int, default=None)
    activation_group.add_argument(
        "--overwrite",
        action="store_true",
        help="Recompute activation files that already exist.",
    )

    upload_group = parser.add_argument_group("upload")
    upload_group.add_argument(
        "--save-to-gcp",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Upload activation outputs to Google Cloud Storage after caching.",
    )
    upload_group.add_argument("--gcp-project-id", default=None)
    upload_group.add_argument("--gcs-bucket-name", default=None)
    upload_group.add_argument("--gcs-prefix", default=None)

    execution_group = parser.add_argument_group("execution")
    execution_group.add_argument(
        "--start-at",
        choices=STAGE_ORDER,
        default=STAGE_ORDER[0],
        help="Stage at which to start the workflow.",
    )
    execution_group.add_argument(
        "--stop-after",
        choices=STAGE_ORDER,
        default=STAGE_ORDER[-1],
        help="Stage after which to stop the workflow.",
    )

    return parser


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse CLI arguments for the workflow."""
    return build_parser().parse_args(argv)


def run_workflow(config: WorkflowConfig) -> None:
    """Execute the requested workflow stages."""
    validate_stage_window(config.start_at, config.stop_after)

    if config.includes_stage("dataset"):
        run_dataset_stage(
            dataset=config.dataset,
            output_path=config.prompt_records_path,
            randomize_template=config.randomize_template,
        )

    if config.includes_stage("completions"):
        run_completions_stage(
            prompt_records_path=config.prompt_records_path,
            completions_path=config.completions_path,
            model_name=config.model_name,
            batch_size=config.batch_size,
            max_new_tokens=config.max_new_tokens,
            temperature=config.temperature,
            top_k=config.top_k,
            dtype=config.dtype,
            device=config.device,
            attn_type=config.attn_type,
        )

    if config.includes_stage("activations"):
        run_activations_stage(
            completions_path=config.completions_path,
            nodes_path=config.nodes_path,
            output_dir=config.output_dir,
            model_name=config.model_name,
            dtype=config.dtype,
            device=config.device,
            attn_type=config.attn_type,
            position_selection_policy=config.position_selection_policy,
            max_samples=config.max_samples,
            overwrite=config.overwrite,
            save_to_gcp=config.save_to_gcp,
            gcp_project_id=config.gcp_project_id,
            gcs_bucket_name=config.gcs_bucket_name,
            gcs_prefix=config.gcs_prefix,
        )


def main(argv: Sequence[str] | None = None) -> None:
    """Run the workflow with the requested stage window."""
    args = parse_args(argv)
    run_workflow(WorkflowConfig.from_args(args))


if __name__ == "__main__":
    main()
