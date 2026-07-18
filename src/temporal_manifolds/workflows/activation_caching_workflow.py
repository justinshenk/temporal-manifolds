"""Single-command workflow for generating and caching temporal activations."""

from __future__ import annotations

import argparse
import os
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any, Literal, Mapping, Sequence, TypeAlias, cast

import yaml
from dotenv import load_dotenv

from temporal_manifolds.activations.extract_activations import POSITION_SELECTION_POLICIES
from temporal_manifolds.dataset.generate import DATASETS, generate_task_dataset

StageName: TypeAlias = Literal[
    "dataset",
    "completions",
    "activations",
]
PositionSelectionPolicy: TypeAlias = Literal["default", "all", "after_assistant"]
CONFIG_ONLY = True

load_dotenv()
GCP_PROJECT_ID = os.getenv("GCP_PROJECT_ID")
GCS_BUCKET_NAME = os.getenv("GCS_BUCKET_NAME")
ENV_CONFIG_FIELDS = {"gcp_project_id", "gcs_bucket_name"}
SCENARIO_CONFIG_FIELDS = {"gcs_prefix"}

REPO_ROOT = Path(__file__).resolve().parents[3]
STAGE_ORDER: tuple[StageName, ...] = (
    "dataset",
    "completions",
    "activations",
)


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
    dtype: str | None
    device: str | None
    attn_type: str
    position_selection_policy: PositionSelectionPolicy
    max_samples: int | None
    overwrite: bool
    save_to_gcp: bool
    upload_worker_count: int
    upload_queue_capacity: int
    gcp_project_id: str | None
    gcs_bucket_name: str | None
    gcs_prefix: str | None
    dataset_gcs_prefix: str | None
    completions_gcs_prefix: str | None
    nodes_gcs_prefix: str | None
    modules: tuple[StageName, ...]

    @classmethod
    def from_mapping(
        cls,
        values: Mapping[str, Any],
        *,
        gcs_prefix: str | None = None,
    ) -> "WorkflowConfig":
        """Build a resolved workflow config from a YAML mapping."""
        expected_keys = (
            {field.name for field in fields(cls)} - ENV_CONFIG_FIELDS - SCENARIO_CONFIG_FIELDS
        )
        unknown_keys = set(values) - expected_keys
        missing_keys = expected_keys - set(values)
        if unknown_keys:
            raise ValueError(f"Unknown activation-caching config keys: {sorted(unknown_keys)}")
        if missing_keys:
            raise ValueError(f"Missing activation-caching config keys: {sorted(missing_keys)}")

        dataset = str(values["dataset"])
        if dataset not in DATASETS:
            raise ValueError(f"dataset must be one of {sorted(DATASETS)}, got {dataset!r}")
        position_selection_policy = str(values["position_selection_policy"])
        if position_selection_policy not in POSITION_SELECTION_POLICIES:
            raise ValueError(
                "position_selection_policy must be one of "
                f"{POSITION_SELECTION_POLICIES}, got {position_selection_policy!r}"
            )
        raw_modules = values["modules"]
        if not isinstance(raw_modules, list) or not raw_modules:
            raise TypeError("modules must be a non-empty YAML list")
        invalid_modules = [module for module in raw_modules if module not in STAGE_ORDER]
        if invalid_modules:
            raise ValueError(
                f"modules entries must be one of {STAGE_ORDER}, got {invalid_modules}"
            )
        modules = tuple(cast(StageName, module) for module in raw_modules)

        boolean_keys = ("overwrite", "save_to_gcp")
        for key in boolean_keys:
            if not isinstance(values[key], bool):
                raise TypeError(f"{key} must be a YAML boolean")
        upload_worker_count = int(values["upload_worker_count"])
        if upload_worker_count < 1:
            raise ValueError("upload_worker_count must be at least 1")
        upload_queue_capacity = int(values["upload_queue_capacity"])
        if upload_queue_capacity < 1:
            raise ValueError("upload_queue_capacity must be at least 1")

        return cls(
            dataset=dataset,
            prompt_records_path=resolve_repo_path(values["prompt_records_path"]),
            completions_path=resolve_repo_path(values["completions_path"]),
            nodes_path=resolve_repo_path(values["nodes_path"]),
            output_dir=resolve_repo_path(values["output_dir"]),
            model_name=str(values["model_name"]),
            batch_size=int(values["batch_size"]),
            max_new_tokens=int(values["max_new_tokens"]),
            temperature=float(values["temperature"]),
            top_k=int(values["top_k"]),
            dtype=None if values["dtype"] is None else str(values["dtype"]),
            device=None if values["device"] is None else str(values["device"]),
            attn_type=str(values["attn_type"]),
            position_selection_policy=cast(PositionSelectionPolicy, position_selection_policy),
            max_samples=None if values["max_samples"] is None else int(values["max_samples"]),
            overwrite=bool(values["overwrite"]),
            save_to_gcp=bool(values["save_to_gcp"]),
            upload_worker_count=upload_worker_count,
            upload_queue_capacity=upload_queue_capacity,
            gcp_project_id=GCP_PROJECT_ID,
            gcs_bucket_name=GCS_BUCKET_NAME,
            gcs_prefix=gcs_prefix,
            dataset_gcs_prefix=(
                None
                if values["dataset_gcs_prefix"] is None
                else str(values["dataset_gcs_prefix"])
            ),
            completions_gcs_prefix=(
                None
                if values["completions_gcs_prefix"] is None
                else str(values["completions_gcs_prefix"])
            ),
            nodes_gcs_prefix=(
                None
                if values["nodes_gcs_prefix"] is None
                else str(values["nodes_gcs_prefix"])
            ),
            modules=modules,
        )

    @classmethod
    def from_yaml(cls, path: str | Path) -> "WorkflowConfig":
        """Load activation-caching options from a scenario YAML file."""
        config_path = resolve_repo_path(path)
        with config_path.open("r", encoding="utf-8") as config_file:
            document = yaml.safe_load(config_file)
        if not isinstance(document, dict):
            raise TypeError(f"Config must be a YAML mapping: {config_path}")

        values = document.get("args")
        if not isinstance(values, dict):
            raise TypeError(f"Config 'args' must be a YAML mapping: {config_path}")
        raw_gcs_prefix = document.get("gcs_prefix", document.get("gcs-prefix"))
        gcs_prefix = None if raw_gcs_prefix is None else str(raw_gcs_prefix)
        return cls.from_mapping(values, gcs_prefix=gcs_prefix)

    def includes_stage(self, stage: StageName) -> bool:
        """Return whether the requested workflow module was explicitly selected."""
        return stage in self.modules


def resolve_repo_path(path: str | Path) -> Path:
    """Resolve repository-relative paths from configuration values."""
    candidate = Path(path).expanduser()
    if candidate.is_absolute():
        return candidate
    return REPO_ROOT / candidate


def run_dataset_stage(
    *,
    dataset: str,
    output_path: Path,
) -> Path:
    """Generate prompt records for the requested activation dataset."""
    generate_task_dataset(
        output_path=output_path,
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
    nodes_gcs_prefix: str | None,
    upload_worker_count: int,
    upload_queue_capacity: int,
) -> Path:
    """Cache selected-node activations from generated completions."""
    from temporal_manifolds.activations.extract_activations import cache_completion_activations

    ensure_nodes_file_available(
        nodes_path,
        save_to_gcp=save_to_gcp,
        gcp_project_id=gcp_project_id,
        gcs_bucket_name=gcs_bucket_name,
        nodes_gcs_prefix=nodes_gcs_prefix,
    )

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
        upload_worker_count=upload_worker_count,
        upload_queue_capacity=upload_queue_capacity,
    )
    return output_dir


def ensure_nodes_file_available(
    nodes_path: Path,
    *,
    save_to_gcp: bool,
    gcp_project_id: str | None,
    gcs_bucket_name: str | None,
    nodes_gcs_prefix: str | None,
) -> None:
    """Download the selected-node file from GCS when it is absent locally."""
    if nodes_path.is_file():
        return

    from temporal_manifolds.utils.gcs_upload import (
        gcs_object_name_for_file,
        maybe_build_gcs_existing_object_fetcher,
    )

    object_name = gcs_object_name_for_file(nodes_path, upload_root=REPO_ROOT)
    fetch_from_gcs = maybe_build_gcs_existing_object_fetcher(
        enabled=save_to_gcp,
        project_id=gcp_project_id,
        bucket_name=gcs_bucket_name,
        prefix=nodes_gcs_prefix,
    )
    if fetch_from_gcs(object_name, nodes_path):
        return

    location = (
        f"{nodes_gcs_prefix.strip('/')}/{object_name}"
        if nodes_gcs_prefix
        else object_name
    )
    raise FileNotFoundError(
        f"Selected-node file was not found locally at {nodes_path}"
        + (
            f" or in GCS at gs://{gcs_bucket_name}/{location}."
            if save_to_gcp
            else ". GCS lookup is disabled because save_to_gcp is false."
        )
    )


def ensure_input_artifact_available(
    artifact_path: Path,
    *,
    artifact_name: str,
    save_to_gcp: bool,
    gcp_project_id: str | None,
    gcs_bucket_name: str | None,
    gcs_prefix: str | None,
) -> None:
    """Download a required workflow input from GCS when absent locally."""
    if artifact_path.is_file():
        return

    from temporal_manifolds.utils.gcs_upload import (
        gcs_object_name_for_file,
        maybe_build_gcs_existing_object_fetcher,
    )

    object_name = gcs_object_name_for_file(artifact_path, upload_root=REPO_ROOT)
    fetch_from_gcs = maybe_build_gcs_existing_object_fetcher(
        enabled=save_to_gcp,
        project_id=gcp_project_id,
        bucket_name=gcs_bucket_name,
        prefix=gcs_prefix,
    )
    if fetch_from_gcs(object_name, artifact_path):
        return

    location = f"{gcs_prefix.strip('/')}/{object_name}" if gcs_prefix else object_name
    raise FileNotFoundError(
        f"{artifact_name} artifact was not found locally at {artifact_path}"
        + (
            f" or in GCS at gs://{gcs_bucket_name}/{location}."
            if save_to_gcp
            else ". GCS lookup is disabled because save_to_gcp is false."
        )
    )


def download_cached_artifact(
    artifact_path: Path,
    *,
    save_to_gcp: bool,
    gcp_project_id: str | None,
    gcs_bucket_name: str | None,
    gcs_prefix: str | None,
) -> bool:
    """Download a stage artifact from GCS, returning whether it was found."""
    from temporal_manifolds.utils.gcs_upload import (
        gcs_object_name_for_file,
        maybe_build_gcs_existing_object_fetcher,
    )

    object_name = gcs_object_name_for_file(artifact_path, upload_root=REPO_ROOT)
    fetch_from_gcs = maybe_build_gcs_existing_object_fetcher(
        enabled=save_to_gcp,
        project_id=gcp_project_id,
        bucket_name=gcs_bucket_name,
        prefix=gcs_prefix,
    )
    return fetch_from_gcs(object_name, artifact_path)

def upload_stage_artifacts(
    artifacts: Sequence[Path],
    *,
    config: WorkflowConfig,
) -> None:
    """Upload module artifacts under the scenario's common GCS prefix."""
    from temporal_manifolds.utils.gcs_upload import upload_files_to_gcs

    upload_files_to_gcs(
        artifacts,
        enabled=config.save_to_gcp,
        project_id=config.gcp_project_id,
        bucket_name=config.gcs_bucket_name,
        prefix=config.gcs_prefix,
        upload_root=REPO_ROOT,
    )


def build_parser() -> argparse.ArgumentParser:
    """Build a parser accepting only the YAML configuration path."""
    parser = argparse.ArgumentParser(
        description="Run activation caching using options from a YAML config.",
    )
    parser.add_argument(
        "--scenario-config",
        type=Path,
        required=True,
        help="YAML file containing all activation-caching options under 'args'.",
    )

    return parser


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse the YAML configuration path for the workflow."""
    return build_parser().parse_args(argv)


def run_workflow(config: WorkflowConfig) -> None:
    """Execute the selected modules in the workflow's fixed order."""

    if config.includes_stage("dataset"):
        dataset_cached = download_cached_artifact(
            config.prompt_records_path,
            save_to_gcp=config.save_to_gcp,
            gcp_project_id=config.gcp_project_id,
            gcs_bucket_name=config.gcs_bucket_name,
            gcs_prefix=config.dataset_gcs_prefix,
        )
        if not dataset_cached:
            dataset_artifact = run_dataset_stage(
                dataset=config.dataset,
                output_path=config.prompt_records_path,
            )
            upload_stage_artifacts([dataset_artifact], config=config)

    if config.includes_stage("completions"):
        ensure_input_artifact_available(
            config.prompt_records_path,
            artifact_name="Dataset",
            save_to_gcp=config.save_to_gcp,
            gcp_project_id=config.gcp_project_id,
            gcs_bucket_name=config.gcs_bucket_name,
            gcs_prefix=config.dataset_gcs_prefix,
        )
        completions_cached = download_cached_artifact(
            config.completions_path,
            save_to_gcp=config.save_to_gcp,
            gcp_project_id=config.gcp_project_id,
            gcs_bucket_name=config.gcs_bucket_name,
            gcs_prefix=config.completions_gcs_prefix,
        )
        if not completions_cached:
            completions_artifact = run_completions_stage(
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
            upload_stage_artifacts([completions_artifact], config=config)

    if config.includes_stage("activations"):
        ensure_input_artifact_available(
            config.completions_path,
            artifact_name="Completions",
            save_to_gcp=config.save_to_gcp,
            gcp_project_id=config.gcp_project_id,
            gcs_bucket_name=config.gcs_bucket_name,
            gcs_prefix=config.completions_gcs_prefix,
        )
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
            nodes_gcs_prefix=config.nodes_gcs_prefix,
            upload_worker_count=config.upload_worker_count,
            upload_queue_capacity=config.upload_queue_capacity,
        )


def main(argv: Sequence[str] | None = None) -> None:
    """Load the YAML configuration and run its selected modules."""
    args = parse_args(argv)
    run_workflow(WorkflowConfig.from_yaml(args.scenario_config))


if __name__ == "__main__":
    main()
