"""Stream selected activation nodes from GCS into matched-condition averages."""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import math
import os
from collections.abc import Mapping
from concurrent.futures import Future, ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import torch
from dotenv import load_dotenv
from google.api_core.exceptions import NotFound
from google.cloud.storage import Bucket
from tqdm import tqdm

from temporal_manifolds.activations.extract_activations import (
    SelectedNodeGroups,
    load_selected_node_groups_from_file,
)
from temporal_manifolds.utils.activation_aggregation import (
    aggregate_activation_payload,
    nodes_for_classes,
)
from temporal_manifolds.utils.gcs_upload import _build_gcs_client


PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_COMPLETIONS_PATH = PROJECT_ROOT / "data" / "completions_completions_256.jsonl"
DEFAULT_BUCKET_NAME = "temporal-research-bucket"
DEFAULT_ACTIVATION_PREFIX = (
    "conversational_after_assistant_residual_stream/"
    "results/feature_geometry_after_assistant_residual_stream"
)
DEFAULT_NODES_OBJECT = (
    "eap-ig-selected-100/data/selected_nodes/"
    "eap_ig_selected_100/final_node_list.pkl"
)
DEFAULT_OUTPUT_OBJECT = (
    f"{DEFAULT_ACTIVATION_PREFIX}/aggregated/"
    "eap_ig_selected_100_matched_assistant_first_token_activations.pt"
)
SEMANTIC_FIELDS = ("task", "base_value", "base_unit")
PHRASING_FIELDS = (
    "template_metadata.prompt_framing",
    "template_metadata.output_format",
)
OUTPUT_FORMAT_VERSION = 1


@dataclass
class MatchedGroup:
    """Completion records belonging to one notebook-style matched condition."""

    semantic_values: tuple[Any, ...]
    representative_metadata: dict[str, Any]
    sample_indices: list[int] = field(default_factory=list)
    phrasing_values: dict[str, list[Any]] = field(default_factory=dict)
    sample_metadata: list[dict[str, Any]] | None = None

    def output_metadata(
        self,
        included_sample_indices: list[int] | None = None,
    ) -> dict[str, Any]:
        """Return the representative metadata row produced by the notebook policy."""
        included_indices = (
            self.sample_indices
            if included_sample_indices is None
            else included_sample_indices
        )
        if not included_indices:
            raise ValueError("Matched metadata requires at least one included sample.")
        included_count = len(included_indices)
        if included_indices == self.sample_indices:
            row = self.representative_metadata.copy()
            phrasing_values = self.phrasing_values
        else:
            if self.sample_metadata is None:
                raise ValueError("Per-sample metadata was not retained for missing-file handling.")
            metadata_by_index = dict(zip(self.sample_indices, self.sample_metadata, strict=True))
            included_metadata = [metadata_by_index[index] for index in included_indices]
            row = included_metadata[0].copy()
            phrasing_values = {}
            for phrasing_field in PHRASING_FIELDS:
                values: list[Any] = []
                for metadata in included_metadata:
                    if phrasing_field in metadata:
                        _append_unique_non_null(values, metadata[phrasing_field])
                if any(phrasing_field in metadata for metadata in included_metadata):
                    phrasing_values[phrasing_field] = values

        row["sample_index"] = included_indices[0]
        row["source_sample_count"] = included_count
        if included_count != len(self.sample_indices):
            row["expected_source_sample_count"] = len(self.sample_indices)
        for phrasing_field, values in phrasing_values.items():
            if len(values) > 1:
                row[phrasing_field] = "<averaged>"
        return row


@dataclass(frozen=True)
class CompletionIndex:
    """Insertion-ordered matched groups and identity of their source JSONL."""

    groups: tuple[MatchedGroup, ...]
    record_count: int
    sha256: str


@dataclass
class ProcessingState:
    """The resumable, much-smaller result accumulated from activation objects."""

    vectors: list[torch.Tensor] = field(default_factory=list)
    metadata: list[dict[str, Any]] = field(default_factory=list)
    source_sample_indices: list[list[int]] = field(default_factory=list)
    missing_activation_objects: list[str] = field(default_factory=list)
    feature_schema: list[dict[str, Any]] | None = None

    @property
    def completed_group_count(self) -> int:
        """Return the number of complete matched conditions in this state."""
        return len(self.vectors)


def flatten_scalar_metadata(
    metadata: Mapping[str, Any],
    prefix: str = "",
) -> dict[str, Any]:
    """Return dotted paths for scalar values, matching the analysis notebook."""
    flattened: dict[str, Any] = {}
    for key, value in metadata.items():
        path = f"{prefix}.{key}" if prefix else key
        if isinstance(value, dict):
            flattened.update(flatten_scalar_metadata(value, path))
        elif not isinstance(value, (list, tuple, set)):
            flattened[path] = value
    return flattened


def _normalized_key_value(value: Any) -> Any:
    """Make null-like scalar values group together as pandas ``dropna=False`` does."""
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return ("<null>",)
    try:
        hash(value)
    except TypeError as error:
        raise ValueError(f"Matched semantic value is not scalar: {value!r}") from error
    return value


def _append_unique_non_null(values: list[Any], value: Any) -> None:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return
    if not any(existing == value for existing in values):
        values.append(value)


def build_completion_index(
    completions_path: Path,
    *,
    retain_sample_metadata: bool = False,
) -> CompletionIndex:
    """Index noncontiguous matched groups without retaining prompts or completions."""
    groups_by_key: dict[tuple[Any, ...], MatchedGroup] = {}
    digest = hashlib.sha256()
    record_count = 0

    with completions_path.open("rb") as completion_file:
        for line_number, line in enumerate(completion_file, start=1):
            digest.update(line)
            if not line.strip():
                continue
            try:
                record = json.loads(line)
                prompt_metadata = record["prompt_metadata"]
            except (json.JSONDecodeError, KeyError, TypeError) as error:
                raise ValueError(
                    f"Invalid completion record at {completions_path}:{line_number}"
                ) from error
            if not isinstance(prompt_metadata, dict):
                raise ValueError(
                    f"prompt_metadata must be an object at {completions_path}:{line_number}"
                )

            flattened = flatten_scalar_metadata(prompt_metadata)
            missing_fields = set(SEMANTIC_FIELDS) - flattened.keys()
            if missing_fields:
                raise ValueError(
                    f"Missing semantic fields at {completions_path}:{line_number}: "
                    f"{sorted(missing_fields)}"
                )
            semantic_values = tuple(flattened[field] for field in SEMANTIC_FIELDS)
            semantic_key = tuple(_normalized_key_value(value) for value in semantic_values)
            group = groups_by_key.get(semantic_key)
            if group is None:
                group = MatchedGroup(
                    semantic_values=semantic_values,
                    representative_metadata=flattened,
                    phrasing_values={
                        field: [] for field in PHRASING_FIELDS if field in flattened
                    },
                    sample_metadata=[] if retain_sample_metadata else None,
                )
                groups_by_key[semantic_key] = group

            group.sample_indices.append(record_count)
            if group.sample_metadata is not None:
                group.sample_metadata.append(flattened)
            for phrasing_field in PHRASING_FIELDS:
                if phrasing_field in flattened:
                    group.phrasing_values.setdefault(phrasing_field, [])
                    _append_unique_non_null(
                        group.phrasing_values[phrasing_field],
                        flattened[phrasing_field],
                    )
            record_count += 1

    if not groups_by_key:
        raise ValueError(f"No completion records found in {completions_path}")
    return CompletionIndex(
        groups=tuple(groups_by_key.values()),
        record_count=record_count,
        sha256=digest.hexdigest(),
    )


def activation_object_name(activation_prefix: str, sample_index: int) -> str:
    """Return the cache object name corresponding to a completion record index."""
    return (
        f"{activation_prefix.strip('/')}/"
        f"activations_sample_{sample_index:05d}.pt"
    )


def limited_output_object(output_object: str, max_groups: int) -> str:
    """Derive a noncanonical output name for a limited smoke run."""
    suffix = f"_first_{max_groups}_matched_groups"
    if output_object.endswith(".pt"):
        return f"{output_object[:-3]}{suffix}.pt"
    return f"{output_object}{suffix}.pt"


def requested_nodes(selected_node_groups: SelectedNodeGroups) -> dict[str, set[int]]:
    """Return the exact union of nodes in every class from the target node list."""
    allowed_nodes = nodes_for_classes(selected_node_groups, set(selected_node_groups))
    if not allowed_nodes:
        raise ValueError("The selected-node file does not contain any nodes.")
    return allowed_nodes


def _validate_requested_nodes(
    payload: dict[str, Any],
    allowed_nodes: dict[str, set[int]],
    source_name: str,
) -> None:
    cached_activations = payload.get("activations")
    if not isinstance(cached_activations, dict):
        raise ValueError(f"{source_name} does not contain an activations mapping.")

    missing: dict[str, list[int]] = {}
    for name, expected_indices in allowed_nodes.items():
        entry = cached_activations.get(name)
        if not isinstance(entry, dict) or "node_indices" not in entry:
            missing[name] = sorted(expected_indices)
            continue
        cached_indices = {int(index) for index in entry["node_indices"]}
        missing_indices = expected_indices - cached_indices
        if missing_indices:
            missing[name] = sorted(missing_indices)
    if missing:
        raise ValueError(f"{source_name} is missing requested nodes: {missing}")


def extract_assistant_feature_vector(
    payload: dict[str, Any],
    allowed_nodes: dict[str, set[int]],
    *,
    expected_sample_index: int | None = None,
    expected_feature_schema: list[dict[str, Any]] | None = None,
    source_name: str = "activation payload",
) -> tuple[torch.Tensor, list[dict[str, Any]]]:
    """Extract exactly the requested nodes at the first cached assistant position."""
    _validate_requested_nodes(payload, allowed_nodes, source_name)
    aggregated = aggregate_activation_payload(
        payload,
        "assistant",
        allowed_nodes,
        residual_stream_layers=set(),
        source_name=source_name,
    )
    sample_index = int(aggregated["sample_index"])
    if expected_sample_index is not None and sample_index != expected_sample_index:
        raise ValueError(
            f"{source_name} contains sample_index={sample_index}, "
            f"expected {expected_sample_index}."
        )

    feature_parts: list[torch.Tensor] = []
    feature_schema: list[dict[str, Any]] = []
    offset = 0
    for activation_type in ("mlp", "attn"):
        for name, tensor in aggregated["activations"][activation_type].items():
            flattened = tensor.ravel()
            next_offset = offset + flattened.numel()
            feature_parts.append(flattened)
            feature_schema.append(
                {
                    "activation_type": activation_type,
                    "name": name,
                    "node_indices": aggregated["node_indices"][name],
                    "value_shape": list(tensor.shape),
                    "feature_start": offset,
                    "feature_stop": next_offset,
                }
            )
            offset = next_offset

    if not feature_parts:
        raise ValueError(f"{source_name} did not yield any selected-node activations.")
    if expected_feature_schema is not None and feature_schema != expected_feature_schema:
        raise ValueError(f"{source_name} has an inconsistent selected-node feature layout.")
    return torch.cat(feature_parts).to(torch.float32), feature_schema


def download_selected_node_groups(
    bucket: Bucket,
    nodes_object: str,
) -> tuple[SelectedNodeGroups, str]:
    """Download and safely deserialize the selected-node definition in memory."""
    with io.BytesIO() as nodes_buffer:
        bucket.blob(nodes_object).download_to_file(nodes_buffer)
        nodes_digest = hashlib.sha256(nodes_buffer.getbuffer()).hexdigest()
        nodes_buffer.seek(0)
        groups = load_selected_node_groups_from_file(nodes_buffer)
    return groups, nodes_digest


def download_activation_vector(
    bucket: Bucket,
    activation_prefix: str,
    sample_index: int,
    allowed_nodes: dict[str, set[int]],
    expected_feature_schema: list[dict[str, Any]] | None,
) -> tuple[torch.Tensor, list[dict[str, Any]], str]:
    """Download one cache to memory, extract its small vector, and release the cache."""
    object_name = activation_object_name(activation_prefix, sample_index)
    with io.BytesIO() as activation_buffer:
        bucket.blob(object_name).download_to_file(activation_buffer)
        activation_buffer.seek(0)
        payload = torch.load(activation_buffer, map_location="cpu", weights_only=True)
        vector, feature_schema = extract_assistant_feature_vector(
            payload,
            allowed_nodes,
            expected_sample_index=sample_index,
            expected_feature_schema=expected_feature_schema,
            source_name=object_name,
        )
    return vector, feature_schema, object_name


def _download_group_vectors(
    executor: ThreadPoolExecutor,
    bucket: Bucket,
    activation_prefix: str,
    group: MatchedGroup,
    allowed_nodes: dict[str, set[int]],
    expected_feature_schema: list[dict[str, Any]] | None,
    *,
    skip_missing: bool,
    progress: tqdm[Any] | None,
) -> tuple[list[torch.Tensor], list[int], list[str], list[dict[str, Any]]]:
    futures: dict[
        Future[tuple[torch.Tensor, list[dict[str, Any]], str]],
        int,
    ] = {
        executor.submit(
            download_activation_vector,
            bucket,
            activation_prefix,
            sample_index,
            allowed_nodes,
            expected_feature_schema,
        ): sample_index
        for sample_index in group.sample_indices
    }
    results: dict[int, tuple[torch.Tensor, list[dict[str, Any]]]] = {}
    missing_objects: list[str] = []

    for future in as_completed(futures):
        sample_index = futures[future]
        try:
            vector, feature_schema, _ = future.result()
            results[sample_index] = (vector, feature_schema)
        except NotFound:
            object_name = activation_object_name(activation_prefix, sample_index)
            if not skip_missing:
                for pending_future in futures:
                    pending_future.cancel()
                raise
            missing_objects.append(object_name)
            if progress is not None:
                progress.write(f"Skipping missing object gs://{bucket.name}/{object_name}")
        except BaseException:
            for pending_future in futures:
                pending_future.cancel()
            raise
        finally:
            if progress is not None:
                progress.update(1)

    included_indices = [index for index in group.sample_indices if index in results]
    if not included_indices:
        raise ValueError(
            "Every activation object is missing for matched condition "
            f"{group.semantic_values!r}."
        )
    ordered_results = [results[index] for index in included_indices]
    feature_schema = expected_feature_schema or ordered_results[0][1]
    for _, result_schema in ordered_results:
        if result_schema != feature_schema:
            raise ValueError(
                f"Inconsistent feature layouts in matched condition {group.semantic_values!r}."
            )
    return (
        [vector for vector, _ in ordered_results],
        included_indices,
        missing_objects,
        feature_schema,
    )


def _run_fingerprint(provenance: dict[str, Any]) -> str:
    identity = provenance.copy()
    identity.pop("completions_path", None)
    encoded = json.dumps(identity, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def _feature_width(feature_schema: list[dict[str, Any]]) -> int:
    """Validate contiguous feature offsets and return the flattened width."""
    expected_start = 0
    for entry in feature_schema:
        start = int(entry["feature_start"])
        stop = int(entry["feature_stop"])
        if start != expected_start or stop <= start:
            raise ValueError("Feature schema contains invalid or noncontiguous offsets.")
        expected_start = stop
    return expected_start


def _matrix_from_vectors(
    vectors: list[torch.Tensor],
    feature_schema: list[dict[str, Any]] | None,
) -> torch.Tensor:
    feature_count = _feature_width(feature_schema or [])
    if vectors:
        matrix = torch.stack(vectors)
        if matrix.ndim != 2 or matrix.shape[1] != feature_count:
            raise ValueError("Activation vectors do not match the selected-node feature schema.")
        return matrix
    return torch.empty((0, feature_count), dtype=torch.float32)


def build_output_payload(
    state: ProcessingState,
    selected_node_groups: SelectedNodeGroups,
    provenance: dict[str, Any],
    run_fingerprint: str,
    *,
    status: str,
    target_group_count: int,
) -> dict[str, Any]:
    """Build the self-describing checkpoint/final artifact."""
    return {
        "format_version": OUTPUT_FORMAT_VERSION,
        "status": status,
        "created_at_utc": datetime.now(UTC).isoformat(),
        "completed_group_count": state.completed_group_count,
        "target_group_count": target_group_count,
        "activation_matrix": _matrix_from_vectors(state.vectors, state.feature_schema),
        "metadata": state.metadata,
        "source_sample_indices": state.source_sample_indices,
        "feature_schema": state.feature_schema or [],
        "selected_node_groups": selected_node_groups,
        "missing_activation_objects": state.missing_activation_objects,
        "provenance": provenance,
        "run_fingerprint": run_fingerprint,
    }


def upload_output_payload(bucket: Bucket, output_object: str, payload: dict[str, Any]) -> None:
    """Serialize and upload a checkpoint/final result without touching local disk."""
    with io.BytesIO() as output_buffer:
        torch.save(payload, output_buffer)
        bucket.blob(output_object).upload_from_file(output_buffer, rewind=True)


def _load_existing_state(
    bucket: Bucket,
    output_object: str,
    run_fingerprint: str,
    target_groups: tuple[MatchedGroup, ...],
) -> tuple[ProcessingState, bool]:
    with io.BytesIO() as checkpoint_buffer:
        bucket.blob(output_object).download_to_file(checkpoint_buffer)
        checkpoint_buffer.seek(0)
        checkpoint = torch.load(checkpoint_buffer, map_location="cpu", weights_only=True)

    if checkpoint.get("format_version") != OUTPUT_FORMAT_VERSION:
        raise ValueError(f"Unsupported checkpoint format in gs://{bucket.name}/{output_object}.")
    if checkpoint.get("run_fingerprint") != run_fingerprint:
        raise ValueError(
            f"Existing output gs://{bucket.name}/{output_object} belongs to a different run. "
            "Choose another --output-object or pass --overwrite."
        )
    if int(checkpoint.get("target_group_count", -1)) != len(target_groups):
        raise ValueError("Existing checkpoint has a mismatched target_group_count.")
    completed_group_count = int(checkpoint["completed_group_count"])
    if not 0 <= completed_group_count <= len(target_groups):
        raise ValueError("Existing checkpoint has an invalid completed_group_count.")
    status = checkpoint.get("status")
    if status not in {"in_progress", "complete"}:
        raise ValueError(f"Existing checkpoint has an invalid status: {status!r}.")
    if status == "complete" and completed_group_count != len(target_groups):
        raise ValueError("Existing checkpoint is marked complete but contains too few groups.")

    matrix = checkpoint["activation_matrix"].to(torch.float32)
    metadata = list(checkpoint["metadata"])
    source_sample_indices = [list(indices) for indices in checkpoint["source_sample_indices"]]
    feature_schema = list(checkpoint["feature_schema"])
    if matrix.ndim != 2 or matrix.shape[1] != _feature_width(feature_schema):
        raise ValueError("Existing checkpoint matrix does not match its feature schema.")
    if completed_group_count and not feature_schema:
        raise ValueError("Existing checkpoint is missing its selected-node feature schema.")
    if not (
        matrix.shape[0]
        == len(metadata)
        == len(source_sample_indices)
        == completed_group_count
    ):
        raise ValueError("Existing checkpoint contains inconsistent matched-condition rows.")
    skip_missing = bool(checkpoint.get("provenance", {}).get("skip_missing", False))
    for group, indices, row in zip(
        target_groups,
        source_sample_indices,
        metadata,
        strict=False,
    ):
        index_set = set(indices)
        indices_in_source_order = [
            index for index in group.sample_indices if index in index_set
        ]
        if not indices or len(index_set) != len(indices) or indices != indices_in_source_order:
            raise ValueError("Existing checkpoint source indices do not match the completion index.")
        if not skip_missing and indices != group.sample_indices:
            raise ValueError("Strict checkpoint is missing expected source activation indices.")
        if row.get("sample_index") != indices[0] or row.get("source_sample_count") != len(indices):
            raise ValueError("Existing checkpoint metadata does not match its source indices.")

    state = ProcessingState(
        vectors=list(matrix.unbind(dim=0)),
        metadata=metadata,
        source_sample_indices=source_sample_indices,
        missing_activation_objects=list(checkpoint.get("missing_activation_objects", [])),
        feature_schema=feature_schema,
    )
    return state, status == "complete"


def process_groups(
    bucket: Bucket,
    activation_prefix: str,
    target_groups: tuple[MatchedGroup, ...],
    allowed_nodes: dict[str, set[int]],
    state: ProcessingState,
    *,
    selected_node_groups: SelectedNodeGroups,
    provenance: dict[str, Any],
    run_fingerprint: str,
    output_object: str,
    download_workers: int,
    checkpoint_every_groups: int,
    skip_missing: bool,
) -> ProcessingState:
    """Process complete semantic groups and periodically checkpoint to one GCS object."""
    completed = state.completed_group_count
    attempted_files = sum(len(group.sample_indices) for group in target_groups[:completed])
    total_files = sum(len(group.sample_indices) for group in target_groups)
    progress = tqdm(
        total=total_files,
        initial=attempted_files,
        desc="Streaming activation caches",
        unit="file",
    )
    try:
        with ThreadPoolExecutor(max_workers=download_workers) as executor:
            for group_index in range(completed, len(target_groups)):
                group = target_groups[group_index]
                vectors, included_indices, missing_objects, feature_schema = (
                    _download_group_vectors(
                        executor,
                        bucket,
                        activation_prefix,
                        group,
                        allowed_nodes,
                        state.feature_schema,
                        skip_missing=skip_missing,
                        progress=progress,
                    )
                )
                state.feature_schema = feature_schema
                state.vectors.append(torch.stack(vectors).mean(dim=0))
                state.metadata.append(group.output_metadata(included_indices))
                state.source_sample_indices.append(included_indices)
                state.missing_activation_objects.extend(missing_objects)

                should_checkpoint = (
                    checkpoint_every_groups > 0
                    and state.completed_group_count < len(target_groups)
                    and state.completed_group_count % checkpoint_every_groups == 0
                )
                if should_checkpoint:
                    payload = build_output_payload(
                        state,
                        selected_node_groups,
                        provenance,
                        run_fingerprint,
                        status="in_progress",
                        target_group_count=len(target_groups),
                    )
                    upload_output_payload(bucket, output_object, payload)
                    progress.write(
                        f"Checkpointed {state.completed_group_count}/{len(target_groups)} "
                        f"groups to gs://{bucket.name}/{output_object}"
                    )
                    del payload
    finally:
        progress.close()
    return state


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line interface."""
    parser = argparse.ArgumentParser(
        description=(
            "Stream GCS activation caches, select the requested nodes at the first "
            "assistant position, average matched conditions, and upload one .pt artifact."
        ),
        epilog="Do not run concurrent jobs that share the same output object.",
    )
    parser.add_argument("--completions-path", type=Path, default=DEFAULT_COMPLETIONS_PATH)
    parser.add_argument("--project-id", default=None)
    parser.add_argument("--bucket-name", default=DEFAULT_BUCKET_NAME)
    parser.add_argument("--activation-prefix", default=DEFAULT_ACTIVATION_PREFIX)
    parser.add_argument("--nodes-object", default=DEFAULT_NODES_OBJECT)
    parser.add_argument("--output-object", default=DEFAULT_OUTPUT_OBJECT)
    parser.add_argument(
        "--download-workers",
        type=int,
        default=4,
        help="Concurrent in-memory downloads; peak memory scales with this value.",
    )
    parser.add_argument(
        "--checkpoint-every-groups",
        type=int,
        default=25,
        help=(
            "Overwrite the output object with the complete resumable state every N matched "
            "groups; 0 disables interim checkpoints."
        ),
    )
    parser.add_argument(
        "--resume",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Resume an existing compatible in-progress output object.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Start from scratch even when the output object already exists.",
    )
    parser.add_argument(
        "--skip-missing",
        action="store_true",
        help="Average available samples when an activation object is absent; fail by default.",
    )
    parser.add_argument(
        "--max-groups",
        type=int,
        default=None,
        help="Process only the first N matched groups (useful for a small end-to-end check).",
    )
    return parser


def run(args: argparse.Namespace) -> str:
    """Execute the in-memory GCS aggregation workflow and return the output URI."""
    if args.download_workers < 1:
        raise ValueError("--download-workers must be at least 1.")
    if args.checkpoint_every_groups < 0:
        raise ValueError("--checkpoint-every-groups cannot be negative.")
    if args.max_groups is not None and args.max_groups < 1:
        raise ValueError("--max-groups must be at least 1 when provided.")
    if not args.completions_path.is_file():
        raise FileNotFoundError(f"Completion index not found: {args.completions_path}")

    output_object = args.output_object
    if args.max_groups is not None and output_object == DEFAULT_OUTPUT_OBJECT:
        output_object = limited_output_object(output_object, args.max_groups)
        print(
            f"Limited run will use noncanonical output gs://{args.bucket_name}/{output_object}",
            flush=True,
        )

    load_dotenv(PROJECT_ROOT / ".env")
    project_id = args.project_id or os.getenv("GCP_PROJECT_ID")
    client = _build_gcs_client(project_id)
    bucket = client.bucket(args.bucket_name)

    completion_index = build_completion_index(
        args.completions_path,
        retain_sample_metadata=args.skip_missing,
    )
    target_groups = completion_index.groups
    if args.max_groups is not None:
        target_groups = target_groups[: args.max_groups]
    selected_node_groups, nodes_sha256 = download_selected_node_groups(
        bucket,
        args.nodes_object,
    )
    allowed_nodes = requested_nodes(selected_node_groups)

    provenance = {
        "bucket_name": args.bucket_name,
        "activation_prefix": args.activation_prefix.strip("/"),
        "nodes_object": args.nodes_object.strip("/"),
        "nodes_sha256": nodes_sha256,
        "completions_path": str(args.completions_path),
        "completions_sha256": completion_index.sha256,
        "completion_record_count": completion_index.record_count,
        "position_policy": "assistant",
        "cached_position_selection_policy": "after_assistant",
        "matched_policy": "matched",
        "semantic_fields": list(SEMANTIC_FIELDS),
        "phrasing_fields": list(PHRASING_FIELDS),
        "target_group_count": len(target_groups),
        "skip_missing": args.skip_missing,
    }
    fingerprint = _run_fingerprint(provenance)
    output_blob = bucket.blob(output_object)
    state = ProcessingState()
    if output_blob.exists() and not args.overwrite:
        if not args.resume:
            raise FileExistsError(
                f"Output already exists: gs://{args.bucket_name}/{output_object}"
            )
        state, is_complete = _load_existing_state(
            bucket,
            output_object,
            fingerprint,
            target_groups,
        )
        if is_complete:
            output_uri = f"gs://{args.bucket_name}/{output_object}"
            print(f"Complete output already exists: {output_uri}")
            return output_uri
        print(
            f"Resuming {state.completed_group_count}/{len(target_groups)} matched groups "
            f"from gs://{args.bucket_name}/{output_object}",
            flush=True,
        )

    state = process_groups(
        bucket,
        args.activation_prefix,
        target_groups,
        allowed_nodes,
        state,
        selected_node_groups=selected_node_groups,
        provenance=provenance,
        run_fingerprint=fingerprint,
        output_object=output_object,
        download_workers=args.download_workers,
        checkpoint_every_groups=args.checkpoint_every_groups,
        skip_missing=args.skip_missing,
    )
    final_payload = build_output_payload(
        state,
        selected_node_groups,
        provenance,
        fingerprint,
        status="complete",
        target_group_count=len(target_groups),
    )
    upload_output_payload(bucket, output_object, final_payload)
    output_uri = f"gs://{args.bucket_name}/{output_object}"
    print(
        f"Uploaded {tuple(final_payload['activation_matrix'].shape)} matched activations "
        f"to {output_uri}",
        flush=True,
    )
    return output_uri


def main() -> None:
    """Run the command-line workflow."""
    run(build_parser().parse_args())


if __name__ == "__main__":
    main()
