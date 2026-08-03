"""Download activation caches from GCS and retain residual-stream positions 0-2.

This is a standalone exploration utility; it is intentionally not registered as a
project workflow or console entry point.
"""

from __future__ import annotations

import argparse
import io
import os
from pathlib import Path
from typing import Any, Iterable

import torch
from dotenv import load_dotenv
from google.cloud import storage
from google.cloud.storage import Blob, Bucket
from tqdm import tqdm


DEFAULT_GCS_URI = (
    "gs://temporal-research-bucket/"
    "conversational_after_assistant_residual_stream/"
    "results/feature_geometry_after_assistant_residual_stream"
)
DEFAULT_OUTPUT = Path("data/residual_stream_positions_0_1_2.pt")
TOKEN_POSITION_INDICES = (0, 1, 2)
OUTPUT_FORMAT_VERSION = 1


def parse_gcs_uri(uri: str) -> tuple[str, str]:
    """Split a ``gs://bucket/prefix`` URI into its bucket and object prefix."""
    if not uri.startswith("gs://"):
        raise ValueError(f"Expected a gs:// URI, got {uri!r}.")
    bucket_name, separator, prefix = uri[5:].partition("/")
    if not bucket_name or not separator or not prefix.strip("/"):
        raise ValueError(f"GCS URI must include both a bucket and prefix: {uri!r}.")
    return bucket_name, prefix.strip("/")


def activation_blobs(bucket: Bucket, prefix: str) -> Iterable[Blob]:
    """Yield activation cache blobs below ``prefix`` in deterministic order."""
    object_prefix = f"{prefix.rstrip('/')}/activations_sample_"
    blobs = (
        blob
        for blob in bucket.list_blobs(prefix=object_prefix)
        if blob.name.endswith(".pt")
    )
    yield from sorted(blobs, key=lambda blob: blob.name)


def extract_residual_stream_positions(
    payload: dict[str, Any],
    *,
    source_object: str,
    position_indices: tuple[int, ...] = TOKEN_POSITION_INDICES,
) -> dict[str, Any]:
    """Extract selected cached-position indices from every residual-stream layer."""
    if payload.get("position_selection_policy") != "after_assistant":
        raise ValueError(
            f"{source_object} was not cached with position_selection_policy="
            "'after_assistant'."
        )

    cached_positions = payload.get("positions")
    if not isinstance(cached_positions, (list, tuple)):
        raise ValueError(f"{source_object} does not contain a positions sequence.")
    required_position_count = max(position_indices) + 1
    if len(cached_positions) < required_position_count:
        raise ValueError(
            f"{source_object} has only {len(cached_positions)} cached positions; "
            f"cannot select indices {list(position_indices)}."
        )

    residual_streams = payload.get("residual_stream_activations")
    if not isinstance(residual_streams, dict) or not residual_streams:
        raise ValueError(f"{source_object} has no residual-stream activations.")

    index = torch.tensor(position_indices, dtype=torch.long)
    selected: dict[str, torch.Tensor] = {}
    for layer_name, tensor in residual_streams.items():
        if not isinstance(tensor, torch.Tensor):
            raise ValueError(f"{source_object}:{layer_name} is not a tensor.")
        if tensor.ndim < 3 or tensor.shape[0] != 1:
            raise ValueError(
                f"{source_object}:{layer_name} has shape {tuple(tensor.shape)}; "
                "expected batch x cached positions x hidden size."
            )
        if tensor.shape[1] != len(cached_positions):
            raise ValueError(
                f"{source_object}:{layer_name} has {tensor.shape[1]} cached positions; "
                f"the payload declares {len(cached_positions)}."
            )
        selected[layer_name] = tensor.index_select(1, index).contiguous()

    metadata = payload.get("metadata")
    sample_index = None
    if isinstance(metadata, list) and metadata and isinstance(metadata[0], dict):
        sample_index = metadata[0].get("sample_index")

    return {
        "source_object": source_object,
        "sample_index": sample_index,
        "cached_position_indices": list(position_indices),
        "absolute_token_positions": [cached_positions[i] for i in position_indices],
        "residual_stream_activations": selected,
    }


def download_and_extract(blob: Blob) -> dict[str, Any]:
    """Download one cache into memory and return only its selected residuals."""
    with io.BytesIO() as buffer:
        blob.download_to_file(buffer)
        buffer.seek(0)
        payload = torch.load(buffer, map_location="cpu", weights_only=True)
    if not isinstance(payload, dict):
        raise ValueError(f"{blob.name} did not contain a mapping payload.")
    return extract_residual_stream_positions(payload, source_object=blob.name)


def combine_samples(samples: list[dict[str, Any]]) -> dict[str, Any]:
    """Stack samples without reducing the three retained token positions."""
    if not samples:
        raise ValueError("At least one extracted sample is required.")

    first_residuals = samples[0]["residual_stream_activations"]
    layer_names = list(first_residuals)
    expected_layers = set(layer_names)
    combined: dict[str, torch.Tensor] = {}
    for layer_name in layer_names:
        layer_tensors: list[torch.Tensor] = []
        expected_shape = tuple(first_residuals[layer_name].shape)
        for sample in samples:
            residuals = sample["residual_stream_activations"]
            if set(residuals) != expected_layers:
                raise ValueError(
                    f"{sample['source_object']} has an inconsistent residual-stream layer set."
                )
            tensor = residuals[layer_name]
            if tuple(tensor.shape) != expected_shape:
                raise ValueError(
                    f"{sample['source_object']}:{layer_name} has shape {tuple(tensor.shape)}; "
                    f"expected {expected_shape}."
                )
            layer_tensors.append(tensor)
        # Each source tensor is (1, 3, d_model), so concatenating on its
        # existing batch dimension produces (num_samples, 3, d_model).
        combined[layer_name] = torch.cat(layer_tensors, dim=0)

    return {
        "sample_indices": [sample["sample_index"] for sample in samples],
        "source_objects": [sample["source_object"] for sample in samples],
        "absolute_token_positions": [
            sample["absolute_token_positions"] for sample in samples
        ],
        "residual_stream_activations": combined,
    }


def run(
    gcs_uri: str,
    output: Path,
    *,
    project_id: str | None = None,
    max_files: int | None = None,
    overwrite: bool = False,
) -> Path:
    """Download matching caches, extract positions 0-2, and save one local file."""
    if output.exists() and not overwrite:
        raise FileExistsError(f"Output already exists: {output}. Pass --overwrite to replace it.")
    if max_files is not None and max_files < 1:
        raise ValueError("max_files must be at least 1 when provided.")

    bucket_name, prefix = parse_gcs_uri(gcs_uri)
    client = storage.Client(project=project_id)
    bucket = client.bucket(bucket_name)
    blobs = activation_blobs(bucket, prefix)

    samples: list[dict[str, Any]] = []
    for blob in tqdm(blobs, total=max_files, desc="Extracting residual streams", unit="file"):
        samples.append(download_and_extract(blob))
        if max_files is not None and len(samples) >= max_files:
            break
    if not samples:
        raise FileNotFoundError(f"No activation cache files found below {gcs_uri}.")

    combined = combine_samples(samples)
    result = {
        "format_version": OUTPUT_FORMAT_VERSION,
        "source_gcs_uri": gcs_uri.rstrip("/"),
        "cached_position_indices": list(TOKEN_POSITION_INDICES),
        "sample_count": len(samples),
        **combined,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(result, output)
    return output


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gcs-uri", default=DEFAULT_GCS_URI)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--project-id",
        default=os.getenv("GCP_PROJECT_ID"),
        help="GCP project used for credentials/quota (defaults to GCP_PROJECT_ID).",
    )
    parser.add_argument(
        "--max-files",
        type=int,
        default=None,
        help="Optional limit for a small exploratory download.",
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    load_dotenv()
    args = parse_args()
    output = run(
        args.gcs_uri,
        args.output,
        project_id=args.project_id,
        max_files=args.max_files,
        overwrite=args.overwrite,
    )
    print(f"Saved extracted residual streams to {output.resolve()}")


if __name__ == "__main__":
    main()
