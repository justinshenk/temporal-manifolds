"""Download GCS activation caches and upload residual positions 0-2 in chunks.

This is a standalone exploration utility; it is intentionally not registered as a
project workflow or console entry point.
"""

from __future__ import annotations

import argparse
import io
import os
import queue
import tempfile
import threading
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
DEFAULT_DESTINATION_PREFIX = "resid_only_0_2"
DEFAULT_CHUNK_SIZE = 5_000
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


def download_extract_pipeline(
    blobs: Iterable[Blob],
    *,
    download_workers: int,
    processing_workers: int,
    queue_capacity: int,
) -> list[dict[str, Any]]:
    """Run bounded download, extraction, and raw-buffer cleanup stages."""
    if download_workers < 1 or processing_workers < 1:
        raise ValueError("download_workers and processing_workers must be at least 1.")
    if queue_capacity < 1:
        raise ValueError("queue_capacity must be at least 1.")

    blob_list = list(blobs)
    if not blob_list:
        return []

    download_jobs: queue.Queue[tuple[int, Blob] | None] = queue.Queue()
    downloaded: queue.Queue[tuple[int, str, io.BytesIO] | None] = queue.Queue(
        maxsize=queue_capacity
    )
    cleanup: queue.Queue[tuple[int | None, dict[str, Any] | None, io.BytesIO] | None]
    cleanup = queue.Queue(maxsize=queue_capacity)
    errors: queue.Queue[tuple[str, str, BaseException]] = queue.Queue()
    stop_event = threading.Event()
    results: list[dict[str, Any] | None] = [None] * len(blob_list)
    progress = tqdm(total=len(blob_list), desc="Extracting residual streams", unit="file")

    def download_worker() -> None:
        while True:
            job = download_jobs.get()
            try:
                if job is None:
                    return
                ordinal, blob = job
                if stop_event.is_set():
                    continue
                buffer = io.BytesIO()
                try:
                    blob.download_to_file(buffer)
                    buffer.seek(0)
                    downloaded.put((ordinal, blob.name, buffer))
                except BaseException as error:
                    buffer.close()
                    errors.put(("download", blob.name, error))
                    stop_event.set()
            finally:
                download_jobs.task_done()

    def processing_worker() -> None:
        while True:
            item = downloaded.get()
            try:
                if item is None:
                    return
                ordinal, object_name, buffer = item
                if stop_event.is_set():
                    cleanup.put((None, None, buffer))
                    continue
                try:
                    payload = torch.load(buffer, map_location="cpu", weights_only=True)
                    if not isinstance(payload, dict):
                        raise ValueError(f"{object_name} did not contain a mapping payload.")
                    result = extract_residual_stream_positions(
                        payload,
                        source_object=object_name,
                    )
                    # The selected tensors are copies. Release the much larger source
                    # payload before handing its raw buffer to the cleanup stage.
                    del payload
                    cleanup.put((ordinal, result, buffer))
                except BaseException as error:
                    errors.put(("processing", object_name, error))
                    stop_event.set()
                    cleanup.put((None, None, buffer))
            finally:
                downloaded.task_done()

    def cleanup_worker() -> None:
        while True:
            item = cleanup.get()
            try:
                if item is None:
                    return
                ordinal, result, buffer = item
                # Closing releases the raw downloaded object from RAM. The selected
                # tensors have their own storage after torch.load/index_select.
                buffer.close()
                if ordinal is not None and result is not None:
                    results[ordinal] = result
                    progress.update(1)
            finally:
                cleanup.task_done()

    download_threads = [
        threading.Thread(target=download_worker, name=f"gcs-download-{index}")
        for index in range(download_workers)
    ]
    processing_threads = [
        threading.Thread(target=processing_worker, name=f"activation-process-{index}")
        for index in range(processing_workers)
    ]
    cleanup_thread = threading.Thread(target=cleanup_worker, name="download-cleanup")
    for thread in [*download_threads, *processing_threads, cleanup_thread]:
        thread.start()

    for ordinal, blob in enumerate(blob_list):
        download_jobs.put((ordinal, blob))
    for _ in download_threads:
        download_jobs.put(None)

    download_jobs.join()
    for thread in download_threads:
        thread.join()
    for _ in processing_threads:
        downloaded.put(None)
    downloaded.join()
    for thread in processing_threads:
        thread.join()
    cleanup.put(None)
    cleanup.join()
    cleanup_thread.join()
    progress.close()

    if not errors.empty():
        stage, object_name, error = errors.get()
        raise RuntimeError(f"Failed during {stage} of {object_name}: {error}") from error
    if any(result is None for result in results):
        raise RuntimeError("The extraction pipeline finished with missing results.")
    return [result for result in results if result is not None]


def combine_samples(samples: list[dict[str, Any]]) -> dict[str, Any]:
    """Stack samples while releasing their individual activation tensors."""
    if not samples:
        raise ValueError("At least one extracted sample is required.")

    first_residuals = samples[0]["residual_stream_activations"]
    layer_names = list(first_residuals)
    expected_layers = set(layer_names)
    for sample in samples:
        residuals = sample["residual_stream_activations"]
        if set(residuals) != expected_layers:
            raise ValueError(
                f"{sample['source_object']} has an inconsistent residual-stream layer set."
            )

    combined: dict[str, torch.Tensor] = {}
    for layer_name in layer_names:
        first_tensor = first_residuals[layer_name]
        expected_shape = tuple(first_tensor.shape)
        layer_output = torch.empty(
            (len(samples), *expected_shape[1:]),
            dtype=first_tensor.dtype,
            device=first_tensor.device,
        )
        for sample_number, sample in enumerate(samples):
            residuals = sample["residual_stream_activations"]
            tensor = residuals.pop(layer_name)
            if tuple(tensor.shape) != expected_shape:
                raise ValueError(
                    f"{sample['source_object']}:{layer_name} has shape {tuple(tensor.shape)}; "
                    f"expected {expected_shape}."
                )
            layer_output[sample_number].copy_(tensor[0])
            # pop() plus dropping this reference releases the per-sample tensor
            # as soon as it has been copied into the chunk tensor.
            del tensor
        combined[layer_name] = layer_output

    return {
        "sample_indices": [sample["sample_index"] for sample in samples],
        "source_objects": [sample["source_object"] for sample in samples],
        "absolute_token_positions": [
            sample["absolute_token_positions"] for sample in samples
        ],
        "residual_stream_activations": combined,
    }


def chunked(items: list[Blob], chunk_size: int) -> Iterable[list[Blob]]:
    """Yield consecutive chunks from an already ordered blob list."""
    if chunk_size < 1:
        raise ValueError("chunk_size must be at least 1.")
    for start in range(0, len(items), chunk_size):
        yield items[start : start + chunk_size]


def chunk_object_name(destination_prefix: str, chunk_index: int) -> str:
    """Return the GCS object name for one extracted activation chunk."""
    prefix = destination_prefix.strip("/")
    if not prefix:
        raise ValueError("destination_prefix must not be empty.")
    return f"{prefix}/residual_stream_positions_0_2_chunk_{chunk_index:05d}.pt"


def upload_chunk(
    bucket: Bucket,
    object_name: str,
    payload: dict[str, Any],
    *,
    overwrite: bool,
) -> None:
    """Serialize through an auto-deleted temporary file and upload one chunk."""
    blob = bucket.blob(object_name)
    if not overwrite and blob.exists():
        raise FileExistsError(
            f"gs://{bucket.name}/{object_name} already exists. Pass --overwrite to replace it."
        )
    # A BytesIO serialization duplicates the entire chunk in RAM. A temporary
    # file keeps peak memory bounded and is deleted automatically on close.
    with tempfile.TemporaryFile(prefix="resid_only_0_2_", suffix=".pt") as output_file:
        torch.save(payload, output_file)
        output_file.flush()
        blob.upload_from_file(
            output_file,
            rewind=True,
            content_type="application/octet-stream",
        )


def run(
    gcs_uri: str,
    *,
    project_id: str | None = None,
    max_files: int | None = None,
    overwrite: bool = False,
    download_workers: int = 8,
    processing_workers: int = 2,
    queue_capacity: int = 16,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    destination_prefix: str = DEFAULT_DESTINATION_PREFIX,
) -> list[str]:
    """Extract matching caches and upload chunked results to the source bucket."""
    if max_files is not None and max_files < 1:
        raise ValueError("max_files must be at least 1 when provided.")
    if chunk_size < 1:
        raise ValueError("chunk_size must be at least 1.")

    bucket_name, prefix = parse_gcs_uri(gcs_uri)
    client = storage.Client(project=project_id)
    bucket = client.bucket(bucket_name)
    blobs = list(activation_blobs(bucket, prefix))
    if max_files is not None:
        blobs = blobs[:max_files]
    if not blobs:
        raise FileNotFoundError(f"No activation cache files found below {gcs_uri}.")

    uploaded_uris: list[str] = []
    sample_offset = 0
    for chunk_index, blob_chunk in enumerate(chunked(blobs, chunk_size)):
        object_name = chunk_object_name(destination_prefix, chunk_index)
        uri = f"gs://{bucket_name}/{object_name}"
        if not overwrite and bucket.blob(object_name).exists():
            print(f"Skipping existing chunk {uri}", flush=True)
            uploaded_uris.append(uri)
            sample_offset += len(blob_chunk)
            continue

        samples = download_extract_pipeline(
            blob_chunk,
            download_workers=download_workers,
            processing_workers=processing_workers,
            queue_capacity=queue_capacity,
        )
        combined = combine_samples(samples)
        result = {
            "format_version": OUTPUT_FORMAT_VERSION,
            "source_gcs_uri": gcs_uri.rstrip("/"),
            "cached_position_indices": list(TOKEN_POSITION_INDICES),
            "chunk_index": chunk_index,
            "sample_offset": sample_offset,
            "sample_count": len(samples),
            **combined,
        }
        # combine_samples consumes the individual activation tensors. Drop the
        # remaining per-sample metadata dictionaries before serialization.
        del samples
        upload_chunk(bucket, object_name, result, overwrite=overwrite)
        uploaded_uris.append(uri)
        chunk_sample_count = int(result["sample_count"])
        print(f"Uploaded {chunk_sample_count} samples to {uri}", flush=True)
        sample_offset += chunk_sample_count
        # Drop every tensor in this chunk before downloading the next one.
        del result, combined
    return uploaded_uris


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gcs-uri", default=DEFAULT_GCS_URI)
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
    parser.add_argument("--chunk-size", type=int, default=DEFAULT_CHUNK_SIZE)
    parser.add_argument(
        "--destination-prefix",
        default=DEFAULT_DESTINATION_PREFIX,
        help="Destination folder in the source GCS bucket.",
    )
    parser.add_argument("--download-workers", type=int, default=8)
    parser.add_argument("--processing-workers", type=int, default=2)
    parser.add_argument(
        "--queue-capacity",
        type=int,
        default=16,
        help="Maximum raw downloaded objects waiting in RAM between stages.",
    )
    return parser.parse_args()


def main() -> None:
    load_dotenv()
    args = parse_args()
    uploaded_uris = run(
        args.gcs_uri,
        project_id=args.project_id,
        max_files=args.max_files,
        overwrite=args.overwrite,
        download_workers=args.download_workers,
        processing_workers=args.processing_workers,
        queue_capacity=args.queue_capacity,
        chunk_size=args.chunk_size,
        destination_prefix=args.destination_prefix,
    )
    print(f"Completed {len(uploaded_uris)} chunk(s).", flush=True)


if __name__ == "__main__":
    main()
