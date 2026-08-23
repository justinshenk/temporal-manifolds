"""Cache the fixed-contract activation for every registered prompt dataset.

The dataset list comes from ``temporal_manifolds.dataset.generate.DATASETS`` so
new dataset modules added to that registry are included automatically. The model
is loaded once and reused across datasets. Each dataset is written to its own
local directory and GCS prefix.
"""

from __future__ import annotations

import argparse
import io
import os
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from tqdm import tqdm

from temporal_manifolds.activations.extraction_policy import (
    PROMPT_TOKEN_POSITION as POSITION,
)
from temporal_manifolds.activations.extraction_policy import (
    TARGET_COMPONENT as COMPONENT,
)
from temporal_manifolds.activations.extraction_policy import TARGET_LAYER as LAYER
from temporal_manifolds.activations.extraction_policy import (
    TARGET_LAYER_COMPONENT,
    canonical_prompt_metadata,
    validate_activation_payload,
)
from temporal_manifolds.dataset.generate import DATASETS, generate_task_dataset
from temporal_manifolds.utils.gcs_upload import (
    maybe_build_gcs_existing_object_fetcher,
    maybe_start_memory_gcs_upload_workers,
)
from temporal_manifolds.utils.mech_interp_toolkit.activation_utils import get_activations
from temporal_manifolds.utils.mech_interp_toolkit.utils import load_model_tokenizer_config

DEFAULT_MODEL_NAME = "Qwen/Qwen3-4B-Instruct-2507"
DEFAULT_OUTPUT_ROOT = Path("results")
DEFAULT_BATCH_SIZE = 128


@dataclass(frozen=True)
class DatasetTarget:
    """Output locations for one registered dataset."""

    name: str
    output_dir_name: str
    gcs_prefix: str


def target_for_dataset(dataset: str) -> DatasetTarget:
    """Return the isolated output locations for a dataset."""
    directory_name = (
        "selected_acts" if dataset == "conversational" else f"{dataset}_selected_acts"
    )
    return DatasetTarget(
        name=dataset,
        output_dir_name=directory_name,
        gcs_prefix=directory_name,
    )


DATASET_TARGETS = tuple(target_for_dataset(dataset) for dataset in DATASETS)
DATASET_TARGETS_BY_NAME = {target.name: target for target in DATASET_TARGETS}


def iter_indexed_batches(
    records: list[dict[str, Any]], batch_size: int
) -> Iterator[tuple[list[int], list[dict[str, Any]]]]:
    """Yield prompt batches together with their dataset indices."""
    if batch_size < 1:
        raise ValueError(f"batch_size must be at least 1, got {batch_size}")
    for start in range(0, len(records), batch_size):
        end = min(start + batch_size, len(records))
        yield list(range(start, end)), records[start:end]


def configure_and_tokenize_left_padded(tokenizer: Any, prompts: list[str]) -> Any:
    """Tokenize a batch with left padding through a chat wrapper or HF tokenizer."""
    underlying_tokenizer = getattr(tokenizer, "tokenizer", tokenizer)
    if underlying_tokenizer.pad_token_id is None:
        underlying_tokenizer.pad_token = underlying_tokenizer.eos_token
    underlying_tokenizer.padding_side = "left"

    if underlying_tokenizer is tokenizer:
        return tokenizer(prompts, padding=True, return_tensors="pt")
    return tokenizer(prompts)


def build_payload(
    *,
    activation: Any,
    records: list[dict[str, Any]],
    sample_indices: list[int],
    batch_index: int,
    model_name: str,
    dataset: str,
) -> dict[str, Any]:
    """Build and validate one fixed-contract activation batch."""
    payload = {
        "dataset": dataset,
        "model_name": model_name,
        "batch_index": batch_index,
        "sample_indices": sample_indices,
        "prompts": [record["text"] for record in records],
        "prompt_metadata": [canonical_prompt_metadata(record) for record in records],
        "layer_component": TARGET_LAYER_COMPONENT,
        "positions": [POSITION],
        "activations": {TARGET_LAYER_COMPONENT: activation.detach().cpu()},
    }
    validate_activation_payload(payload, source_name="Generated activation payload")
    return payload


def cache_dataset(
    target: DatasetTarget,
    *,
    model: Any,
    tokenizer: Any,
    model_name: str,
    output_root: Path,
    batch_size: int,
    max_samples: int | None,
    save_to_gcp: bool,
    gcp_project_id: str | None,
    gcs_bucket_name: str | None,
    upload_worker_count: int,
    upload_queue_capacity: int,
    overwrite: bool,
    skip_uploaded: bool,
) -> None:
    """Generate and cache every prompt from one dataset."""
    import torch

    records = generate_task_dataset(dataset=target.name)
    if max_samples is not None:
        records = records[:max_samples]

    output_dir = output_root / target.output_dir_name
    if not save_to_gcp:
        output_dir.mkdir(parents=True, exist_ok=True)

    object_exists = maybe_build_gcs_existing_object_fetcher(
        enabled=save_to_gcp and skip_uploaded and not overwrite,
        project_id=gcp_project_id,
        bucket_name=gcs_bucket_name,
        prefix=target.gcs_prefix,
        download_existing=False,
    )
    upload_queue, upload_threads, enqueue_upload = maybe_start_memory_gcs_upload_workers(
        enabled=save_to_gcp,
        project_id=gcp_project_id,
        bucket_name=gcs_bucket_name,
        prefix=target.gcs_prefix,
        worker_count=upload_worker_count,
        queue_capacity=upload_queue_capacity,
    )

    try:
        batches = iter_indexed_batches(records, batch_size)
        total_batches = (len(records) + batch_size - 1) // batch_size
        for batch_index, (sample_indices, batch_records) in enumerate(
            tqdm(
                batches,
                total=total_batches,
                desc=f"{target.name}: caching {TARGET_LAYER_COMPONENT} at token {POSITION}",
            )
        ):
            output_file = output_dir / f"activations_batch_{batch_index:05d}.pt"
            if save_to_gcp:
                if object_exists(output_file.name, output_file):
                    continue
            elif output_file.exists() and not overwrite:
                continue

            tokenized = configure_and_tokenize_left_padded(
                tokenizer,
                [record["text"] for record in batch_records],
            )
            activations, _ = get_activations(
                model,
                tokenized,
                [(LAYER, COMPONENT)],
                positions=POSITION,
                return_logits=False,
                clone_tensors=True,
                early_exit=True,
            )
            payload = build_payload(
                activation=activations[(LAYER, COMPONENT)],
                records=batch_records,
                sample_indices=sample_indices,
                batch_index=batch_index,
                model_name=model_name,
                dataset=target.name,
            )

            if save_to_gcp:
                buffer = io.BytesIO()
                torch.save(payload, buffer)
                enqueue_upload(buffer, output_file.name, output_file.name)
            else:
                torch.save(payload, output_file)
    finally:
        if upload_queue is not None:
            for _ in upload_threads:
                upload_queue.put(None)
            for upload_thread in upload_threads:
                upload_thread.join()


def cache_all_selected_acts(
    *,
    targets: Sequence[DatasetTarget] = DATASET_TARGETS,
    model_name: str = DEFAULT_MODEL_NAME,
    output_root: Path = DEFAULT_OUTPUT_ROOT,
    dtype: str | None = None,
    device: str | None = None,
    attn_type: str = "sdpa",
    batch_size: int = DEFAULT_BATCH_SIZE,
    max_samples: int | None = None,
    save_to_gcp: bool = True,
    gcp_project_id: str | None = None,
    gcs_bucket_name: str | None = None,
    upload_worker_count: int = 16,
    upload_queue_capacity: int = 128,
    overwrite: bool = False,
    skip_uploaded: bool = True,
) -> None:
    """Load the model once, then cache every selected dataset."""
    if batch_size < 1:
        raise ValueError(f"batch_size must be at least 1, got {batch_size}")
    if not targets:
        raise ValueError("At least one dataset must be selected.")

    model, tokenizer, _ = load_model_tokenizer_config(
        model_name=model_name,
        device=device,
        dtype=dtype,
        attn_type=attn_type,
    )
    if LAYER >= int(model.config.num_hidden_layers):
        raise ValueError(
            f"Model {model_name!r} has {model.config.num_hidden_layers} layers; "
            f"cannot extract {TARGET_LAYER_COMPONENT}."
        )
    model.eval()

    for target in targets:
        print(f"[all-dataset-caching] Starting dataset {target.name}", flush=True)
        cache_dataset(
            target,
            model=model,
            tokenizer=tokenizer,
            model_name=model_name,
            output_root=output_root,
            batch_size=batch_size,
            max_samples=max_samples,
            save_to_gcp=save_to_gcp,
            gcp_project_id=gcp_project_id,
            gcs_bucket_name=gcs_bucket_name,
            upload_worker_count=upload_worker_count,
            upload_queue_capacity=upload_queue_capacity,
            overwrite=overwrite,
            skip_uploaded=skip_uploaded,
        )
        print(f"[all-dataset-caching] Finished dataset {target.name}", flush=True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset",
        dest="dataset_names",
        action="append",
        choices=sorted(DATASET_TARGETS_BY_NAME),
        default=None,
        help="Restrict the run to one dataset; repeatable. Defaults to all registered datasets.",
    )
    parser.add_argument("--model-name", default=DEFAULT_MODEL_NAME)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--dtype", default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--attn-type", default="sdpa")
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--save-to-gcp", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--gcp-project-id", default=None)
    parser.add_argument("--gcs-bucket-name", default=None)
    parser.add_argument("--upload-worker-count", type=int, default=16)
    parser.add_argument("--upload-queue-capacity", type=int, default=128)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--skip-uploaded",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Skip batches already present in GCS so an interrupted run can resume.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    load_dotenv()
    args = build_parser().parse_args(argv)
    targets = (
        DATASET_TARGETS
        if not args.dataset_names
        else tuple(
            DATASET_TARGETS_BY_NAME[name] for name in dict.fromkeys(args.dataset_names)
        )
    )
    cache_all_selected_acts(
        targets=targets,
        model_name=args.model_name,
        output_root=args.output_root,
        dtype=args.dtype,
        device=args.device,
        attn_type=args.attn_type,
        batch_size=args.batch_size,
        max_samples=args.max_samples,
        save_to_gcp=args.save_to_gcp,
        gcp_project_id=args.gcp_project_id or os.getenv("GCP_PROJECT_ID"),
        gcs_bucket_name=args.gcs_bucket_name or os.getenv("GCS_BUCKET_NAME"),
        upload_worker_count=args.upload_worker_count,
        upload_queue_capacity=args.upload_queue_capacity,
        overwrite=args.overwrite,
        skip_uploaded=args.skip_uploaded,
    )


if __name__ == "__main__":
    main()
