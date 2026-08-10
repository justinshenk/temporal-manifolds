"""Cache layer 21's final-token residual activation for abstract prompts."""

from __future__ import annotations

import argparse
import io
import os
from collections.abc import Iterator
from pathlib import Path
from typing import Any, Sequence

from dotenv import load_dotenv
from tqdm import tqdm

from temporal_manifolds.activations.extraction_policy import (
    PROMPT_TOKEN_POSITION as POSITION,
    TARGET_COMPONENT as COMPONENT,
    TARGET_LAYER as LAYER,
    TARGET_LAYER_COMPONENT,
    canonical_prompt_metadata,
    validate_activation_payload,
)
from temporal_manifolds.dataset.generate_abstract import generate_abstract_prompt_records
from temporal_manifolds.utils.gcs_upload import maybe_start_memory_gcs_upload_workers
from temporal_manifolds.utils.mech_interp_toolkit.activation_utils import get_activations
from temporal_manifolds.utils.mech_interp_toolkit.utils import load_model_tokenizer_config

GCS_PREFIX = "abstract_selected_acts"
DEFAULT_MODEL_NAME = "Qwen/Qwen3-4B-Instruct-2507"
DEFAULT_OUTPUT_DIR = Path("results/abstract_selected_acts")
DEFAULT_BATCH_SIZE = 128


def build_payload(
    *,
    activation: Any,
    records: list[dict[str, Any]],
    sample_indices: list[int],
    batch_index: int,
    model_name: str,
) -> dict[str, Any]:
    """Build the serialized activation payload for one prompt batch."""
    payload = {
        "dataset": "abstract",
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
    """Tokenize a batch with left padding through either a chat wrapper or HF tokenizer."""
    underlying_tokenizer = getattr(tokenizer, "tokenizer", tokenizer)
    if underlying_tokenizer.pad_token_id is None:
        underlying_tokenizer.pad_token = underlying_tokenizer.eos_token
    underlying_tokenizer.padding_side = "left"

    if underlying_tokenizer is tokenizer:
        return tokenizer(prompts, padding=True, return_tensors="pt")

    # ChatTemplateTokenizer applies the model's chat template and performs padded
    # tensor tokenization internally; its public call does not accept HF kwargs.
    return tokenizer(prompts)


def cache_abstract_selected_acts(
    *,
    model_name: str = DEFAULT_MODEL_NAME,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
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
) -> None:
    """Generate abstract prompts and cache batched layer_out/21 at token -1."""
    import torch

    if batch_size < 1:
        raise ValueError(f"batch_size must be at least 1, got {batch_size}")

    records = generate_abstract_prompt_records()
    if max_samples is not None:
        records = records[:max_samples]

    model, tokenizer, _ = load_model_tokenizer_config(
        model_name=model_name,
        device=device,
        dtype=dtype,
        attn_type=attn_type,
    )
    if LAYER >= int(model.config.num_hidden_layers):
        raise ValueError(
            f"Model {model_name!r} has {model.config.num_hidden_layers} layers; "
            f"cannot extract {COMPONENT}/{LAYER}."
        )
    if not save_to_gcp:
        output_dir.mkdir(parents=True, exist_ok=True)
    upload_queue, upload_threads, enqueue_upload = maybe_start_memory_gcs_upload_workers(
        enabled=save_to_gcp,
        project_id=gcp_project_id,
        bucket_name=gcs_bucket_name,
        prefix=GCS_PREFIX,
        worker_count=upload_worker_count,
        queue_capacity=upload_queue_capacity,
    )

    try:
        model.eval()
        batches = iter_indexed_batches(records, batch_size)
        total_batches = (len(records) + batch_size - 1) // batch_size
        for batch_index, (sample_indices, batch_records) in enumerate(
            tqdm(
                batches,
                total=total_batches,
                desc=f"Caching {COMPONENT}/{LAYER} at token {POSITION}",
            )
        ):
            output_file = output_dir / f"activations_batch_{batch_index:05d}.pt"
            if not save_to_gcp and output_file.exists() and not overwrite:
                continue

            # Left padding aligns every prompt's final non-padding token at -1.
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


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-name", default=DEFAULT_MODEL_NAME)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
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
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    load_dotenv()
    args = build_parser().parse_args(argv)
    cache_abstract_selected_acts(
        model_name=args.model_name,
        output_dir=args.output_dir,
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
    )


if __name__ == "__main__":
    main()
