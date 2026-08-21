"""Cache layer_out/17..35 residual activations at prompt tokens -2 and -1 for every dataset.

This is the expanded counterpart to the fixed-contract single-layer, single-position
caching scripts. Because the repository-wide extraction policy in
``temporal_manifolds.activations.extraction_policy`` pins analysis to
``layer_out/21`` at token -1, this script drives the caching hooks directly instead
of going through :func:`get_activations`, and validates its own payload shape.
Uploads land under GCS prefixes carrying a directory-name tag so layer ranges never
mix with each other or with the fixed-contract artifacts: ``expanded_`` for the
default 17..35 sweep, and ``early_`` for the 0..16 sweep run by
scripts/run_activation_caching_early_selected_acts.sh.
"""

from __future__ import annotations

import argparse
import io
import os
from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

from dotenv import load_dotenv
from tqdm import tqdm

from temporal_manifolds.activations.extraction_policy import canonical_prompt_metadata
from temporal_manifolds.dataset.generate import generate_task_dataset
from temporal_manifolds.utils.gcs_upload import (
    maybe_build_gcs_existing_object_fetcher,
    maybe_start_memory_gcs_upload_workers,
)
from temporal_manifolds.utils.mech_interp_toolkit.activation_dict import ActivationDict
from temporal_manifolds.utils.mech_interp_toolkit.hook_utils import (
    gen_cache_hookfn,
    temporary_hooks,
)
from temporal_manifolds.utils.mech_interp_toolkit.utils import load_model_tokenizer_config

COMPONENT = "layer_out"
FIRST_LAYER = 17
LAST_LAYER = 35
POSITIONS = (-2, -1)
GCS_PREFIX_PREFIX = "expanded_"

DEFAULT_MODEL_NAME = "Qwen/Qwen3-4B-Instruct-2507"
DEFAULT_OUTPUT_ROOT = Path("results")
DEFAULT_BATCH_SIZE = 128


@dataclass(frozen=True)
class Scenario:
    """One dataset run: how prompts are generated and where activations land."""

    name: str
    dataset: str
    gcs_prefix: str
    output_dir_name: str
    remove_output_format_constraints: bool = False


# Mirrors the per-scenario shell wrappers in scripts/run_activation_caching_*.sh.
SCENARIOS: tuple[Scenario, ...] = (
    Scenario(
        name="conversational",
        dataset="conversational",
        gcs_prefix="selected_acts",
        output_dir_name="selected_acts",
    ),
    Scenario(
        name="conversational_no_output_format",
        dataset="conversational",
        gcs_prefix="NOF_selected_acts",
        output_dir_name="selected_acts_no_output_format",
        remove_output_format_constraints=True,
    ),
    Scenario(
        name="abstract",
        dataset="abstract",
        gcs_prefix="abstract_selected_acts",
        output_dir_name="abstract_selected_acts",
    ),
    Scenario(
        name="plain_english",
        dataset="plain_english",
        gcs_prefix="plain_english_selected_acts",
        output_dir_name="plain_english_selected_acts",
    ),
    Scenario(
        name="plain_long",
        dataset="plain_long",
        gcs_prefix="plain_long_selected_acts",
        output_dir_name="plain_long_selected_acts",
    ),
    Scenario(
        name="task_only",
        dataset="task_only",
        gcs_prefix="task_only_selected_acts",
        output_dir_name="task_only_selected_acts",
    ),
)
SCENARIOS_BY_NAME = {scenario.name: scenario for scenario in SCENARIOS}


def target_layers(first_layer: int = FIRST_LAYER, last_layer: int = LAST_LAYER) -> tuple[int, ...]:
    """Return the inclusive layer range that is cached for every prompt."""
    if first_layer < 0:
        raise ValueError(f"first_layer must be non-negative, got {first_layer}")
    if first_layer > last_layer:
        raise ValueError(f"first_layer {first_layer} must not exceed last_layer {last_layer}")
    return tuple(range(first_layer, last_layer + 1))


def layer_component_key(layer: int) -> str:
    """Return the serialized key for one cached layer."""
    return f"{COMPONENT}/{layer}"


def expanded_gcs_prefix(scenario_prefix: str, prefix_tag: str = GCS_PREFIX_PREFIX) -> str:
    """Return the tagged GCS prefix for a scenario's fixed-contract prefix.

    The tag keeps one layer range's uploads in their own directories: ``expanded_``
    for the default 17..35 sweep, ``early_`` for the 0..16 sweep, and so on.
    """
    tag = prefix_tag.strip("/")
    if not tag:
        raise ValueError("prefix_tag must be a non-empty directory-name prefix.")
    return f"{tag}{scenario_prefix}"


def validate_expanded_payload(payload: Any, *, source_name: str = "Activation payload") -> None:
    """Reject a serialized batch that does not match the expanded caching contract."""
    import torch

    if not isinstance(payload, Mapping):
        raise ValueError(f"{source_name} must be a mapping.")

    positions = payload.get("positions")
    if not isinstance(positions, list) or positions != list(POSITIONS):
        raise ValueError(
            f"{source_name} must contain positions={list(POSITIONS)!r}; got {positions!r}."
        )

    layer_components = payload.get("layer_components")
    if not isinstance(layer_components, list) or not layer_components:
        raise ValueError(f"{source_name} must contain a non-empty layer_components list.")

    activations = payload.get("activations")
    if not isinstance(activations, Mapping):
        raise ValueError(f"{source_name} must contain an activations mapping.")
    if set(activations) != set(layer_components):
        raise ValueError(
            f"{source_name} activations keys {sorted(map(str, activations))!r} do not match "
            f"layer_components {sorted(map(str, layer_components))!r}."
        )

    row_counts = set()
    for key, activation in activations.items():
        if not isinstance(activation, torch.Tensor):
            raise ValueError(f"{source_name}:{key} must be a torch.Tensor.")
        if activation.ndim != 3 or activation.shape[1] != len(POSITIONS):
            raise ValueError(
                f"{source_name}:{key} must have shape batch x {len(POSITIONS)} positions x "
                f"hidden size; got {tuple(activation.shape)}."
            )
        if activation.shape[2] < 1:
            raise ValueError(f"{source_name}:{key} must have a non-empty hidden dimension.")
        row_counts.add(int(activation.shape[0]))
    if len(row_counts) != 1:
        raise ValueError(f"{source_name} activations disagree on batch size: {sorted(row_counts)}.")

    row_count = row_counts.pop()
    row_fields = ("sample_indices", "prompts", "prompt_metadata")
    for field in row_fields:
        if not isinstance(payload.get(field), list):
            raise ValueError(f"{source_name} must contain {field!r} as a list.")
    row_lengths = {field: len(payload[field]) for field in row_fields}
    if any(length != row_count for length in row_lengths.values()):
        raise ValueError(
            f"{source_name} has {row_count} activation rows but row metadata lengths "
            f"are {row_lengths}."
        )
    if not all(type(index) is int for index in payload["sample_indices"]):
        raise ValueError(f"{source_name} sample_indices must contain only integers.")
    if not all(isinstance(prompt, str) for prompt in payload["prompts"]):
        raise ValueError(f"{source_name} prompts must contain only strings.")
    if not all(isinstance(metadata, Mapping) for metadata in payload["prompt_metadata"]):
        raise ValueError(f"{source_name} prompt_metadata must contain only mappings.")


def build_payload(
    *,
    activations: Mapping[int, Any],
    records: list[dict[str, Any]],
    sample_indices: list[int],
    batch_index: int,
    model_name: str,
    dataset: str,
    scenario_name: str,
    layers: Sequence[int],
) -> dict[str, Any]:
    """Build the serialized multi-layer, multi-position payload for one prompt batch."""
    payload = {
        "dataset": dataset,
        "scenario": scenario_name,
        "model_name": model_name,
        "batch_index": batch_index,
        "sample_indices": sample_indices,
        "prompts": [record["text"] for record in records],
        "prompt_metadata": [canonical_prompt_metadata(record) for record in records],
        "layer_components": [layer_component_key(layer) for layer in layers],
        "positions": list(POSITIONS),
        "activations": {
            layer_component_key(layer): activations[layer].detach().cpu() for layer in layers
        },
    }
    validate_expanded_payload(payload, source_name="Generated activation payload")
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


def _model_device(model: Any) -> Any:
    """Infer the device holding the model's parameters."""
    import torch

    try:
        return next(model.parameters()).device
    except StopIteration:
        return torch.device("cpu")


def extract_expanded_activations(
    model: Any,
    tokenized: Any,
    layers: Sequence[int],
) -> dict[int, Any]:
    """Cache ``layer_out`` at every requested layer for prompt tokens -2 and -1.

    The shared :func:`get_activations` helper enforces the repository's
    single-layer, single-position extraction policy, so the caching hooks are
    driven directly here with the same clone and early-exit behaviour.
    """
    import torch

    device = _model_device(model)
    inputs = {
        key: (value if value.device == device else value.to(device))
        for key, value in dict(tokenized).items()
        if isinstance(value, torch.Tensor)
    }

    layer_components = [(layer, COMPONENT) for layer in layers]
    cached = ActivationDict(model.config, positions=list(POSITIONS), value_type="activation")
    hook_specs_dict = {"fwd": gen_cache_hookfn(layer_components, cached, clone_tensors=True)}

    with torch.no_grad():
        # Early exit stops the forward pass after the deepest cached layer.
        with temporary_hooks(dict(model.named_modules()), hook_specs_dict, early_exit=True):
            model(**inputs)

    if "attention_mask" in inputs:
        cached.attention_mask = inputs["attention_mask"]
    cached.extract_positions()

    missing = [layer for layer in layers if (layer, COMPONENT) not in cached]
    if missing:
        raise RuntimeError(f"Caching hooks produced no activations for layers {missing}.")
    return {layer: cached[(layer, COMPONENT)] for layer in layers}


def cache_scenario_expanded_acts(
    scenario: Scenario,
    *,
    layers: Sequence[int],
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
    prefix_tag: str = GCS_PREFIX_PREFIX,
) -> None:
    """Cache and upload one dataset scenario across every requested layer."""
    import torch

    records = generate_task_dataset(
        dataset=scenario.dataset,
        remove_output_format_constraints=scenario.remove_output_format_constraints,
    )
    if max_samples is not None:
        records = records[:max_samples]

    gcs_prefix = expanded_gcs_prefix(scenario.gcs_prefix, prefix_tag)
    output_dir = output_root / expanded_gcs_prefix(scenario.output_dir_name, prefix_tag)
    if not save_to_gcp:
        output_dir.mkdir(parents=True, exist_ok=True)

    object_exists = maybe_build_gcs_existing_object_fetcher(
        enabled=save_to_gcp and skip_uploaded and not overwrite,
        project_id=gcp_project_id,
        bucket_name=gcs_bucket_name,
        prefix=gcs_prefix,
        download_existing=False,
    )
    upload_queue, upload_threads, enqueue_upload = maybe_start_memory_gcs_upload_workers(
        enabled=save_to_gcp,
        project_id=gcp_project_id,
        bucket_name=gcs_bucket_name,
        prefix=gcs_prefix,
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
                desc=(
                    f"{scenario.name}: caching {COMPONENT}/{layers[0]}..{layers[-1]} "
                    f"at tokens {list(POSITIONS)}"
                ),
            )
        ):
            output_file = output_dir / f"activations_batch_{batch_index:05d}.pt"
            if save_to_gcp:
                if object_exists(output_file.name, output_file):
                    continue
            elif output_file.exists() and not overwrite:
                continue

            # Left padding aligns every prompt's final non-padding tokens at -2 and -1.
            tokenized = configure_and_tokenize_left_padded(
                tokenizer,
                [record["text"] for record in batch_records],
            )
            activations = extract_expanded_activations(model, tokenized, layers)
            payload = build_payload(
                activations=activations,
                records=batch_records,
                sample_indices=sample_indices,
                batch_index=batch_index,
                model_name=model_name,
                dataset=scenario.dataset,
                scenario_name=scenario.name,
                layers=layers,
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


def cache_expanded_selected_acts(
    *,
    scenarios: Sequence[Scenario] = SCENARIOS,
    first_layer: int = FIRST_LAYER,
    last_layer: int = LAST_LAYER,
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
    prefix_tag: str = GCS_PREFIX_PREFIX,
) -> None:
    """Load the model once and cache every scenario across the requested layer range."""
    if batch_size < 1:
        raise ValueError(f"batch_size must be at least 1, got {batch_size}")
    if not scenarios:
        raise ValueError("At least one scenario must be selected.")

    layers = target_layers(first_layer, last_layer)

    model, tokenizer, _ = load_model_tokenizer_config(
        model_name=model_name,
        device=device,
        dtype=dtype,
        attn_type=attn_type,
    )
    num_hidden_layers = int(model.config.num_hidden_layers)
    if layers[-1] >= num_hidden_layers:
        raise ValueError(
            f"Model {model_name!r} has {num_hidden_layers} layers; "
            f"cannot extract {COMPONENT}/{layers[-1]}."
        )

    for scenario in scenarios:
        print(f"[expanded-caching] Starting scenario {scenario.name}", flush=True)
        cache_scenario_expanded_acts(
            scenario,
            layers=layers,
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
            prefix_tag=prefix_tag,
        )
        print(f"[expanded-caching] Finished scenario {scenario.name}", flush=True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--scenario",
        dest="scenario_names",
        action="append",
        choices=sorted(SCENARIOS_BY_NAME),
        default=None,
        help="Restrict the run to one scenario; repeatable. Defaults to every scenario.",
    )
    parser.add_argument(
        "--prefix-tag",
        default=GCS_PREFIX_PREFIX,
        help=(
            "Directory-name prefix for the GCS uploads and local output dirs, so each "
            "layer range lands in its own folders (default: %(default)s)."
        ),
    )
    parser.add_argument("--first-layer", type=int, default=FIRST_LAYER)
    parser.add_argument("--last-layer", type=int, default=LAST_LAYER)
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
    selected = (
        SCENARIOS
        if not args.scenario_names
        else tuple(SCENARIOS_BY_NAME[name] for name in dict.fromkeys(args.scenario_names))
    )
    cache_expanded_selected_acts(
        scenarios=selected,
        first_layer=args.first_layer,
        last_layer=args.last_layer,
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
        prefix_tag=args.prefix_tag,
    )


if __name__ == "__main__":
    main()
