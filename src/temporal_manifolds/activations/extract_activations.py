"""Extract selected-node activations for generated temporal prompt completions.

The input is a JSONL file produced by ``generate_completions.py``. Each record is
expected to contain ``full_text`` plus prompt metadata. The default position policy
caches from the generated ``assistant`` marker through the generated planning header,
stopping before the detailed plan body marker. Other policies can cache the entire
text or every position from the assistant marker through the response. Selected-node
activations and full residual-stream outputs after every model layer are saved.
"""

from __future__ import annotations

import argparse
import io
import json
import pickle
import sys
from pathlib import Path
from typing import Any, BinaryIO, Literal

from tqdm import tqdm


def find_project_root(start: Path) -> Path:
    """Find the repository root by walking upward until src/ is present."""
    for path in (start, *start.parents):
        if (path / "src").is_dir():
            return path
    raise RuntimeError(f"Could not find project root containing src/ from {start}")


PROJECT_ROOT = find_project_root(Path(__file__).resolve())
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

DEFAULT_INPUT_PATH = PROJECT_ROOT / "data" / "model_completions" / "completions_256.jsonl"
DEFAULT_NODES_PATH = PROJECT_ROOT / "data" / "selected_nodes" / "final_node_list.pkl"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "results" / "feature_geometry_new_activations"
DEFAULT_MODEL_NAME = "Qwen/Qwen3-4B-Instruct-2507"

SelectedNode = tuple[tuple[int, str], int]
SelectedNodeGroups = dict[str, list[SelectedNode]]
ActivationSpan = tuple[int, int, str, str]
PositionSelectionPolicy = Literal["default", "all", "after_assistant"]
POSITION_SELECTION_POLICIES: tuple[PositionSelectionPolicy, ...] = (
    "default",
    "all",
    "after_assistant",
)

ASSISTANT_MARKER = "assistant\n"

PLAN_MARKER_PAIRS = (
    ("Strategy:", "Steps:", "assistant_to_strategy_generation"),
    ("Summary:", "Checklist:", "assistant_to_summary_generation"),
    ("Approach:", "Actions:", "assistant_to_approach_generation"),
    ("Allocation rule:", "Schedule:", "assistant_to_allocation_rule_generation"),
)


class RestrictedUnpickler(pickle.Unpickler):
    """Unpickle only primitive containers used by selected-node files."""

    def find_class(self, module: str, name: str) -> Any:
        raise pickle.UnpicklingError(f"Unsupported pickle global: {module}.{name}")


def resolve_path(path: str | Path) -> Path:
    """Resolve repo-relative paths."""
    path = Path(path)
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path


def gcs_object_name_for_file(local_file: Path, upload_root: Path) -> str:
    """Return a stable GCS object name for a local cache file."""
    local_file_abs = local_file.resolve()
    try:
        return local_file_abs.relative_to(PROJECT_ROOT.resolve()).as_posix()
    except ValueError:
        return local_file_abs.relative_to(upload_root.resolve()).as_posix()


def normalize_selected_node_groups(raw_nodes: Any) -> SelectedNodeGroups:
    """Validate and normalize a deserialized selected-node mapping."""
    if not isinstance(raw_nodes, dict):
        raise ValueError(f"Expected selected nodes to be a dict, got {type(raw_nodes)}")

    selected_node_groups: SelectedNodeGroups = {}
    for group_name, group_nodes in raw_nodes.items():
        if not isinstance(group_name, str):
            raise ValueError(f"Selected node group names must be strings: {group_name!r}")

        selected_node_groups[group_name] = []
        for raw_node in sorted(group_nodes):
            if (
                not isinstance(raw_node, tuple)
                or len(raw_node) != 2
                or not isinstance(raw_node[0], str)
                or not isinstance(raw_node[1], int)
            ):
                raise ValueError(f"Invalid selected node entry: {raw_node!r}")

            component_layer, node_index = raw_node
            if "/" not in component_layer:
                raise ValueError(f"Expected component/layer entry, got {component_layer!r}")
            component, layer_text = component_layer.split("/", maxsplit=1)
            selected_node_groups[group_name].append(((int(layer_text), component), node_index))

    return selected_node_groups


def load_selected_node_groups_from_file(nodes_file: BinaryIO) -> SelectedNodeGroups:
    """Load selected nodes from an open binary file without requiring a local path."""
    return normalize_selected_node_groups(RestrictedUnpickler(nodes_file).load())


def load_selected_node_groups(nodes_path: Path) -> SelectedNodeGroups:
    """Load selected nodes as group -> [((layer, component), node_index), ...]."""
    with nodes_path.open("rb") as nodes_file:
        return load_selected_node_groups_from_file(nodes_file)


def get_unique_layer_components(
    selected_node_groups: SelectedNodeGroups,
) -> list[tuple[int, str]]:
    """Return sorted unique layer/component pairs required by selected nodes."""
    return sorted(
        {
            layer_component
            for group_nodes in selected_node_groups.values()
            for layer_component, _ in group_nodes
        }
    )


def get_cache_layer_components(
    selected_node_groups: SelectedNodeGroups,
    num_hidden_layers: int,
) -> list[tuple[int, str]]:
    """Return selected components plus every layer's residual-stream output."""
    layer_components = set(get_unique_layer_components(selected_node_groups))
    layer_components.update((layer, "layer_out") for layer in range(num_hidden_layers))
    return sorted(layer_components)


def group_node_indices(
    selected_node_groups: SelectedNodeGroups,
) -> dict[tuple[int, str], list[int]]:
    """Return unique selected node indices grouped by layer/component."""
    grouped: dict[tuple[int, str], set[int]] = {}
    for group_nodes in selected_node_groups.values():
        for layer_component, node_index in group_nodes:
            grouped.setdefault(layer_component, set()).add(node_index)

    return {
        layer_component: sorted(node_indices) for layer_component, node_indices in grouped.items()
    }


def selected_nodes_include_attention_heads(
    selected_node_groups: SelectedNodeGroups,
) -> bool:
    """Return whether selected nodes include z attention-head entries."""
    return any(
        component == "z"
        for group_nodes in selected_node_groups.values()
        for (_, component), _ in group_nodes
    )


def extract_selected_activations(
    activations: Any,
    selected_node_groups: SelectedNodeGroups,
) -> dict[str, dict[str, Any]]:
    """Extract selected nodes once per layer/component from an ActivationDict."""
    import torch

    selected: dict[str, dict[str, list[int] | torch.Tensor]] = {}
    for (layer, component), node_indices in group_node_indices(selected_node_groups).items():
        activation = activations[(layer, component)]
        index = torch.as_tensor(node_indices, device=activation.device)
        if component == "z":
            node_dim = 2 if activation.ndim == 4 else 1
        else:
            node_dim = 2 if activation.ndim == 3 else 1

        selected[f"{component}/{layer}"] = {
            "node_indices": node_indices,
            "values": activation.index_select(node_dim, index).detach().cpu(),
        }
    return selected


def extract_residual_stream_activations(
    activations: Any,
    num_hidden_layers: int,
) -> dict[str, Any]:
    """Extract full residual-stream activations after every transformer layer."""
    return {
        f"layer_out/{layer}": activations[(layer, "layer_out")].detach().cpu()
        for layer in range(num_hidden_layers)
    }


def load_completion_records(
    input_path: Path,
    max_samples: int | None = None,
) -> list[dict[str, Any]]:
    """Load completion JSONL records, preserving per-record metadata."""
    records: list[dict[str, Any]] = []
    with input_path.open("r", encoding="utf-8") as f:
        for line_number, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            full_text = record.get("full_text")
            if not isinstance(full_text, str) or not full_text:
                raise ValueError(f"Record {line_number} in {input_path} is missing full_text")
            records.append(record)
            if max_samples is not None and len(records) >= max_samples:
                break
    return records


def get_underlying_tokenizer(tokenizer: Any) -> Any:
    """Return the HF tokenizer wrapped by mech_interp_toolkit when present."""
    return getattr(tokenizer, "tokenizer", tokenizer)


def set_pad_token_if_missing(tokenizer: Any) -> None:
    """Set a pad token on wrapped or raw tokenizers when one is missing."""
    hf_tokenizer = get_underlying_tokenizer(tokenizer)
    if getattr(hf_tokenizer, "pad_token_id", None) is None:
        hf_tokenizer.pad_token = hf_tokenizer.eos_token


def tokenize_raw_texts(tokenizer: Any, texts: list[str]) -> dict[str, Any]:
    """Tokenize decoded full-text completions without reapplying a chat template."""
    hf_tokenizer = get_underlying_tokenizer(tokenizer)
    return hf_tokenizer(
        texts,
        add_special_tokens=True,
        return_tensors="pt",
        padding=True,
    )


def find_assistant_plan_span(full_text: str) -> ActivationSpan | None:
    """Return the char span from assistant marker through the generated plan header."""
    assistant_start = full_text.rfind(ASSISTANT_MARKER)
    if assistant_start == -1:
        return None
    search_start = assistant_start + len(ASSISTANT_MARKER)

    matched_span: ActivationSpan | None = None
    matched_header_start: int | None = None
    for header_marker, body_marker, activation_section in PLAN_MARKER_PAIRS:
        header_start = full_text.find(header_marker, search_start)
        if header_start == -1:
            continue

        body_start = full_text.find(body_marker, header_start)
        if body_start == -1:
            continue

        span_end = body_start
        while span_end > assistant_start and full_text[span_end - 1].isspace():
            span_end -= 1
        if span_end <= assistant_start:
            continue

        if matched_header_start is None or header_start < matched_header_start:
            matched_header_start = header_start
            matched_span = (
                assistant_start,
                span_end,
                full_text[assistant_start:span_end],
                activation_section,
            )

    return matched_span


def find_assistant_response_span(full_text: str) -> ActivationSpan | None:
    """Return the final assistant marker and its non-empty response char span."""
    assistant_start = full_text.rfind(ASSISTANT_MARKER)
    if assistant_start == -1:
        return None

    response_start = assistant_start + len(ASSISTANT_MARKER)
    if response_start >= len(full_text):
        return None
    return (
        assistant_start,
        len(full_text),
        full_text[assistant_start:],
        "assistant_response",
    )


def find_activation_span(
    full_text: str,
    *,
    position_selection_policy: PositionSelectionPolicy,
) -> ActivationSpan | None:
    """Return the activation char span selected by the requested policy."""
    if position_selection_policy == "default":
        return find_assistant_plan_span(full_text)
    if position_selection_policy == "all":
        return (0, len(full_text), full_text, "all_positions")
    if position_selection_policy == "after_assistant":
        return find_assistant_response_span(full_text)

    raise ValueError(f"Unsupported position selection policy: {position_selection_policy!r}")


def token_positions_for_char_span(
    tokenizer: Any,
    text: str,
    start_char: int,
    end_char: int,
) -> list[int]:
    """Map a character span to token positions using tokenizer offsets."""
    hf_tokenizer = get_underlying_tokenizer(tokenizer)
    encoding = hf_tokenizer(
        text,
        return_offsets_mapping=True,
        add_special_tokens=True,
    )
    offsets = encoding["offset_mapping"]
    if offsets and isinstance(offsets[0], list):
        offsets = offsets[0]

    positions = [
        token_idx
        for token_idx, (token_start, token_end) in enumerate(offsets)
        if token_end > token_start and token_end > start_char and token_start < end_char
    ]
    if not positions:
        raise ValueError(f"Character span {start_char}:{end_char} did not map to any tokens")
    return positions


def all_token_positions(tokenizer: Any, text: str) -> list[int]:
    """Return every non-padding token position for ``text``."""
    tokenized = tokenize_raw_texts(tokenizer, [text])
    input_ids = tokenized["input_ids"]
    sequence_length = input_ids.shape[-1] if hasattr(input_ids, "shape") else len(input_ids[0])
    return list(range(sequence_length))


def token_positions_for_policy(
    tokenizer: Any,
    text: str,
    span: ActivationSpan,
    *,
    position_selection_policy: PositionSelectionPolicy,
) -> list[int]:
    """Return token positions selected by the requested policy."""
    if position_selection_policy == "all":
        return all_token_positions(tokenizer, text)

    start_char, end_char, _, _ = span
    return token_positions_for_char_span(
        tokenizer,
        text,
        start_char,
        end_char,
    )


def build_activation_metadata(
    record: dict[str, Any],
    sample_index: int,
    char_span: tuple[int, int],
    token_positions: list[int],
    activation_text: str,
    activation_section: str,
    position_selection_policy: PositionSelectionPolicy,
) -> list[dict[str, Any]]:
    """Build metadata for one assistant-to-plan-header activation sample."""
    return [
        {
            "sample_index": sample_index,
            "prompt": record.get("prompt"),
            "prompt_metadata": record.get("prompt_metadata", {}),
            "completion_model_name": record.get("model_name"),
            "activation_section": activation_section,
            "activation_char_span": list(char_span),
            "activation_token_positions": token_positions,
            "activation_text": activation_text,
            "position_selection_policy": position_selection_policy,
        }
    ]


def cache_completion_activations(
    *,
    input_path: Path,
    model_name: str,
    nodes_path: Path,
    output_dir: Path,
    dtype: str | None,
    device: str | None,
    attn_type: str,
    max_samples: int | None,
    overwrite: bool,
    save_to_gcp: bool,
    gcp_project_id: str | None,
    gcs_bucket_name: str | None,
    gcs_prefix: str | None,
    upload_worker_count: int = 4,
    upload_queue_capacity: int = 8,
    position_selection_policy: PositionSelectionPolicy = "default",
) -> None:
    """Cache selected-node and per-layer residual-stream completion activations."""
    import torch

    from ..utils.gcs_upload import maybe_start_memory_gcs_upload_workers
    from ..utils.mech_interp_toolkit.activation_utils import get_activations
    from ..utils.mech_interp_toolkit.utils import load_model_tokenizer_config

    selected_node_groups = load_selected_node_groups(nodes_path)
    records = load_completion_records(input_path, max_samples=max_samples)

    model, tokenizer, _ = load_model_tokenizer_config(
        model_name=model_name,
        device=device,
        dtype=dtype,
        attn_type=attn_type,
    )
    set_pad_token_if_missing(tokenizer)
    num_hidden_layers = int(model.config.num_hidden_layers)
    layer_components = get_cache_layer_components(selected_node_groups, num_hidden_layers)

    if not save_to_gcp:
        output_dir.mkdir(parents=True, exist_ok=True)
    skipped_spans: list[dict[str, Any]] = []
    upload_queue, upload_threads, enqueue_upload = maybe_start_memory_gcs_upload_workers(
        enabled=save_to_gcp,
        project_id=gcp_project_id,
        bucket_name=gcs_bucket_name,
        prefix=gcs_prefix,
        worker_count=upload_worker_count,
        queue_capacity=upload_queue_capacity,
    )

    try:
        for sample_index, record in enumerate(
            tqdm(
                records,
                total=len(records),
                desc="Caching completion activations",
            )
        ):
            output_file = output_dir / f"activations_sample_{sample_index:05d}.pt"
            if not save_to_gcp and output_file.exists() and not overwrite:
                continue

            text = record["full_text"]
            span = find_activation_span(
                text,
                position_selection_policy=position_selection_policy,
            )
            if span is None:
                skipped_spans.append(
                    {
                        "sample_index": sample_index,
                        "error": (
                            "Could not find a non-empty activation span for position policy "
                            f"{position_selection_policy!r}"
                        ),
                        "prompt": record.get("prompt"),
                        "prompt_metadata": record.get("prompt_metadata", {}),
                    }
                )
                continue

            start_char, end_char, activation_text, activation_section = span
            token_positions = token_positions_for_policy(
                tokenizer,
                text,
                span,
                position_selection_policy=position_selection_policy,
            )

            texts = [text]
            tokenized_batch = tokenize_raw_texts(tokenizer, texts)
            activations, logits = get_activations(
                model,
                tokenized_batch,
                layer_components,
                positions=token_positions,
                return_logits=False,
                clone_tensors=True,
            )

            if selected_nodes_include_attention_heads(selected_node_groups):
                activations = activations.split_heads()
            cache_payload: dict[str, Any] = {
                "input_path": str(input_path),
                "model_name": model_name,
                "nodes_path": str(nodes_path),
                "layer_components": layer_components,
                "metadata": build_activation_metadata(
                    record=record,
                    sample_index=sample_index,
                    char_span=(start_char, end_char),
                    token_positions=token_positions,
                    activation_text=activation_text,
                    activation_section=activation_section,
                    position_selection_policy=position_selection_policy,
                ),
                "positions": token_positions,
                "average_positions": False,
                "activation_section": activation_section,
                "position_selection_policy": position_selection_policy,
                "activations": extract_selected_activations(
                    activations,
                    selected_node_groups,
                ),
                "residual_stream_activations": extract_residual_stream_activations(
                    activations,
                    num_hidden_layers,
                ),
            }
            if logits is not None:
                cache_payload["logits"] = logits.detach().cpu()

            if save_to_gcp:
                cache_buffer = io.BytesIO()
                torch.save(cache_payload, cache_buffer)
                enqueue_upload(
                    cache_buffer,
                    gcs_object_name_for_file(output_file, upload_root=output_dir),
                    output_file.name,
                )
            else:
                torch.save(cache_payload, output_file)

        if skipped_spans:
            skipped_path = output_dir / "skipped_assistant_plan_header_spans.jsonl"
            skipped_contents = "".join(
                json.dumps(skipped, ensure_ascii=False) + "\n" for skipped in skipped_spans
            )
            if save_to_gcp:
                skipped_buffer = io.BytesIO(skipped_contents.encode("utf-8"))
                enqueue_upload(
                    skipped_buffer,
                    gcs_object_name_for_file(skipped_path, upload_root=output_dir),
                    skipped_path.name,
                )
            else:
                skipped_path.write_text(skipped_contents, encoding="utf-8")
    finally:
        if upload_queue is not None:
            for _ in upload_threads:
                upload_queue.put(None)
            for upload_thread in upload_threads:
                upload_thread.join()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Cache selected-node activations for generated completions."
    )
    parser.add_argument(
        "--input-path",
        type=Path,
        default=DEFAULT_INPUT_PATH,
        help=f"Completion JSONL input. Defaults to {DEFAULT_INPUT_PATH}.",
    )
    parser.add_argument(
        "--model-name",
        default=DEFAULT_MODEL_NAME,
        help="Model used to extract activations.",
    )
    parser.add_argument(
        "--nodes-path",
        type=Path,
        default=DEFAULT_NODES_PATH,
        help="Pickle file containing selected nodes to cache.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=f"Activation cache output directory. Defaults to {DEFAULT_OUTPUT_DIR}.",
    )
    parser.add_argument("--dtype", default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--attn-type", default="sdpa")
    parser.add_argument(
        "--position-selection-policy",
        choices=POSITION_SELECTION_POLICIES,
        default="default",
        help="Policy used to select completion token positions for activation caching.",
    )
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Recompute batches even when their output files already exist.",
    )
    parser.add_argument(
        "--save-to-gcp",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Upload the output directory to Google Cloud Storage after extraction.",
    )
    parser.add_argument("--gcp-project-id", default=None)
    parser.add_argument("--gcs-bucket-name", default=None)
    parser.add_argument("--gcs-prefix", default=None)
    parser.add_argument("--upload-worker-count", type=int, default=4)
    parser.add_argument("--upload-queue-capacity", type=int, default=8)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cache_completion_activations(
        input_path=resolve_path(args.input_path),
        model_name=args.model_name,
        nodes_path=resolve_path(args.nodes_path),
        output_dir=resolve_path(args.output_dir),
        dtype=args.dtype,
        device=args.device,
        attn_type=args.attn_type,
        max_samples=args.max_samples,
        overwrite=args.overwrite,
        save_to_gcp=args.save_to_gcp,
        gcp_project_id=args.gcp_project_id,
        gcs_bucket_name=args.gcs_bucket_name,
        gcs_prefix=args.gcs_prefix,
        upload_worker_count=args.upload_worker_count,
        upload_queue_capacity=args.upload_queue_capacity,
        position_selection_policy=args.position_selection_policy,
    )


if __name__ == "__main__":
    main()
