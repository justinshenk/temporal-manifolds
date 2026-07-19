"""Aggregate cached activation tensors over semantic response sections."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Literal, TypeAlias

import torch

AggregationPolicy: TypeAlias = Literal[
    "assistant",
    "assisstant",
    "all",
    "strategy",
    "steps",
]
SelectedNodeGroups: TypeAlias = dict[str, list[tuple[tuple[int, str], int]]]
AllowedNodes: TypeAlias = dict[str, set[int]] | None
TensorGroups: TypeAlias = dict[str, dict[str, torch.Tensor]]

PLAN_FORMATS = (
    ("Strategy:", "Steps:"),
    ("Summary:", "Checklist:"),
    ("Approach:", "Actions:"),
)
STEP_START_RE = re.compile(r"(?m)^[ \t]*(?:\d+[.)]|[-*])[ \t]+")
VALID_AGGREGATION_POLICIES = {"assistant", "assisstant", "all", "strategy", "steps"}


def trim_span(text: str, start: int, end: int) -> tuple[int, int]:
    """Remove leading and trailing whitespace from a character span."""
    while start < end and text[start].isspace():
        start += 1
    while end > start and text[end - 1].isspace():
        end -= 1
    return start, end


def find_plan_char_spans(full_text: str) -> tuple[tuple[int, int], list[tuple[int, int]]]:
    """Return the strategy/summary span and each individual step span."""
    assistant_start = full_text.rfind("assistant\n")
    if assistant_start < 0:
        raise ValueError("Could not locate the final assistant response.")
    response_start = assistant_start + len("assistant\n")

    for summary_marker, steps_marker in PLAN_FORMATS:
        summary_start = full_text.find(summary_marker, response_start)
        if summary_start < 0:
            continue
        steps_header_start = full_text.find(
            steps_marker,
            summary_start + len(summary_marker),
        )
        if steps_header_start < 0:
            continue

        summary_span = trim_span(
            full_text,
            summary_start + len(summary_marker),
            steps_header_start,
        )
        steps_text_start = steps_header_start + len(steps_marker)
        matches = list(STEP_START_RE.finditer(full_text, steps_text_start))
        if not matches:
            raise ValueError(f"No steps found after {steps_marker!r}.")

        step_spans = []
        for index, match in enumerate(matches):
            step_end = matches[index + 1].start() if index + 1 < len(matches) else len(full_text)
            step_spans.append(trim_span(full_text, match.end(), step_end))
        return summary_span, step_spans

    raise ValueError("Could not identify a supported strategy/steps output format.")


def cached_indices_for_span(
    offsets: list[tuple[int, int]],
    original_to_cached: dict[int, int],
    char_span: tuple[int, int],
) -> list[int]:
    """Map a character span to indices on the cached-position tensor axis."""
    start, end = char_span
    return [
        original_to_cached[token_index]
        for token_index, (token_start, token_end) in enumerate(offsets)
        if token_index in original_to_cached
        and token_end > token_start
        and token_end > start
        and token_start < end
    ]


def nodes_for_classes(
    selected_node_groups: SelectedNodeGroups,
    node_classes: set[str] | None,
) -> AllowedNodes:
    """Return component/layer node indices belonging to the requested classes."""
    if node_classes is None:
        return None
    unknown_classes = node_classes - selected_node_groups.keys()
    if unknown_classes:
        raise ValueError(
            f"Unknown node classes: {sorted(unknown_classes)}. "
            f"Available classes: {sorted(selected_node_groups)}"
        )

    allowed_nodes: dict[str, set[int]] = {}
    for node_class in node_classes:
        for (layer, component), node_index in selected_node_groups[node_class]:
            allowed_nodes.setdefault(f"{component}/{layer}", set()).add(node_index)
    return allowed_nodes


def activation_tensors_by_type(
    payload: dict[str, Any],
    allowed_nodes: AllowedNodes,
    residual_stream_layers: set[int] | None,
) -> tuple[TensorGroups, dict[str, list[int]]]:
    """Group cached tensors by type and apply node/layer inclusion filters."""
    tensors: TensorGroups = {"mlp": {}, "attn": {}, "residual": {}}
    retained_node_indices: dict[str, list[int]] = {}

    residual_tensors = payload["residual_stream_activations"]
    available_residual_layers = {
        int(name.split("/", 1)[1]) for name in residual_tensors
    }
    if residual_stream_layers is not None:
        unknown_layers = residual_stream_layers - available_residual_layers
        if unknown_layers:
            raise ValueError(
                f"Residual layers are not cached: {sorted(unknown_layers)}. "
                f"Available layers: {sorted(available_residual_layers)}"
            )
    for name, tensor in residual_tensors.items():
        layer = int(name.split("/", 1)[1])
        if residual_stream_layers is None or layer in residual_stream_layers:
            tensors["residual"][name] = tensor

    for name, entry in payload["activations"].items():
        component = name.split("/", 1)[0]
        if component in {"mlp", "mlp_hidden"}:
            activation_type = "mlp"
        elif component in {"z", "attn"}:
            activation_type = "attn"
        else:
            continue

        cached_node_indices = [int(index) for index in entry["node_indices"]]
        if allowed_nodes is None:
            retained_positions = list(range(len(cached_node_indices)))
        else:
            allowed_for_component = allowed_nodes.get(name, set())
            retained_positions = [
                position
                for position, node_index in enumerate(cached_node_indices)
                if node_index in allowed_for_component
            ]
        if not retained_positions:
            continue

        position_index = torch.tensor(retained_positions, dtype=torch.long)
        tensors[activation_type][name] = entry["values"].index_select(2, position_index)
        retained_node_indices[name] = [cached_node_indices[pos] for pos in retained_positions]

    return tensors, retained_node_indices


def aggregate_positions(
    tensor: torch.Tensor,
    position_groups: list[list[int]],
    *,
    keep_group_dimension: bool = False,
) -> torch.Tensor:
    """Average cached positions, optionally retaining one result per group."""
    if tensor.ndim < 3 or tensor.shape[0] != 1:
        raise ValueError(f"Expected batch x positions x features, got {tuple(tensor.shape)}.")
    if not position_groups or any(not indices for indices in position_groups):
        raise ValueError("Every aggregation position group must be non-empty.")
    aggregated = [tensor[0, indices].mean(dim=0) for indices in position_groups]
    return torch.stack(aggregated) if keep_group_dimension else aggregated[0]


def aggregate_tensors_by_type(
    tensors: TensorGroups,
    position_groups: list[list[int]],
    policy: str,
) -> TensorGroups:
    """Apply one positional aggregation policy to every activation type."""
    return {
        activation_type: {
            name: aggregate_positions(
                tensor,
                position_groups,
                keep_group_dimension=policy == "steps",
            )
            for name, tensor in type_tensors.items()
        }
        for activation_type, type_tensors in tensors.items()
    }


def aggregate_activation_file(
    path: str | Path,
    full_text: str,
    tokenizer: Any,
    policy: AggregationPolicy,
    allowed_nodes: AllowedNodes = None,
    residual_stream_layers: set[int] | None = None,
) -> dict[str, Any]:
    """Load and aggregate one activation-cache file by semantic response section."""
    if policy not in VALID_AGGREGATION_POLICIES:
        raise ValueError(f"Unknown aggregation policy: {policy!r}.")
    normalized_policy = "assistant" if policy == "assisstant" else policy
    path = Path(path)
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if payload.get("position_selection_policy") != "after_assistant":
        raise ValueError(
            f"{path.name} was not created with position_selection_policy='after_assistant'."
        )

    cached_positions = [int(position) for position in payload["positions"]]
    original_to_cached = {position: index for index, position in enumerate(cached_positions)}
    offsets = tokenizer(
        full_text,
        add_special_tokens=True,
        return_offsets_mapping=True,
    )["offset_mapping"]
    summary_span, step_spans = find_plan_char_spans(full_text)
    strategy_indices = cached_indices_for_span(offsets, original_to_cached, summary_span)
    step_indices = [
        cached_indices_for_span(offsets, original_to_cached, span) for span in step_spans
    ]
    if not strategy_indices or any(not indices for indices in step_indices):
        raise ValueError(f"A response section in {path.name} mapped to no cached positions.")

    if normalized_policy == "assistant":
        position_groups = [[0]]
    elif normalized_policy == "strategy":
        position_groups = [strategy_indices]
    elif normalized_policy == "steps":
        position_groups = step_indices
    else:
        position_groups = [
            strategy_indices + [index for step in step_indices for index in step]
        ]

    tensors, retained_node_indices = activation_tensors_by_type(
        payload,
        allowed_nodes,
        residual_stream_layers,
    )
    for type_tensors in tensors.values():
        for name, tensor in type_tensors.items():
            if tensor.shape[1] != len(cached_positions):
                raise ValueError(
                    f"{name} has {tensor.shape[1]} positions; "
                    f"expected {len(cached_positions)}."
                )

    return {
        "path": path,
        "sample_index": payload["metadata"][0]["sample_index"],
        "policy": normalized_policy,
        "step_count": len(step_indices),
        "node_indices": retained_node_indices,
        "included_residual_streams": sorted(tensors["residual"]),
        "activations": aggregate_tensors_by_type(tensors, position_groups, normalized_policy),
    }
