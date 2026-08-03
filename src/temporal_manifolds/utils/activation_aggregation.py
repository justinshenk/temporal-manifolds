"""Aggregate activation tensors over cached token positions."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal, TypeAlias

import torch

AggregationPolicy: TypeAlias = Literal[
    "assistant",
    "all",
]
SelectedNodeGroups: TypeAlias = dict[str, list[tuple[tuple[int, str], int]]]
AllowedNodes: TypeAlias = dict[str, set[int]] | None
TensorGroups: TypeAlias = dict[str, dict[str, torch.Tensor]]

VALID_AGGREGATION_POLICIES = {"assistant", "all"}


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
    position_indices: list[int],
) -> torch.Tensor:
    """Average the requested cached positions."""
    if tensor.ndim < 3 or tensor.shape[0] != 1:
        raise ValueError(f"Expected batch x positions x features, got {tuple(tensor.shape)}.")
    if not position_indices:
        raise ValueError("Aggregation position indices must be non-empty.")
    return tensor[0, position_indices].mean(dim=0)


def aggregate_tensors_by_type(
    tensors: TensorGroups,
    position_indices: list[int],
) -> TensorGroups:
    """Apply one positional aggregation policy to every activation type."""
    return {
        activation_type: {
            name: aggregate_positions(tensor, position_indices)
            for name, tensor in type_tensors.items()
        }
        for activation_type, type_tensors in tensors.items()
    }


def aggregate_activation_payload(
    payload: dict[str, Any],
    policy: AggregationPolicy,
    allowed_nodes: AllowedNodes = None,
    residual_stream_layers: set[int] | None = None,
    *,
    source_name: str = "activation payload",
) -> dict[str, Any]:
    """Aggregate an in-memory activation-cache payload over cached positions."""
    if policy not in VALID_AGGREGATION_POLICIES:
        raise ValueError(f"Unknown aggregation policy: {policy!r}.")
    if payload.get("position_selection_policy") != "after_assistant":
        raise ValueError(
            f"{source_name} was not created with position_selection_policy='after_assistant'."
        )

    cached_positions = [int(position) for position in payload["positions"]]
    if not cached_positions:
        raise ValueError(f"{source_name} does not contain any cached positions.")
    if policy == "assistant":
        position_indices = [0]
    else:
        position_indices = list(range(len(cached_positions)))

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
        "sample_index": payload["metadata"][0]["sample_index"],
        "policy": policy,
        "node_indices": retained_node_indices,
        "included_residual_streams": sorted(tensors["residual"]),
        "activations": aggregate_tensors_by_type(tensors, position_indices),
    }


def aggregate_activation_file(
    path: str | Path,
    policy: AggregationPolicy,
    allowed_nodes: AllowedNodes = None,
    residual_stream_layers: set[int] | None = None,
) -> dict[str, Any]:
    """Load and aggregate one activation-cache file over cached positions."""
    if policy not in VALID_AGGREGATION_POLICIES:
        raise ValueError(f"Unknown aggregation policy: {policy!r}.")
    path = Path(path)
    payload = torch.load(path, map_location="cpu", weights_only=True)
    aggregated = aggregate_activation_payload(
        payload,
        policy,
        allowed_nodes,
        residual_stream_layers,
        source_name=path.name,
    )
    return {"path": path, **aggregated}
