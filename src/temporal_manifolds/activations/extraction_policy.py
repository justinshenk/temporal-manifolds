"""Repository-wide contract for residual-stream activation extraction.

Every supported extraction and analysis path is intentionally fixed to the
final token of the formatted prompt at the output of transformer layer 21.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import torch

TARGET_LAYER = 21
TARGET_COMPONENT = "layer_out"
TARGET_LAYER_COMPONENT = f"{TARGET_COMPONENT}/{TARGET_LAYER}"
PROMPT_TOKEN_POSITION = -1
CACHED_POSITION_INDEX = 0

TARGET_LAYER_COMPONENTS = ((TARGET_LAYER, TARGET_COMPONENT),)
CACHED_POSITIONS = (PROMPT_TOKEN_POSITION,)

NOT_APPLICABLE = "N/A"

# The standard conversational caching run defines the serialized prompt-metadata
# contract. Other caching modes retain values for these fields when available and
# use NOT_APPLICABLE otherwise.
PROMPT_METADATA_FIELDS = (
    "template_id",
    "template_metadata",
    "task",
    "task_metadata",
    "quantity",
    "quantity_text",
    "base_value",
    "base_unit",
    "unit_variant",
    "number_format",
    "value",
    "value_text",
    "unit",
)
TEMPLATE_METADATA_FIELDS = ("prompt_framing", "output_format")
TASK_METADATA_FIELDS = (
    "task_family",
    "difficulty",
    "domain",
    "complexity",
    "planning_type",
    "stakes",
    "agency",
)


def _metadata_value(value: Any) -> Any:
    """Represent a missing or inapplicable metadata value consistently."""
    return NOT_APPLICABLE if value is None else value


def _canonical_nested_metadata(
    metadata: Any,
    *,
    fields: tuple[str, ...],
    field_name: str,
) -> dict[str, Any]:
    """Return one nested metadata mapping in canonical field order."""
    if metadata is None:
        metadata = {}
    if not isinstance(metadata, Mapping):
        raise ValueError(f"Prompt record {field_name!r} must be a mapping.")
    return {field: _metadata_value(metadata.get(field)) for field in fields}


def canonical_prompt_metadata(record: Mapping[str, Any]) -> dict[str, Any]:
    """Project a prompt record onto the standard conversational output schema."""
    if not isinstance(record, Mapping):
        raise ValueError("Prompt record must be a mapping.")

    metadata = {
        field: _metadata_value(record.get(field))
        for field in PROMPT_METADATA_FIELDS
        if field not in {"template_metadata", "task_metadata"}
    }
    metadata["template_metadata"] = _canonical_nested_metadata(
        record.get("template_metadata"),
        fields=TEMPLATE_METADATA_FIELDS,
        field_name="template_metadata",
    )
    metadata["task_metadata"] = _canonical_nested_metadata(
        record.get("task_metadata"),
        fields=TASK_METADATA_FIELDS,
        field_name="task_metadata",
    )
    return {field: metadata[field] for field in PROMPT_METADATA_FIELDS}


def validate_model_hook_request(
    *,
    layer_components: Any,
    positions: Any,
) -> None:
    """Reject a model-hook request outside the fixed extraction contract."""
    try:
        normalized_components = tuple(tuple(value) for value in layer_components)
    except TypeError as exc:
        raise ValueError(
            f"Model-hook extraction requires {TARGET_LAYER_COMPONENTS!r}."
        ) from exc
    if normalized_components != TARGET_LAYER_COMPONENTS:
        raise ValueError(
            "Model-hook extraction is restricted to "
            f"{TARGET_LAYER_COMPONENTS!r}; got {normalized_components!r}."
        )
    if type(positions) is not int or positions != PROMPT_TOKEN_POSITION:
        raise ValueError(
            "Model-hook extraction is restricted to prompt token position "
            f"{PROMPT_TOKEN_POSITION}; got {positions!r}."
        )


def validate_extraction_request(*, layer_component: Any, position_index: Any) -> None:
    """Reject a layer/position request outside the fixed extraction contract."""
    if layer_component != TARGET_LAYER_COMPONENT:
        raise ValueError(
            "Activation extraction is restricted to "
            f"{TARGET_LAYER_COMPONENT!r}; got {layer_component!r}."
        )
    if type(position_index) is not int or position_index != CACHED_POSITION_INDEX:
        raise ValueError(
            "Activation extraction is restricted to cached position index "
            f"{CACHED_POSITION_INDEX} (prompt token {PROMPT_TOKEN_POSITION}); "
            f"got {position_index!r}."
        )


def validate_cached_position(cached_position: Any) -> None:
    """Reject analysis of anything except the final formatted-prompt token."""
    if type(cached_position) is not int or cached_position != PROMPT_TOKEN_POSITION:
        raise ValueError(
            "Activation analysis is restricted to prompt token position "
            f"{PROMPT_TOKEN_POSITION}; got {cached_position!r}."
        )


def validate_activation_payload(
    payload: Any,
    *,
    source_name: str = "Activation payload",
) -> torch.Tensor:
    """Validate a serialized batch and return its sole activation tensor."""
    if not isinstance(payload, Mapping):
        raise ValueError(f"{source_name} must be a mapping.")

    layer_component = payload.get("layer_component")
    if layer_component != TARGET_LAYER_COMPONENT:
        raise ValueError(
            f"{source_name} must declare layer_component={TARGET_LAYER_COMPONENT!r}; "
            f"got {layer_component!r}."
        )

    positions = payload.get("positions")
    if not isinstance(positions, list) or positions != [PROMPT_TOKEN_POSITION]:
        raise ValueError(
            f"{source_name} must contain positions=[{PROMPT_TOKEN_POSITION}]; "
            f"got {positions!r}."
        )

    activations = payload.get("activations")
    if not isinstance(activations, Mapping):
        raise ValueError(f"{source_name} must contain an activations mapping.")
    activation_keys = set(activations)
    if activation_keys != {TARGET_LAYER_COMPONENT}:
        raise ValueError(
            f"{source_name} must contain only {TARGET_LAYER_COMPONENT!r}; "
            f"got {sorted(map(str, activation_keys))!r}."
        )

    activation = activations[TARGET_LAYER_COMPONENT]
    if not isinstance(activation, torch.Tensor):
        raise ValueError(
            f"{source_name}:{TARGET_LAYER_COMPONENT} must be a torch.Tensor."
        )
    if activation.ndim != 3 or activation.shape[1] != len(CACHED_POSITIONS):
        raise ValueError(
            f"{source_name}:{TARGET_LAYER_COMPONENT} must have shape "
            "batch x 1 cached position x hidden size; "
            f"got {tuple(activation.shape)}."
        )
    if activation.shape[2] < 1:
        raise ValueError(
            f"{source_name}:{TARGET_LAYER_COMPONENT} must have a non-empty hidden dimension."
        )

    row_fields = ("sample_indices", "prompts", "prompt_metadata")
    for field in row_fields:
        if not isinstance(payload.get(field), list):
            raise ValueError(f"{source_name} must contain {field!r} as a list.")
    row_count = int(activation.shape[0])
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

    return activation
