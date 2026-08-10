"""Fixed layer-21, final-prompt-token activation extraction contract."""

from .extraction_policy import (
    CACHED_POSITION_INDEX,
    CACHED_POSITIONS,
    PROMPT_TOKEN_POSITION,
    TARGET_COMPONENT,
    TARGET_LAYER,
    TARGET_LAYER_COMPONENT,
    TARGET_LAYER_COMPONENTS,
    validate_activation_payload,
    validate_cached_position,
    validate_extraction_request,
    validate_model_hook_request,
)

__all__ = [
    "CACHED_POSITION_INDEX",
    "CACHED_POSITIONS",
    "PROMPT_TOKEN_POSITION",
    "TARGET_COMPONENT",
    "TARGET_LAYER",
    "TARGET_LAYER_COMPONENT",
    "TARGET_LAYER_COMPONENTS",
    "validate_activation_payload",
    "validate_cached_position",
    "validate_extraction_request",
    "validate_model_hook_request",
]
