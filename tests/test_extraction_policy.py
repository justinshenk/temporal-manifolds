from __future__ import annotations

import pytest

from temporal_manifolds.activations.extraction_policy import (
    PROMPT_TOKEN_POSITION,
    TARGET_LAYER_COMPONENTS,
    validate_model_hook_request,
)
from temporal_manifolds.utils.mech_interp_toolkit.activation_utils import get_activations


def test_fixed_model_hook_request_is_accepted() -> None:
    validate_model_hook_request(
        layer_components=list(TARGET_LAYER_COMPONENTS),
        positions=PROMPT_TOKEN_POSITION,
    )


@pytest.mark.parametrize(
    ("layer_components", "positions", "error_match"),
    [
        ([(20, "layer_out")], -1, "restricted to"),
        ([(21, "mlp")], -1, "restricted to"),
        ([(21, "layer_out"), (20, "layer_out")], -1, "restricted to"),
        ([(21, "layer_out")], 0, "prompt token position -1"),
        ([(21, "layer_out")], [-1], "prompt token position -1"),
    ],
)
def test_model_hook_request_rejects_any_wider_scope(
    layer_components: list[tuple[int, str]],
    positions: object,
    error_match: str,
) -> None:
    with pytest.raises(ValueError, match=error_match):
        validate_model_hook_request(
            layer_components=layer_components,
            positions=positions,
        )


def test_low_level_activation_api_enforces_policy_before_running_a_model() -> None:
    with pytest.raises(ValueError, match="restricted to"):
        get_activations(
            None,  # type: ignore[arg-type]
            {},
            [(20, "layer_out")],
            positions=-1,
        )
