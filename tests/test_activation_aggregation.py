"""Tests for semantic activation aggregation utilities."""

from __future__ import annotations

import pytest
import torch

from temporal_manifolds.utils.activation_aggregation import (
    activation_tensors_by_type,
    aggregate_positions,
    aggregate_tensors_by_type,
    find_plan_char_spans,
    nodes_for_classes,
)


def test_find_plan_char_spans_separates_summary_and_steps() -> None:
    text = "user\ntask\nassistant\nSummary: concise plan\nChecklist:\n- first\n- second"

    summary_span, step_spans = find_plan_char_spans(text)

    assert text[slice(*summary_span)] == "concise plan"
    assert [text[slice(*span)] for span in step_spans] == ["first", "second"]


def test_nodes_for_classes_unions_requested_classes() -> None:
    groups = {
        "positive": [((1, "mlp_hidden"), 3), ((2, "z"), 4)],
        "shared": [((1, "mlp_hidden"), 3), ((1, "mlp_hidden"), 8)],
    }

    assert nodes_for_classes(groups, {"positive", "shared"}) == {
        "mlp_hidden/1": {3, 8},
        "z/2": {4},
    }
    with pytest.raises(ValueError, match="Unknown node classes"):
        nodes_for_classes(groups, {"missing"})


def test_tensor_grouping_filters_selected_nodes_and_whole_residual_layers() -> None:
    residual_zero = torch.arange(24).reshape(1, 3, 8)
    residual_one = torch.arange(24, 48).reshape(1, 3, 8)
    payload = {
        "activations": {
            "mlp_hidden/1": {
                "node_indices": [3, 8],
                "values": torch.arange(6).reshape(1, 3, 2),
            },
            "z/2": {
                "node_indices": [4, 7],
                "values": torch.arange(18).reshape(1, 3, 2, 3),
            },
        },
        "residual_stream_activations": {
            "layer_out/0": residual_zero,
            "layer_out/1": residual_one,
        },
    }

    tensors, retained = activation_tensors_by_type(
        payload,
        allowed_nodes={"mlp_hidden/1": {8}, "z/2": {4}},
        residual_stream_layers={1},
    )

    assert tensors["mlp"]["mlp_hidden/1"].shape == (1, 3, 1)
    assert tensors["attn"]["z/2"].shape == (1, 3, 1, 3)
    assert tensors["residual"] == {"layer_out/1": residual_one}
    assert tensors["residual"]["layer_out/1"].shape == (1, 3, 8)
    assert retained == {"mlp_hidden/1": [8], "z/2": [4]}


def test_residual_tensors_use_same_aggregation_policy() -> None:
    residual = torch.tensor([[[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]])
    tensors = {"mlp": {}, "attn": {}, "residual": {"layer_out/0": residual}}

    all_result = aggregate_tensors_by_type(tensors, [[0, 1, 2]], "all")
    step_result = aggregate_tensors_by_type(tensors, [[0], [1, 2]], "steps")

    assert torch.equal(all_result["residual"]["layer_out/0"], torch.tensor([3.0, 4.0]))
    assert torch.equal(
        step_result["residual"]["layer_out/0"],
        torch.tensor([[1.0, 2.0], [4.0, 5.0]]),
    )


def test_aggregate_positions_rejects_empty_groups() -> None:
    with pytest.raises(ValueError, match="must be non-empty"):
        aggregate_positions(torch.ones(1, 2, 3), [[]])
