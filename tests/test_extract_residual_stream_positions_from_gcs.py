"""Tests for the standalone residual-stream exploration script."""

from __future__ import annotations

import io
import importlib.util
from pathlib import Path

import pytest
import torch

SCRIPT_PATH = Path(__file__).parents[1] / "scripts" / "extract_residual_stream_positions_from_gcs.py"
SCRIPT_SPEC = importlib.util.spec_from_file_location(
    "extract_residual_stream_positions_from_gcs",
    SCRIPT_PATH,
)
assert SCRIPT_SPEC is not None and SCRIPT_SPEC.loader is not None
SCRIPT_MODULE = importlib.util.module_from_spec(SCRIPT_SPEC)
SCRIPT_SPEC.loader.exec_module(SCRIPT_MODULE)

download_and_extract = SCRIPT_MODULE.download_and_extract
combine_samples = SCRIPT_MODULE.combine_samples
extract_residual_stream_positions = SCRIPT_MODULE.extract_residual_stream_positions
parse_gcs_uri = SCRIPT_MODULE.parse_gcs_uri


def _payload() -> dict[str, object]:
    return {
        "position_selection_policy": "after_assistant",
        "positions": [10, 11, 12, 13],
        "metadata": [{"sample_index": 7}],
        "residual_stream_activations": {
            "layer_out/0": torch.arange(24).reshape(1, 4, 6),
            "layer_out/1": torch.arange(24, 48).reshape(1, 4, 6),
        },
    }


class FakeBlob:
    name = "prefix/activations_sample_00007.pt"

    def download_to_file(self, destination: io.BytesIO) -> None:
        torch.save(_payload(), destination)


def test_parse_gcs_uri() -> None:
    assert parse_gcs_uri("gs://bucket/some/prefix/") == ("bucket", "some/prefix")
    with pytest.raises(ValueError, match="gs://"):
        parse_gcs_uri("bucket/some/prefix")


def test_extracts_first_three_cached_positions_from_every_layer() -> None:
    result = extract_residual_stream_positions(_payload(), source_object="source.pt")

    assert result["sample_index"] == 7
    assert result["cached_position_indices"] == [0, 1, 2]
    assert result["absolute_token_positions"] == [10, 11, 12]
    residuals = result["residual_stream_activations"]
    assert isinstance(residuals, dict)
    assert torch.equal(residuals["layer_out/0"], torch.arange(18).reshape(1, 3, 6))
    assert torch.equal(residuals["layer_out/1"], torch.arange(24, 42).reshape(1, 3, 6))


def test_rejects_payload_with_too_few_positions() -> None:
    payload = _payload()
    payload["positions"] = [10, 11]
    payload["residual_stream_activations"] = {
        "layer_out/0": torch.zeros(1, 2, 6)
    }

    with pytest.raises(ValueError, match="only 2 cached positions"):
        extract_residual_stream_positions(payload, source_object="source.pt")


def test_downloads_and_loads_payload_in_memory() -> None:
    result = download_and_extract(FakeBlob())  # type: ignore[arg-type]

    assert result["source_object"] == FakeBlob.name
    assert result["absolute_token_positions"] == [10, 11, 12]


def test_combines_samples_as_num_samples_by_three_by_model_width() -> None:
    first = extract_residual_stream_positions(_payload(), source_object="sample_0.pt")
    second_payload = _payload()
    second_payload["metadata"] = [{"sample_index": 8}]
    second_payload["positions"] = [20, 21, 22, 23]
    second = extract_residual_stream_positions(
        second_payload,
        source_object="sample_1.pt",
    )

    combined = combine_samples([first, second])

    assert combined["sample_indices"] == [7, 8]
    assert combined["absolute_token_positions"] == [[10, 11, 12], [20, 21, 22]]
    residuals = combined["residual_stream_activations"]
    assert residuals["layer_out/0"].shape == (2, 3, 6)
    assert residuals["layer_out/1"].shape == (2, 3, 6)
    assert torch.equal(residuals["layer_out/0"][0], torch.arange(18).reshape(3, 6))
