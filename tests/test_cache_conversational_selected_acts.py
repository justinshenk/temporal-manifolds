from __future__ import annotations

import importlib.util
from pathlib import Path

import torch


SCRIPT_PATH = Path.cwd() / "scripts" / "cache_conversational_selected_acts.py"
SPEC = importlib.util.spec_from_file_location("cache_conversational_selected_acts", SCRIPT_PATH)
assert SPEC is not None and SPEC.loader is not None
SCRIPT = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(SCRIPT)


def test_build_payload_contains_only_requested_activation() -> None:
    activation = torch.arange(12).reshape(2, 1, 6)
    records = [
        {"text": "First prompt", "template_id": "first", "unit": "days"},
        {"text": "Second prompt", "template_id": "second", "unit": "weeks"},
    ]

    payload = SCRIPT.build_payload(
        activation=activation,
        records=records,
        sample_indices=[128, 129],
        batch_index=1,
        model_name="test-model",
    )

    assert payload["layer_component"] == "layer_out/21"
    assert payload["positions"] == [-1]
    assert set(payload["activations"]) == {"layer_out/21"}
    assert torch.equal(payload["activations"]["layer_out/21"], activation)
    assert payload["batch_index"] == 1
    assert payload["sample_indices"] == [128, 129]
    assert payload["prompts"] == ["First prompt", "Second prompt"]
    assert payload["prompt_metadata"] == [
        {"template_id": "first", "unit": "days"},
        {"template_id": "second", "unit": "weeks"},
    ]


def test_cli_defaults_to_gcp_selected_acts_contract() -> None:
    args = SCRIPT.build_parser().parse_args([])

    assert args.save_to_gcp is True
    assert SCRIPT.GCS_PREFIX == "selected_acts"
    assert SCRIPT.LAYER == 21
    assert SCRIPT.COMPONENT == "layer_out"
    assert SCRIPT.POSITION == -1
    assert args.batch_size == 128


def test_iter_indexed_batches_preserves_sample_indices() -> None:
    records = [{"text": str(index)} for index in range(5)]

    batches = list(SCRIPT.iter_indexed_batches(records, batch_size=2))

    assert batches == [
        ([0, 1], records[0:2]),
        ([2, 3], records[2:4]),
        ([4], records[4:5]),
    ]
