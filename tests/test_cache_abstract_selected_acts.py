from __future__ import annotations

import importlib.util
from pathlib import Path

import torch


SCRIPT_PATH = Path.cwd() / "scripts" / "cache_abstract_selected_acts.py"
RUNNER_PATH = Path.cwd() / "scripts" / "run_activation_caching_abstract_selected_acts.sh"
SPEC = importlib.util.spec_from_file_location("cache_abstract_selected_acts", SCRIPT_PATH)
assert SPEC is not None and SPEC.loader is not None
SCRIPT = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(SCRIPT)


def test_build_payload_identifies_the_abstract_dataset() -> None:
    activation = torch.arange(12).reshape(2, 1, 6)
    records = [
        {"text": "First prompt", "template_id": "first", "base_unit": "days"},
        {"text": "Second prompt", "template_id": "second", "base_unit": "weeks"},
    ]

    payload = SCRIPT.build_payload(
        activation=activation,
        records=records,
        sample_indices=[128, 129],
        batch_index=1,
        model_name="test-model",
    )

    assert payload["dataset"] == "abstract"
    assert payload["layer_component"] == "layer_out/21"
    assert payload["positions"] == [-1]
    assert torch.equal(payload["activations"]["layer_out/21"], activation)
    assert payload["sample_indices"] == [128, 129]
    assert payload["prompts"] == ["First prompt", "Second prompt"]
    assert payload["prompt_metadata"] == [
        {"template_id": "first", "base_unit": "days"},
        {"template_id": "second", "base_unit": "weeks"},
    ]


def test_cli_defaults_use_abstract_specific_paths() -> None:
    args = SCRIPT.build_parser().parse_args([])

    assert args.save_to_gcp is True
    assert SCRIPT.GCS_PREFIX == "abstract_selected_acts"
    assert SCRIPT.DEFAULT_OUTPUT_DIR == Path("results/abstract_selected_acts")
    assert SCRIPT.LAYER == 21
    assert SCRIPT.COMPONENT == "layer_out"
    assert SCRIPT.POSITION == -1
    assert args.batch_size == 128


def test_abstract_runner_targets_the_abstract_cache_script() -> None:
    runner = RUNNER_PATH.read_text(encoding="utf-8")

    assert "scripts/cache_abstract_selected_acts.py" in runner
    assert "conversational" not in runner
