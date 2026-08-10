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
        {
            "text": "First prompt",
            "template_id": "first",
            "template_metadata": {
                "prompt_framing": "task_available_time",
                "subject_framing": "task",
                "time_framing": "available_time",
            },
            "task_metadata": {
                "task_family": "state_stabilization",
                "planning_type": "stabilization",
                "domain": "abstract",
                "temporal_coverage": "full_range",
            },
            "base_unit": "days",
        },
        {
            "text": "Second prompt",
            "template_id": "second",
            "template_metadata": {
                "prompt_framing": "goal_deadline",
                "subject_framing": "goal",
                "time_framing": "deadline",
            },
            "task_metadata": {
                "task_family": "adaptation",
                "planning_type": "adaptive",
                "domain": "abstract",
                "temporal_coverage": "full_range",
            },
            "base_unit": "weeks",
        },
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
    first_metadata, second_metadata = payload["prompt_metadata"]
    assert list(first_metadata) == [
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
    ]
    assert first_metadata["template_metadata"] == {
        "prompt_framing": "task_available_time",
        "output_format": "N/A",
    }
    assert first_metadata["task_metadata"] == {
        "task_family": "state_stabilization",
        "difficulty": "N/A",
        "domain": "abstract",
        "complexity": "N/A",
        "planning_type": "stabilization",
        "stakes": "N/A",
        "agency": "N/A",
    }
    assert first_metadata["task"] == "N/A"
    assert first_metadata["quantity"] == "N/A"
    assert first_metadata["base_unit"] == "days"
    assert second_metadata["template_metadata"] == {
        "prompt_framing": "goal_deadline",
        "output_format": "N/A",
    }
    assert second_metadata["task_metadata"]["task_family"] == "adaptation"
    assert second_metadata["base_unit"] == "weeks"


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


def test_model_hook_call_is_fixed_to_final_prompt_layer_output(
    monkeypatch, tmp_path: Path
) -> None:
    class Tokenizer:
        pad_token_id = 0
        eos_token = "<eos>"
        padding_side = "right"

        def __call__(self, prompts, **_kwargs):
            assert prompts == ["Prompt"]
            return {"input_ids": torch.tensor([[1, 2]])}

    class Model:
        config = type("Config", (), {"num_hidden_layers": 22})()

        def eval(self) -> None:
            return None

    captured = {}
    monkeypatch.setattr(
        SCRIPT,
        "generate_abstract_prompt_records",
        lambda: [{"text": "Prompt", "template_id": "test"}],
    )
    monkeypatch.setattr(
        SCRIPT,
        "load_model_tokenizer_config",
        lambda **_kwargs: (Model(), Tokenizer(), None),
    )

    def get_activations(_model, _tokenized, layer_components, **kwargs):
        captured["layer_components"] = layer_components
        captured.update(kwargs)
        return {(21, "layer_out"): torch.ones(1, 1, 4)}, None

    monkeypatch.setattr(SCRIPT, "get_activations", get_activations)

    SCRIPT.cache_abstract_selected_acts(
        output_dir=tmp_path,
        batch_size=1,
        save_to_gcp=False,
    )

    assert captured == {
        "layer_components": [(21, "layer_out")],
        "positions": -1,
        "return_logits": False,
        "clone_tensors": True,
        "early_exit": True,
    }
