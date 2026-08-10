from __future__ import annotations

import importlib.util
from pathlib import Path
from types import SimpleNamespace

import torch


SCRIPT_PATH = Path.cwd() / "scripts" / "cache_conversational_selected_acts.py"
NO_OUTPUT_FORMAT_RUNNER_PATH = (
    Path.cwd()
    / "scripts"
    / "run_activation_caching_nof_conversational_selected_acts.sh"
)
SPEC = importlib.util.spec_from_file_location("cache_conversational_selected_acts", SCRIPT_PATH)
assert SPEC is not None and SPEC.loader is not None
SCRIPT = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(SCRIPT)


def test_build_payload_contains_only_requested_activation() -> None:
    activation = torch.arange(12).reshape(2, 1, 6)
    records = [
        {
            "text": "First prompt",
            "template_id": "first",
            "template_metadata": {
                "prompt_framing": "task_available_time",
                "output_format": "strategy_steps",
            },
            "unit": "days",
        },
        {
            "text": "Second prompt",
            "template_id": "second",
            "template_metadata": {"prompt_framing": "goal_deadline"},
            "unit": "weeks",
        },
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
        {
            "template_id": "first",
            "template_metadata": {
                "prompt_framing": "task_available_time",
                "output_format": "strategy_steps",
            },
            "task": "N/A",
            "task_metadata": {
                "task_family": "N/A",
                "difficulty": "N/A",
                "domain": "N/A",
                "complexity": "N/A",
                "planning_type": "N/A",
                "stakes": "N/A",
                "agency": "N/A",
            },
            "quantity": "N/A",
            "quantity_text": "N/A",
            "base_value": "N/A",
            "base_unit": "N/A",
            "unit_variant": "N/A",
            "number_format": "N/A",
            "value": "N/A",
            "value_text": "N/A",
            "unit": "days",
        },
        {
            "template_id": "second",
            "template_metadata": {
                "prompt_framing": "goal_deadline",
                "output_format": "N/A",
            },
            "task": "N/A",
            "task_metadata": {
                "task_family": "N/A",
                "difficulty": "N/A",
                "domain": "N/A",
                "complexity": "N/A",
                "planning_type": "N/A",
                "stakes": "N/A",
                "agency": "N/A",
            },
            "quantity": "N/A",
            "quantity_text": "N/A",
            "base_value": "N/A",
            "base_unit": "N/A",
            "unit_variant": "N/A",
            "number_format": "N/A",
            "value": "N/A",
            "value_text": "N/A",
            "unit": "weeks",
        },
    ]


def test_cli_defaults_to_gcp_selected_acts_contract() -> None:
    args = SCRIPT.build_parser().parse_args([])

    assert args.save_to_gcp is True
    assert SCRIPT.GCS_PREFIX == "selected_acts"
    assert SCRIPT.LAYER == 21
    assert SCRIPT.COMPONENT == "layer_out"
    assert SCRIPT.POSITION == -1
    assert args.batch_size == 128


def test_no_output_format_runner_uses_an_isolated_artifact_namespace() -> None:
    runner = NO_OUTPUT_FORMAT_RUNNER_PATH.read_text(encoding="utf-8")

    assert "scripts/cache_conversational_selected_acts.py" in runner
    assert "--remove-output-format-constraints" in runner
    assert "--gcs-prefix NOF_selected_acts" in runner
    assert "results/selected_acts_no_output_format" in runner


def test_no_output_format_option_is_forwarded_to_dataset_generation(
    monkeypatch, tmp_path: Path
) -> None:
    generated_with = None

    def generate_task_dataset(**kwargs):
        nonlocal generated_with
        generated_with = kwargs
        return []

    model = SimpleNamespace(
        config=SimpleNamespace(num_hidden_layers=22),
        eval=lambda: None,
    )
    monkeypatch.setattr(SCRIPT, "generate_task_dataset", generate_task_dataset)
    monkeypatch.setattr(
        SCRIPT,
        "load_model_tokenizer_config",
        lambda **_kwargs: (model, object(), None),
    )

    SCRIPT.cache_conversational_selected_acts(
        output_dir=tmp_path,
        remove_output_format_constraints=True,
        save_to_gcp=False,
    )

    assert generated_with is not None
    assert generated_with["dataset"] == "conversational"
    assert generated_with["remove_output_format_constraints"] is True


def test_iter_indexed_batches_preserves_sample_indices() -> None:
    records = [{"text": str(index)} for index in range(5)]

    batches = list(SCRIPT.iter_indexed_batches(records, batch_size=2))

    assert batches == [
        ([0, 1], records[0:2]),
        ([2, 3], records[2:4]),
        ([4], records[4:5]),
    ]


def test_chat_tokenizer_wrapper_is_called_without_hugging_face_kwargs() -> None:
    class UnderlyingTokenizer:
        pad_token_id = None
        pad_token = None
        eos_token = "<eos>"
        padding_side = "right"

    class ChatTokenizer:
        def __init__(self) -> None:
            self.tokenizer = UnderlyingTokenizer()
            self.received_prompts = None

        def __call__(self, prompts):
            self.received_prompts = prompts
            return {"input_ids": torch.tensor([[1], [2]])}

    tokenizer = ChatTokenizer()
    prompts = ["short", "a longer prompt"]

    result = SCRIPT.configure_and_tokenize_left_padded(tokenizer, prompts)

    assert tokenizer.tokenizer.padding_side == "left"
    assert tokenizer.tokenizer.pad_token == "<eos>"
    assert tokenizer.received_prompts == prompts
    assert torch.equal(result["input_ids"], torch.tensor([[1], [2]]))


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

    model = SimpleNamespace(
        config=SimpleNamespace(num_hidden_layers=22),
        eval=lambda: None,
    )
    captured = {}

    monkeypatch.setattr(
        SCRIPT,
        "generate_task_dataset",
        lambda **_kwargs: [{"text": "Prompt", "template_id": "test"}],
    )
    monkeypatch.setattr(
        SCRIPT,
        "load_model_tokenizer_config",
        lambda **_kwargs: (model, Tokenizer(), None),
    )

    def get_activations(_model, _tokenized, layer_components, **kwargs):
        captured["layer_components"] = layer_components
        captured.update(kwargs)
        return {(21, "layer_out"): torch.ones(1, 1, 4)}, None

    monkeypatch.setattr(SCRIPT, "get_activations", get_activations)

    SCRIPT.cache_conversational_selected_acts(
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
