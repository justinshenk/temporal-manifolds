from __future__ import annotations

import importlib.util
from pathlib import Path
from types import SimpleNamespace

import torch


SCRIPT_PATH = Path.cwd() / "scripts" / "cache_conversational_selected_acts.py"
PLAIN_ENGLISH_RUNNER_PATH = (
    Path.cwd()
    / "scripts"
    / "run_activation_caching_plain_english_selected_acts.sh"
)
TASK_ONLY_RUNNER_PATH = (
    Path.cwd()
    / "scripts"
    / "run_activation_caching_task_only_selected_acts.sh"
)
INDIRECT_HORIZON_RUNNER_PATH = (
    Path.cwd()
    / "scripts"
    / "run_activation_caching_indirect_horizon_selected_acts.sh"
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


def test_plain_english_runner_uses_the_new_dataset_and_an_isolated_namespace() -> None:
    runner = PLAIN_ENGLISH_RUNNER_PATH.read_text(encoding="utf-8")

    assert "scripts/cache_conversational_selected_acts.py" in runner
    assert "--dataset plain_english" in runner
    assert "--gcs-prefix plain_english_selected_acts" in runner
    assert "results/plain_english_selected_acts" in runner


def test_task_only_runner_mirrors_the_conversational_workflow() -> None:
    runner = TASK_ONLY_RUNNER_PATH.read_text(encoding="utf-8")
    conversational_runner = (
        Path.cwd() / "scripts" / "run_activation_caching_conversational_selected_acts.sh"
    ).read_text(encoding="utf-8")

    # The same fixed-contract extractor and scenario driver as every sibling run.
    assert "scripts/cache_conversational_selected_acts.py" in runner
    assert "scripts/run_activation_caching_scenario.sh" in runner
    assert "scripts/cache_conversational_selected_acts.py" in conversational_runner
    # Differs only by dataset and its isolated artifact namespace.
    assert "--dataset task_only" in runner
    assert "--gcs-prefix task_only_selected_acts" in runner
    assert "results/task_only_selected_acts" in runner


def test_task_only_runner_does_not_override_credentials_or_extraction() -> None:
    runner = TASK_ONLY_RUNNER_PATH.read_text(encoding="utf-8")

    # Credentials stay in .env / ADC, and the layer, component, and position
    # remain constants rather than runner-supplied options.
    assert "GOOGLE_APPLICATION_CREDENTIALS" not in runner
    assert "--gcp-project-id" not in runner
    assert "--gcs-bucket-name" not in runner
    assert "--remove-time-constraints" not in runner


def test_task_only_is_a_supported_caching_dataset() -> None:
    assert "task_only" in SCRIPT.SUPPORTED_DATASETS

    args = SCRIPT.build_parser().parse_args(["--dataset", "task_only"])

    assert args.dataset == "task_only"


def test_indirect_horizon_runner_mirrors_the_conversational_workflow() -> None:
    runner = INDIRECT_HORIZON_RUNNER_PATH.read_text(encoding="utf-8")

    # The same fixed-contract extractor and scenario driver as every sibling run.
    assert "scripts/cache_conversational_selected_acts.py" in runner
    assert "scripts/run_activation_caching_scenario.sh" in runner
    # Differs only by dataset and its isolated artifact namespace.
    assert "--dataset indirect_horizon" in runner
    assert "--gcs-prefix indirect_horizon_selected_acts" in runner
    assert "results/indirect_horizon_selected_acts" in runner


def test_indirect_horizon_runner_does_not_override_credentials_or_extraction() -> None:
    runner = INDIRECT_HORIZON_RUNNER_PATH.read_text(encoding="utf-8")

    assert "GOOGLE_APPLICATION_CREDENTIALS" not in runner
    assert "--gcp-project-id" not in runner
    assert "--gcs-bucket-name" not in runner
    # The dataset has no time-free form; task_only is the horizon-free control.
    assert "--remove-time-constraints" not in runner


def test_indirect_horizon_is_a_supported_caching_dataset() -> None:
    assert "indirect_horizon" in SCRIPT.SUPPORTED_DATASETS

    args = SCRIPT.build_parser().parse_args(["--dataset", "indirect_horizon"])

    assert args.dataset == "indirect_horizon"


def test_event_anchored_is_a_supported_caching_dataset() -> None:
    assert "event_anchored" in SCRIPT.SUPPORTED_DATASETS

    args = SCRIPT.build_parser().parse_args(["--dataset", "event_anchored"])

    assert args.dataset == "event_anchored"


def test_task_only_dataset_is_forwarded_to_generation_and_payload(
    monkeypatch, tmp_path: Path
) -> None:
    generated_with = None
    payload_dataset = None

    def generate_task_dataset(**kwargs):
        nonlocal generated_with
        generated_with = kwargs
        return [{"text": "Prompt", "template_id": "test"}]

    class Tokenizer:
        pad_token_id = 0
        eos_token = "<eos>"
        padding_side = "right"

        def __call__(self, _prompts, **_kwargs):
            return {"input_ids": torch.tensor([[1, 2]])}

    model = SimpleNamespace(
        config=SimpleNamespace(num_hidden_layers=22),
        eval=lambda: None,
    )
    monkeypatch.setattr(SCRIPT, "generate_task_dataset", generate_task_dataset)
    monkeypatch.setattr(
        SCRIPT,
        "load_model_tokenizer_config",
        lambda **_kwargs: (model, Tokenizer(), None),
    )
    monkeypatch.setattr(
        SCRIPT,
        "get_activations",
        lambda *_args, **_kwargs: ({(21, "layer_out"): torch.ones(1, 1, 4)}, None),
    )

    original_build_payload = SCRIPT.build_payload

    def build_payload(**kwargs):
        nonlocal payload_dataset
        payload_dataset = kwargs["dataset"]
        return original_build_payload(**kwargs)

    monkeypatch.setattr(SCRIPT, "build_payload", build_payload)

    SCRIPT.cache_conversational_selected_acts(
        dataset="task_only",
        output_dir=tmp_path,
        batch_size=1,
        save_to_gcp=False,
    )

    assert generated_with is not None
    assert generated_with["dataset"] == "task_only"
    assert payload_dataset == "task_only"


def test_plain_english_dataset_is_forwarded_to_generation_and_payload(
    monkeypatch, tmp_path: Path
) -> None:
    generated_with = None
    payload_dataset = None

    def generate_task_dataset(**kwargs):
        nonlocal generated_with
        generated_with = kwargs
        return [{"text": "Prompt", "template_id": "test"}]

    class Tokenizer:
        pad_token_id = 0
        eos_token = "<eos>"
        padding_side = "right"

        def __call__(self, _prompts, **_kwargs):
            return {"input_ids": torch.tensor([[1, 2]])}

    model = SimpleNamespace(
        config=SimpleNamespace(num_hidden_layers=22),
        eval=lambda: None,
    )
    monkeypatch.setattr(SCRIPT, "generate_task_dataset", generate_task_dataset)
    monkeypatch.setattr(
        SCRIPT,
        "load_model_tokenizer_config",
        lambda **_kwargs: (model, Tokenizer(), None),
    )
    monkeypatch.setattr(
        SCRIPT,
        "get_activations",
        lambda *_args, **_kwargs: ({(21, "layer_out"): torch.ones(1, 1, 4)}, None),
    )

    original_build_payload = SCRIPT.build_payload

    def build_payload(**kwargs):
        nonlocal payload_dataset
        payload_dataset = kwargs["dataset"]
        return original_build_payload(**kwargs)

    monkeypatch.setattr(SCRIPT, "build_payload", build_payload)

    SCRIPT.cache_conversational_selected_acts(
        dataset="plain_english",
        output_dir=tmp_path,
        batch_size=1,
        save_to_gcp=False,
    )

    assert generated_with is not None
    assert generated_with["dataset"] == "plain_english"
    assert generated_with["remove_time_constraints"] is False
    assert payload_dataset == "plain_english"


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
