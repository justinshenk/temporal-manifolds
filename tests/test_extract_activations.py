"""Focused tests for activation extraction helpers."""

from __future__ import annotations

from temporal_manifolds.activations.extract_activations import (
    all_token_positions,
    extract_residual_stream_activations,
    find_activation_span,
    get_cache_layer_components,
    set_pad_token_if_missing,
    tokenize_raw_texts,
)


class FakeTensor:
    def __init__(self, shape: tuple[int, ...]) -> None:
        self.shape = shape

    def detach(self) -> "FakeTensor":
        return self

    def cpu(self) -> "FakeTensor":
        return self


class FakeUnderlyingTokenizer:
    def __init__(self) -> None:
        self.pad_token_id = None
        self.pad_token = None
        self.eos_token = "<eos>"
        self.calls: list[dict[str, object]] = []

    def __call__(self, texts: list[str], **kwargs: object) -> dict[str, FakeTensor]:
        self.calls.append({"texts": texts, **kwargs})
        return {
            "input_ids": FakeTensor((1, 4)),
            "attention_mask": FakeTensor((1, 4)),
        }


class FakeChatTokenizer:
    def __init__(self) -> None:
        self.tokenizer = FakeUnderlyingTokenizer()

    def __call__(self, prompts: list[str]) -> dict[str, FakeTensor]:
        raise AssertionError("chat template wrapper should not tokenize decoded full_text")


def test_tokenize_raw_texts_uses_underlying_tokenizer() -> None:
    tokenizer = FakeChatTokenizer()

    batch = tokenize_raw_texts(tokenizer, ["assistant\nStrategy: test"])

    assert batch["input_ids"].shape == (1, 4)
    assert tokenizer.tokenizer.calls == [
        {
            "texts": ["assistant\nStrategy: test"],
            "add_special_tokens": True,
            "return_tensors": "pt",
            "padding": True,
        }
    ]


def test_all_token_positions_uses_raw_text_tokenization() -> None:
    tokenizer = FakeChatTokenizer()

    assert all_token_positions(tokenizer, "decoded full text") == [0, 1, 2, 3]


def test_set_pad_token_if_missing_updates_underlying_tokenizer() -> None:
    tokenizer = FakeChatTokenizer()

    set_pad_token_if_missing(tokenizer)

    assert tokenizer.tokenizer.pad_token == "<eos>"


def test_find_activation_span_supports_summary_checklist_format() -> None:
    full_text = "user\nTask text\nassistant\nSummary: short plan\nChecklist:\n- first\n"

    span = find_activation_span(full_text, position_selection_policy="default")

    assert span == (
        len("user\nTask text\n"),
        len("user\nTask text\nassistant\nSummary: short plan"),
        "assistant\nSummary: short plan",
        "assistant_to_summary_generation",
    )


def test_find_activation_span_supports_approach_actions_format() -> None:
    full_text = "user\nTask text\nassistant\nApproach: short plan\nActions:\n1. first\n"

    span = find_activation_span(full_text, position_selection_policy="default")

    assert span == (
        len("user\nTask text\n"),
        len("user\nTask text\nassistant\nApproach: short plan"),
        "assistant\nApproach: short plan",
        "assistant_to_approach_generation",
    )


def test_find_activation_span_selects_assistant_marker_and_everything_after_it() -> None:
    full_text = "user\nTask text\nassistant\nFirst line\nSecond line\n"

    span = find_activation_span(full_text, position_selection_policy="after_assistant")

    assert span == (
        len("user\nTask text\n"),
        len(full_text),
        "assistant\nFirst line\nSecond line\n",
        "assistant_response",
    )


def test_after_assistant_policy_uses_final_assistant_marker() -> None:
    full_text = "assistant\nold response\nuser\nfollow-up\nassistant\nnew response"

    span = find_activation_span(full_text, position_selection_policy="after_assistant")

    assert span is not None
    assert span[2] == "assistant\nnew response"


def test_after_assistant_policy_requires_non_empty_response() -> None:
    assert (
        find_activation_span("user\ntask\nassistant\n", position_selection_policy="after_assistant")
        is None
    )


def test_cache_layer_components_include_residual_after_every_layer() -> None:
    selected_nodes = {
        "group": [
            ((1, "z"), 2),
            ((0, "layer_out"), 3),
        ]
    }

    assert get_cache_layer_components(selected_nodes, num_hidden_layers=3) == [
        (0, "layer_out"),
        (1, "layer_out"),
        (1, "z"),
        (2, "layer_out"),
    ]


def test_extract_residual_stream_activations_keeps_full_layer_outputs() -> None:
    layer_zero = FakeTensor((1, 5, 8))
    layer_one = FakeTensor((1, 5, 8))
    activations = {
        (0, "layer_out"): layer_zero,
        (1, "layer_out"): layer_one,
    }

    assert extract_residual_stream_activations(activations, num_hidden_layers=2) == {
        "layer_out/0": layer_zero,
        "layer_out/1": layer_one,
    }
