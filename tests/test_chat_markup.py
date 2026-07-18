"""Verify ChatMarkup registry entries against REAL tokenizers (from cache).

These tests re-prove the token-anatomy facts the capture pipeline depends on.
Models not present in the local HF cache are skipped, never faked.
"""

from __future__ import annotations

import pytest

from src.chat_markup.registry import detect_markup, verify_markup

MODELS = [
    "HuggingFaceTB/SmolLM2-135M-Instruct",
    "Qwen/Qwen3-0.6B",
    "Qwen/Qwen3-4B-Instruct-2507",
    "meta-llama/Llama-3.1-8B-Instruct",
    "google/gemma-3-1b-it",
]


def _load_tokenizer(model_id: str):
    from transformers import AutoTokenizer

    try:
        return AutoTokenizer.from_pretrained(model_id)
    except Exception as e:  # gated / offline
        pytest.skip(f"tokenizer unavailable for {model_id}: {e}")


@pytest.mark.parametrize("model_id", MODELS)
def test_registry_matches_real_tokenizer(model_id):
    markup = detect_markup(model_id)
    tok = _load_tokenizer(model_id)
    ids = verify_markup(markup, tok)
    assert markup.turn_end in ids and markup.turn_start in ids


def test_qwen3_empty_think_block_rendering():
    """enable_thinking=False must render exactly the empty think block we
    append manually on later turns."""
    markup = detect_markup("Qwen/Qwen3-0.6B")
    tok = _load_tokenizer("Qwen/Qwen3-0.6B")
    rendered = tok.apply_chat_template(
        [{"role": "user", "content": "hi"}],
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    assert rendered.endswith(
        f"{markup.turn_start}{markup.assistant_role}{markup.post_role_sep}"
        f"{markup.empty_think_block}"
    ), f"unexpected tail: {rendered[-80:]!r}"


def test_detect_specificity():
    assert detect_markup("Qwen/Qwen3-4B-Instruct-2507").family == "qwen3-instruct-2507"
    assert detect_markup("Qwen/Qwen3-14B").family == "qwen3"
    assert detect_markup("HuggingFaceTB/SmolLM2-135M-Instruct").family == "chatml"
    assert detect_markup("meta-llama/Llama-3.1-8B-Instruct").family == "llama3"
    assert detect_markup("google/gemma-3-27b-it").family == "gemma"
    with pytest.raises(ValueError):
        detect_markup("mistralai/Mistral-7B-Instruct-v0.3")
