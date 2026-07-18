"""Boundary finding on a hand-constructed conversation (tokenizer only, no model)."""

from __future__ import annotations

import pytest

from src.capture.boundaries import find_boundaries
from src.chat_markup.registry import detect_markup
from src.conversation.records import ConversationRecord, Turn


def _tok(model_id):
    from transformers import AutoTokenizer

    try:
        return AutoTokenizer.from_pretrained(model_id)
    except Exception as e:
        pytest.skip(f"tokenizer unavailable: {e}")


def _build_conversation_ids(tok, markup, assistant_texts, prefill=""):
    """Mimic the driver's incremental construction."""
    messages = [{"role": "user", "content": "Plan something."}]
    rendered = tok.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    text = rendered + prefill
    for i, a in enumerate(assistant_texts):
        text += a + markup.turn_end
        if i < len(assistant_texts) - 1:
            text += (
                f"{markup.post_turn_end_sep}{markup.turn_start}{markup.user_role}"
                f"{markup.post_role_sep}Continue.{markup.turn_end}"
                f"{markup.post_turn_end_sep}{markup.turn_start}"
                f"{markup.assistant_role}{markup.post_role_sep}{prefill}"
            )
    return tok.encode(text, add_special_tokens=False)


def _record_for(assistant_texts):
    turns = [Turn(role="user", text="Plan something.", turn_index=0)]
    idx = 1
    for i, a in enumerate(assistant_texts):
        turns.append(Turn(role="assistant", text=a, turn_index=idx, step_index=i or None))
        idx += 1
        if i < len(assistant_texts) - 1:
            turns.append(Turn(role="user", text="Continue.", turn_index=idx))
            idx += 1
    return ConversationRecord(
        sample_uid="test",
        prompt_id="p",
        model_id="m",
        target_horizon_years=1.0,
        turns=turns,
    )


def test_boundaries_smollm2_two_assistant_turns():
    model_id = "HuggingFaceTB/SmolLM2-135M-Instruct"
    tok = _tok(model_id)
    markup = detect_markup(model_id)
    assistant_texts = ["Overview: 1. A (1 day)", "Step: 1\nTime horizon: 1 day\nDo A."]
    ids = _build_conversation_ids(tok, markup, assistant_texts)
    record = _record_for(assistant_texts)
    record.token_ids = ids

    bmap = find_boundaries(ids, tok, markup, record)
    kinds = [b.kind for b in bmap.boundaries]
    # SmolLM2 template injects a system turn: 4 turns in record + 1 system.
    assert kinds.count("turn_start") == 6  # sys,u,a,u,a + trailing none; see below
    # 5 opened turns (sys,u,a,u,a) each closed -> 5 turn_end... but last
    # assistant turn end is present too, and the count of ends == starts.
    assert kinds.count("turn_end") == kinds.count("turn_start") - 0
    roles = {(b.kind, b.role) for b in bmap.boundaries}
    assert ("role", "assistant") in roles and ("role", "user") in roles
    # system boundaries are labeled turn_index=-1
    assert any(b.turn_index == -1 for b in bmap.boundaries)
    # assistant record turns mapped
    a_turns = {b.turn_index for b in bmap.boundaries if b.role == "assistant"}
    assert {2, 4} <= a_turns
    # every claimed position decodes to its stored token_str
    for b in bmap.boundaries:
        assert tok.decode([ids[b.abs_pos]], skip_special_tokens=False) == b.token_str


def test_boundaries_qwen3_with_empty_think():
    model_id = "Qwen/Qwen3-0.6B"
    tok = _tok(model_id)
    markup = detect_markup(model_id)
    prefill = markup.empty_think_block
    assistant_texts = ["Overview: 1. A (1 day)", "Step: 1\nTime horizon: 1 day\nGo."]
    # emulate enable_thinking=False turn-0 render by appending prefill manually
    ids = _build_conversation_ids(tok, markup, assistant_texts, prefill=prefill)
    record = _record_for(assistant_texts)
    record.token_ids = ids

    bmap = find_boundaries(ids, tok, markup, record)
    kinds = [b.kind for b in bmap.boundaries]
    assert kinds.count("think_open") == 2  # one empty block per assistant turn
    assert kinds.count("think_close") == 2
    think_roles = {b.role for b in bmap.boundaries if b.kind.startswith("think")}
    assert think_roles == {"assistant"}
