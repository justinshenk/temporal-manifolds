"""Registry of verified ChatMarkup specs per model family × generation.

`detect_markup(model_id)` picks the spec; `verify_markup(markup, tokenizer)`
proves the spec against the actual tokenizer (single-token round-trips and a
rendered 3-message conversation containing the expected boundary sequence).
Verification failures raise — the capture pipeline must never run with an
unverified token anatomy.
"""

from __future__ import annotations

from .markup import ChatMarkup

# ---------------------------------------------------------------------------
# Registry (facts verified against real tokenizers on 2026-07-17; see
# tests/test_chat_markup.py which re-verifies on every run)
# ---------------------------------------------------------------------------

QWEN3_HYBRID = ChatMarkup(
    family="qwen3",
    turn_end="<|im_end|>",
    turn_start="<|im_start|>",
    assistant_role="assistant",
    think_open="<think>",
    think_close="</think>",
    no_think_mode="template_kwarg",  # enable_thinking=False renders empty block
    notes=(
        "Qwen3 hybrid reasoning models (Qwen3-0.6B..32B). Thinking ON by "
        "default; enable_thinking=False makes the template append an empty "
        "'<think>\\n\\n</think>\\n\\n' inside the assistant turn. "
        "<think>/<\\/think> are single tokens 151667/151668."
    ),
    model_matchers=("qwen3-",),
)

QWEN3_INSTRUCT_2507 = ChatMarkup(
    family="qwen3-instruct-2507",
    turn_end="<|im_end|>",
    turn_start="<|im_start|>",
    assistant_role="assistant",
    think_open="<think>",
    think_close="</think>",
    no_think_mode="none",  # non-thinking refresh: never emits think blocks
    notes=(
        "Qwen3-*-Instruct-2507 non-thinking refresh (paper's target family). "
        "Tokenizer still has the think tokens but the model never emits them "
        "and the template has no enable_thinking switch."
    ),
    model_matchers=("instruct-2507",),
)

CHATML_NO_THINK = ChatMarkup(
    family="chatml",
    turn_end="<|im_end|>",
    turn_start="<|im_start|>",
    assistant_role="assistant",
    no_think_mode="none",
    notes="Plain ChatML instruct models without think tokens (SmolLM2, Qwen2.5).",
    model_matchers=("smollm", "qwen2.5"),
)

LLAMA3 = ChatMarkup(
    family="llama3",
    turn_end="<|eot_id|>",
    turn_start="<|start_header_id|>",
    assistant_role="assistant",
    post_turn_end_sep="",  # llama3 has no newline between <|eot_id|> and header
    post_role_sep="<|end_header_id|>\n\n",
    no_think_mode="none",
    notes=(
        "Llama-3.x instruct. Role header is "
        "'<|start_header_id|>role<|end_header_id|>\\n\\n'. The "
        "<|end_header_id|> token is treated as part of the role span."
    ),
    model_matchers=("llama-3", "llama3"),
)

GEMMA = ChatMarkup(
    family="gemma",
    turn_end="<end_of_turn>",
    turn_start="<start_of_turn>",
    assistant_role="model",
    no_think_mode="none",
    notes="Gemma 2/3 instruct; assistant role word is 'model'.",
    model_matchers=("gemma",),
)

# Order matters: more specific matchers first.
MARKUP_REGISTRY: tuple[ChatMarkup, ...] = (
    QWEN3_INSTRUCT_2507,
    QWEN3_HYBRID,
    CHATML_NO_THINK,
    LLAMA3,
    GEMMA,
)


def detect_markup(model_id: str) -> ChatMarkup:
    """Pick the ChatMarkup for a model id. Raises if the family is unknown."""
    name = model_id.lower()
    for markup in MARKUP_REGISTRY:
        if any(m in name for m in markup.model_matchers):
            return markup
    raise ValueError(
        f"No ChatMarkup registered for model '{model_id}'. Add an entry to "
        "src/chat_markup/registry.py and verify it before running experiments."
    )


# ---------------------------------------------------------------------------
# Verification against the real tokenizer
# ---------------------------------------------------------------------------


def _single_token_id(tokenizer, text: str) -> int:
    ids = tokenizer.encode(text, add_special_tokens=False)
    if len(ids) != 1:
        raise ValueError(
            f"Marker {text!r} does not encode to a single token "
            f"(got {len(ids)} ids: {ids}). ChatMarkup spec is wrong for "
            f"this tokenizer."
        )
    return ids[0]


def verify_markup(markup: ChatMarkup, tokenizer) -> dict[str, int]:
    """Prove the markup spec against a tokenizer.

    Returns {marker_string: token_id} for the single-token markers.
    Raises ValueError on any mismatch.
    """
    ids: dict[str, int] = {}
    ids[markup.turn_end] = _single_token_id(tokenizer, markup.turn_end)
    ids[markup.turn_start] = _single_token_id(tokenizer, markup.turn_start)
    if markup.has_thinking:
        ids[markup.think_open] = _single_token_id(tokenizer, markup.think_open)
        ids[markup.think_close] = _single_token_id(tokenizer, markup.think_close)

    # Render a 3-message conversation and check the boundary structure exists.
    messages = [
        {"role": "user", "content": "U1"},
        {"role": "assistant", "content": "A1"},
        {"role": "user", "content": "Continue."},
    ]
    rendered = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    expected_boundary = (
        f"{markup.turn_end}{markup.post_turn_end_sep}{markup.turn_start}"
    )
    if expected_boundary not in rendered:
        raise ValueError(
            f"Rendered template does not contain expected boundary "
            f"{expected_boundary!r}.\nRendered: {rendered!r}"
        )
    for role in (markup.user_role, markup.assistant_role):
        if role not in rendered:
            raise ValueError(
                f"Role word {role!r} not found in rendered template: {rendered!r}"
            )
    # The generation prompt must end with an assistant header.
    tail = f"{markup.turn_start}{markup.assistant_role}"
    if tail not in rendered.rstrip()[-len(tail) - len(markup.post_role_sep) - 4 :]:
        raise ValueError(
            f"add_generation_prompt did not end with assistant header "
            f"({tail!r}). Rendered tail: {rendered[-80:]!r}"
        )
    return ids
