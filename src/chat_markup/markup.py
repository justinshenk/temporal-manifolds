"""ChatMarkup: the chat-template token anatomy of one model family × generation.

Change-of-turn tokens and thinking delimiters differ per family AND per
generation (e.g. Qwen3 hybrid has <think>/</think> single tokens; the
Qwen3-*-Instruct-2507 refresh shares the tokens but never thinks; Llama-3 and
Gemma have no think tokens at all). Everything the capture pipeline knows
about a template is declared here and VERIFIED against the real tokenizer at
engine init — a mismatch is a hard error, never a silent skip.

Boundary kinds captured at each turn transition (paper: the user->assistant
turn boundary is where horizon geometry collapses, arXiv:2606.05194 §5.2):

    turn_end          <|im_end|> / <|eot_id|> / <end_of_turn>
    post_turn_end_nl  the newline token right after turn_end (ChatML/Gemma)
    turn_start        <|im_start|> / <|start_header_id|> / <start_of_turn>
    role              the role word token(s): 'user' / 'assistant' / 'model'
    post_role_nl      newline(s) after the role header
    think_open        <think>   (families that have it)
    think_close       </think>  (families that have it)
"""

from __future__ import annotations

from dataclasses import dataclass, field

from ..core.schema import BaseSchema

BOUNDARY_KINDS = (
    "turn_end",
    "post_turn_end_nl",
    "turn_start",
    "role",
    "post_role_nl",
    "think_open",
    "think_close",
)


@dataclass
class ChatMarkup(BaseSchema):
    """Declarative spec of a chat template's boundary tokens.

    Token strings must each encode to EXACTLY ONE token id with the family's
    tokenizer (verified in registry.verify_markup), except `role_strs` values
    and newline runs which may span multiple tokens and are matched greedily.
    """

    family: str  # e.g. "qwen3", "chatml", "llama3", "gemma"
    turn_end: str  # e.g. "<|im_end|>"
    turn_start: str  # e.g. "<|im_start|>"
    assistant_role: str  # role word rendered in template, e.g. "assistant"/"model"
    user_role: str = "user"
    # Separators as rendered by the template (may be "" for none).
    post_turn_end_sep: str = "\n"  # between turn_end and next turn_start
    # Everything between the role word and the message content. Plain "\n" for
    # ChatML/Gemma; "<|end_header_id|>\n\n" for Llama-3 (the header-close token
    # is part of the role span for capture purposes).
    post_role_sep: str = "\n"
    # Thinking delimiters; None when family has no thinking tokens.
    think_open: str | None = None
    think_close: str | None = None
    # How to disable thinking:
    #   "none"           - model never thinks, nothing to do
    #   "template_kwarg" - pass enable_thinking=False to apply_chat_template
    #   "empty_block"    - prefill an empty think block "<think>\n\n</think>\n\n"
    no_think_mode: str = "none"
    # Extra notes for humans reading configs/results.
    notes: str = ""
    # Model-name substrings that select this markup (lowercased match).
    model_matchers: tuple[str, ...] = field(default_factory=tuple)

    @property
    def has_thinking(self) -> bool:
        return self.think_open is not None and self.think_close is not None

    @property
    def empty_think_block(self) -> str:
        """The literal string that represents 'no thinking happened'.

        Qwen3 renders exactly this when enable_thinking=False:
        '<think>\\n\\n</think>\\n\\n'. Verified per model in registry tests.
        """
        if not self.has_thinking:
            return ""
        return f"{self.think_open}\n\n{self.think_close}\n\n"
