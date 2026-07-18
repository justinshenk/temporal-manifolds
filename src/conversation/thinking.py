"""ThinkingPolicy: control over chain-of-thought during generation.

Modes:
  disabled  (DEFAULT for all experiments) — the model must not think. For
            Qwen3 hybrid this uses enable_thinking=False (the template renders
            an empty '<think>\n\n</think>\n\n' block); for non-reasoning
            families nothing is needed. The empty block's delimiter tokens
            still exist in the sequence and ARE captured.
  capped(n) — thinking allowed but if <think> is not closed within n new
            tokens, generation is cut and '</think>\n\n' is injected, then
            generation continues. For small models that never stop thinking.
  natural   — the model closes its own think block (large models).

Only `disabled` is used for real experiments right now; capped/natural exist
so the architecture is ready and are exercised by smoke tests only.
"""

from __future__ import annotations

from dataclasses import dataclass

from ..chat_markup.markup import ChatMarkup


@dataclass
class ThinkingPolicy:
    mode: str = "disabled"  # disabled | capped | natural
    cap_tokens: int = 512  # used by capped

    def __post_init__(self):
        if self.mode not in ("disabled", "capped", "natural"):
            raise ValueError(f"Unknown thinking mode: {self.mode}")

    def template_enable_thinking(self, markup: ChatMarkup) -> bool | None:
        """Value for apply_chat_template(enable_thinking=...), or None to omit."""
        if markup.no_think_mode != "template_kwarg":
            return None
        return self.mode != "disabled"

    def validate_for(self, markup: ChatMarkup) -> None:
        if self.mode != "disabled" and not markup.has_thinking:
            raise ValueError(
                f"Thinking mode {self.mode!r} requested but family "
                f"{markup.family!r} has no think tokens."
            )
