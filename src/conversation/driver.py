"""ConversationDriver: run the plan-then-Continue protocol against an Engine.

The conversation is built INCREMENTALLY at the token level. Turn 0 comes from
the tokenizer's own chat template; every later turn appends a manually
constructed continuation block:

    <turn_end already generated>
    {post_turn_end_sep}{turn_start}{user}{post_role_sep}Continue.{turn_end}
    {post_turn_end_sep}{turn_start}{assistant}{post_role_sep}[empty think block]

This (rather than re-rendering the message list each turn) is deliberate:
Qwen3's template strips past think blocks when re-rendering, which would
delete exactly the delimiter tokens we capture. Incremental construction
keeps every turn's boundary anatomy in the final sequence, identically
structured across turns. `verify_turn0_matches_template` proves the manual
construction agrees with the template where they must agree.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from ..chat_markup.markup import ChatMarkup
from ..core.schema import BaseSchema
from ..datasets.schema import PlanningPrompt
from ..engine.base import Engine
from . import parsing
from .records import ConversationRecord, Turn, make_sample_uid
from .thinking import ThinkingPolicy


@dataclass
class ProtocolConfig(BaseSchema):
    continue_message: str = "Continue."
    max_assistant_turns: int = 10  # overview + steps + "Plan Completed"
    max_new_tokens_per_turn: int = 900
    temperature: float = 0.0
    thinking_mode: str = "disabled"  # disabled | capped | natural
    thinking_cap_tokens: int = 512
    seed: int = 0

    def thinking_policy(self) -> ThinkingPolicy:
        return ThinkingPolicy(mode=self.thinking_mode, cap_tokens=self.thinking_cap_tokens)


class ConversationDriver:
    def __init__(self, engine: Engine, markup: ChatMarkup, protocol: ProtocolConfig):
        self.engine = engine
        self.markup = markup
        self.protocol = protocol
        self.policy = protocol.thinking_policy()
        self.policy.validate_for(markup)

        self._turn_end_id = self._single_id(markup.turn_end)
        eos = engine.tokenizer.eos_token_id
        self._stop_ids = tuple({self._turn_end_id} | ({eos} if eos is not None else set()))
        if markup.has_thinking:
            self._think_open_id = self._single_id(markup.think_open)
            self._think_close_id = self._single_id(markup.think_close)

    def _single_id(self, marker: str) -> int:
        ids = self.engine.encode(marker)
        if len(ids) != 1:
            raise ValueError(f"{marker!r} is not a single token: {ids}")
        return ids[0]

    # -- continuation block -------------------------------------------------

    def _assistant_prefill(self) -> str:
        """Text appended after the assistant role header before generation."""
        if (
            self.policy.mode == "disabled"
            and self.markup.has_thinking
            and self.markup.no_think_mode == "template_kwarg"
        ):
            return self.markup.empty_think_block
        return ""

    def continuation_block(self) -> str:
        m = self.markup
        return (
            f"{m.post_turn_end_sep}{m.turn_start}{m.user_role}{m.post_role_sep}"
            f"{self.protocol.continue_message}{m.turn_end}"
            f"{m.post_turn_end_sep}{m.turn_start}{m.assistant_role}{m.post_role_sep}"
            f"{self._assistant_prefill()}"
        )

    def verify_turn0_matches_template(self, first_user_message: str) -> None:
        """Prove that template render of turn 0 ends with the same assistant
        header + prefill our continuation blocks use."""
        rendered = self.engine.render(
            [{"role": "user", "content": first_user_message}],
            add_generation_prompt=True,
            enable_thinking=self.policy.template_enable_thinking(self.markup),
        )
        m = self.markup
        expected_tail = (
            f"{m.turn_end}{m.post_turn_end_sep}{m.turn_start}{m.assistant_role}"
            f"{m.post_role_sep}{self._assistant_prefill()}"
        )
        if not rendered.endswith(expected_tail):
            raise ValueError(
                "Template render disagrees with manual continuation "
                f"construction.\nExpected tail: {expected_tail!r}\n"
                f"Rendered tail:  {rendered[-len(expected_tail) - 40 :]!r}"
            )

    # -- generation with thinking control ------------------------------------

    def _generate_turn(self, input_ids: list[int]) -> tuple[list[int], bool, bool]:
        """Generate one assistant turn. Returns (new_ids, truncated, forced_close)."""
        budget = self.protocol.max_new_tokens_per_turn
        temp = self.protocol.temperature
        forced_close = False

        if self.policy.mode == "capped" and self.markup.has_thinking:
            cap = self.policy.cap_tokens
            first = self.engine.generate_ids(
                input_ids,
                max_new_tokens=cap,
                temperature=temp,
                stop_token_ids=self._stop_ids + (self._think_close_id,),
            )
            opened = self._think_open_id in first
            closed = self._think_close_id in first
            if opened and not closed and first[-1] not in self._stop_ids:
                # Thinking ran past the cap: force-close and continue.
                forced_close = True
                close_ids = self.engine.encode(f"{self.markup.think_close}\n\n")
                prefix = input_ids + first + close_ids
                rest = self.engine.generate_ids(
                    prefix,
                    max_new_tokens=budget,
                    temperature=temp,
                    stop_token_ids=self._stop_ids,
                )
                new_ids = first + close_ids + rest
            elif closed and first[-1] == self._think_close_id:
                # Stopped exactly at </think>: continue into the answer.
                rest = self.engine.generate_ids(
                    input_ids + first,
                    max_new_tokens=budget,
                    temperature=temp,
                    stop_token_ids=self._stop_ids,
                )
                new_ids = first + rest
            else:
                new_ids = first
        else:
            new_ids = self.engine.generate_ids(
                input_ids,
                max_new_tokens=budget,
                temperature=temp,
                stop_token_ids=self._stop_ids,
            )

        truncated = not new_ids or new_ids[-1] not in self._stop_ids
        if truncated:
            # Keep the transcript well-formed for capture: close the turn.
            new_ids = new_ids + [self._turn_end_id]
        elif new_ids[-1] != self._turn_end_id:
            # Stopped on eos that isn't turn_end (e.g. <|endoftext|>): normalize.
            new_ids[-1] = self._turn_end_id
        return new_ids, truncated, forced_close

    # -- main loop -----------------------------------------------------------

    def run(self, prompt: PlanningPrompt) -> ConversationRecord:
        self.verify_turn0_matches_template(prompt.text)

        record = ConversationRecord(
            sample_uid=make_sample_uid(
                prompt.prompt_id,
                self.engine.model_id,
                prompt.target_horizon_years,
                self.protocol.to_dict(),
                self.protocol.seed,
            ),
            prompt_id=prompt.prompt_id,
            model_id=self.engine.model_id,
            target_horizon_years=prompt.target_horizon_years,
            protocol=self.protocol.to_dict(),
        )

        rendered0 = self.engine.render(
            [{"role": "user", "content": prompt.text}],
            add_generation_prompt=True,
            enable_thinking=self.policy.template_enable_thinking(self.markup),
        )
        ids = self.engine.encode(rendered0)
        record.turns.append(Turn(role="user", text=prompt.text, turn_index=0))

        turn_index = 1
        expanded_steps = 0
        for assistant_i in range(self.protocol.max_assistant_turns):
            new_ids, truncated, forced_close = self._generate_turn(ids)
            ids = ids + new_ids

            raw_text = self.engine.decode(new_ids)
            content = raw_text
            if content.endswith(self.markup.turn_end):
                content = content[: -len(self.markup.turn_end)]
            visible = content
            if self.markup.has_thinking:
                visible = parsing.strip_think_block(
                    content, self.markup.think_open, self.markup.think_close
                )
            visible = visible.strip()

            step_idx = parsing.parse_step_index(visible)
            horizon_raw, horizon_years = parsing.parse_step_horizon(visible)
            completed = parsing.is_plan_completed(visible) and step_idx is None

            record.turns.append(
                Turn(
                    role="assistant",
                    text=visible,
                    turn_index=turn_index,
                    step_index=step_idx,
                    step_horizon_text=horizon_raw,
                    step_horizon_years=horizon_years,
                    truncated=truncated,
                    think_forced_closed=forced_close,
                )
            )
            turn_index += 1

            if assistant_i == 0:
                record.n_steps_planned = parsing.count_overview_steps(visible)
            if step_idx is not None:
                expanded_steps += 1

            if completed:
                record.completed = True
                break
            if assistant_i == self.protocol.max_assistant_turns - 1:
                break

            cont = self.continuation_block()
            ids = ids + self.engine.encode(cont)
            record.turns.append(
                Turn(
                    role="user",
                    text=self.protocol.continue_message,
                    turn_index=turn_index,
                )
            )
            turn_index += 1

        record.n_steps_expanded = expanded_steps
        record.token_ids = ids
        return record
