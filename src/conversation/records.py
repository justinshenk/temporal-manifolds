"""Records of a completed multi-stage planning conversation."""

from __future__ import annotations

from dataclasses import dataclass, field

from ..core.schema import BaseSchema, deterministic_id_from_dataclass


@dataclass
class Turn(BaseSchema):
    """One message in the conversation."""

    role: str  # "user" | "assistant"
    text: str  # message content as sent/received (no template markup)
    turn_index: int  # 0-based index within the conversation
    step_index: int | None = None  # which plan step an assistant turn expands
    #   (None: the overview turn, the final
    #   "Plan Completed" turn, or user turns)
    step_horizon_text: str | None = None  # raw "Time horizon: ..." value if parsed
    step_horizon_years: float | None = None  # parsed to years when parseable
    truncated: bool = False  # hit max_new_tokens without a stop token
    think_forced_closed: bool = False  # capped thinking had to inject </think>


@dataclass
class ConversationRecord(BaseSchema):
    """Everything about one sample: prompt, turns, tokens, provenance."""

    sample_uid: str
    prompt_id: str
    model_id: str
    target_horizon_years: float
    turns: list[Turn] = field(default_factory=list)
    token_ids: list[int] = field(default_factory=list)  # final full sequence
    completed: bool = False  # saw "Plan Completed"
    n_steps_planned: int = 0  # steps promised in overview
    n_steps_expanded: int = 0  # steps actually expanded
    protocol: dict = field(default_factory=dict)  # protocol/gen config provenance
    notes: str = ""

    @property
    def assistant_turns(self) -> list[Turn]:
        return [t for t in self.turns if t.role == "assistant"]


def make_sample_uid(
    prompt_id: str,
    model_id: str,
    target_horizon_years: float,
    protocol_cfg: dict,
    seed: int,
) -> str:
    """Deterministic unique id for one (prompt, model, protocol, seed) sample."""

    @dataclass
    class _UidPayload(BaseSchema):
        prompt_id: str
        model_id: str
        target_horizon_years: float
        protocol_cfg: dict
        seed: int

    return deterministic_id_from_dataclass(
        _UidPayload(prompt_id, model_id, target_horizon_years, protocol_cfg, seed),
        digest_bytes=8,
    )
