"""Ecologically valid phrasings of the target time horizon.

Instead of the stilted "The target time horizon is {horizon}.", these read the
way a real requester would state a deadline/scope. Each phrasing keeps the
horizon value in a single contiguous substring so token positions of the
horizon are recoverable.
"""

from __future__ import annotations

from .schema import HorizonPhrasing

DEFAULT_PHRASINGS: tuple[HorizonPhrasing, ...] = (
    HorizonPhrasing(
        phrasing_id="available_time",
        template="You have {horizon} to accomplish this.",
    ),
    HorizonPhrasing(
        phrasing_id="deadline",
        template="Everything must be finished within {horizon}.",
    ),
    HorizonPhrasing(
        phrasing_id="commitment_window",
        template="We are committing to seeing results in {horizon}.",
    ),
    HorizonPhrasing(
        phrasing_id="scope",
        template="Plan on a {horizon} timescale.",
    ),
)


def get_phrasing(phrasing_id: str) -> HorizonPhrasing:
    for p in DEFAULT_PHRASINGS:
        if p.phrasing_id == phrasing_id:
            return p
    raise KeyError(f"Unknown phrasing_id: {phrasing_id}")
