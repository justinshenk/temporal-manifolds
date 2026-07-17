"""Equivalent-phrasing groups for time horizons.

Each group is a list of strings that all express the same horizon in months.
Phrasing-group membership defines the train/test split: held-out groups never
appear in training, so the held-out rows let us measure whether unseen
phrasings collapse onto the same activation point as their training-set
equivalents.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class PhrasingGroup:
    horizon_months: float
    phrasings: tuple[str, ...]


DEFAULT_PHRASING_GROUPS: tuple[PhrasingGroup, ...] = (
    PhrasingGroup(0.25, ("1 week", "7 days", "a week")),
    PhrasingGroup(0.5, ("2 weeks", "14 days", "a fortnight", "half a month")),
    PhrasingGroup(1.0, ("1 month", "4 weeks", "28 days", "one month")),
    PhrasingGroup(2.0, ("2 months", "8 weeks", "60 days", "two months")),
    PhrasingGroup(3.0, ("3 months", "a quarter", "one quarter", "90 days")),
    PhrasingGroup(6.0, ("6 months", "half a year", "two quarters", "180 days")),
    PhrasingGroup(12.0, ("1 year", "12 months", "one year", "365 days")),
    PhrasingGroup(24.0, ("2 years", "24 months", "two years")),
    PhrasingGroup(60.0, ("5 years", "60 months", "half a decade")),
    PhrasingGroup(120.0, ("10 years", "a decade", "120 months")),
)
