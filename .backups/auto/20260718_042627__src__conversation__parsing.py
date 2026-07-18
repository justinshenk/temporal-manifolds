"""Parse assistant responses of the planning protocol.

The response format contract (datasets/generator.py) makes these trivial:

    Step: 3
    Time horizon: 2 months

    <details>

and the final reply is exactly "Plan Completed".
"""

from __future__ import annotations

import re

from ..core.time_value import TimeValue

STEP_RE = re.compile(r"^\s*Step:\s*(\d+)", re.MULTILINE)
HORIZON_RE = re.compile(r"^\s*Time horizon:\s*(.+?)\s*$", re.MULTILINE)
PLAN_COMPLETED_RE = re.compile(r"plan\s+completed", re.IGNORECASE)
# overview lines like "3. Secure funding (6 months)" or "- Step 2: ... (2 weeks)"
OVERVIEW_STEP_RE = re.compile(r"^\s*(?:\d+[.)]|[-*])\s+", re.MULTILINE)


def strip_think_block(text: str, think_open: str, think_close: str) -> str:
    """Remove a leading (possibly empty) think block from a response."""
    stripped = text.lstrip()
    if not stripped.startswith(think_open):
        return text
    end = stripped.find(think_close)
    if end == -1:
        return stripped[len(think_open) :]
    return stripped[end + len(think_close) :].lstrip("\n")


def parse_step_index(text: str) -> int | None:
    m = STEP_RE.search(text)
    return int(m.group(1)) if m else None


def parse_step_horizon(text: str) -> tuple[str | None, float | None]:
    """Return (raw horizon text, parsed years) of the FIRST 'Time horizon:' line."""
    m = HORIZON_RE.search(text)
    if not m:
        return None, None
    raw = m.group(1)
    years = _parse_duration_years(raw)
    return raw, years


def _parse_duration_years(raw: str) -> float | None:
    """Parse free-ish durations: '2 months', '~3 weeks', '1-2 years', '6 mo',
    and schedule-window styles: 'Months 1–2' (=2 months), 'Year 3' (=1 year)."""
    text = raw.lower().strip().rstrip(".")
    text = text.replace("approximately", "").replace("about", "").replace("~", "")
    text = text.replace("–", "-").replace("—", "-")
    # unit-first schedule window: "months 1-2" -> window length 2 months
    win = re.match(r"^([a-z]+)s?\s+(\d+(?:\.\d+)?)\s*(?:-|to)\s*(\d+(?:\.\d+)?)$", text)
    if win:
        unit, lo, hi = win.group(1), float(win.group(2)), float(win.group(3))
        try:
            return TimeValue(max(hi - lo + 1, 1.0), unit + "s").to_years()
        except ValueError:
            pass
    # unit-first single slot: "month 3" -> 1 month
    slot = re.match(r"^([a-z]+?)s?\s+(\d+(?:\.\d+)?)$", text)
    if slot:
        try:
            return TimeValue(1.0, slot.group(1) + "s").to_years()
        except ValueError:
            pass
    # ranges: take the midpoint
    range_m = re.search(
        r"(\d+(?:\.\d+)?)\s*(?:-|–|to)\s*(\d+(?:\.\d+)?)\s*([a-z]+)", text
    )
    if range_m:
        lo, hi, unit = float(range_m.group(1)), float(range_m.group(2)), range_m.group(3)
        try:
            return TimeValue((lo + hi) / 2, unit).to_years()
        except ValueError:
            return None
    m = re.search(r"(\d+(?:\.\d+)?)\s*([a-z]+)", text)
    if not m:
        return None
    try:
        return TimeValue(float(m.group(1)), m.group(2)).to_years()
    except ValueError:
        return None


def is_plan_completed(text: str) -> bool:
    return PLAN_COMPLETED_RE.search(text) is not None


def is_final_completion(text: str) -> bool:
    """True only for a standalone completion reply (not a mention inside a
    longer message — small models sometimes role-play the whole protocol)."""
    stripped = text.strip()
    return PLAN_COMPLETED_RE.search(stripped) is not None and len(stripped) <= 40


def count_overview_steps(text: str) -> int:
    """Count numbered/bulleted step lines in the overview reply."""
    return len(OVERVIEW_STEP_RE.findall(text))
