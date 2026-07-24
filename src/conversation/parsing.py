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
HORIZON_RE = re.compile(r"^\s*Time (horizon|target):\s*(.+?)\s*$", re.MULTILINE)
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
    """Return (raw text, parsed years) of the first 'Time horizon/target:' line.

    The header word carries the semantics: 'Time target:' values are future
    offsets, where slot-forms like 'Year 2' mean 2 years out; 'Time horizon:'
    values are durations, where 'Year 2' is a schedule slot (1 year)."""
    m = HORIZON_RE.search(text)
    if not m:
        return None, None
    raw = m.group(2)
    years = _parse_duration_years(raw, target_mode=m.group(1) == "target")
    return raw, years


def _parse_duration_years(raw: str, target_mode: bool = False) -> float | None:
    """Parse free-ish durations: '2 months', '~3 weeks', '1-2 years', '6 mo',
    compounds '1 year, 6 months', and schedule-window styles: 'Months 1–2'
    (=2 months), 'Year 3' (=1 year as a slot; =3 years as a future target)."""
    text = raw.lower().strip().rstrip(".")
    text = text.replace("approximately", "").replace("about", "").replace("~", "")
    text = text.replace("–", "-").replace("—", "-")
    # compound: "1 year, 6 months" / "1 year and 6 months"
    comp = re.match(
        r"^(\d+(?:\.\d+)?)\s*([a-z]+)(?:\s*,\s*|\s+and\s+)(\d+(?:\.\d+)?)\s*([a-z]+)$",
        text,
    )
    if comp:
        try:
            return (TimeValue(float(comp.group(1)), comp.group(2)).to_years()
                    + TimeValue(float(comp.group(3)), comp.group(4)).to_years())
        except ValueError:
            pass
    # unit-first schedule window: "months 1-2" or "day 2-day 5" -> window length
    win = re.match(
        r"^([a-z]+)\s+(\d+(?:\.\d+)?)\s*(?:-|to)\s*(?:[a-z]+\s+)?(\d+(?:\.\d+)?)$",
        text,
    )
    if win:
        unit, lo, hi = win.group(1), float(win.group(2)), float(win.group(3))
        try:
            return TimeValue(max(hi - lo + 1, 1.0), unit).to_years()
        except ValueError:
            pass
    # unit-first single slot: "month 3" — duration mode: slot 3 (1 month);
    # target mode: 3 months into the future
    slot = re.match(r"^([a-z]+)\s+(\d+(?:\.\d+)?)$", text)
    if slot:
        try:
            value = float(slot.group(2)) if target_mode else 1.0
            return TimeValue(value, slot.group(1)).to_years()
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


ASSIGNMENTS_HEADER_RE = re.compile(r"time assignments?", re.IGNORECASE)
ASSIGNMENT_LINE_RE = re.compile(
    r"^\s*\**\s*Step\s+(\d+)\s*\**\s*:\s*(.+?)\s*$", re.MULTILINE
)


def parse_time_assignments(text: str) -> dict[int, tuple[str, float | None]]:
    """Parse a control-mode 'Time assignments:' reply into
    {step_number: (raw text, target offset in years)}. Returns {} for turns
    without the header, so step turns that merely mention 'Step 2: ...' are
    never misread as assignments."""
    if not ASSIGNMENTS_HEADER_RE.search(text):
        return {}
    out: dict[int, tuple[str, float | None]] = {}
    for m in ASSIGNMENT_LINE_RE.finditer(text):
        raw = m.group(2)
        out[int(m.group(1))] = (raw, _parse_duration_years(raw, target_mode=True))
    return out


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
