"""Planning prompts whose horizon is implied rather than named as a duration.

Every other time-constrained dataset in this package writes the horizon out as a
literal duration: ``Available time: 30 minutes``, ``I have 30 minutes``.  A model
can read those prompts' horizons straight off the surface form.  Here the horizon
is instead *implied by arithmetic the reader has to perform*: two clock times, two
calendar dates, a fraction of a larger allowance, a count of fixed-length passes.
Activations cached from this dataset therefore separate encoding a horizon that was
stated from encoding a horizon that had to be derived.

Two properties are load-bearing and are enforced by the tests:

* **Every prompt is self-contained.** Nothing here depends on the reader's own
  calendar, locale, or timezone: the anchor moment is written into the prompt
  whenever the horizon is expressed against it, and no prompt says "by Friday" or
  "next month" without also saying what today is. ``base_value`` and ``base_unit``
  carry the configured horizon, including the common four-weeks-per-month
  approximation used by the other datasets.
* **Every (task, horizon) cell is rendered at least ten different ways.**  The
  variants below split into three families -- ``arithmetic`` (unit-agnostic, always
  available), ``clock`` (sub-day horizons), and ``calendar`` (day-and-longer
  horizons) -- and the families overlap enough that the thinnest cell in the grid
  still gets ten distinct renderings.

The task set, the per-task unit sets, and the horizon grid are copied from
:mod:`temporal_manifolds.dataset.conversational`, so a point here pairs with its
directly-stated counterpart in the other folders by task and time horizon.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import MAXYEAR, MINYEAR, datetime, timedelta
from typing import Any, Callable, Literal

try:
    from . import conversational
    from .utils import (
        NumberFormat,
        UnitVariant,
        number_to_words,
        render_unit,
        singularize_unit,
        smaller_unit_value,
    )
except ImportError:  # Support running dataset/generate.py directly.
    import conversational  # type: ignore

    from temporal_manifolds.dataset.utils import (  # type: ignore
        NumberFormat,
        UnitVariant,
        number_to_words,
        render_unit,
        singularize_unit,
        smaller_unit_value,
    )


# The request half of every prompt is byte-identical across variants, so the only
# thing that changes between two prompts for the same task is how the horizon is
# expressed.
REQUEST = "Write a plan to {task} that fits the time available."

# A Tuesday at 09:00, chosen so that weekday-bearing prompts start mid-week and so
# that clock arithmetic stays inside one daytime. The day of the month is small
# enough that adding months never has to clamp to a shorter month.
ANCHOR = datetime(2025, 3, 4, 9, 0, 0)

# Clock renderings are only used while the whole horizon fits in one working day,
# which also keeps the earlier "work opened at ..." timestamp after midnight.
MAX_CLOCK_SECONDS = 8 * 60 * 60

# Calendar dates stop being a natural way to write a horizon long before the
# millennium-scale end of the grid; those cells fall back to spans of years.
MAX_CALENDAR_YEARS = 200

SECONDS_PER_UNIT = {"seconds": 1, "minutes": 60, "hours": 3600}
DAYS_PER_UNIT = {"days": 1, "weeks": 7}
YEARS_PER_UNIT = {"years": 1, "decades": 10, "centuries": 100, "millennia": 1000}

MONTH_NAMES = (
    "January",
    "February",
    "March",
    "April",
    "May",
    "June",
    "July",
    "August",
    "September",
    "October",
    "November",
    "December",
)
WEEKDAY_NAMES = (
    "Monday",
    "Tuesday",
    "Wednesday",
    "Thursday",
    "Friday",
    "Saturday",
    "Sunday",
)
DAYS_IN_MONTH = (31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31)


def _is_leap_year(year: int) -> bool:
    return year % 4 == 0 and (year % 100 != 0 or year % 400 == 0)


def _days_in_month(year: int, month: int) -> int:
    if month == 2 and _is_leap_year(year):
        return 29
    return DAYS_IN_MONTH[month - 1]


def add_months(moment: datetime, months: int) -> datetime:
    """Return ``moment`` shifted by whole calendar months.

    Calendar months are not a fixed number of days, so a month-scale horizon is
    only unambiguous if it is expressed as the same day of a later month. The day
    is clamped for months that are too short, which the anchor never triggers.
    """
    total = (moment.month - 1) + months
    year = moment.year + total // 12
    month = total % 12 + 1
    day = min(moment.day, _days_in_month(year, month))
    return moment.replace(year=year, month=month, day=day)


def add_years(moment: datetime, years: int) -> datetime:
    """Return ``moment`` shifted by whole years, clamping a leap day if needed.

    The grid reaches a hundred millennia, which is far outside the range
    :class:`datetime.datetime` can represent. Those horizons are never written as a
    date, so this raises rather than silently truncating and callers use the
    year-span rendering instead.
    """
    year = moment.year + years
    if not MINYEAR <= year <= MAXYEAR:
        raise ValueError(
            f"A horizon of {years} year(s) from {moment.year} cannot be written as a date."
        )
    day = min(moment.day, _days_in_month(year, moment.month))
    return moment.replace(year=year, day=day)


def horizon_seconds(value: int, unit: str) -> int | None:
    """Return the horizon in seconds, or ``None`` for calendar-scale units."""
    multiplier = SECONDS_PER_UNIT.get(unit)
    return None if multiplier is None else value * multiplier


def deadline_moment(value: int, unit: str) -> datetime:
    """Return the exact moment ``value`` ``unit`` after the anchor.

    Every unit in the grid is convertible: sub-day units through elapsed seconds,
    days and weeks through whole days, months through calendar months, and the
    decade-and-longer units through whole years.
    """
    seconds = horizon_seconds(value, unit)
    if seconds is not None:
        return ANCHOR + timedelta(seconds=seconds)
    if unit in DAYS_PER_UNIT:
        return ANCHOR + timedelta(days=value * DAYS_PER_UNIT[unit])
    if unit == "months":
        return add_months(ANCHOR, value)
    if unit in YEARS_PER_UNIT:
        return add_years(ANCHOR, value * YEARS_PER_UNIT[unit])
    raise ValueError(f"Unsupported time unit for an indirect horizon: {unit!r}")


def horizon_years(value: int, unit: str) -> int | None:
    """Return the horizon in whole years when it is a whole number of them."""
    if unit in YEARS_PER_UNIT:
        return value * YEARS_PER_UNIT[unit]
    return None


def _number(value: int, number_format: NumberFormat) -> str:
    return str(value) if number_format == "numeric" else number_to_words(value)


def _duration(value: int, unit: str, number_format: NumberFormat) -> str:
    """Return a written duration such as ``40 minutes`` or ``one minute``."""
    return f"{_number(value, number_format)} {render_unit(value, unit)}"


def _count(value: int, noun: str, number_format: NumberFormat) -> str:
    """Return a counted noun such as ``5 passes`` or ``one pass``."""
    plural = noun if value == 1 else f"{noun}es" if noun.endswith("s") else f"{noun}s"
    return f"{_number(value, number_format)} {plural}"


def _finish(clause: str) -> str:
    """End a sentence without doubling the period of a trailing abbreviation.

    Clock clauses can end on ``a.m.`` or ``p.m.``, whose own period also closes the
    sentence; appending another one would render ``9:40 a.m..``.
    """
    return clause if clause.endswith(".") else f"{clause}."


def _clock_12h(moment: datetime, with_seconds: bool) -> str:
    hour = moment.hour % 12 or 12
    suffix = "a.m." if moment.hour < 12 else "p.m."
    if with_seconds:
        return f"{hour}:{moment.minute:02d}:{moment.second:02d} {suffix}"
    return f"{hour}:{moment.minute:02d} {suffix}"


def _clock_24h(moment: datetime, with_seconds: bool) -> str:
    if with_seconds:
        return f"{moment.hour:02d}:{moment.minute:02d}:{moment.second:02d}"
    return f"{moment.hour:02d}:{moment.minute:02d}"


def _part_of_day(hour: int) -> str:
    if hour < 12:
        return "in the morning"
    if hour < 17:
        return "in the afternoon"
    if hour < 21:
        return "in the evening"
    return "at night"


def _spoken_clock(moment: datetime, with_seconds: bool) -> str:
    """Return a spoken clock time such as ``twenty minutes past nine in the morning``.

    Only "past the hour" phrasing is used. "Twenty to ten" would name a different
    hour than the one the minutes are counted from, which is exactly the kind of
    read-off ambiguity this dataset is meant to avoid.
    """
    hour_word = number_to_words(moment.hour % 12 or 12)
    part = _part_of_day(moment.hour)
    o_clock = f"{hour_word} o'clock {part}"

    pieces = []
    if moment.minute:
        pieces.append(f"{number_to_words(moment.minute)} {render_unit(moment.minute, 'minutes')}")
    if with_seconds and moment.second:
        pieces.append(f"{number_to_words(moment.second)} {render_unit(moment.second, 'seconds')}")
    if not pieces:
        return o_clock
    return f"{' and '.join(pieces)} past {hour_word} {part}"


def _iso_date(moment: datetime) -> str:
    return f"{moment.year:04d}-{moment.month:02d}-{moment.day:02d}"


def _long_date(moment: datetime) -> str:
    return f"{moment.day} {MONTH_NAMES[moment.month - 1]} {moment.year}"


def _us_date(moment: datetime) -> str:
    return f"{MONTH_NAMES[moment.month - 1]} {moment.day}, {moment.year}"


def _weekday_date(moment: datetime) -> str:
    return f"{WEEKDAY_NAMES[moment.weekday()]}, {_long_date(moment)}"


@dataclass(frozen=True)
class Rendering:
    """One rendered situation clause plus the horizon it is written in.

    ``value`` and ``unit`` describe the units the prompt's own arithmetic is
    carried out in, which is not always the canonical horizon: a one-hour horizon
    split into two blocks has to be written in minutes. ``base_value`` and
    ``base_unit`` on the record always keep the canonical answer.
    """

    text: str
    value: int
    unit: str
    unit_variant: UnitVariant = "original"


@dataclass(frozen=True)
class Variant:
    """One way of implying a horizon without writing it down as a duration."""

    id: str
    family: Literal["arithmetic", "clock", "calendar"]
    reference: str
    number_formats: tuple[NumberFormat, ...]
    build: Callable[[int, str, NumberFormat], Rendering | None]

    def render(self, value: int, unit: str, number_format: NumberFormat) -> Rendering | None:
        """Return the rendered clause, or ``None`` when this cell is unsupported."""
        if number_format not in self.number_formats:
            return None
        return self.build(value, unit, number_format)


def _divisible(value: int, unit: str) -> tuple[int, str, UnitVariant] | None:
    """Return a horizon of at least two whole units, dropping a unit if needed.

    A horizon of one unit cannot be split or scaled while staying in that unit, so
    it is restated in the next smallest unit -- one hour becomes sixty minutes.
    Units with no smaller unit configured (seconds) simply skip those variants.
    """
    if value >= 2:
        return value, unit, "original"
    smaller = smaller_unit_value(value, unit)
    if smaller is None:
        return None
    smaller_value, smaller_unit = smaller
    return smaller_value, smaller_unit, "smaller"


def _factor_pair(value: int) -> tuple[int, int] | None:
    """Return ``(multiplier, reference)`` with ``multiplier >= 2`` and a whole product."""
    for reference in range(int(value**0.5), 0, -1):
        if value % reference == 0 and value // reference >= 2:
            return value // reference, reference
    return None


# --------------------------------------------------------------------------- #
# Arithmetic variants: available for every unit in the grid, including the
# decade-and-longer units that no clock or calendar rendering reaches.
# --------------------------------------------------------------------------- #


def _proportion_half(value: int, unit: str, number_format: NumberFormat) -> Rendering:
    return Rendering(
        "The window originally set aside for this was "
        f"{_duration(2 * value, unit, number_format)} long, and it has just been cut in half.",
        value,
        unit,
    )


def _proportion_third(value: int, unit: str, number_format: NumberFormat) -> Rendering:
    return Rendering(
        f"Only a third of the {_duration(3 * value, unit, number_format)} "
        "originally allotted for this is still free.",
        value,
        unit,
    )


def _proportion_quarter(value: int, unit: str, number_format: NumberFormat) -> Rendering:
    return Rendering(
        f"Of the {_duration(4 * value, unit, number_format)} allotted for this, "
        "three quarters have already been spent.",
        value,
        unit,
    )


def _proportion_percent(value: int, unit: str, number_format: NumberFormat) -> Rendering:
    return Rendering(
        f"Eighty percent of the {_duration(5 * value, unit, number_format)} "
        "set aside for this is already gone.",
        value,
        unit,
    )


def _remainder_spent(value: int, unit: str, number_format: NumberFormat) -> Rendering:
    verb = "has" if value == 1 else "have"
    return Rendering(
        f"Of the {_duration(2 * value, unit, number_format)} booked for this, "
        f"{_duration(value, unit, number_format)} {verb} already gone.",
        value,
        unit,
    )


def _elapsed_window(value: int, unit: str, number_format: NumberFormat) -> Rendering:
    return Rendering(
        f"The clock started {_duration(value, unit, number_format)} ago, "
        f"and the whole window runs for {_duration(2 * value, unit, number_format)}.",
        value,
        unit,
    )


def _delayed_start(value: int, unit: str, number_format: NumberFormat) -> Rendering:
    return Rendering(
        f"The start slipped by {_duration(value, unit, number_format)}, and the finish is "
        f"still pinned to {_duration(2 * value, unit, number_format)} after the original "
        "start time.",
        value,
        unit,
    )


def _unit_rate_count(value: int, unit: str, number_format: NumberFormat) -> Rendering:
    return Rendering(
        f"A single pass through this takes exactly one {singularize_unit(unit)}, and there "
        f"is room for exactly {_count(value, 'pass', number_format)} before the cut-off.",
        value,
        unit,
    )


def _two_blocks(value: int, unit: str, number_format: NumberFormat) -> Rendering | None:
    divisible = _divisible(value, unit)
    if divisible is None:
        return None
    total, block_unit, unit_variant = divisible
    first = total - total // 2
    second = total // 2
    return Rendering(
        "The time comes as two back-to-back blocks, "
        f"{_duration(first, block_unit, number_format)} and then "
        f"{_duration(second, block_unit, number_format)}, with nothing available afterwards.",
        total,
        block_unit,
        unit_variant,
    )


def _multiple_reference(value: int, unit: str, number_format: NumberFormat) -> Rendering | None:
    divisible = _divisible(value, unit)
    if divisible is None:
        return None
    total, reference_unit, unit_variant = divisible
    factors = _factor_pair(total)
    if factors is None:
        return None
    multiplier, reference = factors
    scale = "twice that" if multiplier == 2 else f"{_number(multiplier, number_format)} times that"
    return Rendering(
        f"A single run of this normally takes {_duration(reference, reference_unit, number_format)}, "
        f"and you have {scale}.",
        total,
        reference_unit,
        unit_variant,
    )


# --------------------------------------------------------------------------- #
# Clock variants: sub-day horizons written as two or three wall-clock times.
# --------------------------------------------------------------------------- #


def _clock_supported(value: int, unit: str) -> bool:
    seconds = horizon_seconds(value, unit)
    return seconds is not None and 0 < seconds <= MAX_CLOCK_SECONDS


def _clock_12h_variant(value: int, unit: str, number_format: NumberFormat) -> Rendering | None:
    if not _clock_supported(value, unit):
        return None
    with_seconds = unit == "seconds"
    deadline = deadline_moment(value, unit)
    return Rendering(
        _finish(
            f"It is {_clock_12h(ANCHOR, with_seconds)}, and this has to be finished by "
            f"{_clock_12h(deadline, with_seconds)}"
        ),
        value,
        unit,
    )


def _clock_24h_variant(value: int, unit: str, number_format: NumberFormat) -> Rendering | None:
    if not _clock_supported(value, unit):
        return None
    with_seconds = unit == "seconds"
    deadline = deadline_moment(value, unit)
    return Rendering(
        f"The current time is {_clock_24h(ANCHOR, with_seconds)}, and the cut-off is "
        f"{_clock_24h(deadline, with_seconds)}.",
        value,
        unit,
    )


def _clock_spoken_variant(value: int, unit: str, number_format: NumberFormat) -> Rendering | None:
    if not _clock_supported(value, unit):
        return None
    with_seconds = unit == "seconds"
    deadline = deadline_moment(value, unit)
    return Rendering(
        f"It is {_spoken_clock(ANCHOR, with_seconds)}, and the cut-off is "
        f"{_spoken_clock(deadline, with_seconds)}.",
        value,
        unit,
    )


def _clock_three_point_variant(
    value: int, unit: str, number_format: NumberFormat
) -> Rendering | None:
    if not _clock_supported(value, unit):
        return None
    with_seconds = unit == "seconds"
    seconds = horizon_seconds(value, unit)
    assert seconds is not None  # guarded by _clock_supported
    opened = ANCHOR - timedelta(seconds=seconds)
    deadline = deadline_moment(value, unit)
    return Rendering(
        _finish(
            f"Work opened at {_clock_12h(opened, with_seconds)}, it is "
            f"{_clock_12h(ANCHOR, with_seconds)} now, and everything must be finished by "
            f"{_clock_12h(deadline, with_seconds)}"
        ),
        value,
        unit,
    )


# --------------------------------------------------------------------------- #
# Calendar variants: day-and-longer horizons written as two dates or two years.
# --------------------------------------------------------------------------- #


def _span_years(value: int, unit: str) -> int:
    """Return the horizon rounded up to whole years, for range checks only."""
    years = horizon_years(value, unit)
    if years is not None:
        return years
    if unit == "months":
        return -(-value // 12)
    if unit in DAYS_PER_UNIT:
        return -(-(value * DAYS_PER_UNIT[unit]) // 365)
    return 0


def _calendar_supported(value: int, unit: str) -> bool:
    # The range is checked before any date is constructed: the longest horizons in
    # the grid overflow the calendar entirely.
    if unit in SECONDS_PER_UNIT:
        return False
    return _span_years(value, unit) <= MAX_CALENDAR_YEARS


def _date_variant(
    renderer: Callable[[datetime], str],
) -> Callable[[int, str, NumberFormat], Rendering | None]:
    def build(value: int, unit: str, number_format: NumberFormat) -> Rendering | None:
        if not _calendar_supported(value, unit):
            return None
        deadline = deadline_moment(value, unit)
        return Rendering(
            f"Today is {renderer(ANCHOR)}, and the deadline is {renderer(deadline)}.",
            value,
            unit,
        )

    return build


def _date_weekday_variant(value: int, unit: str, number_format: NumberFormat) -> Rendering | None:
    # A weekday only helps place a date that is a handful of weeks away; beyond
    # that it is noise the reader cannot use.
    if unit not in DAYS_PER_UNIT:
        return None
    deadline = deadline_moment(value, unit)
    return Rendering(
        f"Today is {_weekday_date(ANCHOR)}, and the deadline is {_weekday_date(deadline)}.",
        value,
        unit,
    )


def _datetime_span_variant(value: int, unit: str, number_format: NumberFormat) -> Rendering | None:
    # Hours beyond one working day and whole days both need a date *and* a time to
    # pin the horizon down exactly.
    if unit not in {"hours", "days"}:
        return None
    deadline = deadline_moment(value, unit)
    return Rendering(
        f"It is {_clock_24h(ANCHOR, False)} on {_long_date(ANCHOR)}, and the cut-off is "
        f"{_clock_24h(deadline, False)} on {_long_date(deadline)}.",
        value,
        unit,
    )


def _year_span_variant(value: int, unit: str, number_format: NumberFormat) -> Rendering | None:
    years = horizon_years(value, unit)
    if years is None:
        return None
    return Rendering(
        f"The effort runs from the start of {ANCHOR.year} to the start of "
        f"{ANCHOR.year + years}.",
        value,
        unit,
    )


BOTH_FORMATS: tuple[NumberFormat, ...] = ("numeric", "words")
NUMERIC_ONLY: tuple[NumberFormat, ...] = ("numeric",)
WORDS_ONLY: tuple[NumberFormat, ...] = ("words",)


variants: tuple[Variant, ...] = (
    Variant("proportion_half", "arithmetic", "fraction", BOTH_FORMATS, _proportion_half),
    Variant("proportion_third", "arithmetic", "fraction", BOTH_FORMATS, _proportion_third),
    Variant("proportion_quarter", "arithmetic", "fraction", BOTH_FORMATS, _proportion_quarter),
    Variant("proportion_percent", "arithmetic", "percentage", BOTH_FORMATS, _proportion_percent),
    Variant("remainder_spent", "arithmetic", "remainder", BOTH_FORMATS, _remainder_spent),
    Variant("elapsed_window", "arithmetic", "remainder", BOTH_FORMATS, _elapsed_window),
    Variant("delayed_start", "arithmetic", "remainder", BOTH_FORMATS, _delayed_start),
    Variant("unit_rate_count", "arithmetic", "rate", BOTH_FORMATS, _unit_rate_count),
    Variant("two_blocks", "arithmetic", "sum", BOTH_FORMATS, _two_blocks),
    Variant("multiple_reference", "arithmetic", "multiple", BOTH_FORMATS, _multiple_reference),
    Variant("clock_12h", "clock", "wall_clock", NUMERIC_ONLY, _clock_12h_variant),
    Variant("clock_24h", "clock", "wall_clock", NUMERIC_ONLY, _clock_24h_variant),
    Variant("clock_spoken", "clock", "wall_clock", WORDS_ONLY, _clock_spoken_variant),
    Variant("clock_three_point", "clock", "wall_clock", NUMERIC_ONLY, _clock_three_point_variant),
    Variant("date_iso", "calendar", "calendar_date", NUMERIC_ONLY, _date_variant(_iso_date)),
    Variant("date_long", "calendar", "calendar_date", NUMERIC_ONLY, _date_variant(_long_date)),
    Variant("date_us", "calendar", "calendar_date", NUMERIC_ONLY, _date_variant(_us_date)),
    Variant("date_weekday", "calendar", "calendar_date", NUMERIC_ONLY, _date_weekday_variant),
    Variant("datetime_span", "calendar", "calendar_date", NUMERIC_ONLY, _datetime_span_variant),
    Variant("year_span", "calendar", "calendar_year", NUMERIC_ONLY, _year_span_variant),
)

VARIANTS_BY_ID = {variant.id: variant for variant in variants}


# The module keeps the same surface as the template-driven datasets so it can be
# introspected the same way, but the prompts are built by build_prompt_records
# rather than by formatting these strings: a clock or calendar rendering is
# arithmetic on the horizon, not a substitution into a sentence.
templates = [
    {
        "id": variant.id,
        "template": f"<{variant.family}:{variant.id}>\n\n{REQUEST}",
        "prompt_framing": variant.family,
        "horizon_reference": variant.reference,
    }
    for variant in variants
]


def _copy_task_config(config: dict[str, object]) -> dict[str, object]:
    """Copy a task config without sharing its mutable unit set."""
    copied = dict(config)
    copied["units"] = set(config["units"])  # type: ignore[arg-type]
    return copied


tasks = {task: _copy_task_config(config) for task, config in conversational.tasks.items()}

values = list(conversational.values)

def build_prompt_records() -> list[dict[str, Any]]:
    """Return every indirect-horizon prompt with the parameters that generated it.

    The record schema matches the template-driven datasets so the caching scripts
    and :func:`temporal_manifolds.activations.extraction_policy.canonical_prompt_metadata`
    need no special case. ``template_id`` names the variant and
    ``template_metadata.prompt_framing`` names its family, which are the two fields
    that survive into the cached payloads.
    """
    records: list[dict[str, Any]] = []
    for variant in variants:
        template_metadata = {
            "prompt_framing": variant.family,
            "horizon_reference": variant.reference,
        }
        for task, task_config in sorted(tasks.items()):
            task_metadata = {
                key: str(config_value)
                for key, config_value in task_config.items()
                if key != "units"
            }
            for unit in sorted(task_config["units"]):  # type: ignore[arg-type]
                for value in values:
                    for number_format in variant.number_formats:
                        rendering = variant.render(value, unit, number_format)
                        if rendering is None:
                            continue
                        records.append(
                            {
                                "text": f"{rendering.text}\n\n{REQUEST.format(task=task)}",
                                "template_id": variant.id,
                                "template_metadata": template_metadata,
                                "task": task,
                                "task_metadata": task_metadata,
                                "base_value": value,
                                "base_unit": unit,
                                "unit_variant": rendering.unit_variant,
                                "number_format": number_format,
                                "value": rendering.value,
                                "value_text": _number(rendering.value, number_format),
                                "unit": render_unit(rendering.value, rendering.unit),
                            }
                        )
    return records


def build_prompts() -> list[str]:
    """Return the ordered text of every generated prompt."""
    return [record["text"] for record in build_prompt_records()]


__all__ = [
    "ANCHOR",
    "REQUEST",
    "Rendering",
    "Variant",
    "add_months",
    "add_years",
    "build_prompt_records",
    "build_prompts",
    "deadline_moment",
    "horizon_seconds",
    "horizon_years",
    "tasks",
    "templates",
    "values",
    "variants",
]
