"""Utilities for time-constrained dataset generation."""

from __future__ import annotations

from typing import Literal

NumberFormat = Literal["numeric", "words"]
UnitVariant = Literal["original", "smaller"]

NUMBER_FORMATS: tuple[NumberFormat, ...] = ("numeric", "words")

SINGULAR_UNITS = {
    "seconds": "second",
    "minutes": "minute",
    "hours": "hour",
    "days": "day",
    "weeks": "week",
    "months": "month",
    "years": "year",
    "decades": "decade",
    "centuries": "century",
    "millennia": "millennium",
}

NEXT_SMALLEST_UNIT = {
    "minutes": ("seconds", 60),
    "hours": ("minutes", 60),
    "days": ("hours", 24),
    "weeks": ("days", 7),
    "months": ("weeks", 4),
    "years": ("months", 12),
    "decades": ("years", 10),
    "centuries": ("decades", 10),
    "millennia": ("centuries", 10),
}

SUPPORTED_UNITS = set(SINGULAR_UNITS)

ONES = {
    0: "zero",
    1: "one",
    2: "two",
    3: "three",
    4: "four",
    5: "five",
    6: "six",
    7: "seven",
    8: "eight",
    9: "nine",
    10: "ten",
    11: "eleven",
    12: "twelve",
    13: "thirteen",
    14: "fourteen",
    15: "fifteen",
    16: "sixteen",
    17: "seventeen",
    18: "eighteen",
    19: "nineteen",
}

TENS = {
    20: "twenty",
    30: "thirty",
    40: "forty",
    50: "fifty",
    60: "sixty",
    70: "seventy",
    80: "eighty",
    90: "ninety",
}


def number_to_words(value: int) -> str:
    """Return a lowercase English rendering for non-negative integers."""
    if value < 0:
        raise ValueError(f"Unsupported number for word rendering: {value}")
    if value < 20:
        return ONES[value]
    if value < 100:
        tens = value // 10 * 10
        remainder = value % 10
        if remainder == 0:
            return TENS[tens]
        return f"{TENS[tens]} {ONES[remainder]}"
    if value < 1000:
        hundreds = value // 100
        remainder = value % 100
        if remainder == 0:
            return f"{ONES[hundreds]} hundred"
        return f"{ONES[hundreds]} hundred {number_to_words(remainder)}"
    if value < 1_000_000:
        thousands = value // 1000
        remainder = value % 1000
        if remainder == 0:
            return f"{number_to_words(thousands)} thousand"
        return f"{number_to_words(thousands)} thousand {number_to_words(remainder)}"
    raise ValueError(f"Unsupported number for word rendering: {value}")


def singularize_unit(unit: str) -> str:
    """Return the singular form for a time unit."""
    if unit in SINGULAR_UNITS:
        return SINGULAR_UNITS[unit]
    if unit.endswith("s"):
        return unit[:-1]
    return unit


def render_unit(value: int, unit: str) -> str:
    """Return the unit with singular form when the value is one."""
    return singularize_unit(unit) if value == 1 else unit


def smaller_unit_value(value: int, unit: str) -> tuple[int, str] | None:
    """Convert a value to the next smallest configured time unit."""
    conversion = NEXT_SMALLEST_UNIT.get(unit)
    if conversion is None:
        return None

    smaller_unit, multiplier = conversion
    return value * multiplier, smaller_unit


def validate_task_units(task_units: dict[str, set[str]]) -> None:
    """Raise a helpful error when a task declares an unsupported time unit."""
    invalid_units = {
        unit for units in task_units.values() for unit in units if unit not in SUPPORTED_UNITS
    }
    if invalid_units:
        formatted_units = ", ".join(sorted(invalid_units))
        raise ValueError(f"Unsupported time unit(s): {formatted_units}")
