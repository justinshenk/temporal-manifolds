"""Target construction, including the auxiliary decomposition of the horizon.

``log10_time_horizon_months`` is exactly ``log10(stated value) + log10(months per stated unit)``.
The prompt encodes those two facts in different ways -- a numeral or a number word, and a unit
noun -- so a probe that is asked to recover both parts is pushed to keep directions the single
scalar regression would happily collapse. The auxiliary columns are only ever fitted on training
rows; the reported prediction is always the horizon itself.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .cache import UNIT_TO_MONTHS

CANONICAL_UNITS = (
    "second",
    "minute",
    "hour",
    "day",
    "week",
    "month",
    "year",
    "decade",
    "century",
    "millennium",
)
UNIT_ALIASES = {"centuries": "century", "millennia": "millennium"}


def canonical_unit(unit: str) -> str:
    """Map a stated unit onto its singular canonical form."""
    key = str(unit).strip().lower()
    key = UNIT_ALIASES.get(key, key)
    if key.endswith("s") and key[:-1] in set(CANONICAL_UNITS):
        key = key[:-1]
    if key not in set(CANONICAL_UNITS):
        raise ValueError(f"Unrecognised time unit: {unit!r}")
    return key


def unit_offsets() -> np.ndarray:
    """log10 months for one of each canonical unit, in ``CANONICAL_UNITS`` order."""
    return np.array([np.log10(UNIT_TO_MONTHS[unit]) for unit in CANONICAL_UNITS])


def build_targets(
    metadata: pd.DataFrame, *, auxiliary: bool = True
) -> tuple[np.ndarray, list[str]]:
    """Return the target matrix and its column names.

    Column 0 is always the horizon. With ``auxiliary`` the matrix also carries log10 of the
    stated value, log10 of the unit's month factor, and a one-hot unit indicator.
    """
    horizon = metadata["log10_time_horizon_months"].to_numpy(dtype=np.float64)
    columns = [horizon]
    names = ["log10_time_horizon_months"]
    if not auxiliary:
        return np.column_stack(columns), names

    value = np.log10(metadata["base_value"].to_numpy(dtype=np.float64))
    units = np.array([canonical_unit(unit) for unit in metadata["base_unit"]])
    offsets = unit_offsets()
    unit_index = np.array([CANONICAL_UNITS.index(unit) for unit in units])
    columns.extend([value, offsets[unit_index]])
    names.extend(["log10_value", "log10_unit_months"])
    for position, unit in enumerate(CANONICAL_UNITS):
        columns.append((unit_index == position).astype(np.float64))
        names.append(f"unit::{unit}")
    return np.column_stack(columns), names


def compose_from_parts(
    predictions: np.ndarray, names: list[str], *, snap_units: bool = True
) -> np.ndarray:
    """Rebuild the horizon from the predicted value and unit heads.

    The unit indicator head is a 10-way scoreboard; taking its argmax snaps the coarse scale
    onto the discrete grid the prompts actually use, which removes the interpolation error a
    plain scalar regression makes between, say, "weeks" and "months".
    """
    value = predictions[:, names.index("log10_value")]
    if not snap_units:
        return value + predictions[:, names.index("log10_unit_months")]
    indicator_columns = [names.index(f"unit::{unit}") for unit in CANONICAL_UNITS]
    scores = predictions[:, indicator_columns]
    return value + unit_offsets()[np.argmax(scores, axis=1)]
