"""Grouped cross-validation splits and the per-task / per-folder scorecards.

Every prompt variant of a task -- template, phrasing, unit variant, number format -- must land
on the same side of a split, otherwise a fold measures memorisation of a task's wording rather
than a readable time horizon. Splits are therefore keyed on ``task`` alone.

Tasks are not interchangeable: each one occupies its own window of the horizon range, so a
random grouping can leave a fold's tasks with no comparable training neighbours. Folds are
built by dealing tasks in horizon order, which keeps every fold spanning the full range.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

GROUP_FIELD = "task"
TARGET_FIELD = "log10_time_horizon_months"


def balanced_group_folds(
    metadata: pd.DataFrame,
    *,
    n_splits: int = 5,
    group_field: str = GROUP_FIELD,
    target_field: str = TARGET_FIELD,
) -> np.ndarray:
    """Assign each row a fold index, keeping whole tasks together and horizons balanced.

    Tasks are ordered by their mean target and dealt into folds in a serpentine pattern, so
    consecutive tasks in horizon order go to different folds and every fold covers short,
    medium and long horizons.
    """
    if n_splits < 2:
        raise ValueError("Grouped cross-validation needs at least two folds.")
    summary = (
        metadata.groupby(group_field)[target_field]
        .mean()
        .sort_values(kind="mergesort")
        .reset_index()
    )
    if len(summary) < n_splits:
        raise ValueError(f"Only {len(summary)} groups available for {n_splits} folds.")

    fold_of_group: dict[object, int] = {}
    for position, group in enumerate(summary[group_field]):
        cycle, offset = divmod(position, n_splits)
        fold_of_group[group] = offset if cycle % 2 == 0 else n_splits - 1 - offset
    return metadata[group_field].map(fold_of_group).to_numpy(dtype=np.int64)


def regression_metrics(actual: np.ndarray, predicted: np.ndarray) -> dict[str, float]:
    """Return the standard error summary for one slice of predictions."""
    actual = np.asarray(actual, dtype=np.float64)
    predicted = np.asarray(predicted, dtype=np.float64)
    if actual.size == 0:
        return {"n": 0, "rmse": np.nan, "mae": np.nan, "bias": np.nan, "r2": np.nan}
    error = predicted - actual
    variance = float(np.var(actual))
    return {
        "n": int(actual.size),
        "rmse": float(np.sqrt(np.mean(error**2))),
        "mae": float(np.mean(np.abs(error))),
        "bias": float(np.mean(error)),
        "r2": float(1.0 - np.mean(error**2) / variance) if variance > 0 else np.nan,
    }


def grouped_metrics(frame: pd.DataFrame, by: str | list[str]) -> pd.DataFrame:
    """Score predictions within each level of ``by``, sorted worst RMSE first."""
    keys = [by] if isinstance(by, str) else list(by)
    rows = [
        {
            **dict(zip(keys, key if isinstance(key, tuple) else (key,), strict=True)),
            **regression_metrics(group["actual"], group["predicted"]),
        }
        for key, group in frame.groupby(keys, sort=False)
    ]
    return pd.DataFrame(rows).sort_values("rmse", ascending=False).reset_index(drop=True)


def outlier_groups(
    scorecard: pd.DataFrame, *, key: str, tolerance: float = 3.0
) -> list[str]:
    """Flag groups whose RMSE sits far above the median, using a robust MAD threshold."""
    rmse = scorecard["rmse"].to_numpy(dtype=np.float64)
    median = float(np.median(rmse))
    deviation = float(np.median(np.abs(rmse - median))) * 1.4826
    if deviation <= 0:
        return []
    flagged = scorecard.loc[rmse > median + tolerance * deviation, key]
    return flagged.tolist()
