"""Turn the horizon read-out into a classification problem over bands of log10 months.

Regression on this representation bottoms out near 0.5 decades RMSE, but the error is not
uniformly spread: the median miss is about a quarter of a decade and only a few percent of rows
miss by more than a full decade. That shape is much friendlier to a banded question -- "is this
prompt talking about minutes, days, or years?" -- than to a point estimate, provided the bands
are wider than the bulk of the error and their edges do not fall where the mass sits.

Two ways of choosing edges are supported. Unit-aligned edges are fixed a priori from the
vocabulary the prompts actually use, so they carry no selection bias. Optimised edges are chosen
to maximise agreement on a grid; :func:`optimal_edges` does this exactly by dynamic programming,
because accuracy decomposes over the bands: a partition scores the sum over bands of the number
of rows whose true *and* predicted values both land inside that band.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .cache import UNIT_TO_MONTHS
from .targets import CANONICAL_UNITS

BAND_FIELD = "band"


def assign_classes(values: np.ndarray, edges: np.ndarray) -> np.ndarray:
    """Map values onto band indices ``0 .. len(edges) - 2``.

    Values outside the outer edges are clipped into the end bands; callers that want them
    excluded should filter first with :func:`within_range`.
    """
    values = np.asarray(values, dtype=np.float64)
    return np.clip(np.digitize(values, edges[1:-1], right=False), 0, len(edges) - 2)


def within_range(values: np.ndarray, low: float, high: float) -> np.ndarray:
    """Boolean mask for values inside a closed horizon window."""
    values = np.asarray(values, dtype=np.float64)
    return (values >= low) & (values <= high)


def unit_aligned_edges(units: tuple[str, ...] = CANONICAL_UNITS) -> np.ndarray:
    """Band edges placed midway between consecutive units on the log10-months axis.

    These are the natural joints of the data: every prompt states one of ten units, so a band
    boundary halfway between "hours" and "days" separates two things the prompts really do
    distinguish, rather than cutting through the middle of a cluster.
    """
    offsets = np.array([np.log10(UNIT_TO_MONTHS[unit]) for unit in units])
    midpoints = 0.5 * (offsets[:-1] + offsets[1:])
    return np.concatenate([[-np.inf], midpoints, [np.inf]])


def optimal_edges(
    actual: np.ndarray,
    predicted: np.ndarray,
    n_bands: int,
    *,
    grid: np.ndarray | None = None,
    n_grid: int = 160,
    min_share: float = 0.5,
) -> np.ndarray:
    """Choose band edges that maximise agreement between ``actual`` and ``predicted``.

    Accuracy is additive over bands, so the optimum over a candidate grid is found exactly by
    dynamic programming rather than by search. ``hits[a, b]`` -- the number of rows with both
    values inside ``(grid[a], grid[b]]`` -- comes from a 2-D cumulative histogram.

    Raw accuracy is maximised by a single band containing everything, so ``min_share`` forbids
    any band holding less than that fraction of an equal split (``min_share / n_bands`` of the
    rows). Without it the optimiser returns a degenerate partition that scores ~100% by never
    making a distinction.
    """
    actual = np.asarray(actual, dtype=np.float64)
    predicted = np.asarray(predicted, dtype=np.float64)
    if n_bands < 2:
        raise ValueError("At least two bands are required.")
    if not 0.0 < min_share <= 1.0:
        raise ValueError("min_share must fall in (0, 1].")
    if grid is None:
        low = min(actual.min(), predicted.min())
        high = max(actual.max(), predicted.max())
        grid = np.linspace(low, high, n_grid)
    grid = np.asarray(grid, dtype=np.float64)
    size = grid.size

    # Grid position of every row; a band is a closed range of positions [start, stop].
    actual_bin = np.clip(np.searchsorted(grid, actual, side="right") - 1, 0, size - 1)
    predicted_bin = np.clip(np.searchsorted(grid, predicted, side="right") - 1, 0, size - 1)
    joint = np.zeros((size + 1, size + 1))
    np.add.at(joint, (actual_bin + 1, predicted_bin + 1), 1.0)
    cumulative = joint.cumsum(axis=0).cumsum(axis=1)
    marginal = np.concatenate([[0.0], np.bincount(actual_bin, minlength=size).cumsum()])

    positions = np.arange(size)
    starts, stops = positions[:, None], positions[None, :]
    hit_matrix = (
        cumulative[stops + 1, stops + 1]
        - cumulative[starts, stops + 1]
        - cumulative[stops + 1, starts]
        + cumulative[starts, starts]
    ).astype(np.float64)
    band_rows = marginal[stops + 1] - marginal[starts]
    floor = min_share * actual.size / n_bands
    hit_matrix[(stops < starts) | (band_rows < floor)] = -np.inf

    # best[k, stop] = most agreements achievable covering grid positions 0..stop with k bands.
    best = np.full((n_bands + 1, size), -np.inf)
    choice = np.zeros((n_bands + 1, size), dtype=np.int64)
    best[1] = hit_matrix[0]
    for band in range(2, n_bands + 1):
        for stop in range(1, size):
            candidates = best[band - 1, :stop] + hit_matrix[1 : stop + 1, stop]
            split = int(np.argmax(candidates))
            if np.isfinite(candidates[split]):
                best[band, stop] = candidates[split]
                choice[band, stop] = split + 1
    if not np.isfinite(best[n_bands, size - 1]):
        raise ValueError(
            f"No partition into {n_bands} bands satisfies min_share={min_share}; "
            "lower min_share or ask for fewer bands."
        )

    cuts: list[float] = []
    position = size - 1
    for band in range(n_bands, 1, -1):
        split = int(choice[band, position])
        cuts.append(float(grid[split]))
        position = split - 1
    return np.concatenate([[-np.inf], sorted(cuts), [np.inf]])


def one_hot(band_index: np.ndarray, n_bands: int) -> np.ndarray:
    indicators = np.zeros((band_index.size, n_bands))
    indicators[np.arange(band_index.size), band_index] = 1.0
    return indicators


def fit_bands(
    view_scores, band_index: np.ndarray, folds: np.ndarray, config, n_bands: int
) -> tuple[np.ndarray, np.ndarray]:
    """Fit a one-vs-rest ridge over the bands and return ``(predicted band, band scores)``.

    Regressing the one-hot band membership and taking the argmax is a linear discriminant fitted
    from the same streamed sufficient statistics as everything else, so the classifier costs one
    extra solve rather than a new pass over the activations.
    """
    from .pipeline import fit_predict_head  # imported here to avoid a circular import

    scores, _ = fit_predict_head(view_scores, one_hot(band_index, n_bands), folds, config)
    return np.argmax(scores, axis=1), scores


@dataclass(frozen=True)
class ClassificationReport:
    """Accuracy of a banded read-out, sliced the way the regression report is."""

    edges: np.ndarray
    accuracy: float
    balanced_accuracy: float
    within_one_band: float
    per_band: pd.DataFrame
    by_task: pd.DataFrame
    by_folder: pd.DataFrame
    confusion: pd.DataFrame
    predictions: pd.DataFrame

    def band_labels(self) -> list[str]:
        return band_labels(self.edges)

    def summary(self, *, top_tasks: int = 10) -> str:
        return "\n".join(
            [
                f"bands={len(self.edges) - 1}  accuracy={self.accuracy:.4f}  "
                f"balanced={self.balanced_accuracy:.4f}  within-one-band={self.within_one_band:.4f}",
                "",
                "PER BAND",
                self.per_band.to_string(index=False),
                "",
                "PER FOLDER",
                self.by_folder.to_string(index=False),
                "",
                f"PER TASK (worst {top_tasks} of {len(self.by_task)})",
                self.by_task.head(top_tasks).to_string(index=False),
            ]
        )


def _format_horizon(value: float) -> str:
    """Render a log10-months edge as the nearest human-readable duration."""
    if not np.isfinite(value):
        return "inf" if value > 0 else "-inf"
    months = 10.0**value
    for unit in reversed(CANONICAL_UNITS):
        factor = UNIT_TO_MONTHS[unit]
        if months >= factor * 0.999:
            return f"{months / factor:.3g}{unit[:3]}"
    return f"{months / UNIT_TO_MONTHS['second']:.3g}sec"


def band_labels(edges: np.ndarray) -> list[str]:
    return [
        f"{_format_horizon(edges[index])}..{_format_horizon(edges[index + 1])}"
        for index in range(len(edges) - 1)
    ]


def build_report(
    metadata: pd.DataFrame,
    actual_band: np.ndarray,
    predicted_band: np.ndarray,
    edges: np.ndarray,
    folds: np.ndarray,
) -> ClassificationReport:
    """Assemble accuracy overall, per band, per task and per source folder."""
    frame = metadata.assign(
        actual_band=np.asarray(actual_band),
        predicted_band=np.asarray(predicted_band),
        correct=np.asarray(actual_band) == np.asarray(predicted_band),
        fold=np.asarray(folds),
    )
    labels = band_labels(edges)
    n_bands = len(labels)

    per_band = pd.DataFrame(
        [
            {
                "band": index,
                "range": labels[index],
                "n": int((frame["actual_band"] == index).sum()),
                "recall": float(frame.loc[frame["actual_band"] == index, "correct"].mean())
                if (frame["actual_band"] == index).any()
                else np.nan,
                "precision": float(frame.loc[frame["predicted_band"] == index, "correct"].mean())
                if (frame["predicted_band"] == index).any()
                else np.nan,
            }
            for index in range(n_bands)
        ]
    )

    def slice_accuracy(key: str) -> pd.DataFrame:
        grouped = frame.groupby(key)["correct"].agg(["size", "mean"])
        grouped.columns = ["n", "accuracy"]
        return grouped.reset_index().sort_values("accuracy").reset_index(drop=True)

    confusion = pd.crosstab(
        frame["actual_band"], frame["predicted_band"], dropna=False
    ).reindex(index=range(n_bands), columns=range(n_bands), fill_value=0)
    confusion.index = labels
    confusion.columns = labels

    return ClassificationReport(
        edges=edges,
        accuracy=float(frame["correct"].mean()),
        balanced_accuracy=float(per_band["recall"].mean(skipna=True)),
        within_one_band=float(
            (np.abs(frame["actual_band"] - frame["predicted_band"]) <= 1).mean()
        ),
        per_band=per_band,
        by_task=slice_accuracy("task"),
        by_folder=slice_accuracy("source_folder"),
        confusion=confusion,
        predictions=frame,
    )
