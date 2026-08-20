"""End-to-end experiment: cache -> streaming projection -> polynomial head -> scorecards.

This is the entry point the CLI and the tests both use, so the reported numbers and the
reusable API can never drift apart.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from .cache import ActivationCache, load_cache
from .classification import (
    ClassificationReport,
    assign_classes,
    build_report,
    fit_bands,
    optimal_edges,
    unit_aligned_edges,
    within_range,
)
from .evaluation import balanced_group_folds
from .pipeline import (
    FoldArtifacts,
    ModelConfig,
    ViewConfig,
    compute_view_scores,
    cross_validate,
)
from .readout import bin_edges, bin_indicators, blend, expectation_readout
from .report import Scorecards, build_scorecards
from .targets import build_targets, compose_from_parts

DEFAULT_CONFIG = ModelConfig(
    views=(ViewConfig(view="raw", n_supervised=64, n_residual=32),),
    poly_degree=4,
    poly_head=10,
    alpha=0.003,
    n_splits=5,
)
# Bin count and softmax sharpness for the distributional readout. 48 bins put roughly four
# bins per decade, fine enough to separate adjacent units without starving any bin of rows.
N_BINS = 48
BIN_TEMPERATURE = 40.0


@dataclass(frozen=True)
class ExperimentResult:
    """Out-of-fold predictions and every scorecard derived from them."""

    scorecards: Scorecards
    readouts: dict[str, np.ndarray]
    folds: np.ndarray
    metadata: pd.DataFrame
    artifacts: list[FoldArtifacts]
    details: dict[str, object] = field(default_factory=dict)


def select_rows(
    cache: ActivationCache, *, drop_tasks: tuple[str, ...] = (), subsample: int = 0, seed: int = 0
) -> pd.DataFrame:
    """Keep the rows that carry a horizon, minus any tasks the caller has excluded.

    Prompts that state no horizon at all (the ``task_only`` folder) have no target and are
    dropped here; they remain available as label-free baselines for the task-centred views.
    """
    metadata = cache.metadata
    metadata = metadata.loc[metadata["log10_time_horizon_months"].notna()]
    if drop_tasks:
        metadata = metadata.loc[~metadata["task"].isin(drop_tasks)]
    metadata = metadata.reset_index(drop=True)
    if subsample and subsample < len(metadata):
        generator = np.random.default_rng(seed)
        keep = np.sort(generator.choice(len(metadata), size=subsample, replace=False))
        metadata = metadata.iloc[keep].reset_index(drop=True)
    return metadata


def run_experiment(
    cache: str | Path | ActivationCache,
    *,
    config: ModelConfig = DEFAULT_CONFIG,
    drop_tasks: tuple[str, ...] = (),
    subsample: int = 0,
) -> ExperimentResult:
    """Fit the horizon model under grouped cross-validation and score every slice."""
    if not isinstance(cache, ActivationCache):
        cache = load_cache(cache)
    metadata = select_rows(cache, drop_tasks=drop_tasks, subsample=subsample)
    rows = metadata["row"].to_numpy(dtype=np.int64)
    folds = balanced_group_folds(metadata, n_splits=config.n_splits)
    targets, names = build_targets(metadata, auxiliary=True)

    # The head also learns a distribution over horizon bins. All three readouts below are
    # deterministic functions of that one fitted head, so combining them adds no parameters.
    horizon = targets[:, 0]
    indicators, centres = bin_indicators(horizon, bin_edges(horizon, N_BINS))
    targets = np.hstack([targets, indicators])

    predictions, artifacts, details = cross_validate(
        cache, rows, metadata, folds, targets, config
    )
    direct = predictions[:, 0]
    snapped = compose_from_parts(predictions, names, snap_units=True)
    distributional = expectation_readout(
        predictions[:, len(names) :], centres, temperature=BIN_TEMPERATURE
    )
    readouts = {
        "direct": direct,
        "unit_snapped": snapped,
        "distributional": distributional,
        "blended": blend(direct, snapped, distributional),
    }

    scorecards = build_scorecards(metadata, horizon, readouts["blended"], folds)
    details["readout_rmse"] = {
        name: float(np.sqrt(np.mean((estimate - horizon) ** 2)))
        for name, estimate in readouts.items()
    }
    return ExperimentResult(
        scorecards=scorecards,
        readouts=readouts,
        folds=folds,
        metadata=metadata,
        artifacts=artifacts,
        details=details,
    )


def run_band_classification(
    cache: str | Path | ActivationCache,
    *,
    n_bands: int = 4,
    config: ModelConfig = DEFAULT_CONFIG,
    edges: np.ndarray | None = None,
    unit_bands: bool = False,
    window: tuple[float, float] = (-np.inf, np.inf),
    drop_tasks: tuple[str, ...] = (),
    subsample: int = 0,
    min_share: float = 0.5,
) -> tuple[ClassificationReport, dict[str, object]]:
    """Fit a banded classifier over log10 horizon under the same grouped splits.

    Band edges come from one of three places. Explicit ``edges`` are used as given;
    ``unit_bands`` uses the a-priori unit midpoints, which involve no tuning at all; otherwise
    the edges are optimised against the regression model's own out-of-fold predictions. That
    last choice tunes *what question is asked*, not the classifier -- the classifier is fitted
    and scored strictly out of fold either way -- but it does mean the band count and the band
    positions were informed by the full label set, and the reported accuracy should be read
    with that in mind.
    """
    if not isinstance(cache, ActivationCache):
        cache = load_cache(cache)
    metadata = select_rows(cache, drop_tasks=drop_tasks, subsample=subsample)
    horizon = metadata["log10_time_horizon_months"].to_numpy(dtype=np.float64)

    keep = within_range(horizon, *window)
    metadata = metadata.loc[keep].reset_index(drop=True)
    horizon = horizon[keep]
    rows = metadata["row"].to_numpy(dtype=np.int64)
    folds = balanced_group_folds(metadata, n_splits=config.n_splits)
    targets, _ = build_targets(metadata, auxiliary=True)

    if edges is None:
        if unit_bands:
            edges = unit_aligned_edges()
        else:
            reference = run_experiment(cache, config=config, drop_tasks=drop_tasks)
            estimate = pd.Series(
                reference.readouts["blended"], index=reference.metadata["row"]
            ).reindex(metadata["row"])
            edges = optimal_edges(
                horizon, estimate.to_numpy(), n_bands, min_share=min_share
            )

    actual_band = assign_classes(horizon, edges)
    used = int(np.unique(actual_band).size)
    scores, _ = compute_view_scores(
        cache, rows, metadata, folds, targets, config.views[0], config.n_splits
    )
    predicted_band, band_scores = fit_bands([scores], actual_band, folds, config, used)

    report = build_report(metadata, actual_band, predicted_band, edges, folds)
    details = {
        "n_bands": used,
        "edges": [float(edge) for edge in edges],
        "labels": report.band_labels(),
        "window": [float(window[0]), float(window[1])],
        "rows": int(len(metadata)),
        "tasks": int(metadata["task"].nunique()),
        "majority_baseline": float(np.bincount(actual_band).max() / actual_band.size),
    }
    return report, details
