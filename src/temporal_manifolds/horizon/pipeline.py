"""Two-pass streaming fit of log10 time horizon from last-position activations.

Pass one accumulates per-fold second moments of the 2560-dimensional activations. Pass two
turns each fold's training statistics into a supervised + residual projection and scores every
row through it. Everything after that runs on a few dozen columns, so polynomial degree, ridge
strength and component counts can be swept without touching the memmap again.

Optional row views ("raw", "rms", task-centred) are fitted independently and their scores are
concatenated, which lets a model see both a task's absolute position in activation space and
its position relative to the other prompts about the same task.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal, Sequence

import numpy as np
import pandas as pd
from sklearn.preprocessing import PolynomialFeatures

from .cache import ActivationCache
from .projection import LatentProjection, build_projection
from .streaming import SufficientStatistics, ridge_solution

RowView = Literal["raw", "rms", "task_centered", "task_centered_rms"]
CHUNK_SIZE = 4096


@dataclass(frozen=True)
class ViewConfig:
    """One activation view and the projection to extract from it."""

    view: RowView = "raw"
    n_supervised: int = 8
    n_residual: int = 8


@dataclass(frozen=True)
class ModelConfig:
    """The full experiment definition."""

    views: tuple[ViewConfig, ...] = (ViewConfig(),)
    poly_degree: int = 4
    poly_head: int | None = None
    alpha: float = 1e-3
    n_splits: int = 5
    random_state: int = 0
    # A degree-4 expansion grows as O(k^4): 24 latent features would be 20k terms and a 3 GB
    # normal matrix. Anything past this cap is refused rather than silently thrashing swap.
    max_expanded_features: int = 4000


@dataclass
class FoldArtifacts:
    """What a single fold produced, kept for inspection and for refits."""

    projections: list[LatentProjection] = field(default_factory=list)
    coefficients: np.ndarray | None = None
    intercept: float = 0.0


def _rms_normalize(block: np.ndarray) -> np.ndarray:
    scale = np.sqrt(np.mean(block.astype(np.float32) ** 2, axis=1, keepdims=True)) + 1e-6
    return block / scale * np.sqrt(block.shape[1], dtype=np.float32)


def task_means(
    cache: ActivationCache, rows: np.ndarray, groups: np.ndarray, *, rms: bool
) -> tuple[np.ndarray, dict[object, int]]:
    """Average each task's activations without looking at any label.

    Task centring is label-free, so it may be fitted over every row -- including held-out
    tasks -- without leaking the target. It removes the "which task is this" direction that a
    grouped split would otherwise let the model exploit.
    """
    codes = {group: index for index, group in enumerate(pd.unique(groups))}
    totals = np.zeros((len(codes), cache.n_features), dtype=np.float64)
    counts = np.zeros(len(codes), dtype=np.float64)
    index_of_row = np.array([codes[group] for group in groups], dtype=np.int64)
    for position in range(0, rows.size, CHUNK_SIZE):
        slice_rows = rows[position : position + CHUNK_SIZE]
        block = np.asarray(cache.activations[slice_rows], dtype=np.float32)
        if rms:
            block = _rms_normalize(block)
        np.add.at(totals, index_of_row[position : position + CHUNK_SIZE], block)
        np.add.at(counts, index_of_row[position : position + CHUNK_SIZE], 1.0)
    return (totals / counts[:, None]).astype(np.float32), codes


class ViewReader:
    """Materialise one activation view chunk by chunk."""

    def __init__(
        self,
        cache: ActivationCache,
        rows: np.ndarray,
        groups: np.ndarray,
        view: RowView,
    ) -> None:
        self.cache = cache
        self.rows = rows
        self.view = view
        self.rms = view in {"rms", "task_centered_rms"}
        self.centered = view in {"task_centered", "task_centered_rms"}
        self.baselines: np.ndarray | None = None
        self.group_index: np.ndarray | None = None
        if self.centered:
            baselines, codes = task_means(cache, rows, groups, rms=self.rms)
            self.baselines = baselines
            self.group_index = np.array([codes[group] for group in groups], dtype=np.int64)

    def block(self, start: int, stop: int) -> np.ndarray:
        block = np.asarray(self.cache.activations[self.rows[start:stop]], dtype=np.float32)
        if self.rms:
            block = _rms_normalize(block)
        if self.baselines is not None and self.group_index is not None:
            block = block - self.baselines[self.group_index[start:stop]]
        return block


def accumulate_fold_statistics(
    reader: ViewReader, targets: np.ndarray, folds: np.ndarray, n_splits: int
) -> list[SufficientStatistics]:
    """Stream the activations once, gathering statistics separately for each fold."""
    n_targets = targets.shape[1]
    per_fold = [
        SufficientStatistics.zeros(reader.cache.n_features, n_targets) for _ in range(n_splits)
    ]
    total = reader.rows.size
    for start in range(0, total, CHUNK_SIZE):
        stop = min(start + CHUNK_SIZE, total)
        block = reader.block(start, stop)
        chunk_folds = folds[start:stop]
        for fold in np.unique(chunk_folds):
            mask = chunk_folds == fold
            per_fold[int(fold)].partial_fit(block[mask], targets[start:stop][mask])
    return per_fold


def score_all_folds(
    reader: ViewReader,
    projections: Sequence[LatentProjection],
    *,
    out: np.ndarray | None = None,
) -> np.ndarray:
    """Return ``(n_splits, n_rows, k)`` scores: every row seen through every fold's projection.

    Pass ``out`` (typically a memmap) to keep the result off the heap; at full scale the array
    is a few hundred megabytes and competes with the 2560x2560 statistics for the same RAM.
    """
    n_components = projections[0].n_components
    shape = (len(projections), reader.rows.size, n_components)
    scores = np.empty(shape, dtype=np.float32) if out is None else out
    if scores.shape != shape:
        raise ValueError(f"Output buffer is {scores.shape}, expected {shape}.")
    total = reader.rows.size
    for start in range(0, total, CHUNK_SIZE):
        stop = min(start + CHUNK_SIZE, total)
        block = reader.block(start, stop)
        for index, projection in enumerate(projections):
            scores[index, start:stop] = projection.transform(block)
    return scores


def _expander(config: ModelConfig, n_features: int) -> tuple[PolynomialFeatures, int]:
    head = n_features if config.poly_head is None else min(config.poly_head, n_features)
    return PolynomialFeatures(degree=config.poly_degree, include_bias=False), head


def expand(features: np.ndarray, poly: PolynomialFeatures, head: int) -> np.ndarray:
    """Polynomially expand the leading columns and carry the rest through linearly."""
    expanded = poly.fit_transform(features[:, :head])
    if head < features.shape[1]:
        expanded = np.hstack([expanded, features[:, head:]])
    return expanded


def fit_head(
    features: np.ndarray,
    targets: np.ndarray,
    train_mask: np.ndarray,
    config: ModelConfig,
) -> tuple[np.ndarray, np.ndarray, PolynomialFeatures, int]:
    """Accumulate the expanded design in chunks, then solve the ridge in closed form."""
    poly, head = _expander(config, features.shape[1])
    probe = expand(features[:1, :], poly, head)
    if probe.shape[1] > config.max_expanded_features:
        raise ValueError(
            f"Degree {config.poly_degree} over {head} features yields {probe.shape[1]} terms, "
            f"above the {config.max_expanded_features} cap; lower the degree or set poly_head."
        )
    stats = SufficientStatistics.zeros(probe.shape[1], targets.shape[1])
    train_rows = np.flatnonzero(train_mask)
    for start in range(0, train_rows.size, CHUNK_SIZE):
        block = train_rows[start : start + CHUNK_SIZE]
        stats.partial_fit(expand(features[block], poly, head), targets[block])
    coefficients, intercepts = ridge_solution(stats, config.alpha, scale=stats.feature_scale())
    return coefficients, intercepts, poly, head


def predict_head(
    features: np.ndarray,
    coefficients: np.ndarray,
    intercepts: np.ndarray,
    poly: PolynomialFeatures,
    head: int,
) -> np.ndarray:
    out = np.empty((features.shape[0], coefficients.shape[1]), dtype=np.float64)
    for start in range(0, features.shape[0], CHUNK_SIZE):
        block = slice(start, min(start + CHUNK_SIZE, features.shape[0]))
        out[block] = expand(features[block], poly, head) @ coefficients + intercepts
    return out


def compute_view_scores(
    cache: ActivationCache,
    rows: np.ndarray,
    metadata: pd.DataFrame,
    folds: np.ndarray,
    targets: np.ndarray,
    view_config: ViewConfig,
    n_splits: int,
    *,
    random_state: int = 0,
) -> tuple[np.ndarray, list[LatentProjection]]:
    """Fit one view's per-fold projection and score every row through each of them.

    This is the only part of the fit that reads the memmap, so callers sweeping polynomial
    degree or ridge strength should compute the scores once and reuse them.
    """
    reader = ViewReader(cache, rows, metadata["task"].to_numpy(), view_config.view)
    per_fold = accumulate_fold_statistics(reader, targets, folds, n_splits)
    total = per_fold[0].copy()
    for stats in per_fold[1:]:
        total = total + stats
    projections = [
        build_projection(
            total - per_fold[fold],
            n_supervised=view_config.n_supervised,
            n_residual=view_config.n_residual,
            random_state=random_state,
        )
        for fold in range(n_splits)
    ]
    return score_all_folds(reader, projections), projections


def fit_predict_head(
    view_scores: Sequence[np.ndarray], targets: np.ndarray, folds: np.ndarray, config: ModelConfig
) -> tuple[np.ndarray, list[FoldArtifacts]]:
    """Expand the latent scores, ridge-fit per fold, and return out-of-fold predictions."""
    artifacts = [FoldArtifacts() for _ in range(config.n_splits)]
    predictions = np.full((targets.shape[0], targets.shape[1]), np.nan)
    for fold in range(config.n_splits):
        features = np.hstack([np.asarray(scores)[fold] for scores in view_scores])
        train_mask = folds != fold
        coefficients, intercepts, poly, head = fit_head(features, targets, train_mask, config)
        test_rows = np.flatnonzero(~train_mask)
        predictions[test_rows] = predict_head(
            features[test_rows], coefficients, intercepts, poly, head
        )
        artifacts[fold].coefficients = coefficients
        artifacts[fold].intercept = float(np.mean(intercepts))
    return predictions, artifacts


def cross_validate(
    cache: ActivationCache,
    rows: np.ndarray,
    metadata: pd.DataFrame,
    folds: np.ndarray,
    targets: np.ndarray,
    config: ModelConfig,
) -> tuple[np.ndarray, list[FoldArtifacts], dict[str, object]]:
    """Run the whole two-pass fit and return out-of-fold predictions for every row."""
    view_scores: list[np.ndarray] = []
    details: dict[str, object] = {"views": []}
    all_projections: list[list[LatentProjection]] = []

    for view_config in config.views:
        scores, projections = compute_view_scores(
            cache,
            rows,
            metadata,
            folds,
            targets,
            view_config,
            config.n_splits,
            random_state=config.random_state,
        )
        view_scores.append(scores)
        all_projections.append(projections)
        details["views"].append(
            {
                "view": view_config.view,
                "n_supervised": projections[0].n_supervised,
                "n_residual": projections[0].n_residual,
            }
        )

    predictions, artifacts = fit_predict_head(view_scores, targets, folds, config)
    for fold, artifact in enumerate(artifacts):
        artifact.projections = [projections[fold] for projections in all_projections]
    details["n_expanded_features"] = int(artifacts[0].coefficients.shape[0])
    return predictions, artifacts, details
