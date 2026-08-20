"""Polynomial ridge regression from Kernel PCA coordinates to the fixed time-horizon target.

The manifold explorer answers whether the embedding *organizes* points by
``log10_time_horizon_months``. This module goes one step further and fits a model that
predicts the horizon from the embedding coordinates, reporting held-out scores.

The split is **task-disjoint**: every prompt sharing a ``task`` value lands entirely in the
training set or entirely in the test set, never both. Each task recurs at many horizons, so
a random split would place near-duplicate neighbours of a test point in training and inflate
the held-out score into a memorization measure rather than a generalization one.

This module never changes how the caller aggregated its points. When ``task`` is not one of
the aggregation fields, a point can mix tasks and carries only its first row's label; the
split still runs on those labels, and the caller is responsible for saying so.
"""

from __future__ import annotations

import io
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, BinaryIO

import joblib
import numpy as np
import pandas as pd
from sklearn import __version__ as sklearn_version
from sklearn.linear_model import Ridge
from sklearn.model_selection import GroupKFold, GroupShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler

REGRESSION_MODEL_ARTIFACT_KIND = "temporal-manifolds.manifold-regression"
REGRESSION_MODEL_ARTIFACT_VERSION = 1

#: The regression target matches the explorer's fixed optimization target.
TARGET_FEATURE = "log10_time_horizon_months"

#: Grouping column that must never straddle the train/test boundary.
GROUP_FEATURE = "task"


@dataclass(frozen=True)
class RegressionModel:
    """A fitted polynomial ridge model plus the coordinates it consumes."""

    pipeline: Pipeline
    coordinate_features: tuple[str, ...]
    degree: int
    alpha: float
    interaction_only: bool
    standardize: bool
    metrics: dict[str, Any]

    @property
    def feature_count(self) -> int:
        return len(self.coordinate_features)

    def predict(self, coordinates: np.ndarray) -> np.ndarray:
        """Predict ``log10_time_horizon_months`` from embedding coordinates."""

        values = np.asarray(coordinates, dtype=np.float64)
        if values.ndim != 2 or values.shape[1] != self.feature_count:
            raise ValueError(
                f"Expected coordinates with {self.feature_count} columns "
                f"({', '.join(self.coordinate_features)}), got shape {values.shape}."
            )
        return np.asarray(self.pipeline.predict(values), dtype=np.float64)


@dataclass
class RegressionFitResult:
    """A fitted regression with its split, per-split scores, and cross-validated scores."""

    model: RegressionModel
    metrics: dict[str, Any]
    train_indices: np.ndarray
    test_indices: np.ndarray
    train_prediction: np.ndarray
    test_prediction: np.ndarray
    train_actual: np.ndarray
    test_actual: np.ndarray
    warnings: list[str]


def _scores(actual: np.ndarray, predicted: np.ndarray) -> tuple[float, float]:
    """Return (R², RMSE) for one split.

    R² is computed against the split's own mean. On a task-disjoint test split this can be
    negative, which correctly signals that the model does worse than predicting that mean.
    """

    actual = np.asarray(actual, dtype=np.float64)
    predicted = np.asarray(predicted, dtype=np.float64)
    if not len(actual):
        return float("nan"), float("nan")
    residual = float(np.sum(np.square(actual - predicted)))
    total = float(np.sum(np.square(actual - actual.mean())))
    r2 = float(1.0 - residual / total) if total > np.finfo(np.float64).eps else float("nan")
    rmse = float(np.sqrt(residual / len(actual)))
    return r2, rmse


def _build_pipeline(
    *, degree: int, alpha: float, interaction_only: bool, standardize: bool
) -> Pipeline:
    steps: list[tuple[str, Any]] = []
    if standardize:
        # Polynomial powers of unscaled coordinates span wildly different magnitudes, which
        # makes a single ridge penalty act unevenly across terms.
        steps.append(("scale", StandardScaler()))
    steps.append(
        (
            "polynomial",
            PolynomialFeatures(
                degree=degree, interaction_only=interaction_only, include_bias=False
            ),
        )
    )
    steps.append(("ridge", Ridge(alpha=alpha)))
    return Pipeline(steps)


def task_disjoint_split(
    groups: Sequence[Any],
    *,
    test_fraction: float,
    random_state: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Split point indices so no ``task`` appears in both sides.

    Returns ``(train_indices, test_indices)``. Raises when the grouping cannot produce a
    non-empty split on both sides, which happens when too few distinct tasks are present.
    """

    group_values = np.asarray(pd.Series(list(groups)).astype(str))
    unique_groups = np.unique(group_values)
    if len(unique_groups) < 2:
        raise ValueError(
            "A task-disjoint split needs at least two distinct "
            f"`{GROUP_FEATURE}` values, but the prepared points contain "
            f"{len(unique_groups)}. Widen the metadata filters."
        )
    if not 0.0 < test_fraction < 1.0:
        raise ValueError("The test fraction must be between 0 and 1.")

    splitter = GroupShuffleSplit(
        n_splits=1, test_size=test_fraction, random_state=random_state
    )
    train_indices, test_indices = next(
        splitter.split(np.zeros(len(group_values)), groups=group_values)
    )
    if not len(train_indices) or not len(test_indices):
        raise ValueError(
            "The task-disjoint split produced an empty side. Adjust the test fraction or "
            "include more tasks."
        )
    return np.asarray(train_indices), np.asarray(test_indices)


def fit_manifold_regression(
    coordinates: np.ndarray,
    target: np.ndarray,
    groups: Sequence[Any],
    *,
    coordinate_features: Sequence[str],
    degree: int = 2,
    alpha: float = 1.0,
    interaction_only: bool = False,
    standardize: bool = True,
    test_fraction: float = 0.25,
    random_state: int = 42,
    cross_validation_folds: int = 0,
) -> RegressionFitResult:
    """Fit polynomial ridge on embedding coordinates with a task-disjoint train/test split.

    Rows whose target is missing are dropped: unconstrained prompts state no horizon, so
    there is nothing for them to supervise or score against.
    """

    coordinates = np.asarray(coordinates, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    coordinate_features = tuple(coordinate_features)
    if coordinates.ndim != 2:
        raise ValueError("Embedding coordinates must be a two-dimensional matrix.")
    if coordinates.shape[1] != len(coordinate_features):
        raise ValueError("Coordinate columns and their names are misaligned.")
    if len(target) != len(coordinates) or len(groups) != len(coordinates):
        raise ValueError("Coordinates, targets, and task groups are misaligned.")
    if degree < 1:
        raise ValueError("Polynomial degree must be at least 1.")
    if alpha < 0:
        raise ValueError("Ridge alpha must not be negative.")

    warnings: list[str] = []
    group_values = pd.Series(list(groups)).astype(str).to_numpy()
    usable = np.isfinite(target) & np.isfinite(coordinates).all(axis=1)
    dropped = int((~usable).sum())
    if dropped:
        warnings.append(
            f"{dropped:,} point(s) without a finite {TARGET_FEATURE} were excluded from "
            "the regression."
        )
    coordinates = coordinates[usable]
    target = target[usable]
    group_values = group_values[usable]
    if len(coordinates) < 4:
        raise ValueError(
            "At least four points with a finite time horizon are required to fit and score "
            "a regression."
        )

    train_indices, test_indices = task_disjoint_split(
        group_values, test_fraction=test_fraction, random_state=random_state
    )

    pipeline = _build_pipeline(
        degree=degree, alpha=alpha, interaction_only=interaction_only, standardize=standardize
    )
    pipeline.fit(coordinates[train_indices], target[train_indices])
    train_prediction = np.asarray(pipeline.predict(coordinates[train_indices]))
    test_prediction = np.asarray(pipeline.predict(coordinates[test_indices]))

    train_r2, train_rmse = _scores(target[train_indices], train_prediction)
    test_r2, test_rmse = _scores(target[test_indices], test_prediction)

    term_count = int(pipeline.named_steps["polynomial"].n_output_features_)
    if term_count >= len(train_indices):
        warnings.append(
            f"The degree-{degree} expansion has {term_count:,} terms for "
            f"{len(train_indices):,} training point(s); the fit is underdetermined and leans "
            "entirely on the ridge penalty."
        )

    metrics: dict[str, Any] = {
        "train_r2": train_r2,
        "train_rmse": train_rmse,
        "test_r2": test_r2,
        "test_rmse": test_rmse,
        "train_points": int(len(train_indices)),
        "test_points": int(len(test_indices)),
        "train_tasks": int(len(np.unique(group_values[train_indices]))),
        "test_tasks": int(len(np.unique(group_values[test_indices]))),
        "total_tasks": int(len(np.unique(group_values))),
        "excluded_points": dropped,
        "degree": degree,
        "alpha": alpha,
        "interaction_only": bool(interaction_only),
        "standardize": bool(standardize),
        "test_fraction": float(test_fraction),
        "random_state": int(random_state),
        "polynomial_terms": term_count,
        "target_feature": TARGET_FEATURE,
        "group_feature": GROUP_FEATURE,
        "coordinate_features": list(coordinate_features),
    }

    # Cross-validation reuses the same task-disjoint rule, so every fold's held-out score is
    # measured on tasks the fold never trained on.
    if cross_validation_folds and cross_validation_folds >= 2:
        distinct_groups = int(len(np.unique(group_values)))
        folds = int(min(cross_validation_folds, distinct_groups))
        if folds < 2:
            warnings.append(
                "Cross-validation needs at least two distinct tasks and was skipped."
            )
        else:
            if folds < cross_validation_folds:
                warnings.append(
                    f"Cross-validation used {folds} fold(s), limited by the number of "
                    "distinct tasks."
                )
            fold_r2: list[float] = []
            fold_rmse: list[float] = []
            splitter = GroupKFold(n_splits=folds)
            for fold_train, fold_test in splitter.split(
                coordinates, target, groups=group_values
            ):
                fold_pipeline = _build_pipeline(
                    degree=degree,
                    alpha=alpha,
                    interaction_only=interaction_only,
                    standardize=standardize,
                )
                fold_pipeline.fit(coordinates[fold_train], target[fold_train])
                predicted = np.asarray(fold_pipeline.predict(coordinates[fold_test]))
                r2, rmse = _scores(target[fold_test], predicted)
                fold_r2.append(r2)
                fold_rmse.append(rmse)
            metrics.update(
                {
                    "cv_folds": folds,
                    "cv_r2_mean": float(np.nanmean(fold_r2)),
                    "cv_r2_std": float(np.nanstd(fold_r2)),
                    "cv_rmse_mean": float(np.nanmean(fold_rmse)),
                    "cv_rmse_std": float(np.nanstd(fold_rmse)),
                    "cv_r2_folds": [float(value) for value in fold_r2],
                }
            )

    model = RegressionModel(
        pipeline=pipeline,
        coordinate_features=coordinate_features,
        degree=degree,
        alpha=alpha,
        interaction_only=bool(interaction_only),
        standardize=bool(standardize),
        metrics=metrics,
    )
    return RegressionFitResult(
        model=model,
        metrics=metrics,
        train_indices=train_indices,
        test_indices=test_indices,
        train_prediction=train_prediction,
        test_prediction=test_prediction,
        train_actual=target[train_indices],
        test_actual=target[test_indices],
        warnings=warnings,
    )


def regression_scores_table(metrics: Mapping[str, Any]) -> pd.DataFrame:
    """Return the train/test R² and RMSE table."""

    rows = [
        {
            "split": "Train",
            "points": metrics["train_points"],
            "tasks": metrics["train_tasks"],
            "R²": metrics["train_r2"],
            "RMSE": metrics["train_rmse"],
        },
        {
            "split": "Test (task-disjoint)",
            "points": metrics["test_points"],
            "tasks": metrics["test_tasks"],
            "R²": metrics["test_r2"],
            "RMSE": metrics["test_rmse"],
        },
    ]
    if "cv_r2_mean" in metrics:
        rows.append(
            {
                "split": f"Cross-validated ({metrics['cv_folds']} folds)",
                "points": metrics["train_points"] + metrics["test_points"],
                "tasks": metrics["total_tasks"],
                "R²": metrics["cv_r2_mean"],
                "RMSE": metrics["cv_rmse_mean"],
            }
        )
    return pd.DataFrame(rows)


def validate_regression_model(model: Any) -> int:
    """Validate a fitted regression model and return its coordinate count."""

    if not isinstance(model, RegressionModel):
        raise ValueError("The selected file does not contain a manifold regression model.")
    if not isinstance(model.pipeline, Pipeline):
        raise ValueError("The regression model has an invalid estimator pipeline.")
    if not model.coordinate_features:
        raise ValueError("The regression model does not name its coordinate features.")
    if model.degree < 1:
        raise ValueError("The regression model has an invalid polynomial degree.")
    return len(model.coordinate_features)


def serialize_regression_model(
    model: RegressionModel,
    *,
    metadata: Mapping[str, Any] | None = None,
) -> bytes:
    """Serialize a fitted regression model and provenance as a versioned joblib artifact."""

    feature_count = validate_regression_model(model)
    artifact = {
        "kind": REGRESSION_MODEL_ARTIFACT_KIND,
        "version": REGRESSION_MODEL_ARTIFACT_VERSION,
        "model": model,
        "sklearn_version": sklearn_version,
        "feature_count": feature_count,
        "metadata": {
            **dict(metadata or {}),
            "target_feature": TARGET_FEATURE,
            "group_feature": GROUP_FEATURE,
            "coordinate_features": list(model.coordinate_features),
            "degree": model.degree,
            "alpha": model.alpha,
        },
    }
    buffer = io.BytesIO()
    joblib.dump(artifact, buffer, compress=3)
    return buffer.getvalue()


def load_regression_model(
    source: str | Path | bytes | BinaryIO,
) -> tuple[RegressionModel, dict[str, Any]]:
    """Load a trusted regression artifact and return its provenance.

    Joblib and pickle files can execute arbitrary code while loading. Callers must only pass
    files from trusted sources.
    """

    if isinstance(source, bytes):
        source = io.BytesIO(source)
    elif hasattr(source, "seek"):
        source.seek(0)
    try:
        payload = joblib.load(source)
    except Exception as exc:  # noqa: BLE001 - normalize artifact errors for UI callers
        raise ValueError(f"The regression model file could not be loaded: {exc}") from exc

    if (
        not isinstance(payload, Mapping)
        or payload.get("kind") != REGRESSION_MODEL_ARTIFACT_KIND
    ):
        raise ValueError("The selected file is not a supported regression model artifact.")
    if payload.get("version") != REGRESSION_MODEL_ARTIFACT_VERSION:
        raise ValueError(
            f"Unsupported regression artifact version: {payload.get('version')!r}."
        )
    model = payload.get("model")
    feature_count = validate_regression_model(model)
    if payload.get("feature_count") != feature_count:
        raise ValueError("The regression artifact's feature count does not match its model.")
    metadata = payload.get("metadata", {})
    if not isinstance(metadata, Mapping):
        raise ValueError("The regression artifact contains invalid provenance metadata.")
    provenance = {
        **dict(metadata),
        "artifact_kind": REGRESSION_MODEL_ARTIFACT_KIND,
        "artifact_version": REGRESSION_MODEL_ARTIFACT_VERSION,
        "sklearn_version": payload.get("sklearn_version"),
        "feature_count": feature_count,
    }
    return model, provenance


def evaluate_regression_model(
    model: RegressionModel,
    coordinates: np.ndarray,
    target: np.ndarray,
) -> dict[str, Any]:
    """Score a loaded regression model on the current points.

    Every supplied point is unseen from the artifact's perspective only if it was not part of
    the model's original training set, which this function cannot verify. The caller must say
    so in the UI rather than presenting these as held-out scores.
    """

    coordinates = np.asarray(coordinates, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    usable = np.isfinite(target) & np.isfinite(coordinates).all(axis=1)
    if usable.sum() < 2:
        raise ValueError(
            "At least two points with a finite time horizon are required to score a model."
        )
    predicted = model.predict(coordinates[usable])
    r2, rmse = _scores(target[usable], predicted)
    return {
        "r2": r2,
        "rmse": rmse,
        "points": int(usable.sum()),
        "excluded_points": int((~usable).sum()),
    }
