"""Learn a single-point transform that contracts equal-horizon variation.

The transform is fitted with supervision from ``log10_time_horizon_months``,
but inference requires only ``PLS1``, ``PLS2``, ``PLS3``, and
``reconstruction_residual_rms``.

The input feature vector is

    x = [PLS1, PLS2, PLS3, rho, rho**2, ..., rho**degree]

where rho is reconstruction residual RMS. A regularized Fisher metric learns
three directions that maximize variation between horizon centroids relative to
variation among points with the same horizon. A similarity-Procrustes map then
returns the three learned coordinates to the original PLS units, orientation,
and centroid. This avoids a meaningless improvement from globally shrinking
every coordinate.

Fit and use the transform from Python::

    frame = pd.read_csv("input.csv")
    transform, fit_metadata, _ = fit_metric_transform(frame)

    transform, artifact = load_metric_transform("equivalent_horizon_metric.joblib")
    z_new = transform.transform([pls1, pls2, pls3], residual_rms)
"""

from __future__ import annotations

from io import BytesIO
from dataclasses import dataclass
from pathlib import Path
from typing import Any, BinaryIO

import joblib
import numpy as np
import pandas as pd
from scipy.linalg import eigh
from sklearn.model_selection import GroupKFold


PLS_COLUMNS = ("PLS1", "PLS2", "PLS3")
RESIDUAL_COLUMN = "reconstruction_residual_rms"
HORIZON_COLUMN = "log10_time_horizon_months"
TRANSFORMED_COLUMNS = tuple(f"{name}_metric" for name in PLS_COLUMNS)
DEFAULT_RESIDUAL_DEGREE = 6
DEFAULT_RIDGE = 1e-6


@dataclass(frozen=True)
class EquivalentHorizonMetricTransform:
    """Fitted three-dimensional equivalent-horizon metric transform."""

    residual_degree: int
    feature_mean: np.ndarray
    feature_std: np.ndarray
    projection: np.ndarray
    latent_mean: np.ndarray
    output_mean: np.ndarray
    rotation: np.ndarray
    similarity_scale: float
    residual_rms_min: float
    residual_rms_max: float

    def _validate(self, pls_scores, residual_rms):
        scores = np.asarray(pls_scores, dtype=np.float64)
        residual = np.asarray(residual_rms, dtype=np.float64)
        single = scores.ndim == 1
        if single:
            scores = scores[None, :]
        if scores.ndim != 2 or scores.shape[1] != 3:
            raise ValueError("pls_scores must have shape (3,) or (n, 3)")
        if residual.ndim == 0:
            residual = residual.reshape(1)
        residual = residual.reshape(-1)
        if len(residual) == 1 and len(scores) != 1:
            residual = np.repeat(residual, len(scores))
        if len(residual) != len(scores):
            raise ValueError("residual_rms must be scalar or have one value per row")
        if not np.isfinite(scores).all() or not np.isfinite(residual).all():
            raise ValueError("Inputs must contain only finite values")
        return scores, residual, single

    def _features(self, scores, residual):
        powers = [residual[:, None] ** power for power in range(1, self.residual_degree + 1)]
        return np.column_stack([scores, *powers])

    def transform(self, pls_scores, residual_rms, *, clip: bool = True):
        """Transform one point or a batch without task, folder, or horizon."""
        scores, residual, single = self._validate(pls_scores, residual_rms)
        if clip:
            residual = np.clip(
                residual, self.residual_rms_min, self.residual_rms_max
            )
        features = self._features(scores, residual)
        standardized = (features - self.feature_mean) / self.feature_std
        latent = standardized @ self.projection
        transformed = self.output_mean + self.similarity_scale * (
            latent - self.latent_mean
        ) @ self.rotation
        return transformed[0] if single else transformed

    def transform_dataframe(
        self,
        frame: pd.DataFrame,
        *,
        clip: bool = True,
    ) -> pd.DataFrame:
        """Return a copy with ``PLS1_metric``--``PLS3_metric`` appended."""
        result = frame.copy()
        _require_columns(result, (*PLS_COLUMNS, RESIDUAL_COLUMN))
        transformed = self.transform(
            result.loc[:, PLS_COLUMNS].to_numpy(dtype=np.float64),
            result[RESIDUAL_COLUMN].to_numpy(dtype=np.float64),
            clip=clip,
        )
        for index, column in enumerate(TRANSFORMED_COLUMNS):
            result[column] = transformed[:, index]
        return result


def _require_columns(frame: pd.DataFrame, columns) -> None:
    missing = sorted(set(columns).difference(frame.columns))
    if missing:
        raise ValueError(f"Missing required columns: {missing}")


def _features(scores, residual, degree):
    powers = [residual[:, None] ** power for power in range(1, degree + 1)]
    return np.column_stack([scores, *powers])


def _horizon_labels(horizon):
    # The input values are exact in the supplied table; rounding only protects
    # against harmless CSV floating-point round trips.
    return np.round(np.asarray(horizon, dtype=np.float64), 12)


def _scatter_matrices(features, horizon):
    """Equal-horizon-weighted within and between scatter matrices."""
    labels = np.unique(horizon)
    dimension = features.shape[1]
    within = np.zeros((dimension, dimension), dtype=np.float64)
    centroids = []
    for label in labels:
        points = features[horizon == label]
        centroid = points.mean(axis=0)
        centroids.append(centroid)
        centered = points - centroid
        within += centered.T @ centered / len(points)
    within /= len(labels)
    centroids = np.asarray(centroids)
    centered_centroids = centroids - centroids.mean(axis=0)
    between = centered_centroids.T @ centered_centroids / len(labels)
    return within, between


def _fit_similarity(latent, target):
    latent_mean = latent.mean(axis=0)
    output_mean = target.mean(axis=0)
    centered_latent = latent - latent_mean
    centered_target = target - output_mean
    left, singular_values, right_transpose = np.linalg.svd(
        centered_latent.T @ centered_target,
        full_matrices=False,
    )
    rotation = left @ right_transpose
    similarity_scale = singular_values.sum() / np.square(centered_latent).sum()
    return latent_mean, output_mean, rotation, float(similarity_scale)


def fit_metric_transform(
    frame: pd.DataFrame,
    *,
    residual_degree: int = DEFAULT_RESIDUAL_DEGREE,
    ridge: float = DEFAULT_RIDGE,
) -> tuple[EquivalentHorizonMetricTransform, dict[str, Any], pd.DataFrame]:
    """Fit the supervised metric; inference remains target- and group-free."""
    if residual_degree < 1:
        raise ValueError("residual_degree must be at least 1")
    if not np.isfinite(ridge) or ridge < 0:
        raise ValueError("ridge must be finite and non-negative")
    fit_frame = frame.copy().reset_index(drop=True)
    _require_columns(
        fit_frame,
        (*PLS_COLUMNS, RESIDUAL_COLUMN, HORIZON_COLUMN),
    )
    required_numeric = (*PLS_COLUMNS, RESIDUAL_COLUMN, HORIZON_COLUMN)
    if not np.isfinite(fit_frame.loc[:, required_numeric].to_numpy(float)).all():
        raise ValueError("Required numeric columns contain NaN or infinite values")

    scores = fit_frame.loc[:, PLS_COLUMNS].to_numpy(dtype=np.float64)
    residual = fit_frame[RESIDUAL_COLUMN].to_numpy(dtype=np.float64)
    horizon = _horizon_labels(fit_frame[HORIZON_COLUMN])
    features = _features(scores, residual, residual_degree)
    feature_mean = features.mean(axis=0)
    feature_std = features.std(axis=0)
    feature_std[feature_std < 1e-12] = 1.0
    standardized = (features - feature_mean) / feature_std

    within, between = _scatter_matrices(standardized, horizon)
    eigenvalues, eigenvectors = eigh(
        between,
        within + ridge * np.eye(within.shape[0]),
    )
    order = np.argsort(eigenvalues)[::-1]
    projection = eigenvectors[:, order[:3]]
    projection /= np.linalg.norm(projection, axis=0, keepdims=True)
    leading_eigenvalues = eigenvalues[order[:3]]

    latent = standardized @ projection
    latent_mean, output_mean, rotation, similarity_scale = _fit_similarity(
        latent, scores
    )
    transform = EquivalentHorizonMetricTransform(
        residual_degree=residual_degree,
        feature_mean=feature_mean,
        feature_std=feature_std,
        projection=projection,
        latent_mean=latent_mean,
        output_mean=output_mean,
        rotation=rotation,
        similarity_scale=similarity_scale,
        residual_rms_min=float(residual.min()),
        residual_rms_max=float(residual.max()),
    )
    fit_metadata = {
        "residual_degree": residual_degree,
        "ridge": ridge,
        "leading_generalized_eigenvalues": leading_eigenvalues.tolist(),
        "input_rows": int(len(frame)),
        "fit_rows": int(len(fit_frame)),
        "folders": fit_frame["source_folder"].value_counts().to_dict()
        if "source_folder" in fit_frame
        else {},
    }
    return transform, fit_metadata, fit_frame


def fit_coordinate_metric_transform(
    frame: pd.DataFrame,
    coordinate_columns: tuple[str, str, str],
    *,
    residual_degree: int = DEFAULT_RESIDUAL_DEGREE,
    ridge: float = DEFAULT_RIDGE,
) -> tuple[
    pd.DataFrame,
    EquivalentHorizonMetricTransform,
    dict[str, Any],
    dict[str, dict[str, float]],
]:
    """Fit and apply the metric to any named three-dimensional coordinate basis."""
    if len(set(coordinate_columns)) != 3:
        raise ValueError("coordinate_columns must contain three distinct column names")
    _require_columns(
        frame,
        (*coordinate_columns, RESIDUAL_COLUMN, HORIZON_COLUMN),
    )

    fit_frame = pd.DataFrame(
        {
            **{
                target: pd.to_numeric(frame[source], errors="coerce")
                for target, source in zip(PLS_COLUMNS, coordinate_columns, strict=True)
            },
            RESIDUAL_COLUMN: pd.to_numeric(frame[RESIDUAL_COLUMN], errors="coerce"),
            HORIZON_COLUMN: pd.to_numeric(frame[HORIZON_COLUMN], errors="coerce"),
        }
    )
    if "source_folder" in frame:
        fit_frame["source_folder"] = frame["source_folder"].to_numpy()

    transform, fit_metadata, _ = fit_metric_transform(
        fit_frame,
        residual_degree=residual_degree,
        ridge=ridge,
    )
    coordinates = fit_frame.loc[:, PLS_COLUMNS].to_numpy(dtype=np.float64)
    residual = fit_frame[RESIDUAL_COLUMN].to_numpy(dtype=np.float64)
    horizon = _horizon_labels(fit_frame[HORIZON_COLUMN])
    transformed_coordinates = transform.transform(coordinates, residual)

    result = frame.copy()
    output_columns = tuple(f"{column}_metric" for column in coordinate_columns)
    for index, output_column in enumerate(output_columns):
        result[output_column] = transformed_coordinates[:, index]

    before = _distance_metrics(coordinates, horizon)
    after = _distance_metrics(transformed_coordinates, horizon)
    improvement = {
        key: (
            1.0 - after[key] / before[key]
            if np.isfinite(before[key])
            and np.isfinite(after[key])
            and abs(before[key]) > np.finfo(np.float64).eps
            else float("nan")
        )
        for key in before
    }
    fit_metadata = {
        **fit_metadata,
        "coordinate_columns": list(coordinate_columns),
        "output_columns": list(output_columns),
    }
    diagnostics = {"before": before, "after": after, "improvement": improvement}
    return result, transform, fit_metadata, diagnostics


def apply_coordinate_metric_transform(
    frame: pd.DataFrame,
    transform: EquivalentHorizonMetricTransform,
    coordinate_columns: tuple[str, str, str],
    *,
    output_columns: tuple[str, str, str] | None = None,
    clip: bool = True,
) -> pd.DataFrame:
    """Apply a fitted metric to named coordinates without refitting it."""

    if len(set(coordinate_columns)) != 3:
        raise ValueError("coordinate_columns must contain three distinct column names")
    if output_columns is None:
        output_columns = tuple(f"{column}_metric" for column in coordinate_columns)
    if len(set(output_columns)) != 3:
        raise ValueError("output_columns must contain three distinct column names")
    collisions = sorted(set(output_columns).intersection(frame.columns))
    if collisions:
        raise ValueError(f"Metric output columns already exist: {collisions}")
    _require_columns(frame, (*coordinate_columns, RESIDUAL_COLUMN))

    coordinates = frame.loc[:, coordinate_columns].apply(
        pd.to_numeric, errors="coerce"
    ).to_numpy(dtype=np.float64)
    residual = pd.to_numeric(frame[RESIDUAL_COLUMN], errors="coerce").to_numpy(
        dtype=np.float64
    )
    if not np.isfinite(coordinates).all() or not np.isfinite(residual).all():
        raise ValueError("Metric input coordinates and residual RMS must be finite")

    transformed = transform.transform(coordinates, residual, clip=clip)
    result = frame.copy()
    for index, output_column in enumerate(output_columns):
        result[output_column] = transformed[:, index]
    return result


def coordinate_metric_diagnostics(
    frame: pd.DataFrame,
    coordinate_columns: tuple[str, str, str],
    output_columns: tuple[str, str, str],
) -> dict[str, dict[str, float]]:
    """Compare equal-horizon geometry before and after an applied transform."""

    _require_columns(frame, (*coordinate_columns, *output_columns, HORIZON_COLUMN))
    before_coordinates = frame.loc[:, coordinate_columns].apply(
        pd.to_numeric, errors="coerce"
    ).to_numpy(dtype=np.float64)
    after_coordinates = frame.loc[:, output_columns].apply(
        pd.to_numeric, errors="coerce"
    ).to_numpy(dtype=np.float64)
    horizon = _horizon_labels(pd.to_numeric(frame[HORIZON_COLUMN], errors="coerce"))
    if not (
        np.isfinite(before_coordinates).all()
        and np.isfinite(after_coordinates).all()
        and np.isfinite(horizon).all()
    ):
        raise ValueError("Metric diagnostics require finite coordinates and horizons")

    before = _distance_metrics(before_coordinates, horizon)
    after = _distance_metrics(after_coordinates, horizon)
    improvement = {
        key: (
            1.0 - after[key] / before[key]
            if np.isfinite(before[key])
            and np.isfinite(after[key])
            and abs(before[key]) > np.finfo(np.float64).eps
            else float("nan")
        )
        for key in before
    }
    return {"before": before, "after": after, "improvement": improvement}


def _distance_metrics(coordinates, horizon):
    """Scale-aware metrics, with equal weight assigned to every horizon."""
    labels = np.unique(horizon)
    centroids = []
    pairwise_mean_squared = []
    for label in labels:
        points = coordinates[horizon == label]
        centroid = points.mean(axis=0)
        centroids.append(centroid)
        if len(points) > 1:
            scatter = np.mean(np.sum(np.square(points - centroid), axis=1))
            # Mean squared distance over distinct unordered pairs.
            pairwise_mean_squared.append(
                2.0 * len(points) / (len(points) - 1.0) * scatter
            )
    centroids = np.asarray(centroids)
    equal_horizon_pairwise_rms = (
        float(np.sqrt(np.mean(pairwise_mean_squared)))
        if pairwise_mean_squared
        else float("nan")
    )
    horizon_centroid_rms = float(
        np.sqrt(
            np.mean(
                np.sum(
                    np.square(centroids - centroids.mean(axis=0)), axis=1
                )
            )
        )
    )
    return {
        "equal_horizon_pairwise_rms": equal_horizon_pairwise_rms,
        "horizon_centroid_rms": horizon_centroid_rms,
        "normalized_equal_horizon_rms": (
            equal_horizon_pairwise_rms / horizon_centroid_rms
            if np.isfinite(equal_horizon_pairwise_rms)
            and horizon_centroid_rms > np.finfo(np.float64).eps
            else float("nan")
        ),
    }


def _trajectory_distortion(before, after, trajectory, horizon):
    step_ratios = []
    distance_correlations = []
    for group in np.unique(trajectory):
        mask = trajectory == group
        order = np.argsort(horizon[mask])
        group_before = before[mask][order]
        group_after = after[mask][order]
        if len(group_before) < 2:
            continue
        before_steps = np.linalg.norm(np.diff(group_before, axis=0), axis=1)
        after_steps = np.linalg.norm(np.diff(group_after, axis=0), axis=1)
        valid = before_steps > 1e-12
        step_ratios.extend((after_steps[valid] / before_steps[valid]).tolist())
        if len(group_before) >= 4:
            first, second = np.triu_indices(len(group_before), 1)
            before_distances = np.linalg.norm(
                group_before[first] - group_before[second], axis=1
            )
            after_distances = np.linalg.norm(
                group_after[first] - group_after[second], axis=1
            )
            if np.std(before_distances) > 0 and np.std(after_distances) > 0:
                distance_correlations.append(
                    np.corrcoef(before_distances, after_distances)[0, 1]
                )
    step_ratios = np.asarray(step_ratios)
    return {
        "median_step_length_ratio": float(np.median(step_ratios)),
        "median_absolute_step_length_change": float(
            np.median(np.abs(step_ratios - 1.0))
        ),
        "median_within_trajectory_distance_correlation": float(
            np.median(distance_correlations)
        ),
    }


def _cross_folder_centroid_rms(frame, coordinates, horizon):
    folders = frame["source_folder"].astype(str).to_numpy()
    squared_distances = []
    for label in np.unique(horizon):
        present = np.unique(folders[horizon == label])
        centroids = [
            coordinates[(horizon == label) & (folders == folder)].mean(axis=0)
            for folder in present
        ]
        for first in range(len(centroids)):
            for second in range(first + 1, len(centroids)):
                squared_distances.append(
                    np.sum(np.square(centroids[first] - centroids[second]))
                )
    return {
        "pair_count": len(squared_distances),
        "rms": float(np.sqrt(np.mean(squared_distances))),
    }


def validate_transform(
    frame: pd.DataFrame,
    *,
    residual_degree: int = DEFAULT_RESIDUAL_DEGREE,
    ridge: float = DEFAULT_RIDGE,
    folds: int = 5,
) -> dict[str, Any]:
    """Hold out complete tasks, including all their source-folder variants."""
    analysis_frame = frame.copy().reset_index(drop=True)
    _require_columns(
        analysis_frame,
        (*PLS_COLUMNS, RESIDUAL_COLUMN, HORIZON_COLUMN, "task", "source_folder"),
    )
    scores = analysis_frame.loc[:, PLS_COLUMNS].to_numpy(dtype=np.float64)
    residual = analysis_frame[RESIDUAL_COLUMN].to_numpy(dtype=np.float64)
    horizon = _horizon_labels(analysis_frame[HORIZON_COLUMN])
    task = analysis_frame["task"].astype(str).to_numpy()
    trajectory = (
        analysis_frame["source_folder"].astype(str)
        + "||"
        + analysis_frame["task"].astype(str)
    ).to_numpy()
    fold_count = min(folds, len(np.unique(task)))
    oof = np.empty_like(scores)
    fold_metrics = []
    fold_baselines = []

    for train, test in GroupKFold(fold_count).split(scores, groups=task):
        train_frame = analysis_frame.iloc[train].reset_index(drop=True)
        transform, _, _ = fit_metric_transform(
            train_frame,
            residual_degree=residual_degree,
            ridge=ridge,
        )
        oof[test] = transform.transform(scores[test], residual[test])
        fold_metrics.append(
            {
                **_distance_metrics(oof[test], horizon[test]),
                **_trajectory_distortion(
                    scores[test], oof[test], trajectory[test], horizon[test]
                ),
            }
        )
        fold_baselines.append(_distance_metrics(scores[test], horizon[test]))

    def average(items):
        return {
            key: float(np.mean([item[key] for item in items]))
            for key in items[0]
        }

    fold_mean = average(fold_metrics)
    fold_baseline = average(fold_baselines)
    combined = {
        **_distance_metrics(oof, horizon),
        **_trajectory_distortion(scores, oof, trajectory, horizon),
    }
    by_folder = {}
    folders = analysis_frame["source_folder"].astype(str).to_numpy()
    for folder in sorted(np.unique(folders)):
        mask = folders == folder
        by_folder[folder] = {
            "before": _distance_metrics(scores[mask], horizon[mask]),
            "after": _distance_metrics(oof[mask], horizon[mask]),
        }

    return {
        "protocol": f"{fold_count}-fold GroupKFold; complete tasks held out",
        "residual_degree": residual_degree,
        "ridge": ridge,
        "input_rows": int(len(frame)),
        "analysis_rows": int(len(analysis_frame)),
        "folders_analyzed": analysis_frame["source_folder"].value_counts().to_dict(),
        "fold_mean_baseline": fold_baseline,
        "fold_mean_transformed": fold_mean,
        "fold_mean_improvement": {
            "equal_horizon_pairwise_rms": 1.0
            - fold_mean["equal_horizon_pairwise_rms"]
            / fold_baseline["equal_horizon_pairwise_rms"],
            "normalized_equal_horizon_rms": 1.0
            - fold_mean["normalized_equal_horizon_rms"]
            / fold_baseline["normalized_equal_horizon_rms"],
        },
        "combined_out_of_fold": combined,
        "full_data_baseline": _distance_metrics(scores, horizon),
        "by_folder_combined_out_of_fold": by_folder,
        "cross_folder_horizon_centroid_distance": {
            "before": _cross_folder_centroid_rms(analysis_frame, scores, horizon),
            "after": _cross_folder_centroid_rms(analysis_frame, oof, horizon),
        },
        "note": (
            "The normalized metric divides equal-horizon RMS distance by the "
            "RMS spread of horizon centroids, preventing trivial global shrinkage."
        ),
    }


def _metric_transform_artifact(
    transform: EquivalentHorizonMetricTransform,
    fit_metadata: dict[str, Any],
    validation: dict[str, Any] | None = None,
) -> dict[str, Any]:
    coordinate_columns = list(fit_metadata.get("coordinate_columns", PLS_COLUMNS))
    output_columns = list(
        fit_metadata.get(
            "output_columns",
            [f"{column}_metric" for column in coordinate_columns],
        )
    )
    return {
        "kind": "temporal-manifolds.equivalent-horizon-metric",
        "version": 1,
        "parameters": {
            "residual_degree": transform.residual_degree,
            "feature_mean": transform.feature_mean,
            "feature_std": transform.feature_std,
            "projection": transform.projection,
            "latent_mean": transform.latent_mean,
            "output_mean": transform.output_mean,
            "rotation": transform.rotation,
            "similarity_scale": transform.similarity_scale,
            "residual_rms_min": transform.residual_rms_min,
            "residual_rms_max": transform.residual_rms_max,
        },
        "metadata": {
            **fit_metadata,
            "coordinate_columns": coordinate_columns,
            "input_columns": [*coordinate_columns, RESIDUAL_COLUMN],
            "output_columns": output_columns,
            "inference_uses": [*coordinate_columns, RESIDUAL_COLUMN],
            "inference_does_not_use": [
                "task",
                "source_folder",
                HORIZON_COLUMN,
                "other trajectory points",
            ],
        },
        "validation": validation or {},
    }


def serialize_metric_transform(
    transform: EquivalentHorizonMetricTransform,
    fit_metadata: dict[str, Any],
    validation: dict[str, Any] | None = None,
) -> bytes:
    """Serialize a fitted metric to an in-memory joblib artifact."""

    buffer = BytesIO()
    joblib.dump(_metric_transform_artifact(transform, fit_metadata, validation), buffer)
    return buffer.getvalue()


def save_metric_transform(
    path: str | Path,
    transform: EquivalentHorizonMetricTransform,
    fit_metadata: dict[str, Any],
    validation: dict[str, Any] | None = None,
) -> None:
    """Save a plain dictionary so loading does not depend on this class path."""

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(_metric_transform_artifact(transform, fit_metadata, validation), path)


def load_metric_transform(
    source: str | Path | BinaryIO | bytes,
) -> tuple[EquivalentHorizonMetricTransform, dict[str, Any]]:
    """Load and validate an equivalent-horizon metric artifact."""

    payload = BytesIO(source) if isinstance(source, bytes) else source
    artifact = joblib.load(payload)
    if not isinstance(artifact, dict) or artifact.get("kind") != (
        "temporal-manifolds.equivalent-horizon-metric"
    ):
        raise ValueError("Unrecognized equivalent-horizon metric artifact")
    if artifact.get("version") != 1:
        raise ValueError("Unsupported equivalent-horizon metric artifact version")
    try:
        parameters = artifact["parameters"]
        transform = EquivalentHorizonMetricTransform(
            residual_degree=int(parameters["residual_degree"]),
            feature_mean=np.asarray(parameters["feature_mean"], dtype=np.float64),
            feature_std=np.asarray(parameters["feature_std"], dtype=np.float64),
            projection=np.asarray(parameters["projection"], dtype=np.float64),
            latent_mean=np.asarray(parameters["latent_mean"], dtype=np.float64),
            output_mean=np.asarray(parameters["output_mean"], dtype=np.float64),
            rotation=np.asarray(parameters["rotation"], dtype=np.float64),
            similarity_scale=float(parameters["similarity_scale"]),
            residual_rms_min=float(parameters["residual_rms_min"]),
            residual_rms_max=float(parameters["residual_rms_max"]),
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("Invalid equivalent-horizon metric parameters") from exc

    feature_count = 3 + transform.residual_degree
    arrays_and_shapes = (
        (transform.feature_mean, (feature_count,)),
        (transform.feature_std, (feature_count,)),
        (transform.projection, (feature_count, 3)),
        (transform.latent_mean, (3,)),
        (transform.output_mean, (3,)),
        (transform.rotation, (3, 3)),
    )
    if transform.residual_degree < 1 or any(
        array.shape != shape or not np.isfinite(array).all()
        for array, shape in arrays_and_shapes
    ):
        raise ValueError("Invalid equivalent-horizon metric parameter shapes")
    if np.any(transform.feature_std <= 0):
        raise ValueError("Equivalent-horizon feature scales must be positive")
    scalar_parameters = (
        transform.similarity_scale,
        transform.residual_rms_min,
        transform.residual_rms_max,
    )
    if not np.isfinite(scalar_parameters).all() or transform.similarity_scale <= 0:
        raise ValueError("Invalid equivalent-horizon metric scalar parameters")
    if transform.residual_rms_min > transform.residual_rms_max:
        raise ValueError("Invalid equivalent-horizon residual range")

    metadata = artifact.get("metadata", {})
    if not isinstance(metadata, dict):
        raise ValueError("Invalid equivalent-horizon metric metadata")
    coordinate_columns = metadata.get("coordinate_columns", PLS_COLUMNS)
    output_columns = metadata.get("output_columns", TRANSFORMED_COLUMNS)
    if (
        not isinstance(coordinate_columns, (list, tuple))
        or len(coordinate_columns) != 3
        or not all(isinstance(column, str) for column in coordinate_columns)
        or not isinstance(output_columns, (list, tuple))
        or len(output_columns) != 3
        or not all(isinstance(column, str) for column in output_columns)
    ):
        raise ValueError("Invalid equivalent-horizon coordinate metadata")
    return transform, artifact
