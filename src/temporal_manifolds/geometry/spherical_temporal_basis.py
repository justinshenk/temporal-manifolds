"""Spherical temporal basis from ``spherical_temporal_basis_no_conv.ipynb``."""

from __future__ import annotations

import io
import json
from collections.abc import Mapping
from typing import Any

import numpy as np
import pandas as pd
from scipy import linalg

MODEL_KIND = "temporal_manifolds.spherical_temporal_basis"
MODEL_VERSION = 3
LEGACY_MODEL_VERSIONS = frozenset({2})
SOURCE_COLUMN = "source_folder"
TIME_COLUMN = "time_horizon_months"
TASK_COLUMN = "task"
FEATURES = (
    "PLS1",
    "PLS2",
    "PLS3",
    "reconstruction_residual_PC1",
    "reconstruction_residual_PC2",
    "reconstruction_residual_PC3",
)
OUTPUT_COLUMNS = (
    "spherical_time_1",
    "spherical_time_2",
    "spherical_log_radius",
)
DEFAULT_PARAMETERS = {
    "within_weight": 0.5,
    "pair_weight": 5.0,
    # ``local_weight`` now penalizes local curvature (second differences)
    # instead of rewarding first-difference energy. Rewarding first
    # differences can select noisy, zigzagging trajectories.
    "local_weight": 0.02,
    # Explicitly makes the selected angular subspace predictive of log-time.
    "time_order_weight": 1.0,
    "ridge_fraction": 0.10,
}
LEGACY_PARAMETERS = frozenset({"within_weight", "pair_weight", "local_weight", "ridge_fraction"})


def _scatter(rows: np.ndarray) -> np.ndarray:
    rows = np.asarray(rows, dtype=np.float64)
    if rows.ndim != 2 or len(rows) == 0:
        raise ValueError("Scatter requires a non-empty two-dimensional array.")
    return rows.T @ rows / len(rows)


def _common_horizons(frame: pd.DataFrame, sources: list[str]) -> list[float]:
    horizon_sets = [
        set(frame.loc[frame[SOURCE_COLUMN] == source, TIME_COLUMN].unique()) for source in sources
    ]
    return sorted(set.intersection(*horizon_sets)) if horizon_sets else []


def _balanced_row_weights(frame: pd.DataFrame) -> np.ndarray:
    sizes = frame.groupby([SOURCE_COLUMN, TIME_COLUMN], observed=True)[SOURCE_COLUMN].transform(
        "size"
    )
    weights = 1.0 / sizes.to_numpy(dtype=np.float64)
    return weights / weights.sum()


def _fit_preprocessor(frame: pd.DataFrame) -> dict[str, Any]:
    values = frame.loc[:, FEATURES].to_numpy(dtype=np.float64)
    weights = _balanced_row_weights(frame)
    center = np.average(values, axis=0, weights=weights)
    centered = values - center
    scale = np.sqrt(np.maximum(np.average(centered**2, axis=0, weights=weights), 1e-12))
    sphere_matrix = np.diag(1.0 / scale)
    radius = np.linalg.norm(centered @ sphere_matrix, axis=1)
    log_radius = np.log(np.maximum(radius, 1e-12))
    radius_center = float(np.average(log_radius, weights=weights))
    radius_scale = max(
        float(np.sqrt(np.average((log_radius - radius_center) ** 2, weights=weights))),
        1e-8,
    )
    return {
        "center": center,
        "scale": scale,
        "sphere_matrix": sphere_matrix,
        "radius_center": radius_center,
        "radius_scale": radius_scale,
    }


def _pretransform(values: np.ndarray, preprocessor: Mapping[str, Any]) -> np.ndarray:
    centered = np.asarray(values, dtype=np.float64) - preprocessor["center"]
    q = centered @ preprocessor["sphere_matrix"]
    radius = np.linalg.norm(q, axis=-1, keepdims=True)
    direction = q / np.maximum(radius, 1e-12)
    log_radius = np.log(np.maximum(radius, 1e-12))
    standardized_radius = (log_radius - preprocessor["radius_center"]) / preprocessor[
        "radius_scale"
    ]
    return np.concatenate([direction, standardized_radius], axis=-1)


def _cell_means(
    frame: pd.DataFrame,
    values: np.ndarray,
    horizons: list[float],
    sources: list[str],
) -> np.ndarray:
    work = frame[[TIME_COLUMN, SOURCE_COLUMN]].reset_index(drop=True).copy()
    value_columns = [f"v{index}" for index in range(values.shape[1])]
    work[value_columns] = values
    grouped = work.groupby([TIME_COLUMN, SOURCE_COLUMN], observed=True)[value_columns].mean()
    return np.stack(
        [
            np.stack([grouped.loc[(horizon, source)].to_numpy() for source in sources])
            for horizon in horizons
        ]
    )


def _prepare_statistics(
    frame: pd.DataFrame, preprocessor: Mapping[str, Any], horizons: list[float]
) -> dict[str, Any]:
    """Build alignment statistics on shared horizons and time statistics on all.

    Folder, within-horizon, and matched-task terms require comparable source
    support, so they use only horizons shared by every source. Temporal order
    and smoothness are computed separately for each source and then averaged,
    allowing a source with additional horizons to contribute without receiving
    extra source weight.
    """
    sources = sorted(frame[SOURCE_COLUMN].unique())
    shared_frame = frame.loc[frame[TIME_COLUMN].isin(horizons)].reset_index(drop=True)
    shared_values = _pretransform(
        shared_frame.loc[:, FEATURES].to_numpy(dtype=np.float64), preprocessor
    )
    means = _cell_means(shared_frame, shared_values, horizons, sources)
    pooled_means = means.mean(axis=1)
    offsets = means - pooled_means[:, None, :]
    folder_scatter = _scatter(offsets.reshape(-1, shared_values.shape[1]))

    within_scatters = []
    keys = shared_frame[[TIME_COLUMN, SOURCE_COLUMN]].reset_index(drop=True)
    for indices in keys.groupby([TIME_COLUMN, SOURCE_COLUMN], observed=True).groups.values():
        rows = shared_values[np.asarray(list(indices), dtype=int)]
        within_scatters.append(_scatter(rows - rows.mean(axis=0)))
    within_scatter = np.mean(within_scatters, axis=0)

    pair_rows: list[np.ndarray] = []
    pair_frame = (
        shared_frame[[TIME_COLUMN, SOURCE_COLUMN, TASK_COLUMN]].reset_index(drop=True).copy()
    )
    value_columns = [f"p{index}" for index in range(shared_values.shape[1])]
    pair_frame[value_columns] = shared_values
    for _, group in pair_frame.groupby([TIME_COLUMN, TASK_COLUMN], observed=True):
        source_means = group.groupby(SOURCE_COLUMN, observed=True)[value_columns].mean().to_numpy()
        for left in range(len(source_means)):
            for right in range(left + 1, len(source_means)):
                pair_rows.append((source_means[left] - source_means[right]) / np.sqrt(2.0))
    pair_scatter = _scatter(np.asarray(pair_rows)) if pair_rows else np.zeros_like(folder_scatter)

    all_values = _pretransform(frame.loc[:, FEATURES].to_numpy(dtype=np.float64), preprocessor)
    all_curve_frame = frame[[TIME_COLUMN, SOURCE_COLUMN]].reset_index(drop=True).copy()
    all_value_columns = [f"a{index}" for index in range(all_values.shape[1])]
    all_curve_frame[all_value_columns] = all_values
    source_time_scatters = []
    source_order_scatters = []
    source_roughness_scatters = []
    for source in sources:
        curve = (
            all_curve_frame.loc[all_curve_frame[SOURCE_COLUMN] == source]
            .groupby(TIME_COLUMN, observed=True)[all_value_columns]
            .mean()
            .sort_index()
        )
        curve_values = curve.to_numpy()
        centered_curve = curve_values - curve_values.mean(axis=0)
        source_time_scatters.append(_scatter(centered_curve))

        log_time = np.log10(curve.index.to_numpy(dtype=np.float64))
        centered_time = log_time - log_time.mean()
        time_scale = max(float(np.sqrt(np.mean(centered_time**2))), 1e-12)
        standardized_time = centered_time / time_scale
        time_covariance = np.mean(centered_curve * standardized_time[:, None], axis=0)
        source_order_scatters.append(np.outer(time_covariance, time_covariance))

        if len(curve_values) > 1:
            source_spacing = np.diff(log_time)
            source_slopes = (
                np.diff(curve_values, axis=0) / np.maximum(source_spacing, 1e-12)[:, None]
            )
            if len(source_slopes) > 1:
                midpoint_spacing = (source_spacing[:-1] + source_spacing[1:]) / 2.0
                curvature = (
                    np.diff(source_slopes, axis=0) / np.maximum(midpoint_spacing, 1e-12)[:, None]
                )
                source_roughness_scatters.append(_scatter(curvature))

    zero_scatter = np.zeros_like(folder_scatter)

    return {
        "sources": sources,
        "horizons": horizons,
        "pooled_means": pooled_means,
        "folder": folder_scatter,
        "within": within_scatter,
        "pair": pair_scatter,
        "time": np.mean(source_time_scatters, axis=0),
        "time_order": np.mean(source_order_scatters, axis=0),
        "roughness": (
            np.mean(source_roughness_scatters, axis=0)
            if source_roughness_scatters
            else zero_scatter
        ),
    }


def _canonical_sign(vector: np.ndarray) -> np.ndarray:
    pivot = int(np.argmax(np.abs(vector)))
    return vector if vector[pivot] >= 0 else -vector


def _solve_basis(
    statistics: Mapping[str, Any],
    *,
    within_weight: float,
    pair_weight: float,
    local_weight: float,
    time_order_weight: float,
    ridge_fraction: float,
) -> tuple[np.ndarray, np.ndarray]:
    direction_dim = len(FEATURES)

    def trace_matched(matrix: np.ndarray, reference: np.ndarray) -> np.ndarray:
        matrix_trace = np.trace(matrix)
        if matrix_trace <= 1e-12:
            return matrix
        return matrix * (np.trace(reference) / matrix_trace)

    folder = statistics["folder"][:direction_dim, :direction_dim]
    within = statistics["within"][:direction_dim, :direction_dim]
    pair = statistics["pair"][:direction_dim, :direction_dim]
    temporal = statistics["time"][:direction_dim, :direction_dim]
    time_order = statistics["time_order"][:direction_dim, :direction_dim]
    roughness = statistics["roughness"][:direction_dim, :direction_dim]
    alignment = (
        folder
        + within_weight * trace_matched(within, folder)
        + pair_weight * trace_matched(pair, folder)
        + local_weight * trace_matched(roughness, folder)
    )
    temporal = temporal + time_order_weight * trace_matched(time_order, temporal)
    ridge_scale = max(float(np.trace(alignment)) / direction_dim, 1e-10)
    denominator = (alignment + alignment.T) / 2 + (
        ridge_fraction * ridge_scale * np.eye(direction_dim)
    )
    eigenvalues, eigenvectors = linalg.eigh((temporal + temporal.T) / 2, denominator)
    order = np.argsort(eigenvalues)[::-1]
    angular_axes, _ = np.linalg.qr(eigenvectors[:, order[:2]])

    direction_curve = statistics["pooled_means"][:, :direction_dim]
    centered_curve = direction_curve @ angular_axes
    centered_curve -= centered_curve.mean(axis=0)
    log_horizons = np.log10(np.asarray(statistics["horizons"], dtype=np.float64))
    time_vector, *_ = np.linalg.lstsq(
        centered_curve, log_horizons - log_horizons.mean(), rcond=None
    )
    if np.linalg.norm(time_vector) > 1e-12:
        first = time_vector / np.linalg.norm(time_vector)
        second = np.array([-first[1], first[0]])
        angular_axes = angular_axes @ np.column_stack([first, second])
    angular_axes[:, 1] = _canonical_sign(angular_axes[:, 1])

    basis = np.zeros((direction_dim + 1, 3))
    basis[:direction_dim, :2] = angular_axes
    basis[-1, 2] = 1.0
    return basis, eigenvalues[order]


def _validate_frame(frame: pd.DataFrame, *, fitting: bool) -> None:
    required = {*FEATURES}
    if fitting:
        required.update({SOURCE_COLUMN, TIME_COLUMN, TASK_COLUMN})
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"Spherical temporal basis requires columns: {', '.join(missing)}.")
    values = frame.loc[:, FEATURES].to_numpy(dtype=np.float64)
    if not np.isfinite(values).all():
        raise ValueError("Spherical temporal basis features must all be finite.")


def fit_spherical_temporal_basis(
    frame: pd.DataFrame,
    *,
    residual_pca_center: np.ndarray,
    residual_pca_components: np.ndarray,
    excluded_sources: tuple[str, ...] = ("conv",),
    within_weight: float = DEFAULT_PARAMETERS["within_weight"],
    pair_weight: float = DEFAULT_PARAMETERS["pair_weight"],
    local_weight: float = DEFAULT_PARAMETERS["local_weight"],
    time_order_weight: float = DEFAULT_PARAMETERS["time_order_weight"],
    ridge_fraction: float = DEFAULT_PARAMETERS["ridge_fraction"],
) -> dict[str, Any]:
    """Fit a radius-fixed spherical model with supervised log-time ordering.

    ``time_order_weight`` promotes an angular direction correlated with
    log-horizon. ``local_weight`` penalizes local curvature, which suppresses
    noisy horizon-to-horizon zigzags without forcing the trajectory to be a
    straight line. Labels are used only during fitting; transformation remains
    a fixed pointwise operation.
    """
    _validate_frame(frame, fitting=True)
    parameters = {
        "within_weight": float(within_weight),
        "pair_weight": float(pair_weight),
        "local_weight": float(local_weight),
        "time_order_weight": float(time_order_weight),
        "ridge_fraction": float(ridge_fraction),
    }
    if not all(np.isfinite(value) and value >= 0 for value in parameters.values()):
        raise ValueError("Spherical temporal basis parameters must be finite and non-negative.")
    residual_pca_center = np.asarray(residual_pca_center, dtype=np.float64)
    residual_pca_components = np.asarray(residual_pca_components, dtype=np.float64)
    if (
        residual_pca_center.ndim != 1
        or residual_pca_components.shape != (3, len(residual_pca_center))
        or not np.isfinite(residual_pca_center).all()
        or not np.isfinite(residual_pca_components).all()
    ):
        raise ValueError("Residual PCA parameters must contain three finite components.")

    excluded_sources = tuple(str(source) for source in excluded_sources)
    fit_frame = frame.loc[~frame[SOURCE_COLUMN].astype(str).isin(excluded_sources)].copy()
    if fit_frame.empty:
        raise ValueError("At least one row is required to fit the basis.")
    time = pd.to_numeric(fit_frame[TIME_COLUMN], errors="coerce").to_numpy(dtype=np.float64)
    if not np.isfinite(time).all() or np.any(time <= 0):
        raise ValueError("Time horizons must be finite and positive.")
    fit_frame[TIME_COLUMN] = time
    sources = sorted(fit_frame[SOURCE_COLUMN].unique())
    if len(sources) < 2:
        raise ValueError("At least two source folders are required to fit the basis.")
    horizons = _common_horizons(fit_frame, sources)
    if len(horizons) < 2:
        raise ValueError("At least two horizons shared by all sources are required.")
    shared_training = fit_frame.loc[fit_frame[TIME_COLUMN].isin(horizons)].reset_index(drop=True)
    preprocessor = _fit_preprocessor(shared_training)
    statistics = _prepare_statistics(fit_frame.reset_index(drop=True), preprocessor, horizons)
    basis, eigenvalues = _solve_basis(statistics, **parameters)
    return {
        "kind": MODEL_KIND,
        "version": MODEL_VERSION,
        "features": list(FEATURES),
        "output_columns": list(OUTPUT_COLUMNS),
        "preprocess_mode": "sphere_zscore",
        "time_mode": "all_source_curves",
        "trace_balance": True,
        "parameters": parameters,
        "center": preprocessor["center"],
        "scale": preprocessor["scale"],
        "sphere_matrix": preprocessor["sphere_matrix"],
        "radius_center": preprocessor["radius_center"],
        "radius_scale": preprocessor["radius_scale"],
        "basis": basis,
        "generalized_eigenvalues": eigenvalues,
        "residual_pca_center": residual_pca_center.copy(),
        "residual_pca_components": residual_pca_components.copy(),
        "training_sources": sources,
        "excluded_training_sources": list(excluded_sources),
        "training_horizons": horizons,
        "training_rows": len(fit_frame),
        "shared_training_rows": len(shared_training),
    }


def validate_spherical_temporal_basis(model: Any) -> dict[str, Any]:
    """Validate and normalize a fitted model or uploaded artifact payload."""
    if not isinstance(model, Mapping) or model.get("kind") != MODEL_KIND:
        raise ValueError("The selected file is not a spherical temporal basis model.")
    version = model.get("version")
    if version != MODEL_VERSION and version not in LEGACY_MODEL_VERSIONS:
        raise ValueError(f"Unsupported spherical temporal basis version: {model.get('version')!r}.")
    normalized = dict(model)
    if tuple(normalized.get("features", ())) != FEATURES:
        raise ValueError("The model uses an unsupported feature schema.")
    if tuple(normalized.get("output_columns", ())) != OUTPUT_COLUMNS:
        raise ValueError("The model uses an unsupported output schema.")
    shapes = {
        "center": (6,),
        "scale": (6,),
        "sphere_matrix": (6, 6),
        "basis": (7, 3),
        "generalized_eigenvalues": (6,),
    }
    for key, shape in shapes.items():
        array = np.asarray(normalized.get(key), dtype=np.float64)
        if array.shape != shape or not np.isfinite(array).all():
            raise ValueError(f"The model contains invalid {key!r} values.")
        normalized[key] = array
    residual_center = np.asarray(normalized.get("residual_pca_center"), dtype=np.float64)
    residual_components = np.asarray(normalized.get("residual_pca_components"), dtype=np.float64)
    if (
        residual_center.ndim != 1
        or residual_components.shape != (3, len(residual_center))
        or not np.isfinite(residual_center).all()
        or not np.isfinite(residual_components).all()
    ):
        raise ValueError("The model contains invalid residual PCA parameters.")
    normalized["residual_pca_center"] = residual_center
    normalized["residual_pca_components"] = residual_components
    for key in ("radius_center", "radius_scale"):
        value = float(normalized.get(key, np.nan))
        if not np.isfinite(value) or (key == "radius_scale" and value <= 0):
            raise ValueError(f"The model contains an invalid {key!r} value.")
        normalized[key] = value
    parameters = normalized.get("parameters")
    expected_parameters = (
        set(DEFAULT_PARAMETERS) if version == MODEL_VERSION else set(LEGACY_PARAMETERS)
    )
    if not isinstance(parameters, Mapping) or set(parameters) != expected_parameters:
        raise ValueError("The model contains invalid algorithm parameters.")
    normalized["parameters"] = {key: float(parameters[key]) for key in expected_parameters}
    if not all(np.isfinite(value) and value >= 0 for value in normalized["parameters"].values()):
        raise ValueError("The model contains invalid algorithm parameters.")
    return normalized


def transform_spherical_temporal_basis(
    frame: pd.DataFrame, model: Mapping[str, Any]
) -> pd.DataFrame:
    """Append the two angular coordinates and standardized log-radius coordinate."""
    _validate_frame(frame, fitting=False)
    model = validate_spherical_temporal_basis(model)
    preprocessor = {
        "center": model["center"],
        "sphere_matrix": model["sphere_matrix"],
        "radius_center": model["radius_center"],
        "radius_scale": model["radius_scale"],
    }
    values = frame.loc[:, FEATURES].to_numpy(dtype=np.float64)
    transformed = _pretransform(values, preprocessor) @ model["basis"]
    result = frame.copy()
    result.loc[:, OUTPUT_COLUMNS] = transformed
    return result


def serialize_spherical_temporal_basis(model: Mapping[str, Any]) -> bytes:
    """Serialize a validated model as a compressed, non-pickle NumPy artifact."""
    normalized = validate_spherical_temporal_basis(model)
    array_keys = {
        "center",
        "scale",
        "sphere_matrix",
        "basis",
        "generalized_eigenvalues",
        "residual_pca_center",
        "residual_pca_components",
    }
    metadata = {key: value for key, value in normalized.items() if key not in array_keys}
    buffer = io.BytesIO()
    np.savez_compressed(
        buffer,
        metadata=np.asarray(json.dumps(metadata, allow_nan=False)),
        **{key: normalized[key] for key in array_keys},
    )
    return buffer.getvalue()


def load_spherical_temporal_basis(source: bytes) -> dict[str, Any]:
    """Load and validate a compressed model artifact without enabling pickle."""
    try:
        with np.load(io.BytesIO(source), allow_pickle=False) as artifact:
            expected = {
                "metadata",
                "center",
                "scale",
                "sphere_matrix",
                "basis",
                "generalized_eigenvalues",
                "residual_pca_center",
                "residual_pca_components",
            }
            if set(artifact.files) != expected:
                raise ValueError("The artifact has an unsupported array schema.")
            payload = json.loads(str(artifact["metadata"].item()))
            payload.update({key: artifact[key].copy() for key in expected - {"metadata"}})
    except (OSError, ValueError, json.JSONDecodeError, TypeError) as exc:
        raise ValueError(f"The spherical temporal basis file could not be loaded: {exc}") from exc
    return validate_spherical_temporal_basis(payload)
