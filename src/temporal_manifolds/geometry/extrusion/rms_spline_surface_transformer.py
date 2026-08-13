"""Fit and apply an RMS-conditioned extrusion of a 3D parametric spline.

The surface is

    S(t, r) = C(t) + E(u(r))
    u(r)    = [log10(r) - log10(r_ref)] / q_scale
    E(u)    = sum_{k=1}^degree B_k u^k

where C(t) is the supplied 3D spline.  Because E(0) = 0, the original
spline is an exact edge of the fitted surface.  No source/dataset label is
used by the fitted transform.
"""

from __future__ import annotations

import argparse
import io
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, BinaryIO, Iterable

import joblib
import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar
from scipy.spatial import cKDTree
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score
from sklearn.model_selection import GroupKFold


COORDINATE_COLUMNS = ("PLS1", "PLS2", "PLS3")


def _load_curve_artifact(path: str | Path) -> dict[str, Any]:
    """Load the supplied artifact even when its container package is absent.

    Only the lightweight CurveModel container is replaced.  Its fitted SciPy
    spline predictors are loaded unchanged.
    """

    try:
        artifact = joblib.load(path)
    except ModuleNotFoundError as error:
        if not str(error.name).startswith("temporal_manifolds"):
            raise
        import joblib.numpy_pickle as numpy_pickle

        original_find_class = numpy_pickle.NumpyUnpickler.find_class

        class CurveModel:  # pragma: no cover - compatibility container
            pass

        def find_class(unpickler: Any, module: str, name: str) -> Any:
            if module == "temporal_manifolds.viz.curve_fitting" and name == "CurveModel":
                return CurveModel
            return original_find_class(unpickler, module, name)

        numpy_pickle.NumpyUnpickler.find_class = find_class
        try:
            artifact = joblib.load(path)
        finally:
            numpy_pickle.NumpyUnpickler.find_class = original_find_class

    if not isinstance(artifact, dict) or "model" not in artifact:
        raise TypeError("Expected a temporal_manifolds curve artifact dictionary")
    return artifact


def _poly_features(u: np.ndarray, degree: int) -> np.ndarray:
    u = np.asarray(u, dtype=float).reshape(-1)
    return np.column_stack([u**power for power in range(1, degree + 1)])


def _metric_block(
    true_t: np.ndarray,
    predicted_t: np.ndarray,
    original: np.ndarray,
    base_curve_at_true_t: np.ndarray,
    predicted_offset: np.ndarray,
    projected_surface: np.ndarray,
) -> dict[str, float]:
    displacement = original - base_curve_at_true_t
    offset_error = displacement - predicted_offset
    parameter_error = predicted_t - true_t
    denominator = float(np.sum(displacement**2))
    displacement_r2 = 1.0 - float(np.sum(offset_error**2)) / denominator
    return {
        "n": int(len(true_t)),
        "offset_rmse_per_coordinate": float(np.sqrt(np.mean(offset_error**2))),
        "displacement_r2_from_base_curve": displacement_r2,
        "projected_surface_rmse_per_coordinate": float(
            np.sqrt(np.mean((original - projected_surface) ** 2))
        ),
        "projected_surface_median_euclidean_distance": float(
            np.median(np.linalg.norm(original - projected_surface, axis=1))
        ),
        "parameter_r2": float(r2_score(true_t, predicted_t)),
        "parameter_rmse_log10_months": float(np.sqrt(np.mean(parameter_error**2))),
        "parameter_mae_log10_months": float(np.mean(np.abs(parameter_error))),
        "parameter_median_ae_log10_months": float(np.median(np.abs(parameter_error))),
        "parameter_correlation": float(np.corrcoef(true_t, predicted_t)[0, 1]),
    }


@dataclass
class RMSSplineSurfaceTransformer:
    """A source-agnostic polynomial extrusion around a fixed 3D spline."""

    predictors: tuple[Any, Any, Any]
    parameter_center: float
    parameter_scale: float
    curve_parameter_bounds: np.ndarray
    projection_parameter_bounds: np.ndarray
    coordinate_features: tuple[str, str, str]
    parameter_feature: str
    residual_feature: str
    residual_reference: float
    log_residual_scale: float
    residual_bounds: np.ndarray
    degree: int
    coefficients: np.ndarray
    ridge_alpha: float = 1e-3

    ARTIFACT_KIND = "rms_conditioned_spline_surface"
    ARTIFACT_VERSION = 1

    @classmethod
    def fit(
        cls,
        frame: pd.DataFrame,
        curve_artifact: dict[str, Any] | str | Path,
        *,
        base_source: str = "abst",
        fit_sources: Iterable[str] = ("conv", "nof_conv"),
        source_feature: str = "source_folder",
        residual_feature: str = "residual_rms",
        degree: int = 3,
        ridge_alpha: float = 1e-3,
        residual_reference: float | None = None,
    ) -> "RMSSplineSurfaceTransformer":
        if not isinstance(curve_artifact, dict):
            curve_artifact = _load_curve_artifact(curve_artifact)
        curve_model = curve_artifact["model"]
        parameter_feature = curve_model.parameter_feature
        coordinate_features = tuple(curve_model.coordinate_features)
        fit_sources = tuple(fit_sources)

        fit_mask = frame[source_feature].isin(fit_sources).to_numpy()
        base_mask = frame[source_feature].eq(base_source).to_numpy()
        if not np.any(fit_mask):
            raise ValueError(f"No rows found for fit sources {fit_sources!r}")
        if residual_reference is None:
            if not np.any(base_mask):
                raise ValueError(f"No rows found for base source {base_source!r}")
            # Constraining the anchor to the observed base range and selecting
            # its lower edge gave the best grouped validation fit.
            residual_reference = float(frame.loc[base_mask, residual_feature].min())

        parameter_bounds = np.array(
            [frame[parameter_feature].min(), frame[parameter_feature].max()], dtype=float
        )
        residual = frame.loc[fit_mask, residual_feature].to_numpy(dtype=float)
        if np.any(~np.isfinite(residual)) or np.any(residual <= 0):
            raise ValueError("residual_rms must be finite and strictly positive")
        log_residual = np.log10(residual)
        log_scale = float(np.std(log_residual))
        if log_scale <= 0:
            raise ValueError("residual_rms has no variation")

        transformer = cls(
            predictors=tuple(curve_model.predictors),
            parameter_center=float(curve_model.parameter_center),
            parameter_scale=float(curve_model.parameter_scale),
            curve_parameter_bounds=np.asarray(
                curve_model.training_parameter_bounds, dtype=float
            ),
            projection_parameter_bounds=parameter_bounds,
            coordinate_features=coordinate_features,
            parameter_feature=parameter_feature,
            residual_feature=residual_feature,
            residual_reference=float(residual_reference),
            log_residual_scale=log_scale,
            residual_bounds=np.array(
                [float(residual_reference), float(np.max(residual))], dtype=float
            ),
            degree=int(degree),
            coefficients=np.zeros((3, int(degree)), dtype=float),
            ridge_alpha=float(ridge_alpha),
        )

        true_t = frame.loc[fit_mask, parameter_feature].to_numpy(dtype=float)
        coordinates = frame.loc[fit_mask, coordinate_features].to_numpy(dtype=float)
        displacement = coordinates - transformer.curve(true_t)
        u = transformer.extrusion_coordinate(residual, clip=False)
        ridge = Ridge(alpha=ridge_alpha, fit_intercept=False)
        ridge.fit(_poly_features(u, degree), displacement)
        transformer.coefficients = np.asarray(ridge.coef_, dtype=float)
        transformer.validate()
        return transformer

    def validate(self) -> None:
        """Validate the fitted state before evaluation or serialization."""

        if len(self.predictors) != 3 or len(self.coordinate_features) != 3:
            raise ValueError("An RMS spline surface requires exactly three coordinates")
        if not np.isfinite(self.parameter_center):
            raise ValueError("parameter_center must be finite")
        if not np.isfinite(self.parameter_scale) or self.parameter_scale == 0:
            raise ValueError("parameter_scale must be finite and non-zero")
        if int(self.degree) < 1:
            raise ValueError("degree must be at least one")
        if np.asarray(self.coefficients).shape != (3, int(self.degree)):
            raise ValueError("coefficients must have shape (3, degree)")
        if not np.isfinite(self.coefficients).all():
            raise ValueError("coefficients must be finite")
        for name, bounds in (
            ("curve_parameter_bounds", self.curve_parameter_bounds),
            ("projection_parameter_bounds", self.projection_parameter_bounds),
            ("residual_bounds", self.residual_bounds),
        ):
            values = np.asarray(bounds, dtype=float)
            if values.shape != (2,) or not np.isfinite(values).all() or values[0] >= values[1]:
                raise ValueError(f"{name} must contain two increasing finite values")
        if np.any(np.asarray(self.residual_bounds) <= 0):
            raise ValueError("residual_bounds must be strictly positive")
        if not np.isfinite(self.residual_reference) or self.residual_reference <= 0:
            raise ValueError("residual_reference must be finite and strictly positive")
        if not np.isfinite(self.log_residual_scale) or self.log_residual_scale <= 0:
            raise ValueError("log_residual_scale must be finite and strictly positive")

    def curve(self, parameter: np.ndarray | float) -> np.ndarray:
        parameter_array = np.asarray(parameter, dtype=float)
        flat = parameter_array.reshape(-1)
        standardized = (flat - self.parameter_center) / self.parameter_scale
        values = np.column_stack([predictor(standardized) for predictor in self.predictors])
        values = values.reshape(parameter_array.shape + (3,))
        return values

    def curve_derivative(self, parameter: np.ndarray | float) -> np.ndarray:
        parameter_array = np.asarray(parameter, dtype=float)
        flat = parameter_array.reshape(-1)
        standardized = (flat - self.parameter_center) / self.parameter_scale
        values = np.column_stack(
            [predictor.derivative()(standardized) for predictor in self.predictors]
        ) / self.parameter_scale
        return values.reshape(parameter_array.shape + (3,))

    def extrusion_coordinate(
        self, residual_rms: np.ndarray | float, *, clip: bool = True
    ) -> np.ndarray:
        residual = np.asarray(residual_rms, dtype=float)
        if np.any(~np.isfinite(residual)) or np.any(residual <= 0):
            raise ValueError("residual_rms must be finite and strictly positive")
        if clip:
            residual = np.clip(residual, self.residual_bounds[0], self.residual_bounds[1])
        return (np.log10(residual) - np.log10(self.residual_reference)) / self.log_residual_scale

    def extrusion_offset(
        self, residual_rms: np.ndarray | float, *, clip: bool = True
    ) -> np.ndarray:
        residual = np.asarray(residual_rms, dtype=float)
        u = self.extrusion_coordinate(residual, clip=clip)
        features = _poly_features(u, self.degree)
        offsets = features @ self.coefficients.T
        return offsets.reshape(residual.shape + (3,))

    def extrusion_derivative(
        self, residual_rms: np.ndarray | float, *, clip: bool = True
    ) -> np.ndarray:
        residual = np.asarray(residual_rms, dtype=float)
        u = self.extrusion_coordinate(residual, clip=clip).reshape(-1)
        derivative = np.zeros((len(u), 3), dtype=float)
        for power in range(1, self.degree + 1):
            derivative += (
                power * u[:, None] ** (power - 1) * self.coefficients[:, power - 1]
            )
        return derivative.reshape(residual.shape + (3,))

    def surface(
        self,
        parameter: np.ndarray | float,
        residual_rms: np.ndarray | float,
        *,
        clip: bool = True,
    ) -> np.ndarray:
        parameter_array, residual_array = np.broadcast_arrays(
            np.asarray(parameter, dtype=float), np.asarray(residual_rms, dtype=float)
        )
        return self.curve(parameter_array) + self.extrusion_offset(
            residual_array, clip=clip
        )

    def dewarp(
        self, coordinates: np.ndarray, residual_rms: np.ndarray, *, clip: bool = True
    ) -> np.ndarray:
        coordinates = np.atleast_2d(np.asarray(coordinates, dtype=float))
        residual = np.asarray(residual_rms, dtype=float).reshape(-1)
        if len(coordinates) != len(residual):
            raise ValueError("coordinates and residual_rms must have the same row count")
        return coordinates - self.extrusion_offset(residual, clip=clip)

    def rewarp(
        self, dewarped_coordinates: np.ndarray, residual_rms: np.ndarray, *, clip: bool = True
    ) -> np.ndarray:
        coordinates = np.atleast_2d(np.asarray(dewarped_coordinates, dtype=float))
        residual = np.asarray(residual_rms, dtype=float).reshape(-1)
        return coordinates + self.extrusion_offset(residual, clip=clip)

    def _project_dewarped(
        self,
        dewarped_coordinates: np.ndarray,
        *,
        parameter_bounds: tuple[float, float] | None = None,
        grid_size: int = 12001,
        refine: bool = True,
    ) -> np.ndarray:
        points = np.atleast_2d(np.asarray(dewarped_coordinates, dtype=float))
        bounds = (
            tuple(np.asarray(parameter_bounds, dtype=float))
            if parameter_bounds is not None
            else tuple(self.projection_parameter_bounds)
        )
        grid = np.linspace(bounds[0], bounds[1], int(grid_size))
        curve_grid = self.curve(grid)
        _, nearest_index = cKDTree(curve_grid).query(points)
        estimates = grid[nearest_index].copy()
        if not refine:
            return estimates

        half_width = 4.0 * (bounds[1] - bounds[0]) / (grid_size - 1)
        for row, (point, index) in enumerate(zip(points, nearest_index)):
            lower = max(bounds[0], grid[index] - half_width)
            upper = min(bounds[1], grid[index] + half_width)
            result = minimize_scalar(
                lambda value: float(np.sum((point - self.curve(value)) ** 2)),
                bounds=(lower, upper),
                method="bounded",
                options={"xatol": 1e-9},
            )
            estimates[row] = result.x
        return estimates

    def predict_parameter(
        self,
        coordinates: np.ndarray,
        residual_rms: np.ndarray,
        *,
        clip: bool = True,
        parameter_bounds: tuple[float, float] | None = None,
        grid_size: int = 12001,
        refine: bool = True,
    ) -> np.ndarray:
        dewarped = self.dewarp(coordinates, residual_rms, clip=clip)
        return self._project_dewarped(
            dewarped,
            parameter_bounds=parameter_bounds,
            grid_size=grid_size,
            refine=refine,
        )

    def project(
        self,
        coordinates: np.ndarray,
        residual_rms: np.ndarray,
        *,
        clip: bool = True,
        parameter_bounds: tuple[float, float] | None = None,
        grid_size: int = 12001,
        refine: bool = True,
    ) -> dict[str, np.ndarray]:
        coordinates = np.atleast_2d(np.asarray(coordinates, dtype=float))
        residual = np.asarray(residual_rms, dtype=float).reshape(-1)
        parameter = self.predict_parameter(
            coordinates,
            residual,
            clip=clip,
            parameter_bounds=parameter_bounds,
            grid_size=grid_size,
            refine=refine,
        )
        surface_points = self.surface(parameter, residual, clip=clip)
        residual_vectors = coordinates - surface_points
        curve_tangent = self.curve_derivative(parameter)
        extrusion_tangent = self.extrusion_derivative(residual, clip=clip)
        normals = np.cross(curve_tangent, extrusion_tangent)
        normal_norm = np.linalg.norm(normals, axis=1, keepdims=True)
        normals = np.divide(
            normals,
            normal_norm,
            out=np.zeros_like(normals),
            where=normal_norm > 1e-12,
        )
        signed_distance = np.sum(residual_vectors * normals, axis=1)
        return {
            "parameter": parameter,
            "extrusion_coordinate": self.extrusion_coordinate(residual, clip=clip),
            "surface_points": surface_points,
            "dewarped_coordinates": self.dewarp(coordinates, residual, clip=clip),
            "residual_vectors": residual_vectors,
            "surface_distance": np.linalg.norm(residual_vectors, axis=1),
            "signed_normal_distance": signed_distance,
            "surface_normals": normals,
        }

    def to_artifact(self, metadata: dict[str, Any] | None = None) -> dict[str, Any]:
        self.validate()
        return {
            "artifact_kind": self.ARTIFACT_KIND,
            "artifact_version": self.ARTIFACT_VERSION,
            "model": {
                "predictors": self.predictors,
                "parameter_center": self.parameter_center,
                "parameter_scale": self.parameter_scale,
                "curve_parameter_bounds": self.curve_parameter_bounds,
                "projection_parameter_bounds": self.projection_parameter_bounds,
                "coordinate_features": self.coordinate_features,
                "parameter_feature": self.parameter_feature,
                "residual_feature": self.residual_feature,
                "residual_reference": self.residual_reference,
                "log_residual_scale": self.log_residual_scale,
                "residual_bounds": self.residual_bounds,
                "degree": self.degree,
                "coefficients": self.coefficients,
                "ridge_alpha": self.ridge_alpha,
            },
            "metadata": metadata or {},
        }

    def save(self, path: str | Path, metadata: dict[str, Any] | None = None) -> None:
        joblib.dump(self.to_artifact(metadata), path)

    @classmethod
    def from_artifact(cls, artifact: dict[str, Any]) -> "RMSSplineSurfaceTransformer":
        """Restore and validate a transformer from its portable artifact dictionary."""

        if not isinstance(artifact, dict) or artifact.get("artifact_kind") != cls.ARTIFACT_KIND:
            raise ValueError("Not an RMS-conditioned spline-surface artifact")
        if artifact.get("artifact_version") != cls.ARTIFACT_VERSION:
            raise ValueError(
                "Unsupported RMS spline-surface artifact version: "
                f"{artifact.get('artifact_version')!r}"
            )
        try:
            transformer = cls(**artifact["model"])
            transformer.validate()
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError(f"Malformed RMS spline-surface artifact: {error}") from error
        return transformer

    @classmethod
    def load_artifact(
        cls, source: str | Path | BinaryIO | bytes
    ) -> tuple["RMSSplineSurfaceTransformer", dict[str, Any]]:
        """Load a transformer and its metadata from a path, stream, or byte payload."""

        artifact = joblib.load(io.BytesIO(source) if isinstance(source, bytes) else source)
        transformer = cls.from_artifact(artifact)
        metadata = dict(artifact.get("metadata") or {})
        metadata.setdefault("artifact_kind", cls.ARTIFACT_KIND)
        metadata.setdefault("artifact_version", cls.ARTIFACT_VERSION)
        return transformer, metadata

    @classmethod
    def load(cls, path: str | Path | BinaryIO) -> "RMSSplineSurfaceTransformer":
        return cls.load_artifact(path)[0]


@dataclass(frozen=True)
class RMSSplineSurfaceEvaluationResult:
    """A sampled RMS-conditioned surface plus point-projection diagnostics."""

    model: RMSSplineSurfaceTransformer
    grid_parameter: np.ndarray
    grid_residual: np.ndarray
    grid_extrusion: np.ndarray
    grid_xyz: np.ndarray
    point_xyz: np.ndarray
    point_projection: np.ndarray
    point_parameter: np.ndarray
    point_extrusion: np.ndarray
    metrics: dict[str, Any]
    warnings: tuple[str, ...]


def serialize_rms_spline_surface(
    transformer: RMSSplineSurfaceTransformer,
    metadata: dict[str, Any] | None = None,
) -> bytes:
    """Serialize a transformer to the artifact bytes used by the explorer."""

    buffer = io.BytesIO()
    joblib.dump(transformer.to_artifact(metadata), buffer)
    return buffer.getvalue()


def add_surface_parameter_columns(
    frame: pd.DataFrame,
    transformer: RMSSplineSurfaceTransformer,
    *,
    clip: bool = True,
) -> pd.DataFrame:
    """Add the nearest-surface spline ``t`` and extrusion ``u`` coordinates."""

    required = [*transformer.coordinate_features, transformer.residual_feature]
    missing = [column for column in required if column not in frame]
    if missing:
        raise ValueError(
            "Surface coordinates cannot be computed because the projection is missing: "
            + ", ".join(missing)
        )

    result = frame.copy()
    points = result.loc[:, list(transformer.coordinate_features)].apply(
        pd.to_numeric, errors="coerce"
    ).to_numpy(dtype=float)
    residual = pd.to_numeric(
        result[transformer.residual_feature], errors="coerce"
    ).to_numpy(dtype=float)
    finite = np.isfinite(points).all(axis=1) & np.isfinite(residual) & (residual > 0)

    parameter = np.full(len(result), np.nan, dtype=float)
    extrusion = np.full(len(result), np.nan, dtype=float)
    if np.any(finite):
        projection = transformer.project(points[finite], residual[finite], clip=clip)
        parameter[finite] = projection["parameter"]
        extrusion[finite] = projection["extrusion_coordinate"]
    result["t"] = parameter
    result["u"] = extrusion
    return result


def evaluate_rms_spline_surface(
    transformer: RMSSplineSurfaceTransformer,
    points: np.ndarray,
    residual_rms: np.ndarray,
    *,
    true_parameter: np.ndarray | None = None,
    parameter_samples: int = 120,
    residual_samples: int = 30,
    parameter_padding_fraction: float = 0.0,
    residual_padding_fraction: float = 0.1,
    residual_bounds: tuple[float, float] | None = None,
    clip: bool = True,
) -> RMSSplineSurfaceEvaluationResult:
    """Project points and sample a regular parameter-by-log-RMS display grid."""

    transformer.validate()
    if not 20 <= int(parameter_samples) <= 1_000:
        raise ValueError("parameter_samples must be between 20 and 1,000")
    if not 2 <= int(residual_samples) <= 300:
        raise ValueError("residual_samples must be between 2 and 300")
    if not 0 <= float(parameter_padding_fraction) <= 1:
        raise ValueError("parameter_padding_fraction must be between 0 and 1")
    if not 0 <= float(residual_padding_fraction) <= 2:
        raise ValueError("residual_padding_fraction must be between 0 and 2")

    values = np.asarray(points, dtype=float)
    residual = np.asarray(residual_rms, dtype=float).reshape(-1)
    if values.ndim != 2 or values.shape[1] != 3:
        raise ValueError("points must have shape (n, 3)")
    if len(values) != len(residual):
        raise ValueError("points and residual_rms must have the same row count")
    parameter = None
    if true_parameter is not None:
        parameter = np.asarray(true_parameter, dtype=float).reshape(-1)
        if len(parameter) != len(values):
            raise ValueError("true_parameter must have the same row count as points")

    finite = np.isfinite(values).all(axis=1) & np.isfinite(residual) & (residual > 0)
    if parameter is not None:
        finite &= np.isfinite(parameter)
        parameter = parameter[finite]
    current_points = values[finite]
    current_residual = residual[finite]
    if not len(current_points):
        raise ValueError("At least one finite point with positive residual RMS is required")

    projection = transformer.project(current_points, current_residual, clip=clip)
    projected = projection["surface_points"]
    evaluated_parameter = projection["parameter"]
    point_extrusion = projection["extrusion_coordinate"]

    parameter_low, parameter_high = map(float, transformer.projection_parameter_bounds)
    parameter_span = parameter_high - parameter_low
    parameter_low -= float(parameter_padding_fraction) * parameter_span
    parameter_high += float(parameter_padding_fraction) * parameter_span

    if residual_bounds is None:
        residual_low = float(np.min(current_residual))
        residual_high = float(np.max(current_residual))
        if np.isclose(np.log10(residual_low), np.log10(residual_high)):
            residual_low, residual_high = map(float, transformer.residual_bounds)
    else:
        requested = np.asarray(residual_bounds, dtype=float)
        if (
            requested.shape != (2,)
            or not np.isfinite(requested).all()
            or np.any(requested <= 0)
            or requested[0] >= requested[1]
        ):
            raise ValueError("Manual residual bounds must be two increasing positive values")
        residual_low, residual_high = map(float, requested)
    log_low, log_high = np.log10([residual_low, residual_high])
    log_span = log_high - log_low
    log_low -= float(residual_padding_fraction) * log_span
    log_high += float(residual_padding_fraction) * log_span
    residual_low, residual_high = np.power(10.0, [log_low, log_high])

    parameter_axis = np.linspace(parameter_low, parameter_high, int(parameter_samples))
    residual_axis = np.geomspace(residual_low, residual_high, int(residual_samples))
    grid_parameter, grid_residual = np.meshgrid(parameter_axis, residual_axis)
    grid_xyz = transformer.surface(grid_parameter, grid_residual, clip=clip)
    grid_extrusion = transformer.extrusion_coordinate(grid_residual, clip=clip)

    surface_distance = projection["surface_distance"]
    clipped_points = int(
        np.sum(
            (current_residual < transformer.residual_bounds[0])
            | (current_residual > transformer.residual_bounds[1])
        )
    )
    metrics: dict[str, Any] = {
        "input_points": int(len(values)),
        "current_points": int(len(current_points)),
        "dropped_nonfinite": int((~finite).sum()),
        "current_rmse_3d": float(np.sqrt(np.mean(np.square(surface_distance)))),
        "current_mae_3d": float(np.mean(surface_distance)),
        "current_max_error_3d": float(np.max(surface_distance)),
        "clipped_residual_points": clipped_points if clip else 0,
        "display_parameter_bounds": [parameter_low, parameter_high],
        "display_residual_bounds": [float(residual_low), float(residual_high)],
        "display_extrusion_bounds": [
            float(np.min(grid_extrusion)),
            float(np.max(grid_extrusion)),
        ],
    }
    if parameter is not None:
        parameter_error = evaluated_parameter - parameter
        metrics.update(
            {
                "parameter_rmse": float(np.sqrt(np.mean(np.square(parameter_error)))),
                "parameter_mae": float(np.mean(np.abs(parameter_error))),
            }
        )

    warnings: list[str] = []
    if metrics["dropped_nonfinite"]:
        warnings.append(
            f"Dropped {metrics['dropped_nonfinite']:,} non-finite or non-positive row(s)."
        )
    if clip and clipped_points:
        warnings.append(
            f"Clipped residual RMS for {clipped_points:,} point(s) to the fitted range."
        )
    return RMSSplineSurfaceEvaluationResult(
        model=transformer,
        grid_parameter=grid_parameter,
        grid_residual=grid_residual,
        grid_extrusion=grid_extrusion,
        grid_xyz=grid_xyz,
        point_xyz=current_points,
        point_projection=projected,
        point_parameter=evaluated_parameter,
        point_extrusion=point_extrusion,
        metrics=metrics,
        warnings=tuple(warnings),
    )


def grouped_cross_validation(
    frame: pd.DataFrame,
    curve_artifact: dict[str, Any],
    *,
    degree: int,
    residual_reference: float,
    ridge_alpha: float = 1e-3,
    source_feature: str = "source_folder",
    fit_sources: tuple[str, ...] = ("conv", "nof_conv"),
    group_feature: str = "task",
    n_splits: int = 8,
) -> tuple[dict[str, Any], np.ndarray, np.ndarray]:
    curve_model = curve_artifact["model"]
    parameter_feature = curve_model.parameter_feature
    coordinate_features = tuple(curve_model.coordinate_features)
    selected = frame[source_feature].isin(fit_sources).to_numpy()
    target = frame.loc[selected].reset_index(drop=True)

    template_model = RMSSplineSurfaceTransformer.fit(
        frame,
        curve_artifact,
        degree=degree,
        ridge_alpha=ridge_alpha,
        residual_reference=residual_reference,
        fit_sources=fit_sources,
    )
    true_t = target[parameter_feature].to_numpy(dtype=float)
    coordinates = target.loc[:, coordinate_features].to_numpy(dtype=float)
    base_curve = template_model.curve(true_t)
    displacement = coordinates - base_curve
    residual = target[template_model.residual_feature].to_numpy(dtype=float)
    log_residual = np.log10(residual)
    groups = target[group_feature].to_numpy()
    splits = min(int(n_splits), len(np.unique(groups)))

    predicted_offset = np.zeros_like(displacement)
    predicted_t = np.zeros(len(target), dtype=float)
    projected_surface = np.zeros_like(coordinates)

    for train, test in GroupKFold(splits).split(log_residual, displacement, groups):
        fold_scale = float(np.std(log_residual[train]))
        train_u = (
            log_residual[train] - np.log10(residual_reference)
        ) / fold_scale
        test_u = (
            log_residual[test] - np.log10(residual_reference)
        ) / fold_scale
        ridge = Ridge(alpha=ridge_alpha, fit_intercept=False)
        ridge.fit(_poly_features(train_u, degree), displacement[train])
        fold_offset = ridge.predict(_poly_features(test_u, degree))
        predicted_offset[test] = fold_offset
        dewarped = coordinates[test] - fold_offset
        fold_t = template_model._project_dewarped(dewarped)
        predicted_t[test] = fold_t
        projected_surface[test] = template_model.curve(fold_t) + fold_offset

    metrics: dict[str, Any] = {
        "validation": "grouped by task",
        "folds": splits,
        "degree": degree,
        "residual_reference": residual_reference,
        "all": _metric_block(
            true_t,
            predicted_t,
            coordinates,
            base_curve,
            predicted_offset,
            projected_surface,
        ),
        "by_source": {},
    }
    for source in fit_sources:
        mask = target[source_feature].eq(source).to_numpy()
        metrics["by_source"][source] = _metric_block(
            true_t[mask],
            predicted_t[mask],
            coordinates[mask],
            base_curve[mask],
            predicted_offset[mask],
            projected_surface[mask],
        )
    return metrics, predicted_t, predicted_offset


def augment_frame(
    frame: pd.DataFrame, transformer: RMSSplineSurfaceTransformer
) -> pd.DataFrame:
    result = frame.copy()
    coordinates = result.loc[:, transformer.coordinate_features].to_numpy(dtype=float)
    residual = result[transformer.residual_feature].to_numpy(dtype=float)
    projection = transformer.project(coordinates, residual)
    offset = transformer.extrusion_offset(residual)

    result["extrusion_u"] = projection["extrusion_coordinate"]
    for axis, name in enumerate(transformer.coordinate_features):
        result[f"extrusion_offset_{name}"] = offset[:, axis]
        result[f"dewarped_{name}"] = projection["dewarped_coordinates"][:, axis]
        result[f"surface_{name}"] = projection["surface_points"][:, axis]
    result["surface_parameter_hat"] = projection["parameter"]
    result["surface_parameter_error"] = (
        projection["parameter"] - result[transformer.parameter_feature].to_numpy(dtype=float)
    )
    result["surface_distance"] = projection["surface_distance"]
    result["surface_signed_normal_distance"] = projection["signed_normal_distance"]
    result["flat_parameter"] = projection["parameter"]
    result["flat_extrusion"] = projection["extrusion_coordinate"]
    result["flat_normal"] = projection["signed_normal_distance"]
    return result


def _sample_indices(
    frame: pd.DataFrame, source: str, maximum: int, random_state: int = 42
) -> np.ndarray:
    index = frame.index[frame["source_folder"].eq(source)].to_numpy()
    if len(index) <= maximum:
        return index
    rng = np.random.default_rng(random_state)
    # Retain both range endpoints, then sample the remainder.
    t = frame.loc[index, "log10_time_horizon_months"].to_numpy()
    endpoints = np.unique([index[np.argmin(t)], index[np.argmax(t)]])
    remainder = np.setdiff1d(index, endpoints)
    chosen = rng.choice(remainder, size=maximum - len(endpoints), replace=False)
    return np.sort(np.concatenate([endpoints, chosen]))


def make_static_figure(
    augmented: pd.DataFrame,
    transformer: RMSSplineSurfaceTransformer,
    output_path: str | Path,
) -> None:
    import matplotlib.pyplot as plt

    colors = {"abst": "#6b7280", "nof_conv": "#0ea5e9", "conv": "#ef4444"}
    labels = {"abst": "abst", "nof_conv": "nof_conv", "conv": "conv"}
    figure = plt.figure(figsize=(18, 6.2), constrained_layout=True)
    axes = [figure.add_subplot(1, 3, index + 1, projection="3d") for index in range(3)]

    t_grid = np.linspace(*transformer.projection_parameter_bounds, 140)
    curve = transformer.curve(t_grid)
    residual_grid = np.geomspace(
        transformer.residual_bounds[0], transformer.residual_bounds[1], 30
    )
    tt, rr = np.meshgrid(t_grid, residual_grid)
    surface = transformer.surface(tt, rr)

    for source, maximum in (("abst", 300), ("nof_conv", 450), ("conv", 450)):
        index = _sample_indices(augmented, source, maximum)
        points = augmented.loc[index, list(COORDINATE_COLUMNS)].to_numpy()
        axes[0].scatter(
            points[:, 0], points[:, 1], points[:, 2], s=8, alpha=0.42,
            color=colors[source], label=labels[source], depthshade=False,
        )
    axes[0].plot(curve[:, 0], curve[:, 1], curve[:, 2], color="#111827", lw=2.2)
    axes[0].set_title("Original PLS coordinates")

    axes[1].plot_surface(
        surface[:, :, 0], surface[:, :, 1], surface[:, :, 2],
        color="#8b5cf6", alpha=0.17, linewidth=0, antialiased=True,
    )
    for source in ("nof_conv", "conv"):
        index = _sample_indices(augmented, source, 550)
        columns = [f"surface_{name}" for name in COORDINATE_COLUMNS]
        points = augmented.loc[index, columns].to_numpy()
        axes[1].scatter(
            points[:, 0], points[:, 1], points[:, 2], s=9, alpha=0.58,
            color=colors[source], label=labels[source], depthshade=False,
        )
    axes[1].plot(curve[:, 0], curve[:, 1], curve[:, 2], color="#111827", lw=2.2)
    axes[1].set_title("Projected onto fitted RMS surface")

    for source in ("nof_conv", "conv"):
        index = _sample_indices(augmented, source, 550)
        columns = [f"dewarped_{name}" for name in COORDINATE_COLUMNS]
        points = augmented.loc[index, columns].to_numpy()
        axes[2].scatter(
            points[:, 0], points[:, 1], points[:, 2], s=9, alpha=0.48,
            color=colors[source], label=labels[source], depthshade=False,
        )
    axes[2].plot(curve[:, 0], curve[:, 1], curve[:, 2], color="#111827", lw=2.4)
    axes[2].set_title("Dewarped: extrusion removed")

    for axis in axes:
        axis.set_xlabel("PLS1")
        axis.set_ylabel("PLS2")
        axis.set_zlabel("PLS3")
        axis.view_init(elev=21, azim=-56)
        axis.legend(loc="upper left", fontsize=8, frameon=False)
        axis.grid(True, alpha=0.22)
    figure.savefig(output_path, dpi=190, bbox_inches="tight")
    plt.close(figure)


def make_flattened_figure(
    augmented: pd.DataFrame, output_path: str | Path
) -> None:
    import matplotlib.pyplot as plt

    colors = {"nof_conv": "#0ea5e9", "conv": "#ef4444"}
    figure = plt.figure(figsize=(12, 5.8), constrained_layout=True)
    axes = [figure.add_subplot(1, 2, index + 1, projection="3d") for index in range(2)]
    for source in ("nof_conv", "conv"):
        index = _sample_indices(augmented, source, 650)
        points = augmented.loc[
            index, ["flat_parameter", "flat_extrusion", "flat_normal"]
        ].to_numpy()
        axes[0].scatter(
            points[:, 0], points[:, 1], points[:, 2], s=9, alpha=0.5,
            color=colors[source], label=source, depthshade=False,
        )
        projected = points.copy()
        projected[:, 2] = 0.0
        axes[1].scatter(
            projected[:, 0], projected[:, 1], projected[:, 2], s=9, alpha=0.55,
            color=colors[source], label=source, depthshade=False,
        )
    for axis, title in zip(
        axes,
        ("Straightened coordinates before projection", "Final surface (normal = 0)"),
    ):
        axis.set_xlabel("Spline parameter (log10 months)")
        axis.set_ylabel("Log-RMS extrusion coordinate")
        axis.set_zlabel("Signed normal distance")
        axis.set_title(title)
        axis.view_init(elev=23, azim=-58)
        axis.legend(frameon=False)
        axis.grid(True, alpha=0.22)
    figure.savefig(output_path, dpi=190, bbox_inches="tight")
    plt.close(figure)


def make_interactive_payload(
    augmented: pd.DataFrame,
    transformer: RMSSplineSurfaceTransformer,
    output_path: str | Path,
) -> None:
    def rounded(values: np.ndarray) -> list[Any]:
        return np.round(np.asarray(values, dtype=float), 4).tolist()

    payload: dict[str, Any] = {"sources": {}, "surface": {}, "curve": {}}
    for source, maximum in (("abst", 180), ("nof_conv", 260), ("conv", 260)):
        index = _sample_indices(augmented, source, maximum)
        raw = augmented.loc[index, list(COORDINATE_COLUMNS)].to_numpy()
        projected = augmented.loc[
            index, [f"surface_{name}" for name in COORDINATE_COLUMNS]
        ].to_numpy()
        payload["sources"][source] = {
            "raw": {"x": rounded(raw[:, 0]), "y": rounded(raw[:, 1]), "z": rounded(raw[:, 2])},
            "projected": {
                "x": rounded(projected[:, 0]),
                "y": rounded(projected[:, 1]),
                "z": rounded(projected[:, 2]),
            },
            "t": rounded(augmented.loc[index, "log10_time_horizon_months"].to_numpy()),
            "r": rounded(augmented.loc[index, "residual_rms"].to_numpy()),
        }
    t_grid = np.linspace(*transformer.projection_parameter_bounds, 85)
    residual_grid = np.geomspace(
        transformer.residual_bounds[0], transformer.residual_bounds[1], 26
    )
    tt, rr = np.meshgrid(t_grid, residual_grid)
    surface = transformer.surface(tt, rr)
    payload["surface"] = {
        "x": rounded(surface[:, :, 0]),
        "y": rounded(surface[:, :, 1]),
        "z": rounded(surface[:, :, 2]),
    }
    curve = transformer.curve(t_grid)
    payload["curve"] = {
        "x": rounded(curve[:, 0]),
        "y": rounded(curve[:, 1]),
        "z": rounded(curve[:, 2]),
        "t": rounded(t_grid),
    }
    Path(output_path).write_text(json.dumps(payload, separators=(",", ":")), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", required=True)
    parser.add_argument("--curve", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    frame = pd.read_csv(args.csv)
    curve_artifact = _load_curve_artifact(args.curve)

    base_mask = frame["source_folder"].eq("abst")
    residual_reference = float(frame.loc[base_mask, "residual_rms"].min())
    candidate_metrics: dict[str, Any] = {}
    for degree in (1, 2, 3, 4):
        metrics, _, _ = grouped_cross_validation(
            frame,
            curve_artifact,
            degree=degree,
            residual_reference=residual_reference,
        )
        candidate_metrics[str(degree)] = metrics

    transformer = RMSSplineSurfaceTransformer.fit(
        frame,
        curve_artifact,
        degree=3,
        residual_reference=residual_reference,
    )
    selected_metrics = candidate_metrics["3"]
    metadata = {
        "fit_sources": ["conv", "nof_conv"],
        "base_source": "abst",
        "source_label_used_as_predictor": False,
        "selection_reason": (
            "Cubic chosen for the lowest balanced geometric error while retaining "
            "a low-order source-agnostic surface."
        ),
        "cross_validation": selected_metrics,
    }
    model_path = output_dir / "rms_spline_surface_model.joblib"
    transformer.save(model_path, metadata=metadata)

    augmented = augment_frame(frame, transformer)
    csv_path = output_dir / "all_pls_with_rms_surface_projection.csv"
    augmented.to_csv(csv_path, index=False)

    diagnostics = {
        "model": {
            "surface_equation": "S(t,r)=C(t)+sum(B_k*u(r)^k), k=1..3",
            "u_equation": "u=(log10(r)-log10(r_ref))/q_scale",
            "degree": transformer.degree,
            "residual_reference": transformer.residual_reference,
            "log_residual_scale": transformer.log_residual_scale,
            "residual_fit_bounds": transformer.residual_bounds.tolist(),
            "projection_parameter_bounds": transformer.projection_parameter_bounds.tolist(),
            "original_curve_training_bounds": transformer.curve_parameter_bounds.tolist(),
            "coefficients_rows_are_PLS_axes": transformer.coefficients.tolist(),
            "uses_source_folder_at_transform_time": False,
        },
        "selected_grouped_cross_validation": selected_metrics,
        "candidate_degrees": candidate_metrics,
        "notes": [
            "The CSV source is named 'nof_conv'; this corresponds to the user's 'of_conv'.",
            "RMS values outside the fitted range are clipped by default.",
            "The supplied spline is extrapolated from its stored upper training bound to the CSV maximum parameter where needed.",
        ],
    }
    (output_dir / "spline_surface_diagnostics.json").write_text(
        json.dumps(diagnostics, indent=2), encoding="utf-8"
    )
    make_static_figure(
        augmented, transformer, output_dir / "spline_surface_final_3d.png"
    )
    make_flattened_figure(
        augmented, output_dir / "spline_surface_straightened_3d.png"
    )
    make_interactive_payload(
        augmented, transformer, output_dir / "interactive_payload.json"
    )

    summary = selected_metrics["all"]
    print(json.dumps({
        "model": str(model_path),
        "csv": str(csv_path),
        "parameter_r2": summary["parameter_r2"],
        "parameter_rmse_log10_months": summary["parameter_rmse_log10_months"],
        "offset_rmse_per_coordinate": summary["offset_rmse_per_coordinate"],
    }, indent=2))


if __name__ == "__main__":
    main()
