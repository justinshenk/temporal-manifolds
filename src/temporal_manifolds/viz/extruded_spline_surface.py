"""Fit, load, and evaluate spline curves extruded into parametric 3D surfaces.

The app-facing fit accepts a saved cubic ``CurveModel`` and derives the least-squares
extrusion direction from displayed points. The legacy portable artifact remains a plain
joblib dictionary containing SciPy spline objects.
"""

from __future__ import annotations

import io
from dataclasses import dataclass
from pathlib import Path
from typing import Any, BinaryIO

import joblib
import numpy as np
from scipy.optimize import minimize_scalar

from temporal_manifolds.viz.curve_fitting import CurveModel


EXTRUDED_SURFACE_ARTIFACT_KIND = "temporal_manifolds_extruded_surface"
EXTRUDED_SURFACE_ARTIFACT_VERSION = 2


@dataclass(frozen=True)
class ExtrudedSplineSurface:
    """Parametric surface ``S(t,u) = C(t) + u*d1 [+ u²*d2]``."""

    predictors: tuple[Any, Any, Any]
    parameter_center: float
    parameter_scale: float
    parameter_bounds: np.ndarray
    direction: np.ndarray
    parameter_feature: str
    coordinate_features: tuple[str, str, str]
    quadratic_direction: np.ndarray | None = None

    @classmethod
    def from_artifact(cls, artifact: dict[str, Any]) -> "ExtrudedSplineSurface":
        """Build and validate a surface from the portable dictionary artifact."""

        if artifact.get("artifact_kind") != EXTRUDED_SURFACE_ARTIFACT_KIND:
            raise ValueError("This is not a supported extruded-surface artifact.")
        try:
            coefficients = artifact.get("extrusion_coefficients")
            if coefficients is not None:
                coefficients = np.asarray(coefficients, dtype=np.float64)
                if coefficients.ndim != 2 or coefficients.shape not in {(1, 3), (2, 3)}:
                    raise ValueError("extrusion_coefficients must have shape (1, 3) or (2, 3)")
                linear_direction = coefficients[0]
                quadratic_direction = coefficients[1] if len(coefficients) == 2 else None
            else:
                linear_direction = np.asarray(
                    artifact["extrusion_direction"], dtype=np.float64
                )
                saved_quadratic = artifact.get("quadratic_extrusion_direction")
                quadratic_direction = (
                    None
                    if saved_quadratic is None
                    else np.asarray(saved_quadratic, dtype=np.float64)
                )
            surface = cls(
                predictors=tuple(artifact["curve_predictors"]),
                parameter_center=float(artifact["parameter_center"]),
                parameter_scale=float(artifact["parameter_scale"]),
                parameter_bounds=np.asarray(
                    artifact["training_parameter_bounds"], dtype=np.float64
                ),
                direction=linear_direction,
                parameter_feature=artifact["parameter_feature"],
                coordinate_features=tuple(artifact["coordinate_features"]),
                quadratic_direction=quadratic_direction,
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"The extruded-surface artifact is malformed: {exc}") from exc
        _validate_surface(surface)
        return surface

    def curve(self, parameter_values: Any) -> np.ndarray:
        """Evaluate the center curve ``C(t)`` in the saved coordinate order."""

        values = np.asarray(parameter_values, dtype=np.float64)
        if not np.isfinite(values).all():
            raise ValueError("Surface parameter values must be finite.")
        normalized = (values - self.parameter_center) / self.parameter_scale
        coordinates = np.stack(
            [np.asarray(predictor(normalized), dtype=np.float64) for predictor in self.predictors],
            axis=-1,
        )
        if coordinates.shape != (*values.shape, 3):
            raise ValueError("The saved curve predictors returned an invalid shape.")
        if not np.isfinite(coordinates).all():
            raise ValueError("The saved curve predictors returned non-finite coordinates.")
        return coordinates

    def evaluate(self, parameter_values: Any, extrusion_values: Any = 0.0) -> np.ndarray:
        """Evaluate ``S(t, u)`` for scalar or broadcast-compatible inputs."""

        parameter, extrusion = np.broadcast_arrays(
            np.asarray(parameter_values, dtype=np.float64),
            np.asarray(extrusion_values, dtype=np.float64),
        )
        if not np.isfinite(extrusion).all():
            raise ValueError("Surface extrusion values must be finite.")
        displacement = extrusion[..., np.newaxis] * self.direction
        if self.quadratic_direction is not None:
            displacement = (
                displacement
                + np.square(extrusion)[..., np.newaxis] * self.quadratic_direction
            )
        return self.curve(parameter) + displacement

    @property
    def extrusion_degree(self) -> int:
        """Polynomial degree of the cross-curve extrusion."""

        return 2 if self.quadratic_direction is not None else 1

    @property
    def extrusion_coefficients(self) -> np.ndarray:
        """Return the vector coefficients ``[d1]`` or ``[d1, d2]``."""

        if self.quadratic_direction is None:
            return self.direction[np.newaxis, :].copy()
        return np.stack([self.direction, self.quadratic_direction])

    def project(self, points: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Project 3D points onto the bounded parametric surface.

        The closest curve parameter is found while minimizing over the linear or quadratic
        extrusion coordinate. This generalizes the axis-aligned PLS2 construction used by
        the original standalone helper.
        """

        values = np.asarray(points, dtype=np.float64)
        scalar_input = values.ndim == 1
        values = np.atleast_2d(values)
        if values.ndim != 2 or values.shape[1] != 3:
            raise ValueError("Points must have shape (3,) or (n, 3).")
        if not np.isfinite(values).all():
            raise ValueError("Projection points must be finite.")

        lower, upper = self.parameter_bounds
        parameters = np.empty(len(values), dtype=np.float64)
        extrusions = np.empty(len(values), dtype=np.float64)
        projected = np.empty_like(values)

        for index, point in enumerate(values):
            def objective(parameter: float) -> float:
                residual = self.curve(parameter) - point
                _, squared_error = _best_extrusion_value(
                    -residual,
                    self.direction,
                    self.quadratic_direction,
                )
                return squared_error

            result = minimize_scalar(objective, bounds=(lower, upper), method="bounded")
            if not result.success or not np.isfinite(result.x):
                raise ValueError("A point could not be projected onto the extruded surface.")
            parameters[index] = result.x
            base = self.curve(result.x)
            extrusions[index], _ = _best_extrusion_value(
                point - base,
                self.direction,
                self.quadratic_direction,
            )
            projected[index] = self.evaluate(parameters[index], extrusions[index])

        if scalar_input:
            return projected[0], parameters[0], extrusions[0]
        return projected, parameters, extrusions


@dataclass(frozen=True)
class ExtrudedSurfaceEvaluationResult:
    """A sampled display grid plus current-point projection diagnostics."""

    model: ExtrudedSplineSurface
    grid_parameter: np.ndarray
    grid_extrusion: np.ndarray
    grid_xyz: np.ndarray
    point_xyz: np.ndarray
    point_projection: np.ndarray
    point_parameter: np.ndarray
    point_extrusion: np.ndarray
    metrics: dict[str, Any]
    warnings: tuple[str, ...]


@dataclass(frozen=True)
class ExtrusionDirectionFitResult:
    """Least-squares extrusion direction fitted around a fixed cubic spline."""

    model: ExtrudedSplineSurface
    point_parameter: np.ndarray
    point_xyz: np.ndarray
    curve_xyz: np.ndarray
    point_projection: np.ndarray
    point_extrusion: np.ndarray
    metrics: dict[str, Any]
    warnings: tuple[str, ...]


def _best_extrusion_value(
    residual: np.ndarray,
    linear_direction: np.ndarray,
    quadratic_direction: np.ndarray | None,
) -> tuple[float, float]:
    """Return the scalar ``u`` minimizing ``||r - u*d1 - u**2*d2||²``."""

    residual = np.asarray(residual, dtype=np.float64).reshape(1, 3)
    value = _best_extrusion_values(residual, linear_direction, quadratic_direction)[0]
    prediction = value * linear_direction
    if quadratic_direction is not None:
        prediction = prediction + value**2 * quadratic_direction
    difference = residual[0] - prediction
    return float(value), float(np.dot(difference, difference))


def _best_extrusion_values(
    residuals: np.ndarray,
    linear_direction: np.ndarray,
    quadratic_direction: np.ndarray | None,
) -> np.ndarray:
    residuals = np.asarray(residuals, dtype=np.float64)
    linear = np.asarray(linear_direction, dtype=np.float64)
    linear_norm_squared = float(np.dot(linear, linear))
    if quadratic_direction is None or np.linalg.norm(quadratic_direction) <= 1e-12:
        return residuals @ linear / linear_norm_squared

    quadratic = np.asarray(quadratic_direction, dtype=np.float64)
    # Solve the stationary cubic for every point with vectorized Cardano formulas.
    cubic = 2.0 * float(np.dot(quadratic, quadratic))
    square = 3.0 * float(np.dot(linear, quadratic))
    linear_coefficients = linear_norm_squared - 2.0 * (residuals @ quadratic)
    constants = -(residuals @ linear)
    p = linear_coefficients / cubic - square**2 / (3.0 * cubic**2)
    q = (
        2.0 * square**3 / (27.0 * cubic**3)
        - square * linear_coefficients / (3.0 * cubic**2)
        + constants / cubic
    )
    discriminant = np.square(q / 2.0) + np.power(p / 3.0, 3)
    shift = square / (3.0 * cubic)
    candidates = np.full((len(residuals), 3), np.nan, dtype=np.float64)

    one_root = discriminant >= 0
    square_root = np.sqrt(np.maximum(discriminant[one_root], 0.0))
    candidates[one_root, 0] = (
        np.cbrt(-q[one_root] / 2.0 + square_root)
        + np.cbrt(-q[one_root] / 2.0 - square_root)
        - shift
    )

    three_roots = ~one_root
    if np.any(three_roots):
        p_three = p[three_roots]
        denominator = np.sqrt(np.maximum(-np.power(p_three / 3.0, 3), 0.0))
        cosine = np.clip(-q[three_roots] / (2.0 * denominator), -1.0, 1.0)
        theta = np.arccos(cosine) / 3.0
        radius = 2.0 * np.sqrt(np.maximum(-p_three / 3.0, 0.0))
        for index in range(3):
            candidates[three_roots, index] = (
                radius * np.cos(theta - 2.0 * np.pi * index / 3.0) - shift
            )

    predictions = (
        candidates[..., np.newaxis] * linear
        + np.square(candidates)[..., np.newaxis] * quadratic
    )
    squared_errors = np.sum(
        np.square(residuals[:, np.newaxis, :] - predictions), axis=2
    )
    squared_errors[~np.isfinite(squared_errors)] = np.inf
    best = np.argmin(squared_errors, axis=1)
    return candidates[np.arange(len(residuals)), best]


def _validate_surface(surface: ExtrudedSplineSurface) -> None:
    if len(surface.predictors) != 3 or not all(callable(item) for item in surface.predictors):
        raise ValueError("The extruded surface must contain three callable curve predictors.")
    if not np.isfinite(surface.parameter_center):
        raise ValueError("The extruded surface has an invalid parameter center.")
    if not np.isfinite(surface.parameter_scale) or surface.parameter_scale <= 0:
        raise ValueError("The extruded surface parameter scale must be positive and finite.")
    if surface.parameter_bounds.shape != (2,) or not np.isfinite(surface.parameter_bounds).all():
        raise ValueError("The extruded surface has invalid parameter bounds.")
    if surface.parameter_bounds[0] >= surface.parameter_bounds[1]:
        raise ValueError("The extruded surface parameter bounds must increase.")
    if surface.direction.shape != (3,) or not np.isfinite(surface.direction).all():
        raise ValueError("The extrusion direction must be a finite three-vector.")
    linear_norm = np.linalg.norm(surface.direction)
    quadratic_norm = 0.0
    if surface.quadratic_direction is not None:
        quadratic = np.asarray(surface.quadratic_direction, dtype=np.float64)
        if quadratic.shape != (3,) or not np.isfinite(quadratic).all():
            raise ValueError("The quadratic extrusion direction must be a finite three-vector.")
        quadratic_norm = np.linalg.norm(quadratic)
    if max(linear_norm, quadratic_norm) <= np.finfo(np.float64).eps:
        raise ValueError("At least one extrusion coefficient must be non-zero.")
    if (
        len(surface.coordinate_features) != 3
        or len(set(surface.coordinate_features)) != 3
        or not all(isinstance(name, str) and name for name in surface.coordinate_features)
    ):
        raise ValueError("The extruded surface must name three distinct coordinates.")
    if not isinstance(surface.parameter_feature, str) or not surface.parameter_feature:
        raise ValueError("The extruded surface parameter feature must be named.")
    surface.curve(surface.parameter_bounds)


def load_extruded_surface(
    source: bytes | bytearray | str | Path | BinaryIO,
) -> tuple[ExtrudedSplineSurface, dict[str, Any]]:
    """Load a portable extruded-surface artifact and return its optional metadata."""

    try:
        payload = joblib.load(io.BytesIO(source) if isinstance(source, (bytes, bytearray)) else source)
    except Exception as exc:  # noqa: BLE001 - normalize trusted-artifact load failures
        raise ValueError(f"Extruded-surface artifact could not be loaded: {exc}") from exc
    if not isinstance(payload, dict):
        raise ValueError("This is not a supported extruded-surface artifact.")
    version = payload.get("artifact_version")
    if version not in (None, 1, EXTRUDED_SURFACE_ARTIFACT_VERSION):
        raise ValueError(f"Unsupported extruded-surface artifact version: {version!r}.")
    surface = ExtrudedSplineSurface.from_artifact(payload)
    metadata = dict(payload.get("metadata") or {})
    metadata.update(
        {
            "artifact_kind": payload["artifact_kind"],
            "artifact_version": version,
        }
    )
    return surface, metadata


def serialize_extruded_surface(
    surface: ExtrudedSplineSurface,
    metadata: dict[str, Any] | None = None,
) -> bytes:
    """Serialize an extruded spline surface as a portable versioned joblib artifact."""

    _validate_surface(surface)
    payload = {
        "artifact_kind": EXTRUDED_SURFACE_ARTIFACT_KIND,
        "artifact_version": EXTRUDED_SURFACE_ARTIFACT_VERSION,
        "curve_predictors": surface.predictors,
        "parameter_center": float(surface.parameter_center),
        "parameter_scale": float(surface.parameter_scale),
        "training_parameter_bounds": np.asarray(
            surface.parameter_bounds, dtype=np.float64
        ).copy(),
        "extrusion_direction": np.asarray(surface.direction, dtype=np.float64).copy(),
        "extrusion_coefficients": surface.extrusion_coefficients,
        "extrusion_degree": surface.extrusion_degree,
        "parameter_feature": surface.parameter_feature,
        "coordinate_features": surface.coordinate_features,
        "metadata": dict(metadata or {}),
    }
    buffer = io.BytesIO()
    joblib.dump(payload, buffer, compress=3)
    return buffer.getvalue()


def load_surface(source: bytes | bytearray | str | Path | BinaryIO) -> ExtrudedSplineSurface:
    """Compatibility helper matching the original standalone reference API."""

    surface, _ = load_extruded_surface(source)
    return surface


def _fit_quadratic_extrusion(
    residuals: np.ndarray,
    *,
    max_iterations: int,
    tolerance: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, int, bool]:
    """Fit ``r_i ~= u_i*d1 + u_i**2*d2`` with deterministic multi-start ALS."""

    _, _, right_vectors = np.linalg.svd(residuals, full_matrices=False)
    seeds: list[np.ndarray] = []
    for vector in right_vectors:
        scores = residuals @ vector
        seeds.extend(
            [
                scores,
                np.sign(scores) * np.sqrt(np.abs(scores)),
                np.sqrt(np.maximum(scores - np.min(scores), 0.0)),
            ]
        )

    best: tuple[float, np.ndarray, np.ndarray, np.ndarray, int, bool] | None = None
    epsilon = np.finfo(np.float64).eps
    for seed in seeds:
        u = np.asarray(seed, dtype=np.float64).copy()
        rms = float(np.sqrt(np.mean(np.square(u))))
        if not np.isfinite(rms) or rms <= epsilon:
            continue
        u /= rms
        previous_error = np.inf
        converged = False
        iteration = 0
        linear = np.zeros(3, dtype=np.float64)
        quadratic = np.zeros(3, dtype=np.float64)
        for iteration in range(1, max_iterations + 1):
            design = np.column_stack([u, np.square(u)])
            coefficients, _, _, _ = np.linalg.lstsq(design, residuals, rcond=None)
            linear, quadratic = coefficients
            updated_u = _best_extrusion_values(residuals, linear, quadratic)
            updated_rms = float(np.sqrt(np.mean(np.square(updated_u))))
            if not np.isfinite(updated_rms) or updated_rms <= epsilon:
                break
            updated_u /= updated_rms
            # Refitting after rescaling preserves the represented curve and its conditioning.
            updated_design = np.column_stack([updated_u, np.square(updated_u)])
            updated_coefficients, _, _, _ = np.linalg.lstsq(
                updated_design, residuals, rcond=None
            )
            updated_linear, updated_quadratic = updated_coefficients
            prediction = (
                updated_u[:, np.newaxis] * updated_linear
                + np.square(updated_u)[:, np.newaxis] * updated_quadratic
            )
            error = float(np.sum(np.square(residuals - prediction)))
            relative_change = (
                np.inf
                if not np.isfinite(previous_error)
                else abs(previous_error - error) / max(abs(previous_error), epsilon)
            )
            u = updated_u
            linear = updated_linear
            quadratic = updated_quadratic
            if relative_change <= tolerance:
                converged = True
                break
            previous_error = error

        prediction = (
            u[:, np.newaxis] * linear
            + np.square(u)[:, np.newaxis] * quadratic
        )
        error = float(np.sum(np.square(residuals - prediction)))
        candidate = (error, linear.copy(), quadratic.copy(), u.copy(), iteration, converged)
        if best is None or candidate[0] < best[0]:
            best = candidate

    if best is None:
        raise ValueError("The quadratic extrusion could not be initialized from these points.")
    _, linear, quadratic, u, iterations, converged = best
    dominant_coordinate = int(np.argmax(np.abs(linear)))
    if np.abs(linear[dominant_coordinate]) > 1e-12 and linear[dominant_coordinate] < 0:
        linear = -linear
        u = -u
    return linear, quadratic, u, iterations, converged


def fit_extrusion_direction(
    curve_model: CurveModel,
    parameter_values: np.ndarray,
    points: np.ndarray,
    *,
    degree: int = 1,
    max_iterations: int = 200,
    tolerance: float = 1e-10,
) -> ExtrusionDirectionFitResult:
    """Fit a linear or quadratic extrusion around a fixed cubic spline.

    For fixed parameter values, each point has residual ``r_i = p_i - C(t_i)``.  The
    Degree 1 uses the global SVD solution. Degree 2 alternates between least-squares vector
    coefficients and exact per-point scalar projections, using deterministic multi-starts.
    """

    if not isinstance(curve_model, CurveModel):
        raise ValueError("The extrusion input must be a saved curve model.")
    if curve_model.algorithm != "spline" or int(curve_model.parameters.get("degree", 0)) not in (2, 3):
        raise ValueError("The extrusion input must be a degree-2 or degree-3 spline curve.")
    if degree not in {1, 2}:
        raise ValueError("Extrusion degree must be 1 or 2.")
    if max_iterations < 1:
        raise ValueError("Quadratic extrusion iterations must be positive.")
    if not np.isfinite(tolerance) or tolerance <= 0:
        raise ValueError("Quadratic extrusion tolerance must be positive and finite.")

    parameter = np.asarray(parameter_values, dtype=np.float64).reshape(-1)
    xyz = np.asarray(points, dtype=np.float64)
    if xyz.ndim != 2 or xyz.shape != (len(parameter), 3):
        raise ValueError("Points must have shape (n, 3) and match the parameter values.")
    finite = np.isfinite(parameter) & np.isfinite(xyz).all(axis=1)
    parameter = parameter[finite]
    xyz = xyz[finite]
    if not len(parameter):
        raise ValueError("At least one finite displayed point is required to fit an extrusion.")

    curve_xyz = curve_model.predict(parameter)
    residuals = xyz - curve_xyz
    residual_energy = float(np.sum(np.square(residuals)))
    warnings: list[str] = []
    quadratic_direction = None
    iterations = 0
    converged = True
    if residual_energy <= np.finfo(np.float64).eps:
        coordinate_span = np.ptp(xyz, axis=0)
        direction = np.zeros(3, dtype=np.float64)
        direction[int(np.argmax(coordinate_span))] = 1.0
        warnings.append(
            "The displayed points already lie on the cubic spline; the extrusion direction "
            "is not identifiable, so a deterministic coordinate direction was used."
        )
        captured_energy = 0.0
        if degree == 2:
            quadratic_direction = np.zeros(3, dtype=np.float64)
    else:
        if degree == 1:
            _, singular_values, right_vectors = np.linalg.svd(residuals, full_matrices=False)
            direction = np.asarray(right_vectors[0], dtype=np.float64)
            captured_energy = float(np.square(singular_values[0]))
            # SVD directions have arbitrary sign. Keep saved results deterministic.
            dominant_coordinate = int(np.argmax(np.abs(direction)))
            if direction[dominant_coordinate] < 0:
                direction = -direction
            point_extrusion = residuals @ direction
        else:
            direction, quadratic_direction, point_extrusion, iterations, converged = (
                _fit_quadratic_extrusion(
                    residuals,
                    max_iterations=max_iterations,
                    tolerance=tolerance,
                )
            )
            quadratic_prediction = (
                point_extrusion[:, np.newaxis] * direction
                + np.square(point_extrusion)[:, np.newaxis] * quadratic_direction
            )
            captured_energy = residual_energy - float(
                np.sum(np.square(residuals - quadratic_prediction))
            )

    if residual_energy <= np.finfo(np.float64).eps:
        point_extrusion = np.zeros(len(residuals), dtype=np.float64)
    point_projection = (
        curve_xyz
        + point_extrusion[:, np.newaxis] * direction
        + (
            0.0
            if quadratic_direction is None
            else np.square(point_extrusion)[:, np.newaxis] * quadratic_direction
        )
    )
    curve_distance = np.linalg.norm(residuals, axis=1)
    surface_distance = np.linalg.norm(xyz - point_projection, axis=1)
    surface = ExtrudedSplineSurface(
        predictors=curve_model.predictors,
        parameter_center=float(curve_model.parameter_center),
        parameter_scale=float(curve_model.parameter_scale),
        parameter_bounds=np.asarray(
            curve_model.training_parameter_bounds, dtype=np.float64
        ).copy(),
        direction=direction,
        parameter_feature=curve_model.parameter_feature,
        coordinate_features=curve_model.coordinate_features,
        quadratic_direction=quadratic_direction,
    )
    _validate_surface(surface)
    dropped_nonfinite = int((~finite).sum())
    if dropped_nonfinite:
        warnings.append(f"Dropped {dropped_nonfinite:,} non-finite displayed row(s).")
    if degree == 2 and not converged:
        warnings.append(
            f"Quadratic extrusion reached {max_iterations:,} iterations before convergence."
        )
    return ExtrusionDirectionFitResult(
        model=surface,
        point_parameter=parameter,
        point_xyz=xyz,
        curve_xyz=curve_xyz,
        point_projection=point_projection,
        point_extrusion=point_extrusion,
        metrics={
            "input_points": int(len(finite)),
            "fit_points": int(len(parameter)),
            "dropped_nonfinite": dropped_nonfinite,
            "curve_rmse_3d": float(np.sqrt(np.mean(np.square(curve_distance)))),
            "surface_rmse_3d": float(np.sqrt(np.mean(np.square(surface_distance)))),
            "surface_mae_3d": float(np.mean(surface_distance)),
            "surface_max_error_3d": float(np.max(surface_distance)),
            "captured_residual_variance": (
                1.0 if residual_energy <= np.finfo(np.float64).eps
                else captured_energy / residual_energy
            ),
            "extrusion_degree": degree,
            "optimization_iterations": iterations,
            "optimization_converged": converged,
        },
        warnings=tuple(warnings),
    )


def evaluate_extruded_surface(
    model: ExtrudedSplineSurface,
    points: np.ndarray,
    *,
    point_parameter: np.ndarray | None = None,
    parameter_samples: int = 120,
    extrusion_samples: int = 30,
    parameter_padding_fraction: float = 0.0,
    extrusion_padding_fraction: float = 0.1,
    extrusion_bounds: tuple[float, float] | None = None,
) -> ExtrudedSurfaceEvaluationResult:
    """Project current points and sample a regular ``(t, u)`` display grid."""

    _validate_surface(model)
    if not 20 <= int(parameter_samples) <= 1_000:
        raise ValueError("parameter_samples must be between 20 and 1,000.")
    if not 2 <= int(extrusion_samples) <= 300:
        raise ValueError("extrusion_samples must be between 2 and 300.")
    if not 0 <= float(parameter_padding_fraction) <= 1:
        raise ValueError("parameter_padding_fraction must be between 0 and 1.")
    if not 0 <= float(extrusion_padding_fraction) <= 2:
        raise ValueError("extrusion_padding_fraction must be between 0 and 2.")

    values = np.asarray(points, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] != 3:
        raise ValueError("Points must have shape (n, 3).")
    parameter = None
    if point_parameter is not None:
        parameter = np.asarray(point_parameter, dtype=np.float64).reshape(-1)
        if len(parameter) != len(values):
            raise ValueError("Point parameters must match the number of points.")
    finite = np.isfinite(values).all(axis=1)
    if parameter is not None:
        finite &= np.isfinite(parameter)
        parameter = parameter[finite]
    current_points = values[finite]
    if not len(current_points):
        raise ValueError("At least one finite current point is required.")
    if parameter is None:
        projected, evaluated_parameter, point_extrusion = model.project(current_points)
    else:
        curve_xyz = model.curve(parameter)
        residuals = current_points - curve_xyz
        point_extrusion = _best_extrusion_values(
            residuals,
            model.direction,
            model.quadratic_direction,
        )
        projected = model.evaluate(parameter, point_extrusion)
        evaluated_parameter = parameter

    parameter_low, parameter_high = model.parameter_bounds
    parameter_span = parameter_high - parameter_low
    parameter_low -= float(parameter_padding_fraction) * parameter_span
    parameter_high += float(parameter_padding_fraction) * parameter_span

    if extrusion_bounds is None:
        extrusion_low = float(np.min(point_extrusion))
        extrusion_high = float(np.max(point_extrusion))
        if np.isclose(extrusion_low, extrusion_high):
            coordinate_span = float(np.max(np.ptp(current_points, axis=0)))
            coefficient_scale = max(
                np.linalg.norm(model.direction),
                (
                    0.0
                    if model.quadratic_direction is None
                    else np.linalg.norm(model.quadratic_direction)
                ),
                np.finfo(np.float64).eps,
            )
            half_width = max(coordinate_span / (2 * coefficient_scale), 0.5)
            extrusion_low -= half_width
            extrusion_high += half_width
    else:
        bounds = np.asarray(extrusion_bounds, dtype=np.float64)
        if bounds.shape != (2,) or not np.isfinite(bounds).all() or bounds[0] >= bounds[1]:
            raise ValueError("Manual extrusion bounds must be two increasing finite values.")
        extrusion_low, extrusion_high = map(float, bounds)
    extrusion_span = extrusion_high - extrusion_low
    extrusion_low -= float(extrusion_padding_fraction) * extrusion_span
    extrusion_high += float(extrusion_padding_fraction) * extrusion_span

    parameter_axis = np.linspace(parameter_low, parameter_high, int(parameter_samples))
    extrusion_axis = np.linspace(extrusion_low, extrusion_high, int(extrusion_samples))
    grid_parameter, grid_extrusion = np.meshgrid(parameter_axis, extrusion_axis)
    grid_xyz = model.evaluate(grid_parameter, grid_extrusion)

    residual_distance = np.linalg.norm(current_points - projected, axis=1)
    warnings: list[str] = []
    dropped_nonfinite = int((~finite).sum())
    if dropped_nonfinite:
        warnings.append(f"Dropped {dropped_nonfinite:,} non-finite current row(s).")
    return ExtrudedSurfaceEvaluationResult(
        model=model,
        grid_parameter=grid_parameter,
        grid_extrusion=grid_extrusion,
        grid_xyz=grid_xyz,
        point_xyz=current_points,
        point_projection=projected,
        point_parameter=evaluated_parameter,
        point_extrusion=point_extrusion,
        metrics={
            "input_points": int(len(values)),
            "current_points": int(len(current_points)),
            "dropped_nonfinite": dropped_nonfinite,
            "current_rmse_3d": float(np.sqrt(np.mean(np.square(residual_distance)))),
            "current_mae_3d": float(np.mean(residual_distance)),
            "current_max_error_3d": float(np.max(residual_distance)),
            "parameter_aligned_projection": parameter is not None,
            "display_parameter_bounds": [float(parameter_low), float(parameter_high)],
            "display_extrusion_bounds": [float(extrusion_low), float(extrusion_high)],
        },
        warnings=tuple(warnings),
    )
