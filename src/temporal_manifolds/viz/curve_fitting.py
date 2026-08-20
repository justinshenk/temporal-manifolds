"""Fit, evaluate, and serialize parameterized curves in three dimensions."""

from __future__ import annotations

import ast
import io
from dataclasses import dataclass
from pathlib import Path
from typing import Any, BinaryIO

import joblib
import numpy as np
from scipy import __version__ as scipy_version
from scipy.interpolate import BSpline
from scipy.sparse.csgraph import dijkstra, minimum_spanning_tree
from scipy.spatial import cKDTree
from scipy.spatial.distance import pdist, squareform
from sklearn import __version__ as sklearn_version
from sklearn.metrics import r2_score

CURVE_MODEL_ARTIFACT_VERSION = 4
CURVE_ALGORITHMS = {
    "spline": "Endpoint-constrained geometric spline",
}
CURVE_DESCRIPTIONS = {
    "spline": (
        "Fits a constrained SciPy BSpline through the 3D point cloud. A dummy parameter "
        "t from 0 to 1 is inferred geometrically, while the supplied endpoints are "
        "enforced exactly at t=0 and t=1."
    ),
}


@dataclass(frozen=True)
class CurveModel:
    """A reusable mapping from one scalar parameter to three chart coordinates."""

    algorithm: str
    predictors: tuple[Any, Any, Any]
    parameter_feature: str
    coordinate_features: tuple[str, str, str]
    parameter_center: float
    parameter_scale: float
    training_parameter_bounds: np.ndarray
    training_coordinate_bounds: np.ndarray
    parameters: dict[str, Any]

    def predict(self, parameter_values: np.ndarray | list[float]) -> np.ndarray:
        """Predict coordinates for raw (unnormalized) parameter values."""

        values = np.asarray(parameter_values, dtype=np.float64).reshape(-1)
        if not np.isfinite(values).all():
            raise ValueError("Curve parameter values must be finite.")
        normalized = (values - self.parameter_center) / self.parameter_scale
        predictions = np.column_stack(
            [_predict_coordinate(predictor, normalized) for predictor in self.predictors]
        )
        if predictions.shape != (len(values), 3):
            raise ValueError("The fitted curve returned an invalid prediction shape.")
        return np.asarray(predictions, dtype=np.float64)

    def extrapolation_mask(self, parameter_values: np.ndarray | list[float]) -> np.ndarray:
        """Identify values outside the parameter interval used for fitting."""

        values = np.asarray(parameter_values, dtype=np.float64).reshape(-1)
        bounds = np.asarray(self.training_parameter_bounds, dtype=np.float64)
        return (values < bounds[0]) | (values > bounds[1])


@dataclass(frozen=True)
class CurveFitResult:
    model: CurveModel
    curve_parameter: np.ndarray
    curve_xyz: np.ndarray
    point_parameter: np.ndarray
    point_xyz: np.ndarray
    point_prediction: np.ndarray
    metrics: dict[str, Any]
    warnings: tuple[str, ...]

    @property
    def algorithm(self) -> str:
        return self.model.algorithm


@dataclass(frozen=True)
class CurveEvaluationResult:
    model: CurveModel
    curve_parameter: np.ndarray
    curve_xyz: np.ndarray
    point_parameter: np.ndarray
    point_xyz: np.ndarray
    point_prediction: np.ndarray
    metrics: dict[str, Any]
    warnings: tuple[str, ...]

    @property
    def algorithm(self) -> str:
        return self.model.algorithm


CurveDisplayResult = CurveFitResult | CurveEvaluationResult


@dataclass(frozen=True)
class TangentExtrapolatingBSpline:
    """Evaluate a BSpline inside its fit range and tangent lines outside it."""

    spline: BSpline
    lower_bound: float
    upper_bound: float
    derivative_order: int = 0

    def __call__(self, values: np.ndarray | float) -> np.ndarray:
        array = np.asarray(values, dtype=np.float64)
        flat = array.reshape(-1)
        clipped = np.clip(flat, self.lower_bound, self.upper_bound)
        if self.derivative_order == 0:
            result = np.asarray(self.spline(clipped), dtype=np.float64)
            derivative = self.spline.derivative()
            below = flat < self.lower_bound
            above = flat > self.upper_bound
            result[below] = self.spline(self.lower_bound) + derivative(self.lower_bound) * (
                flat[below] - self.lower_bound
            )
            result[above] = self.spline(self.upper_bound) + derivative(self.upper_bound) * (
                flat[above] - self.upper_bound
            )
        elif self.derivative_order == 1:
            derivative = self.spline.derivative()
            result = np.asarray(derivative(clipped), dtype=np.float64)
        else:
            result = np.asarray(
                self.spline.derivative(self.derivative_order)(clipped), dtype=np.float64
            )
            result[(flat < self.lower_bound) | (flat > self.upper_bound)] = 0.0
        return result.reshape(array.shape)

    def derivative(self, nu: int = 1) -> "TangentExtrapolatingBSpline":
        """Return a callable derivative compatible with SciPy spline predictors."""

        if nu < 0:
            raise ValueError("Derivative order must be non-negative.")
        return TangentExtrapolatingBSpline(
            self.spline,
            self.lower_bound,
            self.upper_bound,
            self.derivative_order + int(nu),
        )


def _predict_coordinate(predictor: Any, values: np.ndarray) -> np.ndarray:
    return np.asarray(predictor(values), dtype=np.float64)


def parse_spline_quantiles(value: str) -> list[float]:
    """Parse a Python list of strictly increasing interior quantiles."""

    try:
        parsed = ast.literal_eval(value.strip())
    except (SyntaxError, ValueError) as exc:
        raise ValueError(
            "Spline quantiles must be a valid Python list, such as [0.25, 0.5, 0.75]."
        ) from exc
    if not isinstance(parsed, list):
        raise ValueError("Spline quantiles must be entered as a Python list.")
    if any(isinstance(item, bool) or not isinstance(item, (int, float)) for item in parsed):
        raise ValueError("Every spline quantile must be a finite number.")
    quantiles = [float(item) for item in parsed]
    if not np.isfinite(quantiles).all():
        raise ValueError("Every spline quantile must be a finite number.")
    if any(right <= left for left, right in zip(quantiles, quantiles[1:], strict=False)):
        raise ValueError("Spline quantiles must be strictly increasing with no duplicates.")
    if quantiles and (quantiles[0] <= 0 or quantiles[-1] >= 1):
        raise ValueError("Spline quantiles must lie strictly between 0 and 1.")
    return quantiles


def _fit_splines(
    parameter: np.ndarray,
    coordinates: np.ndarray,
    *,
    degree: int,
    smoothing: float,
    endpoints: np.ndarray,
    knots: list[float] | None = None,
) -> tuple[Any, Any, Any]:
    if not 1 <= degree <= 5:
        raise ValueError("Spline degree must be between 1 and 5.")
    if len(parameter) <= degree:
        raise ValueError(f"A degree-{degree} spline requires at least {degree + 1} values.")
    if not np.isfinite(smoothing) or smoothing < 0:
        raise ValueError("Spline smoothing must be finite and non-negative.")
    boundary = np.asarray(endpoints, dtype=np.float64)
    if boundary.shape != (2, 3) or not np.isfinite(boundary).all():
        raise ValueError("Spline endpoints must contain two finite 3D coordinates.")
    interior_knots = np.asarray(knots or [], dtype=np.float64)
    if len(interior_knots):
        if not np.isfinite(interior_knots).all() or np.any(np.diff(interior_knots) <= 0):
            raise ValueError("Spline knots must be finite and strictly increasing.")
        if interior_knots[0] <= parameter[0] or interior_knots[-1] >= parameter[-1]:
            raise ValueError("Spline knots must lie strictly inside the fitted parameter range.")
    else:
        automatic_count = min(12, max(1, len(parameter) // 8))
        interior_knots = np.quantile(parameter, np.linspace(0, 1, automatic_count + 2)[1:-1])
        interior_knots = np.unique(interior_knots)
    knot_vector = np.concatenate(
        [
            np.repeat(parameter[0], degree + 1),
            interior_knots,
            np.repeat(parameter[-1], degree + 1),
        ]
    )
    design = BSpline.design_matrix(parameter, knot_vector, degree, extrapolate=False).toarray()
    coefficient_count = design.shape[1]
    if coefficient_count < 2:
        raise ValueError("The spline basis cannot represent two constrained endpoints.")
    free_design = design[:, 1:-1]
    fixed_fit = np.outer(design[:, 0], boundary[0]) + np.outer(design[:, -1], boundary[1])
    difference = np.diff(np.eye(coefficient_count), n=2, axis=0)
    free_difference = difference[:, 1:-1]
    fixed_difference = np.outer(difference[:, 0], boundary[0]) + np.outer(
        difference[:, -1], boundary[1]
    )
    coefficients = np.empty((coefficient_count, 3), dtype=np.float64)
    coefficients[0] = boundary[0]
    coefficients[-1] = boundary[1]
    if coefficient_count > 2:
        penalty = float(smoothing) * len(parameter)
        system = free_design.T @ free_design
        right = free_design.T @ (coordinates - fixed_fit)
        if len(difference) and penalty:
            system += penalty * (free_difference.T @ free_difference)
            right -= penalty * (free_difference.T @ fixed_difference)
        coefficients[1:-1] = np.linalg.lstsq(system, right, rcond=None)[0]
    return tuple(
        TangentExtrapolatingBSpline(
            BSpline(knot_vector, coefficients[:, index], degree, extrapolate=False),
            float(parameter[0]),
            float(parameter[-1]),
        )
        for index in range(3)
    )  # type: ignore[return-value]


def _merge_duplicate_parameters(
    parameter: np.ndarray,
    coordinates: np.ndarray,
    reducer: str,
) -> tuple[np.ndarray, np.ndarray]:
    if reducer not in {"mean", "median"}:
        raise ValueError("Duplicate parameter reducer must be 'mean' or 'median'.")
    unique_parameter, inverse = np.unique(parameter, return_inverse=True)
    reduced = np.empty((len(unique_parameter), 3), dtype=np.float64)
    reduction = np.mean if reducer == "mean" else np.median
    for index in range(len(unique_parameter)):
        reduced[index] = reduction(coordinates[inverse == index], axis=0)
    return unique_parameter, reduced


def _coordinate_metrics(actual: np.ndarray, predicted: np.ndarray) -> dict[str, Any]:
    if len(actual) == 0:
        return {
            "count": 0,
            "rmse_3d": None,
            "mae_3d": None,
            "r2": None,
            "coordinate_rmse": [None, None, None],
        }
    difference = predicted - actual
    distance = np.linalg.norm(difference, axis=1)
    coordinate_rmse = np.sqrt(np.mean(np.square(difference), axis=0))
    score = None
    if len(actual) >= 2 and np.any(np.ptp(actual, axis=0) > np.finfo(np.float64).eps):
        score = float(r2_score(actual, predicted, multioutput="variance_weighted"))
    return {
        "count": int(len(actual)),
        "rmse_3d": float(np.sqrt(np.mean(np.square(distance)))),
        "mae_3d": float(np.mean(distance)),
        "r2": score,
        "coordinate_rmse": coordinate_rmse.tolist(),
    }


def _mst_backbone(
    coordinates: np.ndarray,
    start_point: np.ndarray | None = None,
    end_point: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Return an endpoint-oriented path through the Euclidean minimum spanning tree."""

    values = np.asarray(coordinates, dtype=np.float64)
    if start_point is not None and end_point is not None:
        values = np.vstack([values, start_point, end_point])
    unique_coordinates = np.unique(values, axis=0)
    if len(unique_coordinates) < 3:
        raise ValueError("A geometric spline requires at least three distinct 3D points.")
    distances = squareform(pdist(unique_coordinates, metric="euclidean"))
    tree = minimum_spanning_tree(distances)
    graph = tree + tree.T
    if start_point is None or end_point is None:
        first_distances = dijkstra(graph, indices=0)
        endpoint = int(np.argmax(first_distances))
        second_distances, predecessors = dijkstra(graph, indices=endpoint, return_predecessors=True)
        opposite = int(np.argmax(second_distances))
    else:
        endpoint = int(np.flatnonzero(np.all(unique_coordinates == start_point, axis=1))[0])
        opposite = int(np.flatnonzero(np.all(unique_coordinates == end_point, axis=1))[0])
        _, predecessors = dijkstra(graph, indices=endpoint, return_predecessors=True)
    path = [opposite]
    while path[-1] != endpoint:
        predecessor = int(predecessors[path[-1]])
        if predecessor < 0:
            raise ValueError("Could not construct a connected geometric spline backbone.")
        path.append(predecessor)
    path.reverse()
    backbone = unique_coordinates[path]
    chord_lengths = np.linalg.norm(np.diff(backbone, axis=0), axis=1)
    cumulative = np.concatenate([[0.0], np.cumsum(chord_lengths)])
    if cumulative[-1] <= np.finfo(np.float64).eps:
        raise ValueError("The geometric spline points have no measurable extent.")
    return backbone, cumulative / cumulative[-1]


def geometric_spline_endpoint_defaults(coordinates: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Choose deterministic endpoint defaults from the point-cloud MST diameter."""

    finite = np.asarray(coordinates, dtype=np.float64)
    if finite.ndim != 2 or finite.shape[1] != 3:
        raise ValueError("Coordinates must have shape (n_points, 3).")
    finite = finite[np.isfinite(finite).all(axis=1)]
    if len(finite) > 3_000:
        indices = np.linspace(0, len(finite) - 1, 3_000).round().astype(int)
        finite = finite[indices]
    backbone, _ = _mst_backbone(finite)
    return backbone[0].copy(), backbone[-1].copy()


def project_onto_curve_parameter(
    model: CurveModel,
    coordinates: np.ndarray,
    *,
    grid_size: int = 4_001,
) -> np.ndarray:
    """Estimate nearest-curve parameter values by dense Euclidean projection."""

    points = np.asarray(coordinates, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError("Coordinates must have shape (n_points, 3).")
    if not np.isfinite(points).all():
        raise ValueError("Coordinates must be finite for geometric curve projection.")
    if grid_size < 100:
        raise ValueError("Curve projection grid size must be at least 100.")
    bounds = np.asarray(model.training_parameter_bounds, dtype=np.float64)
    parameter_grid = np.linspace(bounds[0], bounds[1], int(grid_size))
    curve_grid = model.predict(parameter_grid)
    _, nearest = cKDTree(curve_grid).query(points)
    return parameter_grid[np.asarray(nearest, dtype=int)]


def _expanded_coordinate_bounds(
    coordinate_bounds: np.ndarray, padding_fraction: float
) -> np.ndarray:
    bounds = np.asarray(coordinate_bounds, dtype=np.float64)
    if bounds.shape != (3, 2) or not np.isfinite(bounds).all():
        raise ValueError("Curve display coordinate bounds must have finite shape (3, 2).")
    if np.any(bounds[:, 0] > bounds[:, 1]):
        raise ValueError("Curve display coordinate bounds must be ordered.")
    spans = bounds[:, 1] - bounds[:, 0]
    fallback = np.maximum(np.abs(bounds).max(axis=1), 1.0) * 1e-9
    padding = np.maximum(spans, fallback) * float(padding_fraction)
    return np.column_stack([bounds[:, 0] - padding, bounds[:, 1] + padding])


def _sample_curve(
    model: CurveModel,
    sample_count: int,
    padding_fraction: float,
    coordinate_bounds: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if not 20 <= sample_count <= 5_000:
        raise ValueError("Curve sample count must be between 20 and 5,000.")
    if not np.isfinite(padding_fraction) or not 0 <= padding_fraction <= 1:
        raise ValueError("Curve padding fraction must be between 0 and 1.")
    display_bounds = _expanded_coordinate_bounds(coordinate_bounds, padding_fraction)
    lower, upper = np.asarray(model.training_parameter_bounds, dtype=np.float64)
    parameter_padding = float(padding_fraction) * (upper - lower)
    lower -= parameter_padding
    upper += parameter_padding
    curve_parameter = np.linspace(lower, upper, sample_count)
    return curve_parameter, model.predict(curve_parameter), display_bounds


def _fit_parameterized_spline(
    parameter_values: np.ndarray,
    coordinates: np.ndarray,
    *,
    endpoint_coordinates: np.ndarray,
    parameters: dict[str, Any] | None = None,
    sample_count: int = 240,
    padding_fraction: float = 0.0,
    display_coordinate_bounds: np.ndarray | None = None,
    duplicate_reducer: str = "mean",
    max_fit_points: int | None = None,
    validation_fraction: float = 0.2,
    random_state: int = 42,
    parameter_feature: str = "parameter",
    coordinate_features: tuple[str, str, str] = ("x", "y", "z"),
) -> CurveFitResult:
    """Internal spline fit for parameters inferred by the geometric optimizer."""

    if len(coordinate_features) != 3 or len(set(coordinate_features)) != 3:
        raise ValueError("A curve requires three distinct coordinate feature names.")
    if not 0 <= validation_fraction < 0.5:
        raise ValueError("Validation fraction must be at least 0 and below 0.5.")

    parameter = np.asarray(parameter_values, dtype=np.float64).reshape(-1)
    xyz = np.asarray(coordinates, dtype=np.float64)
    if xyz.ndim != 2 or xyz.shape[1] != 3 or len(xyz) != len(parameter):
        raise ValueError("Coordinates must have shape (n_points, 3) and match the parameter.")
    finite = np.isfinite(parameter) & np.isfinite(xyz).all(axis=1)
    dropped_nonfinite = int((~finite).sum())
    parameter = parameter[finite]
    xyz = xyz[finite]
    if len(parameter) < 3:
        raise ValueError("At least three finite points are required to fit a curve.")

    unique_parameter, unique_xyz = _merge_duplicate_parameters(parameter, xyz, duplicate_reducer)
    endpoints = np.asarray(endpoint_coordinates, dtype=np.float64)
    if endpoints.shape != (2, 3) or not np.isfinite(endpoints).all():
        raise ValueError("Spline endpoints must contain two finite 3D coordinates.")
    if np.allclose(endpoints[0], endpoints[1]):
        raise ValueError("Spline start and end coordinates must be different.")
    for endpoint_parameter, endpoint_xyz in zip((0.0, 1.0), endpoints, strict=True):
        matches = np.isclose(unique_parameter, endpoint_parameter, atol=1e-12)
        if np.any(matches):
            unique_xyz[np.flatnonzero(matches)[0]] = endpoint_xyz
        else:
            unique_parameter = np.append(unique_parameter, endpoint_parameter)
            unique_xyz = np.vstack([unique_xyz, endpoint_xyz])
    order = np.argsort(unique_parameter)
    unique_parameter = unique_parameter[order]
    unique_xyz = unique_xyz[order]
    if len(unique_parameter) < 3:
        raise ValueError("At least three distinct parameter values are required to fit a curve.")
    center = float(np.mean(unique_parameter))
    scale = float(np.std(unique_parameter))
    if scale <= np.finfo(np.float64).eps:
        raise ValueError("The curve parameter must vary.")
    normalized = (unique_parameter - center) / scale

    if max_fit_points is None:
        max_fit_points = len(unique_parameter)
    if max_fit_points < 3:
        raise ValueError("Maximum fit points must be at least 3.")
    if len(unique_parameter) > max_fit_points:
        selected = np.linspace(0, len(unique_parameter) - 1, max_fit_points).round().astype(int)
        unique_parameter = unique_parameter[selected]
        unique_xyz = unique_xyz[selected]
        normalized = normalized[selected]

    rng = np.random.default_rng(random_state)
    validation_count = int(np.floor(len(normalized) * validation_fraction))
    validation_count = min(validation_count, max(0, len(normalized) - 3))
    validation_indices = (
        np.sort(rng.choice(len(normalized), size=validation_count, replace=False))
        if validation_count
        else np.array([], dtype=int)
    )
    train_mask = np.ones(len(normalized), dtype=bool)
    train_mask[validation_indices] = False
    train_t = normalized[train_mask]
    train_xyz = unique_xyz[train_mask]
    order = np.argsort(train_t)
    train_t = train_t[order]
    train_xyz = train_xyz[order]

    model_parameters = dict(parameters or {})
    model_parameters.setdefault("degree", min(3, len(train_t) - 1))
    model_parameters.setdefault("smoothing", 0.15)
    raw_quantiles = model_parameters.get("knot_quantiles")
    raw_count = model_parameters.get("knot_count")
    if raw_quantiles is not None and raw_count is not None:
        raise ValueError("Specify spline knot quantiles or a knot count, not both.")
    if raw_quantiles is not None:
        if not isinstance(raw_quantiles, list):
            raise ValueError("Spline knot quantiles must be provided as a Python list.")
        quantiles = parse_spline_quantiles(repr(raw_quantiles))
        model_parameters["knot_quantiles"] = quantiles
    elif raw_count is not None:
        if isinstance(raw_count, bool) or not isinstance(raw_count, int) or raw_count < 0:
            raise ValueError("Spline knot count must be a non-negative integer.")
        quantiles = np.linspace(0, 1, raw_count + 2)[1:-1].tolist()
        model_parameters["knot_count"] = raw_count
    else:
        quantiles = []
    normalized_knots = np.quantile(train_t, quantiles).astype(float).tolist() if quantiles else None
    if normalized_knots is not None:
        if len(np.unique(normalized_knots)) != len(normalized_knots):
            raise ValueError(
                "The selected knot quantiles produce duplicate positions. Use fewer or "
                "more widely separated quantiles."
            )
        model_parameters["resolved_knots"] = [
            float(knot * scale + center) for knot in normalized_knots
        ]
    predictors = _fit_splines(
        train_t,
        train_xyz,
        degree=model_parameters["degree"],
        smoothing=model_parameters["smoothing"],
        endpoints=endpoints,
        knots=normalized_knots,
    )

    model = CurveModel(
        algorithm="spline",
        predictors=predictors,
        parameter_feature=str(parameter_feature),
        coordinate_features=tuple(coordinate_features),
        parameter_center=center,
        parameter_scale=scale,
        training_parameter_bounds=np.array(
            [float(unique_parameter.min()), float(unique_parameter.max())], dtype=np.float64
        ),
        training_coordinate_bounds=np.column_stack(
            [unique_xyz.min(axis=0), unique_xyz.max(axis=0)]
        ),
        parameters=model_parameters,
    )
    if display_coordinate_bounds is None:
        display_coordinate_bounds = model.training_coordinate_bounds
    curve_parameter, curve_xyz, display_bounds = _sample_curve(
        model,
        sample_count,
        padding_fraction,
        display_coordinate_bounds,
    )
    point_prediction = model.predict(parameter)
    training_prediction = model.predict(unique_parameter[train_mask])
    validation_prediction = (
        model.predict(unique_parameter[validation_indices])
        if len(validation_indices)
        else np.empty((0, 3), dtype=np.float64)
    )
    training_metrics = _coordinate_metrics(unique_xyz[train_mask], training_prediction)
    validation_metrics = _coordinate_metrics(unique_xyz[validation_indices], validation_prediction)
    warnings = []
    if dropped_nonfinite:
        warnings.append(f"Dropped {dropped_nonfinite:,} non-finite row(s).")
    if len(validation_indices) == 0:
        warnings.append("Held-out validation was disabled or the fit had too few unique values.")
    metrics = {
        "input_points": int(len(finite)),
        "finite_points": int(finite.sum()),
        "unique_parameter_values": int(len(unique_parameter)),
        "duplicates_merged": int(len(parameter) - len(np.unique(parameter))),
        "dropped_nonfinite": dropped_nonfinite,
        "fit_points": int(len(train_t)),
        "validation_points": int(len(validation_indices)),
        "train_rmse_3d": training_metrics["rmse_3d"],
        "train_mae_3d": training_metrics["mae_3d"],
        "train_r2": training_metrics["r2"],
        "train_coordinate_rmse": training_metrics["coordinate_rmse"],
        "validation_rmse_3d": validation_metrics["rmse_3d"],
        "validation_mae_3d": validation_metrics["mae_3d"],
        "validation_r2": validation_metrics["r2"],
        "validation_coordinate_rmse": validation_metrics["coordinate_rmse"],
        "display_padding_fraction": float(padding_fraction),
        "display_coordinate_bounds": display_bounds.tolist(),
    }
    return CurveFitResult(
        model=model,
        curve_parameter=curve_parameter,
        curve_xyz=curve_xyz,
        point_parameter=parameter,
        point_xyz=xyz,
        point_prediction=point_prediction,
        metrics=metrics,
        warnings=tuple(warnings),
    )


def fit_geometric_spline(
    coordinates: np.ndarray,
    *,
    start_point: np.ndarray,
    end_point: np.ndarray,
    parameters: dict[str, Any] | None = None,
    sample_count: int = 240,
    padding_fraction: float = 0.0,
    display_coordinate_bounds: np.ndarray | None = None,
    max_fit_points: int | None = None,
    random_state: int = 42,
    coordinate_features: tuple[str, str, str] = ("x", "y", "z"),
    refinement_iterations: int = 4,
) -> CurveFitResult:
    """Fit a 3D principal spline whose dummy parameter is inferred geometrically."""

    if not 1 <= refinement_iterations <= 20:
        raise ValueError("Geometric spline refinement iterations must be between 1 and 20.")
    xyz = np.asarray(coordinates, dtype=np.float64)
    if xyz.ndim != 2 or xyz.shape[1] != 3:
        raise ValueError("Coordinates must have shape (n_points, 3).")
    finite = np.isfinite(xyz).all(axis=1)
    dropped_nonfinite = int((~finite).sum())
    xyz = xyz[finite]
    if len(xyz) < 4:
        raise ValueError("At least four finite points are required for a geometric spline.")
    start = np.asarray(start_point, dtype=np.float64).reshape(-1)
    end = np.asarray(end_point, dtype=np.float64).reshape(-1)
    if start.shape != (3,) or end.shape != (3,) or not np.isfinite([start, end]).all():
        raise ValueError("Spline start and end must each contain three finite coordinates.")
    if np.allclose(start, end):
        raise ValueError("Spline start and end coordinates must be different.")
    endpoints = np.vstack([start, end])

    if max_fit_points is None:
        max_fit_points = min(len(xyz), 3_000)
    if max_fit_points < 4:
        raise ValueError("Maximum geometric spline fit points must be at least 4.")
    rng = np.random.default_rng(random_state)
    if len(xyz) > max_fit_points:
        backbone_indices = np.sort(rng.choice(len(xyz), size=max_fit_points, replace=False))
        backbone_source = xyz[backbone_indices]
    else:
        backbone_source = xyz
    backbone, backbone_parameter = _mst_backbone(backbone_source, start, end)
    _, nearest_backbone = cKDTree(backbone).query(xyz)
    parameter = backbone_parameter[np.asarray(nearest_backbone, dtype=int)]

    model_parameters = dict(parameters or {})
    model_parameters["parameterization"] = "geometric_principal_curve"
    model_parameters["endpoint_constraint"] = "exact"
    model_parameters["start_point"] = start.tolist()
    model_parameters["end_point"] = end.tolist()
    model_parameters["spline_implementation"] = "scipy.interpolate.BSpline"
    model_parameters["extrapolation"] = "endpoint_tangent_linear"
    model_parameters["refinement_iterations"] = int(refinement_iterations)
    fitted: CurveFitResult | None = None
    for _ in range(refinement_iterations):
        fitted = _fit_parameterized_spline(
            parameter,
            xyz,
            endpoint_coordinates=endpoints,
            parameters=model_parameters,
            sample_count=sample_count,
            padding_fraction=padding_fraction,
            display_coordinate_bounds=display_coordinate_bounds,
            duplicate_reducer="mean",
            max_fit_points=max_fit_points,
            validation_fraction=0.0,
            random_state=random_state,
            parameter_feature="t",
            coordinate_features=coordinate_features,
        )
        updated_parameter = project_onto_curve_parameter(fitted.model, xyz)
        if np.max(np.abs(updated_parameter - parameter)) < 1e-5:
            parameter = updated_parameter
            break
        parameter = updated_parameter
    assert fitted is not None
    fitted = _fit_parameterized_spline(
        parameter,
        xyz,
        endpoint_coordinates=endpoints,
        parameters=model_parameters,
        sample_count=sample_count,
        padding_fraction=padding_fraction,
        display_coordinate_bounds=display_coordinate_bounds,
        duplicate_reducer="mean",
        max_fit_points=max_fit_points,
        validation_fraction=0.0,
        random_state=random_state,
        parameter_feature="t",
        coordinate_features=coordinate_features,
    )
    parameter = project_onto_curve_parameter(fitted.model, xyz, grid_size=10_001)
    point_prediction = fitted.model.predict(parameter)
    geometric_metrics = _coordinate_metrics(xyz, point_prediction)
    metrics = {
        **fitted.metrics,
        "input_points": int(len(finite)),
        "finite_points": int(finite.sum()),
        "dropped_nonfinite": dropped_nonfinite,
        "fit_points": int(len(xyz)),
        "validation_points": 0,
        "train_rmse_3d": geometric_metrics["rmse_3d"],
        "train_mae_3d": geometric_metrics["mae_3d"],
        "train_r2": geometric_metrics["r2"],
        "train_coordinate_rmse": geometric_metrics["coordinate_rmse"],
        "geometric_rmse_3d": geometric_metrics["rmse_3d"],
        "geometric_mae_3d": geometric_metrics["mae_3d"],
    }
    warnings = [
        warning for warning in fitted.warnings if not warning.startswith("Held-out validation")
    ]
    if dropped_nonfinite:
        warnings.append(f"Dropped {dropped_nonfinite:,} non-finite row(s).")
    return CurveFitResult(
        model=fitted.model,
        curve_parameter=fitted.curve_parameter,
        curve_xyz=fitted.curve_xyz,
        point_parameter=parameter,
        point_xyz=xyz,
        point_prediction=point_prediction,
        metrics=metrics,
        warnings=tuple(dict.fromkeys(warnings)),
    )


def evaluate_curve_model(
    model: CurveModel,
    parameter_values: np.ndarray,
    coordinates: np.ndarray,
    *,
    sample_count: int = 240,
    padding_fraction: float = 0.0,
    display_coordinate_bounds: np.ndarray | None = None,
) -> CurveEvaluationResult:
    """Evaluate a saved curve on current points and construct a chart line."""

    _validate_curve_model(model)
    parameter = np.asarray(parameter_values, dtype=np.float64).reshape(-1)
    xyz = np.asarray(coordinates, dtype=np.float64)
    if xyz.ndim != 2 or xyz.shape != (len(parameter), 3):
        raise ValueError("Coordinates must have shape (n_points, 3) and match the parameter.")
    finite = np.isfinite(parameter) & np.isfinite(xyz).all(axis=1)
    parameter = parameter[finite]
    xyz = xyz[finite]
    if not len(parameter):
        raise ValueError("At least one finite current point is required to preview a curve.")
    if display_coordinate_bounds is None:
        display_coordinate_bounds = np.column_stack([xyz.min(axis=0), xyz.max(axis=0)])
    curve_parameter, curve_xyz, display_bounds = _sample_curve(
        model,
        sample_count,
        padding_fraction,
        display_coordinate_bounds,
    )
    point_prediction = model.predict(parameter)
    point_metrics = _coordinate_metrics(xyz, point_prediction)
    extrapolation = model.extrapolation_mask(parameter)
    warnings = []
    if np.any(~finite):
        warnings.append(f"Dropped {int((~finite).sum()):,} non-finite current row(s).")
    if extrapolation.any():
        warnings.append(
            f"{int(extrapolation.sum()):,} current point(s) lie outside the fitted parameter range."
        )
    return CurveEvaluationResult(
        model=model,
        curve_parameter=curve_parameter,
        curve_xyz=curve_xyz,
        point_parameter=parameter,
        point_xyz=xyz,
        point_prediction=point_prediction,
        metrics={
            "current_points": int(len(parameter)),
            "current_rmse_3d": point_metrics["rmse_3d"],
            "current_mae_3d": point_metrics["mae_3d"],
            "current_r2": point_metrics["r2"],
            "current_coordinate_rmse": point_metrics["coordinate_rmse"],
            "extrapolation_points": int(extrapolation.sum()),
            "display_padding_fraction": float(padding_fraction),
            "display_coordinate_bounds": display_bounds.tolist(),
        },
        warnings=tuple(warnings),
    )


def _validate_curve_model(model: CurveModel) -> None:
    if not isinstance(model, CurveModel) or model.algorithm not in CURVE_ALGORITHMS:
        raise ValueError("The artifact does not contain a supported fitted curve model.")
    if model.parameters.get("parameterization") != "geometric_principal_curve":
        raise ValueError("Only geometric spline curve artifacts are supported.")
    if model.parameter_feature != "t":
        raise ValueError("A geometric spline artifact must use the dummy parameter 't'.")
    if model.parameters.get("endpoint_constraint") != "exact":
        raise ValueError("A geometric spline artifact must contain exact endpoint constraints.")
    if model.parameters.get("extrapolation") != "endpoint_tangent_linear":
        raise ValueError("A geometric spline artifact must use endpoint-tangent extrapolation.")
    start = np.asarray(model.parameters.get("start_point"), dtype=np.float64)
    end = np.asarray(model.parameters.get("end_point"), dtype=np.float64)
    if start.shape != (3,) or end.shape != (3,) or not np.isfinite([start, end]).all():
        raise ValueError("The fitted curve has invalid endpoint coordinates.")
    if len(model.predictors) != 3 or len(model.coordinate_features) != 3:
        raise ValueError("The fitted curve must contain exactly three coordinate predictors.")
    if not np.isfinite(model.parameter_center) or not np.isfinite(model.parameter_scale):
        raise ValueError("The fitted curve has invalid parameter normalization.")
    if model.parameter_scale <= 0:
        raise ValueError("The fitted curve parameter scale must be positive.")
    parameter_bounds = np.asarray(model.training_parameter_bounds, dtype=np.float64)
    coordinate_bounds = np.asarray(model.training_coordinate_bounds, dtype=np.float64)
    if parameter_bounds.shape != (2,) or not np.isfinite(parameter_bounds).all():
        raise ValueError("The fitted curve has invalid parameter bounds.")
    if parameter_bounds[0] >= parameter_bounds[1]:
        raise ValueError("The fitted curve parameter bounds must increase.")
    if coordinate_bounds.shape != (3, 2) or not np.isfinite(coordinate_bounds).all():
        raise ValueError("The fitted curve has invalid coordinate bounds.")
    model.predict(parameter_bounds)
    predicted_endpoints = model.predict(parameter_bounds)
    if not np.allclose(predicted_endpoints, np.vstack([start, end]), rtol=1e-9, atol=1e-10):
        raise ValueError("The fitted curve does not satisfy its endpoint constraints.")


def serialize_curve_model(model: CurveModel, metadata: dict[str, Any] | None = None) -> bytes:
    """Serialize a curve and provenance in a versioned joblib artifact."""

    _validate_curve_model(model)
    payload = {
        "artifact_kind": "temporal_manifolds_curve_model",
        "artifact_version": CURVE_MODEL_ARTIFACT_VERSION,
        "model": model,
        "metadata": dict(metadata or {}),
        "sklearn_version": sklearn_version,
        "scipy_version": scipy_version,
    }
    buffer = io.BytesIO()
    joblib.dump(payload, buffer, compress=3)
    return buffer.getvalue()


def load_curve_model(
    source: bytes | bytearray | str | Path | BinaryIO,
) -> tuple[CurveModel, dict[str, Any]]:
    """Load and validate a versioned curve artifact from bytes, a path, or a file."""

    try:
        payload = joblib.load(
            io.BytesIO(source) if isinstance(source, (bytes, bytearray)) else source
        )
    except Exception as exc:  # noqa: BLE001 - normalize untrusted artifact errors
        raise ValueError(f"Curve artifact could not be loaded: {exc}") from exc
    if (
        not isinstance(payload, dict)
        or payload.get("artifact_kind") != "temporal_manifolds_curve_model"
    ):
        raise ValueError("This is not a supported curve model artifact.")
    version = payload.get("artifact_version")
    if version != CURVE_MODEL_ARTIFACT_VERSION:
        raise ValueError(f"Unsupported curve artifact version: {version!r}.")
    model = payload.get("model")
    _validate_curve_model(model)
    metadata = dict(payload.get("metadata") or {})
    metadata.update(
        {
            "artifact_kind": payload["artifact_kind"],
            "artifact_version": version,
            "sklearn_version": payload.get("sklearn_version"),
            "scipy_version": payload.get("scipy_version"),
        }
    )
    return model, metadata
