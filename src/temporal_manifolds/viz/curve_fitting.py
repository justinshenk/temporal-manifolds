"""Fit, evaluate, and serialize parameterized curves in three dimensions."""

from __future__ import annotations

import io
from dataclasses import dataclass
from pathlib import Path
from typing import Any, BinaryIO

import joblib
import numpy as np
from scipy import __version__ as scipy_version
from scipy.interpolate import UnivariateSpline
from sklearn import __version__ as sklearn_version
from sklearn.metrics import r2_score

CURVE_MODEL_ARTIFACT_VERSION = 1
CURVE_ALGORITHMS = {
    "spline": "Smoothing spline",
    "polynomial": "Polynomial ridge",
}
CURVE_DESCRIPTIONS = {
    "spline": (
        "Fits one smoothing spline per displayed coordinate. Higher smoothing follows the "
        "large-scale trajectory instead of individual points."
    ),
    "polynomial": (
        "Fits one regularized polynomial per displayed coordinate. This gives a compact, "
        "globally smooth curve that can extrapolate beyond the fitted parameter range."
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


def _predict_coordinate(predictor: Any, values: np.ndarray) -> np.ndarray:
    if isinstance(predictor, np.ndarray):
        return np.polynomial.polynomial.polyval(values, predictor)
    return np.asarray(predictor(values), dtype=np.float64)


def _fit_polynomial(
    parameter: np.ndarray,
    coordinates: np.ndarray,
    *,
    degree: int,
    alpha: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if not 1 <= degree <= 12:
        raise ValueError("Polynomial degree must be between 1 and 12.")
    if not np.isfinite(alpha) or alpha < 0:
        raise ValueError("Polynomial ridge alpha must be finite and non-negative.")
    if degree >= len(parameter):
        raise ValueError(
            f"Polynomial degree {degree} requires at least {degree + 1} unique parameter values."
        )
    design = np.polynomial.polynomial.polyvander(parameter, degree)
    penalty = np.eye(degree + 1, dtype=np.float64) * alpha
    penalty[0, 0] = 0.0
    coefficients = np.linalg.solve(design.T @ design + penalty, design.T @ coordinates)
    return tuple(coefficients[:, index] for index in range(3))  # type: ignore[return-value]


def _fit_splines(
    parameter: np.ndarray,
    coordinates: np.ndarray,
    *,
    degree: int,
    smoothing: float,
) -> tuple[UnivariateSpline, UnivariateSpline, UnivariateSpline]:
    if not 1 <= degree <= 5:
        raise ValueError("Spline degree must be between 1 and 5.")
    if len(parameter) <= degree:
        raise ValueError(f"A degree-{degree} spline requires at least {degree + 1} values.")
    if not np.isfinite(smoothing) or smoothing < 0:
        raise ValueError("Spline smoothing must be finite and non-negative.")
    predictors = []
    for index in range(3):
        variance = max(float(np.var(coordinates[:, index])), np.finfo(np.float64).eps)
        smoothing_budget = float(smoothing) * len(parameter) * variance
        predictors.append(
            UnivariateSpline(parameter, coordinates[:, index], k=degree, s=smoothing_budget)
        )
    return tuple(predictors)  # type: ignore[return-value]


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


def _curve_boundary_parameter(
    model: CurveModel,
    start: float,
    direction: float,
    display_bounds: np.ndarray,
) -> float:
    """Walk outwards from a fitted endpoint and stop at the display-box boundary."""

    def is_inside(parameter_value: float) -> bool:
        try:
            predicted = model.predict([parameter_value])[0]
        except (FloatingPointError, ValueError, OverflowError):
            return False
        return bool(
            np.isfinite(predicted).all()
            and np.all(predicted >= display_bounds[:, 0])
            and np.all(predicted <= display_bounds[:, 1])
        )

    if not is_inside(start):
        return start
    fitted_span = float(np.ptp(model.training_parameter_bounds))
    step = max(fitted_span / 100.0, np.finfo(np.float64).eps)
    inside_parameter = start
    for _ in range(80):
        outside_parameter = inside_parameter + direction * step
        if not is_inside(outside_parameter):
            inside = inside_parameter
            outside = outside_parameter
            for _ in range(40):
                midpoint = (inside + outside) / 2.0
                if is_inside(midpoint):
                    inside = midpoint
                else:
                    outside = midpoint
            return inside
        inside_parameter = outside_parameter
        step *= 1.25
        if abs(inside_parameter - start) >= fitted_span * 50.0:
            break
    return inside_parameter


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
    lower = _curve_boundary_parameter(model, lower, -1.0, display_bounds)
    upper = _curve_boundary_parameter(model, upper, 1.0, display_bounds)
    curve_parameter = np.linspace(lower, upper, sample_count)
    return curve_parameter, model.predict(curve_parameter), display_bounds


def fit_curve(
    parameter_values: np.ndarray,
    coordinates: np.ndarray,
    *,
    algorithm: str = "spline",
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
    """Fit a 3D curve, using one observed scalar to order the points."""

    if algorithm not in CURVE_ALGORITHMS:
        raise ValueError(f"Unknown curve algorithm: {algorithm!r}.")
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

    unique_parameter, unique_xyz = _merge_duplicate_parameters(
        parameter, xyz, duplicate_reducer
    )
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
    if algorithm == "polynomial":
        model_parameters.setdefault("degree", min(3, len(train_t) - 1))
        model_parameters.setdefault("alpha", 1e-3)
        predictors = _fit_polynomial(train_t, train_xyz, **model_parameters)
    else:
        model_parameters.setdefault("degree", min(3, len(train_t) - 1))
        model_parameters.setdefault("smoothing", 0.15)
        predictors = _fit_splines(train_t, train_xyz, **model_parameters)

    model = CurveModel(
        algorithm=algorithm,
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
    validation_metrics = _coordinate_metrics(
        unique_xyz[validation_indices], validation_prediction
    )
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


def load_curve_model(source: bytes | bytearray | str | Path | BinaryIO) -> tuple[CurveModel, dict[str, Any]]:
    """Load and validate a versioned curve artifact from bytes, a path, or a file."""

    try:
        payload = joblib.load(io.BytesIO(source) if isinstance(source, (bytes, bytearray)) else source)
    except Exception as exc:  # noqa: BLE001 - normalize untrusted artifact errors
        raise ValueError(f"Curve artifact could not be loaded: {exc}") from exc
    if not isinstance(payload, dict) or payload.get("artifact_kind") != "temporal_manifolds_curve_model":
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
