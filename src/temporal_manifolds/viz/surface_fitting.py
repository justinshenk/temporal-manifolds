"""Fit and persist extrapolatable height-field surface models.

The helpers in this module intentionally have no Streamlit or Plotly dependency. This keeps
the numerical work unit-testable and makes a fitted ``z = f(x, y)`` model reusable outside
the visualization app.
"""

from __future__ import annotations

import io
from dataclasses import dataclass
from math import comb
from pathlib import Path
from platform import python_version
from time import perf_counter
from typing import Any, BinaryIO, Mapping

import joblib
import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy import __version__ as scipy_version
from scipy.interpolate import RBFInterpolator
from scipy.spatial import Delaunay, QhullError, cKDTree
from sklearn import __version__ as sklearn_version
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import (
    ConstantKernel,
    Matern,
    RBF,
    RationalQuadratic,
    WhiteKernel,
)
from sklearn.linear_model import HuberRegressor, LinearRegression, RANSACRegressor, Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures
from sklearn.svm import SVR


SURFACE_ALGORITHMS = {
    "plane": "Plane / robust plane",
    "polynomial": "Polynomial ridge",
    "rbf": "Radial basis function",
    "svr": "Support vector regression",
    "gaussian_process": "Gaussian process",
}

SURFACE_DESCRIPTIONS = {
    "plane": (
        "A global plane. Least squares is the simplest baseline; Huber and RANSAC resist outliers."
    ),
    "polynomial": (
        "A global curved surface with ridge regularization. Good for broad, smooth trends."
    ),
    "rbf": (
        "A flexible smooth interpolator or smoother that is defined outside the data hull. "
        "Long-range extrapolation can be unstable."
    ),
    "svr": (
        "A kernel regressor defined for arbitrary coordinates. Linear and polynomial kernels "
        "extend trends; an RBF kernel tends back toward its baseline far from the fit."
    ),
    "gaussian_process": (
        "A probabilistic surface with predictive uncertainty. Far from the fit it reverts toward "
        "its learned mean, and cubic scaling limits it to smaller samples."
    ),
}

SURFACE_MODEL_ARTIFACT_KIND = "temporal-manifolds.activation-surface"
SURFACE_MODEL_ARTIFACT_VERSION = 1

# Avoid accidental multi-minute or out-of-memory fits. A user-specified cap can be lower.
ALGORITHM_POINT_CAPS: dict[str, int | None] = {
    "plane": None,
    "polynomial": 20_000,
    "rbf": 8_000,
    "svr": 5_000,
    "gaussian_process": 1_000,
}


def surface_point_cap(algorithm: str, parameters: Mapping[str, Any] | None = None) -> int | None:
    """Return the parameter-aware safety cap for a surface algorithm."""

    if algorithm not in ALGORITHM_POINT_CAPS:
        raise ValueError(f"Unknown surface algorithm: {algorithm!r}.")
    cap = ALGORITHM_POINT_CAPS[algorithm]
    if algorithm == "rbf" and parameters is not None and parameters.get("neighbors") is None:
        return min(cap or 1_000, 1_000)
    return cap


def _equation_expression(
    intercept: float, coefficients: NDArray[np.float64], terms: list[str]
) -> str:
    expression = f"{float(intercept):.12g}"
    for coefficient, term in zip(coefficients, terms, strict=True):
        sign = "+" if coefficient >= 0 else "-"
        expression += f" {sign} {abs(float(coefficient)):.12g}*{term}"
    return expression


@dataclass
class SurfaceModel:
    """A fitted surface predictor whose public inputs use raw PC coordinates.

    Plot masks and display clipping are intentionally not applied here: callers can evaluate
    any finite coordinate pair, including points outside ``training_bounds``.
    """

    algorithm: str
    parameters: dict[str, Any]
    estimator: Any
    coordinate_center: NDArray[np.float64]
    coordinate_scale: NDArray[np.float64]
    training_bounds: NDArray[np.float64]
    training_output_range: tuple[float, float]
    input_features: tuple[str, str] = ("X", "Y")
    output_feature: str = "Z"

    def _coordinates(self, coordinates: ArrayLike) -> NDArray[np.float64]:
        try:
            values = np.asarray(coordinates, dtype=np.float64)
        except (TypeError, ValueError) as exc:
            raise ValueError("Surface coordinates must be numeric pairs.") from exc
        if values.ndim == 1:
            if values.shape != (2,):
                raise ValueError("Surface coordinates must have shape (n, 2) or (2,).")
            values = values.reshape(1, 2)
        if values.ndim != 2 or values.shape[1] != 2:
            raise ValueError("Surface coordinates must have shape (n, 2) or (2,).")
        if not np.all(np.isfinite(values)):
            raise ValueError("Surface coordinates must be finite.")
        return values

    def predict(
        self, coordinates: ArrayLike, *, return_std: bool = False
    ) -> NDArray[np.float64] | tuple[NDArray[np.float64], NDArray[np.float64]]:
        """Predict the output PC for raw ``[[PC1, PC2], ...]`` coordinates."""

        values = self._coordinates(coordinates)
        if return_std and self.algorithm != "gaussian_process":
            raise ValueError("Predictive uncertainty is only available for Gaussian processes.")
        if not len(values):
            empty = np.empty(0, dtype=np.float64)
            return (empty, empty.copy()) if return_std else empty
        normalized = (values - self.coordinate_center) / self.coordinate_scale
        prediction, standard_deviation = _predict_model_in_chunks(
            self.estimator,
            normalized,
            uncertainty=return_std,
        )
        expected_shape = (len(values),)
        if prediction.shape != expected_shape:
            raise ValueError(
                "The fitted model returned an invalid prediction count: "
                f"expected {len(values):,}, received {len(prediction):,}."
            )
        if return_std:
            if standard_deviation is None:  # Defensive guard for malformed artifacts.
                raise ValueError("The fitted model does not provide predictive uncertainty.")
            if standard_deviation.shape != expected_shape:
                raise ValueError(
                    "The fitted model returned an invalid uncertainty count: "
                    f"expected {len(values):,}, received {len(standard_deviation):,}."
                )
            return prediction, standard_deviation
        return prediction

    def extrapolation_mask(self, coordinates: ArrayLike) -> NDArray[np.bool_]:
        """Mark rows outside either raw-coordinate training interval."""

        values = self._coordinates(coordinates)
        return np.any(
            (values < self.training_bounds[:, 0]) | (values > self.training_bounds[:, 1]),
            axis=1,
        )

    @property
    def equation(self) -> str | None:
        """Return a readable exact formula for plane and polynomial models."""

        if self.algorithm == "plane":
            estimator = getattr(self.estimator, "estimator_", self.estimator)
            coefficients = np.asarray(estimator.coef_, dtype=np.float64).reshape(-1)
            if coefficients.shape != (2,):
                return None
            raw_coefficients = coefficients / self.coordinate_scale
            raw_intercept = float(estimator.intercept_) - float(
                np.dot(coefficients, self.coordinate_center / self.coordinate_scale)
            )
            return (
                f"{self.output_feature} = "
                f"{_equation_expression(raw_intercept, raw_coefficients, list(self.input_features))}"
            )

        if self.algorithm == "polynomial":
            try:
                features = self.estimator.named_steps["polynomialfeatures"]
                ridge = self.estimator.named_steps["ridge"]
            except (AttributeError, KeyError):
                return None
            terms = [
                term.replace(" ", "*").replace("^", "**")
                for term in features.get_feature_names_out(("u", "v"))
            ]
            coefficients = np.asarray(ridge.coef_, dtype=np.float64).reshape(-1)
            if len(coefficients) != len(terms):
                return None
            definitions = []
            for normalized_name, feature, center, scale in zip(
                ("u", "v"),
                self.input_features,
                self.coordinate_center,
                self.coordinate_scale,
                strict=True,
            ):
                definitions.append(
                    f"{normalized_name} = ({feature} - ({float(center):.12g})) / "
                    f"{float(scale):.12g}"
                )
            expression = _equation_expression(float(ridge.intercept_), coefficients, terms)
            return "\n".join([*definitions, f"{self.output_feature} = {expression}"])

        return None


@dataclass
class SurfaceFitResult:
    """Reusable model, serializable surface grid, and fit diagnostics."""

    algorithm: str
    parameters: dict[str, Any]
    model: SurfaceModel
    grid_x: NDArray[np.float64]
    grid_y: NDArray[np.float64]
    grid_z: NDArray[np.float64]
    grid_uncertainty: NDArray[np.float64] | None
    point_x: NDArray[np.float64]
    point_y: NDArray[np.float64]
    point_z: NDArray[np.float64]
    point_prediction: NDArray[np.float64]
    metrics: dict[str, float | int | None]
    warnings: tuple[str, ...]


@dataclass
class SurfaceEvaluationResult:
    """Preview grid and current-data diagnostics for a previously fitted model."""

    algorithm: str
    parameters: dict[str, Any]
    model: SurfaceModel
    grid_x: NDArray[np.float64]
    grid_y: NDArray[np.float64]
    grid_z: NDArray[np.float64]
    grid_uncertainty: NDArray[np.float64] | None
    point_x: NDArray[np.float64]
    point_y: NDArray[np.float64]
    point_z: NDArray[np.float64]
    point_prediction: NDArray[np.float64]
    metrics: dict[str, float | int | None]
    warnings: tuple[str, ...]


class _CallablePredictor:
    """Give SciPy interpolators a small sklearn-like prediction interface."""

    def __init__(self, interpolator: Any) -> None:
        self.interpolator = interpolator

    def predict(self, coordinates: NDArray[np.float64]) -> NDArray[np.float64]:
        return np.asarray(self.interpolator(coordinates), dtype=np.float64).reshape(-1)


def default_surface_parameters(algorithm: str, point_count: int) -> dict[str, Any]:
    """Return conservative defaults for an advertised algorithm."""

    if algorithm not in SURFACE_ALGORITHMS:
        raise ValueError(f"Unknown surface algorithm: {algorithm!r}.")
    point_count = max(1, int(point_count))
    defaults: dict[str, dict[str, Any]] = {
        "plane": {
            "estimator": "least_squares",
            "huber_epsilon": 1.35,
            "huber_alpha": 0.0001,
            "ransac_min_samples": 0.5,
            "ransac_residual_threshold": None,
            "ransac_max_trials": 100,
        },
        "polynomial": {"degree": 2, "alpha": 0.01, "interaction_only": False},
        "rbf": {
            "kernel": "thin_plate_spline",
            "smoothing": 0.01,
            "neighbors": min(point_count, max(10, min(80, int(4 * np.sqrt(point_count))))),
            "epsilon": 1.0,
            "degree": None,
        },
        "svr": {"kernel": "linear", "c": 10.0, "epsilon": 0.1, "gamma": "scale"},
        "gaussian_process": {
            "kernel": "matern_1.5",
            "length_scale": 1.0,
            "noise_level": 0.01,
            "optimize": True,
            "optimizer_restarts": 0,
        },
    }
    return defaults[algorithm].copy()


def _as_float_vector(values: ArrayLike, name: str) -> NDArray[np.float64]:
    try:
        vector = np.asarray(values, dtype=np.float64).reshape(-1)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} coordinates must be numeric.") from exc
    return vector


def _collapse_duplicate_coordinates(
    coordinates: NDArray[np.float64],
    values: NDArray[np.float64],
    reducer: str,
) -> tuple[NDArray[np.float64], NDArray[np.float64], int]:
    if reducer not in {"mean", "median"}:
        raise ValueError("duplicate_reducer must be 'mean' or 'median'.")
    unique_coordinates, inverse, counts = np.unique(
        coordinates, axis=0, return_inverse=True, return_counts=True
    )
    duplicates_merged = int(len(values) - len(unique_coordinates))
    if duplicates_merged == 0:
        return coordinates.copy(), values.copy(), 0
    if reducer == "mean":
        totals = np.bincount(inverse, weights=values)
        unique_values = totals / counts
    elif reducer == "median":
        order = np.argsort(inverse, kind="stable")
        sorted_values = values[order]
        group_starts = np.concatenate([[0], np.cumsum(counts)[:-1]])
        unique_values = np.array(
            [
                np.median(sorted_values[start : start + count])
                for start, count in zip(group_starts, counts, strict=True)
            ],
            dtype=np.float64,
        )
    return unique_coordinates, unique_values, duplicates_merged


def _subsample_indices(
    coordinates: NDArray[np.float64], cap: int, random_state: int
) -> NDArray[np.int64]:
    """Sample deterministically while preserving coordinate extrema."""

    if cap >= len(coordinates):
        return np.arange(len(coordinates), dtype=np.int64)
    extrema = np.unique(
        np.array(
            [
                np.argmin(coordinates[:, 0]),
                np.argmax(coordinates[:, 0]),
                np.argmin(coordinates[:, 1]),
                np.argmax(coordinates[:, 1]),
            ],
            dtype=np.int64,
        )
    )
    if len(extrema) >= cap:
        return np.sort(extrema[:cap])
    candidates = np.setdiff1d(np.arange(len(coordinates)), extrema, assume_unique=True)
    rng = np.random.default_rng(random_state)
    selected = rng.choice(candidates, size=cap - len(extrema), replace=False)
    return np.sort(np.concatenate([extrema, selected])).astype(np.int64)


def _validation_split_indices(
    coordinates: NDArray[np.float64],
    validation_count: int,
    random_state: int,
) -> tuple[NDArray[np.int64], NDArray[np.int64]]:
    """Keep coordinate extrema in training so a held-out hull remains well-defined."""

    extrema = np.unique(
        np.array(
            [
                np.argmin(coordinates[:, 0]),
                np.argmax(coordinates[:, 0]),
                np.argmin(coordinates[:, 1]),
                np.argmax(coordinates[:, 1]),
            ],
            dtype=np.int64,
        )
    )
    candidates = np.setdiff1d(np.arange(len(coordinates)), extrema, assume_unique=True)
    validation_count = min(validation_count, len(candidates))
    rng = np.random.default_rng(random_state)
    validation = np.sort(rng.choice(candidates, size=validation_count, replace=False))
    training = np.setdiff1d(np.arange(len(coordinates)), validation, assume_unique=True)
    return training.astype(np.int64), validation.astype(np.int64)


def _rbf_tail_degree(parameters: Mapping[str, Any]) -> int:
    kernel = str(parameters.get("kernel", "thin_plate_spline"))
    minimum_degree = {
        "multiquadric": 0,
        "linear": 0,
        "thin_plate_spline": 1,
        "cubic": 1,
        "quintic": 2,
    }
    degree_value = parameters.get("degree")
    if degree_value is not None:
        degree = int(degree_value)
        if degree < -1:
            raise ValueError("RBF polynomial-tail degree must be -1 or greater.")
        kernel_minimum = minimum_degree.get(kernel, 0)
        if degree != -1 and degree < kernel_minimum:
            raise ValueError(
                f"The {kernel.replace('_', ' ')} kernel needs polynomial-tail degree "
                f"{kernel_minimum} or greater (or -1 to disable the tail)."
            )
        return degree
    return minimum_degree.get(kernel, 0)


def _minimum_points(algorithm: str, parameters: Mapping[str, Any]) -> int:
    if algorithm == "polynomial":
        degree = int(parameters.get("degree", 2))
        if not 1 <= degree <= 6:
            raise ValueError("Polynomial degree must be between 1 and 6.")
        if bool(parameters.get("interaction_only", False)):
            return 3 if degree == 1 else 4
        # Number of terms for two variables, including the fitted intercept.
        return comb(degree + 2, 2)
    if algorithm == "rbf":
        tail_degree = _rbf_tail_degree(parameters)
        tail_terms = 0 if tail_degree == -1 else comb(tail_degree + 2, 2)
        neighbors_value = parameters.get("neighbors")
        if neighbors_value not in (None, 0) and int(neighbors_value) < tail_terms:
            raise ValueError(
                f"This RBF polynomial tail needs at least {tail_terms} local neighbors."
            )
        return max(4, tail_terms)
    return 4


def _make_gaussian_process_kernel(parameters: Mapping[str, Any]) -> Any:
    length_scale = float(parameters.get("length_scale", 1.0))
    noise_level = float(parameters.get("noise_level", 0.01))
    if length_scale <= 0 or noise_level <= 0:
        raise ValueError("Gaussian-process length scale and noise level must be positive.")
    kernel_name = str(parameters.get("kernel", "matern_1.5"))
    if kernel_name == "rbf":
        base_kernel = RBF(length_scale, length_scale_bounds=(1e-3, 1e3))
    elif kernel_name == "matern_0.5":
        base_kernel = Matern(length_scale, length_scale_bounds=(1e-3, 1e3), nu=0.5)
    elif kernel_name == "matern_1.5":
        base_kernel = Matern(length_scale, length_scale_bounds=(1e-3, 1e3), nu=1.5)
    elif kernel_name == "matern_2.5":
        base_kernel = Matern(length_scale, length_scale_bounds=(1e-3, 1e3), nu=2.5)
    elif kernel_name == "rational_quadratic":
        base_kernel = RationalQuadratic(
            length_scale=length_scale,
            alpha=1.0,
            length_scale_bounds=(1e-3, 1e3),
            alpha_bounds=(1e-3, 1e3),
        )
    else:
        raise ValueError(f"Unknown Gaussian-process kernel: {kernel_name!r}.")
    return ConstantKernel(1.0, constant_value_bounds=(1e-3, 1e3)) * base_kernel + WhiteKernel(
        noise_level, noise_level_bounds=(1e-8, 1e1)
    )


def _fit_model(
    algorithm: str,
    coordinates: NDArray[np.float64],
    values: NDArray[np.float64],
    parameters: Mapping[str, Any],
    random_state: int,
) -> Any:
    if algorithm == "plane":
        estimator = str(parameters.get("estimator", "least_squares"))
        if estimator == "least_squares":
            model: Any = LinearRegression()
        elif estimator == "huber":
            epsilon = float(parameters.get("huber_epsilon", 1.35))
            alpha = float(parameters.get("huber_alpha", 0.0001))
            if epsilon < 1.0 or alpha < 0:
                raise ValueError("Huber epsilon must be at least 1 and alpha cannot be negative.")
            model = HuberRegressor(epsilon=epsilon, alpha=alpha, max_iter=1_000)
        elif estimator == "ransac":
            threshold = parameters.get("ransac_residual_threshold")
            threshold = None if threshold in (None, 0) else float(threshold)
            model = RANSACRegressor(
                estimator=LinearRegression(),
                min_samples=float(parameters.get("ransac_min_samples", 0.5)),
                residual_threshold=threshold,
                max_trials=int(parameters.get("ransac_max_trials", 100)),
                random_state=random_state,
            )
        else:
            raise ValueError(f"Unknown plane estimator: {estimator!r}.")
        return model.fit(coordinates, values)

    if algorithm == "polynomial":
        degree = int(parameters.get("degree", 2))
        alpha = float(parameters.get("alpha", 0.01))
        if not 1 <= degree <= 6 or alpha < 0:
            raise ValueError("Polynomial degree must be 1–6 and ridge alpha cannot be negative.")
        model = make_pipeline(
            PolynomialFeatures(
                degree=degree,
                include_bias=False,
                interaction_only=bool(parameters.get("interaction_only", False)),
            ),
            Ridge(alpha=alpha),
        )
        return model.fit(coordinates, values)

    if algorithm == "rbf":
        kernel = str(parameters.get("kernel", "thin_plate_spline"))
        allowed_kernels = {
            "linear",
            "thin_plate_spline",
            "cubic",
            "quintic",
            "multiquadric",
            "inverse_multiquadric",
            "inverse_quadratic",
            "gaussian",
        }
        if kernel not in allowed_kernels:
            raise ValueError(f"Unknown RBF kernel: {kernel!r}.")
        smoothing = float(parameters.get("smoothing", 0.01))
        if smoothing < 0:
            raise ValueError("RBF smoothing cannot be negative.")
        neighbors_value = parameters.get("neighbors")
        neighbors = None if neighbors_value in (None, 0) else int(neighbors_value)
        if neighbors is not None and neighbors < 3:
            raise ValueError("RBF neighbors must be at least 3.")
        if neighbors is not None:
            # A held-out validation fit contains fewer points than the final fit.
            neighbors = min(neighbors, len(coordinates))
        degree_value = parameters.get("degree")
        degree = None if degree_value is None else int(degree_value)
        scale_dependent = {
            "multiquadric",
            "inverse_multiquadric",
            "inverse_quadratic",
            "gaussian",
        }
        epsilon = float(parameters.get("epsilon", 1.0)) if kernel in scale_dependent else None
        if epsilon is not None and epsilon <= 0:
            raise ValueError("RBF epsilon must be positive.")
        interpolator = RBFInterpolator(
            coordinates,
            values,
            neighbors=neighbors,
            smoothing=smoothing,
            kernel=kernel,
            epsilon=epsilon,
            degree=degree,
        )
        return _CallablePredictor(interpolator)

    if algorithm == "svr":
        kernel = str(parameters.get("kernel", "rbf"))
        if kernel not in {"rbf", "linear", "poly"}:
            raise ValueError(f"Unknown SVR kernel: {kernel!r}.")
        c_value = float(parameters.get("c", 10.0))
        epsilon = float(parameters.get("epsilon", 0.1))
        gamma_value = parameters.get("gamma", "scale")
        gamma: str | float
        gamma = gamma_value if gamma_value in {"scale", "auto"} else float(gamma_value)
        if c_value <= 0 or epsilon < 0 or (isinstance(gamma, float) and gamma <= 0):
            raise ValueError("SVR C and gamma must be positive; epsilon cannot be negative.")
        return SVR(
            kernel=kernel,
            C=c_value,
            epsilon=epsilon,
            gamma=gamma,
            degree=int(parameters.get("degree", 3)),
        ).fit(coordinates, values)

    if algorithm == "gaussian_process":
        optimize = bool(parameters.get("optimize", True))
        restarts = int(parameters.get("optimizer_restarts", 0))
        if not 0 <= restarts <= 5:
            raise ValueError("Gaussian-process optimizer restarts must be between 0 and 5.")
        return GaussianProcessRegressor(
            kernel=_make_gaussian_process_kernel(parameters),
            alpha=1e-10,
            optimizer="fmin_l_bfgs_b" if optimize else None,
            n_restarts_optimizer=restarts if optimize else 0,
            normalize_y=True,
            random_state=random_state,
        ).fit(coordinates, values)

    raise ValueError(f"Unknown surface algorithm: {algorithm!r}.")


def _predict_model(
    model: Any,
    coordinates: NDArray[np.float64],
    *,
    uncertainty: bool = False,
) -> tuple[NDArray[np.float64], NDArray[np.float64] | None]:
    if uncertainty and isinstance(model, GaussianProcessRegressor):
        prediction, standard_deviation = model.predict(coordinates, return_std=True)
        return (
            np.asarray(prediction, dtype=np.float64).reshape(-1),
            np.asarray(standard_deviation, dtype=np.float64).reshape(-1),
        )
    prediction = model.predict(coordinates)
    return np.asarray(prediction, dtype=np.float64).reshape(-1), None


def _predict_model_in_chunks(
    model: Any,
    coordinates: NDArray[np.float64],
    *,
    uncertainty: bool = False,
    chunk_size: int = 5_000,
) -> tuple[NDArray[np.float64], NDArray[np.float64] | None]:
    """Bound prediction memory for large grids and kernel estimators."""

    if isinstance(model, GaussianProcessRegressor):
        training_count = max(1, len(model.X_train_))
        chunk_size = min(chunk_size, max(250, 2_000_000 // training_count))
    if len(coordinates) <= chunk_size:
        return _predict_model(model, coordinates, uncertainty=uncertainty)
    predictions: list[NDArray[np.float64]] = []
    uncertainties: list[NDArray[np.float64]] = []
    for start in range(0, len(coordinates), chunk_size):
        prediction, standard_deviation = _predict_model(
            model,
            coordinates[start : start + chunk_size],
            uncertainty=uncertainty,
        )
        predictions.append(prediction)
        if standard_deviation is not None:
            uncertainties.append(standard_deviation)
    return (
        np.concatenate(predictions),
        np.concatenate(uncertainties) if uncertainties else None,
    )


def validate_surface_model(model: Any) -> SurfaceModel:
    """Validate a fitted, raw-coordinate surface predictor."""

    if not isinstance(model, SurfaceModel):
        raise ValueError("The artifact does not contain a supported surface model.")
    if model.algorithm not in SURFACE_ALGORITHMS:
        raise ValueError(f"The surface model uses an unsupported algorithm: {model.algorithm!r}.")
    if not isinstance(model.parameters, dict):
        raise ValueError("The surface model contains invalid algorithm parameters.")

    center = np.asarray(model.coordinate_center, dtype=np.float64)
    scale = np.asarray(model.coordinate_scale, dtype=np.float64)
    bounds = np.asarray(model.training_bounds, dtype=np.float64)
    if center.shape != (2,) or not np.all(np.isfinite(center)):
        raise ValueError("The surface model contains an invalid coordinate center.")
    if scale.shape != (2,) or not np.all(np.isfinite(scale)) or np.any(scale <= 0):
        raise ValueError("The surface model contains an invalid coordinate scale.")
    if bounds.shape != (2, 2) or not np.all(np.isfinite(bounds)):
        raise ValueError("The surface model contains invalid training bounds.")
    if np.any(bounds[:, 0] > bounds[:, 1]):
        raise ValueError("The surface model's training bounds are reversed.")
    output_range = np.asarray(model.training_output_range, dtype=np.float64)
    if (
        output_range.shape != (2,)
        or not np.all(np.isfinite(output_range))
        or output_range[0] > output_range[1]
    ):
        raise ValueError("The surface model contains an invalid training-output range.")
    if (
        not isinstance(model.input_features, tuple)
        or len(model.input_features) != 2
        or len(set(model.input_features)) != 2
        or not all(isinstance(name, str) and name for name in model.input_features)
        or not isinstance(model.output_feature, str)
        or not model.output_feature
    ):
        raise ValueError("The surface model contains invalid feature names.")
    if not callable(getattr(model.estimator, "predict", None)):
        raise ValueError("The surface model does not contain a predictor.")
    try:
        prediction, _ = _predict_model(model.estimator, np.zeros((1, 2), dtype=np.float64))
    except Exception as exc:  # noqa: BLE001 - normalize third-party estimator errors
        raise ValueError(f"The surface model's predictor is not fitted: {exc}") from exc
    if prediction.shape != (1,) or not np.all(np.isfinite(prediction)):
        raise ValueError("The surface model's predictor returned an invalid test prediction.")
    return model


def serialize_surface_model(
    model: SurfaceModel, *, metadata: Mapping[str, Any] | None = None
) -> bytes:
    """Serialize a fitted surface and provenance as a versioned joblib artifact."""

    model = validate_surface_model(model)
    artifact = {
        "kind": SURFACE_MODEL_ARTIFACT_KIND,
        "version": SURFACE_MODEL_ARTIFACT_VERSION,
        "model": model,
        "algorithm": model.algorithm,
        "input_features": model.input_features,
        "output_feature": model.output_feature,
        "python_version": python_version(),
        "joblib_version": joblib.__version__,
        "numpy_version": np.__version__,
        "scipy_version": scipy_version,
        "sklearn_version": sklearn_version,
        "metadata": dict(metadata or {}),
    }
    buffer = io.BytesIO()
    joblib.dump(artifact, buffer, compress=3)
    return buffer.getvalue()


def load_surface_model(
    source: str | Path | bytes | BinaryIO,
) -> tuple[SurfaceModel, dict[str, Any]]:
    """Load a trusted surface artifact and return its model and provenance.

    Joblib and pickle files can execute arbitrary code while loading. Callers must only pass
    artifacts from trusted sources.
    """

    if isinstance(source, bytes):
        source = io.BytesIO(source)
    elif hasattr(source, "seek"):
        source.seek(0)
    try:
        payload = joblib.load(source)
    except Exception as exc:  # noqa: BLE001 - normalize artifact errors for UI callers
        raise ValueError(f"The surface model file could not be loaded: {exc}") from exc
    if not isinstance(payload, Mapping) or payload.get("kind") != SURFACE_MODEL_ARTIFACT_KIND:
        raise ValueError("The selected file is not a supported surface model artifact.")
    if payload.get("version") != SURFACE_MODEL_ARTIFACT_VERSION:
        raise ValueError(f"Unsupported surface artifact version: {payload.get('version')!r}.")

    model = validate_surface_model(payload.get("model"))
    if payload.get("algorithm") != model.algorithm:
        raise ValueError("The surface artifact's algorithm does not match its model.")
    if tuple(payload.get("input_features", ())) != model.input_features:
        raise ValueError("The surface artifact's input features do not match its model.")
    if payload.get("output_feature") != model.output_feature:
        raise ValueError("The surface artifact's output feature does not match its model.")
    metadata = payload.get("metadata", {})
    if not isinstance(metadata, Mapping):
        raise ValueError("The surface artifact contains invalid provenance metadata.")
    provenance = {
        **dict(metadata),
        "artifact_kind": SURFACE_MODEL_ARTIFACT_KIND,
        "artifact_version": SURFACE_MODEL_ARTIFACT_VERSION,
        "algorithm": model.algorithm,
        "input_features": model.input_features,
        "output_feature": model.output_feature,
        "equation": model.equation,
        "python_version": payload.get("python_version"),
        "joblib_version": payload.get("joblib_version"),
        "numpy_version": payload.get("numpy_version"),
        "scipy_version": payload.get("scipy_version"),
        "sklearn_version": payload.get("sklearn_version"),
    }
    return model, provenance


def _short_error(error: BaseException) -> str:
    message = str(error).strip().splitlines()
    return message[0] if message else error.__class__.__name__


def _regression_metrics(
    actual: NDArray[np.float64], prediction: NDArray[np.float64]
) -> dict[str, float | int | None]:
    finite = np.isfinite(prediction)
    predicted_count = int(finite.sum())
    total_count = int(len(actual))
    coverage = float(finite.mean()) if len(finite) else 0.0
    if predicted_count == 0:
        return {
            "count": 0,
            "predicted_count": 0,
            "total_count": total_count,
            "coverage": coverage,
            "rmse": None,
            "mae": None,
            "r2": None,
            "nrmse": None,
        }
    actual_finite = actual[finite]
    prediction_finite = prediction[finite]
    residual = prediction_finite - actual_finite
    rmse = float(np.sqrt(np.mean(np.square(residual))))
    mae = float(np.mean(np.abs(residual)))
    total_variance = float(np.sum(np.square(actual_finite - np.mean(actual_finite))))
    r2 = (
        float(1.0 - np.sum(np.square(residual)) / total_variance)
        if predicted_count >= 2 and total_variance > np.finfo(np.float64).eps
        else None
    )
    interquartile_range = float(np.subtract(*np.percentile(actual_finite, [75, 25])))
    nrmse = rmse / interquartile_range if interquartile_range > 0 else None
    return {
        "count": predicted_count,
        "predicted_count": predicted_count,
        "total_count": total_count,
        "coverage": coverage,
        "rmse": rmse,
        "mae": mae,
        "r2": r2,
        "nrmse": nrmse,
    }


def _prefixed_metrics(
    prefix: str, values: Mapping[str, float | int | None]
) -> dict[str, float | int | None]:
    return {f"{prefix}_{key}": value for key, value in values.items()}


def _surface_domain_mask(
    fit_coordinates: NDArray[np.float64],
    grid_coordinates: NDArray[np.float64],
    mode: str,
    density_factor: float,
) -> NDArray[np.bool_]:
    if mode == "full_grid":
        return np.ones(len(grid_coordinates), dtype=bool)
    if mode not in {"convex_hull", "dense_support"}:
        raise ValueError("domain_mask must be 'convex_hull', 'dense_support', or 'full_grid'.")
    try:
        mask = Delaunay(fit_coordinates).find_simplex(grid_coordinates) >= 0
    except QhullError as exc:
        raise ValueError("The fit coordinates cannot form a two-dimensional convex hull.") from exc
    if mode == "dense_support":
        if not np.isfinite(density_factor) or density_factor <= 0:
            raise ValueError("density_factor must be positive.")
        tree = cKDTree(fit_coordinates)
        fit_distances = tree.query(fit_coordinates, k=2)[0][:, 1]
        reference_distance = float(np.quantile(fit_distances, 0.9))
        if reference_distance > 0:
            grid_distances = tree.query(grid_coordinates, k=1)[0]
            mask &= grid_distances <= density_factor * reference_distance
    return mask


def evaluate_surface_model(
    model: SurfaceModel,
    x: ArrayLike,
    y: ArrayLike,
    z: ArrayLike,
    *,
    input_features: tuple[str, str] | None = None,
    output_feature: str | None = None,
    grid_resolution: int = 60,
    padding_fraction: float = 0.0,
    domain_mask: str = "full_grid",
    density_factor: float = 3.0,
    clip_mode: str = "none",
) -> SurfaceEvaluationResult:
    """Evaluate a saved model against current points and over a chart-oriented grid.

    ``input_features`` names the supplied ``x`` and ``y`` arrays. Their order may differ from
    the model's saved input order; predictions are reordered internally while returned grid and
    point coordinates remain in chart X/Y order. Domain masking and clipping are display-only.
    """

    total_started = perf_counter()
    model = validate_surface_model(model)
    if not 15 <= int(grid_resolution) <= 200:
        raise ValueError("grid_resolution must be between 15 and 200.")
    if not 0 <= float(padding_fraction) <= 0.5:
        raise ValueError("padding_fraction must be between 0 and 0.5.")
    if domain_mask not in {"convex_hull", "dense_support", "full_grid"}:
        raise ValueError("domain_mask must be 'convex_hull', 'dense_support', or 'full_grid'.")
    if domain_mask == "dense_support" and (
        not np.isfinite(float(density_factor)) or float(density_factor) <= 0
    ):
        raise ValueError("density_factor must be positive.")
    if clip_mode not in {"none", "observed_range", "robust_range"}:
        raise ValueError("clip_mode must be 'none', 'observed_range', or 'robust_range'.")

    chart_features = model.input_features if input_features is None else tuple(input_features)
    if (
        len(chart_features) != 2
        or len(set(chart_features)) != 2
        or not all(isinstance(name, str) and name for name in chart_features)
        or set(chart_features) != set(model.input_features)
    ):
        raise ValueError(
            "The current X/Y components must match the saved surface inputs "
            f"{model.input_features!r}."
        )
    current_output = model.output_feature if output_feature is None else output_feature
    if current_output != model.output_feature:
        raise ValueError(
            f"The current Z component must be the saved output {model.output_feature!r}."
        )
    model_order = [chart_features.index(name) for name in model.input_features]

    x_values = _as_float_vector(x, chart_features[0])
    y_values = _as_float_vector(y, chart_features[1])
    z_values = _as_float_vector(z, current_output)
    if not (len(x_values) == len(y_values) == len(z_values)):
        raise ValueError("X, Y, and Z must contain the same number of values.")
    input_points = len(x_values)
    coordinate_finite = np.isfinite(x_values) & np.isfinite(y_values)
    point_finite = coordinate_finite & np.isfinite(z_values)
    extent_coordinates = np.column_stack([x_values[coordinate_finite], y_values[coordinate_finite]])
    chart_coordinates = np.column_stack([x_values[point_finite], y_values[point_finite]])
    point_values = z_values[point_finite]
    dropped_nonfinite = int(input_points - len(point_values))
    if not len(extent_coordinates):
        raise ValueError("At least one finite current X/Y pair is required for a preview.")
    if not len(point_values):
        raise ValueError(
            "At least one current point with finite X, Y, and Z is required for diagnostics."
        )

    support_coordinates = np.unique(extent_coordinates, axis=0)
    if domain_mask != "full_grid":
        if len(support_coordinates) < 3:
            raise ValueError(
                "At least three unique current X/Y locations are required for hull masking."
            )
        if np.linalg.matrix_rank(support_coordinates - np.mean(support_coordinates, axis=0)) < 2:
            raise ValueError(
                "The current X/Y coordinates are collinear; hull masking is not defined."
            )

    chart_minimum = np.min(extent_coordinates, axis=0)
    chart_maximum = np.max(extent_coordinates, axis=0)
    chart_span = chart_maximum - chart_minimum
    fallback_features: list[str] = []
    for chart_index, feature in enumerate(chart_features):
        if chart_span[chart_index] > np.finfo(np.float64).eps:
            continue
        model_index = model.input_features.index(feature)
        saved_low, saved_high = model.training_bounds[model_index]
        saved_span = float(saved_high - saved_low)
        if saved_span <= np.finfo(np.float64).eps:
            raise ValueError(
                f"Neither the current points nor the saved fit define a range for {feature}."
            )
        chart_minimum[chart_index] = saved_low
        chart_maximum[chart_index] = saved_high
        chart_span[chart_index] = saved_span
        fallback_features.append(feature)

    x_span, y_span = map(float, chart_span)
    x_grid = np.linspace(
        float(chart_minimum[0] - padding_fraction * x_span),
        float(chart_maximum[0] + padding_fraction * x_span),
        int(grid_resolution),
    )
    y_grid = np.linspace(
        float(chart_minimum[1] - padding_fraction * y_span),
        float(chart_maximum[1] + padding_fraction * y_span),
        int(grid_resolution),
    )
    grid_x, grid_y = np.meshgrid(x_grid, y_grid)
    grid_chart_coordinates = np.column_stack([grid_x.ravel(), grid_y.ravel()])
    point_model_coordinates = chart_coordinates[:, model_order]
    extent_model_coordinates = extent_coordinates[:, model_order]
    grid_model_coordinates = grid_chart_coordinates[:, model_order]

    prediction_started = perf_counter()
    point_prediction = model.predict(point_model_coordinates)
    if model.algorithm == "gaussian_process":
        grid_prediction, grid_uncertainty = model.predict(grid_model_coordinates, return_std=True)
    else:
        grid_prediction = model.predict(grid_model_coordinates)
        grid_uncertainty = None
    prediction_seconds = perf_counter() - prediction_started

    support_model_coordinates = support_coordinates[:, model_order]
    normalized_support = (
        support_model_coordinates - model.coordinate_center
    ) / model.coordinate_scale
    normalized_grid = (grid_model_coordinates - model.coordinate_center) / model.coordinate_scale
    domain = _surface_domain_mask(
        normalized_support,
        normalized_grid,
        domain_mask,
        float(density_factor),
    )
    point_extrapolation = model.extrapolation_mask(extent_model_coordinates)
    grid_extrapolation = model.extrapolation_mask(grid_model_coordinates)

    invalid_grid_prediction = ~np.isfinite(grid_prediction)
    invalid_grid_prediction_count = int(invalid_grid_prediction.sum())
    grid_prediction[invalid_grid_prediction] = np.nan
    invalid_grid_uncertainty_count = 0
    if grid_uncertainty is not None:
        invalid_grid_uncertainty = ~np.isfinite(grid_uncertainty)
        invalid_grid_uncertainty_count = int(invalid_grid_uncertainty.sum())
        grid_uncertainty[invalid_grid_uncertainty | invalid_grid_prediction] = np.nan
    grid_prediction[~domain] = np.nan
    if grid_uncertainty is not None:
        grid_uncertainty[~domain] = np.nan
    finite_supported = np.isfinite(grid_prediction)
    training_low, training_high = model.training_output_range
    if finite_supported.any():
        overshoot = (grid_prediction[finite_supported] < training_low) | (
            grid_prediction[finite_supported] > training_high
        )
        overshoot_fraction = float(np.mean(overshoot))
    else:
        overshoot_fraction = None

    if clip_mode == "observed_range":
        clip_low, clip_high = np.min(point_values), np.max(point_values)
        grid_prediction = np.clip(grid_prediction, clip_low, clip_high)
    elif clip_mode == "robust_range":
        clip_low, clip_high = np.percentile(point_values, [1, 99])
        grid_prediction = np.clip(grid_prediction, clip_low, clip_high)

    warnings: list[str] = []
    if fallback_features:
        warnings.append(
            "The current points have no range for "
            f"{', '.join(fallback_features)}; the preview uses the saved training interval."
        )
    if dropped_nonfinite:
        warnings.append(
            f"Excluded {dropped_nonfinite:,} row(s) with non-finite X, Y, or Z from "
            "current-data diagnostics. Finite X/Y pairs still define the preview extent."
        )
    if invalid_grid_prediction_count:
        warnings.append(
            f"The loaded model returned non-finite values for "
            f"{invalid_grid_prediction_count:,} preview-grid cell(s)."
        )
    if invalid_grid_uncertainty_count:
        warnings.append(
            f"The loaded model returned non-finite uncertainty for "
            f"{invalid_grid_uncertainty_count:,} preview-grid cell(s)."
        )
    extrapolation_fraction = float(np.mean(point_extrapolation))
    if extrapolation_fraction > 0:
        warnings.append(
            f"{extrapolation_fraction:.1%} of current points are outside at least one saved "
            "training-coordinate interval."
        )
    predicted_finite = int(np.isfinite(point_prediction).sum())
    if predicted_finite < len(point_prediction):
        warnings.append(
            f"The loaded model returned non-finite values for "
            f"{len(point_prediction) - predicted_finite:,} current point(s)."
        )

    evaluation_metrics = _regression_metrics(point_values, point_prediction)
    metrics: dict[str, float | int | None] = {
        "input_points": int(input_points),
        "finite_points": int(len(point_values)),
        "unique_points": int(len(support_coordinates)),
        "dropped_nonfinite": dropped_nonfinite,
        "prediction_seconds": float(prediction_seconds),
        "total_seconds": float(perf_counter() - total_started),
        "grid_coverage": float(np.isfinite(grid_prediction).mean()),
        "point_extrapolation_fraction": extrapolation_fraction,
        "grid_extrapolation_fraction": (
            float(np.mean(grid_extrapolation[finite_supported])) if finite_supported.any() else None
        ),
        "overshoot_fraction": overshoot_fraction,
    }
    metrics.update(_prefixed_metrics("evaluation", evaluation_metrics))
    return SurfaceEvaluationResult(
        algorithm=model.algorithm,
        parameters=model.parameters.copy(),
        model=model,
        grid_x=grid_x,
        grid_y=grid_y,
        grid_z=grid_prediction.reshape(grid_x.shape),
        grid_uncertainty=(
            None if grid_uncertainty is None else grid_uncertainty.reshape(grid_x.shape)
        ),
        point_x=chart_coordinates[:, 0].copy(),
        point_y=chart_coordinates[:, 1].copy(),
        point_z=point_values.copy(),
        point_prediction=point_prediction,
        metrics=metrics,
        warnings=tuple(warnings),
    )


def fit_surface(
    x: ArrayLike,
    y: ArrayLike,
    z: ArrayLike,
    *,
    algorithm: str = "rbf",
    parameters: Mapping[str, Any] | None = None,
    grid_resolution: int = 60,
    padding_fraction: float = 0.0,
    domain_mask: str = "convex_hull",
    density_factor: float = 3.0,
    clip_mode: str = "none",
    duplicate_reducer: str = "mean",
    normalize_coordinates: bool = True,
    max_fit_points: int | None = 3_000,
    validation_fraction: float = 0.2,
    random_state: int = 42,
    input_features: tuple[str, str] = ("X", "Y"),
    output_feature: str = "Z",
) -> SurfaceFitResult:
    """Fit ``z = f(x, y)`` and evaluate it on a regular grid.

    Non-finite rows are dropped, duplicate ``(x, y)`` locations are reduced, and a held-out
    validation split is evaluated before a final model is fitted to all retained points.
    """

    if algorithm not in SURFACE_ALGORITHMS:
        raise ValueError(f"Unknown surface algorithm: {algorithm!r}.")
    if not 15 <= int(grid_resolution) <= 200:
        raise ValueError("grid_resolution must be between 15 and 200.")
    if not 0 <= float(padding_fraction) <= 0.5:
        raise ValueError("padding_fraction must be between 0 and 0.5.")
    if not 0 <= float(validation_fraction) < 0.5:
        raise ValueError("validation_fraction must be at least 0 and less than 0.5.")
    if clip_mode not in {"none", "observed_range", "robust_range"}:
        raise ValueError("clip_mode must be 'none', 'observed_range', or 'robust_range'.")
    if domain_mask not in {"convex_hull", "dense_support", "full_grid"}:
        raise ValueError("domain_mask must be 'convex_hull', 'dense_support', or 'full_grid'.")
    if domain_mask == "dense_support" and (
        not np.isfinite(float(density_factor)) or float(density_factor) <= 0
    ):
        raise ValueError("density_factor must be positive.")
    if max_fit_points is not None and int(max_fit_points) < 4:
        raise ValueError("max_fit_points must be at least 4 or None.")
    input_features = tuple(input_features)
    if (
        len(input_features) != 2
        or not all(isinstance(name, str) and name for name in input_features)
        or not isinstance(output_feature, str)
        or not output_feature
    ):
        raise ValueError("Surface input and output feature names must be non-empty strings.")
    total_started = perf_counter()

    x_values = _as_float_vector(x, "X")
    y_values = _as_float_vector(y, "Y")
    z_values = _as_float_vector(z, "Z")
    if not (len(x_values) == len(y_values) == len(z_values)):
        raise ValueError("X, Y, and Z must contain the same number of values.")
    if not len(x_values):
        raise ValueError("At least four points are required to fit a surface.")

    input_points = len(x_values)
    finite = np.isfinite(x_values) & np.isfinite(y_values) & np.isfinite(z_values)
    coordinates = np.column_stack([x_values[finite], y_values[finite]])
    values = z_values[finite]
    dropped_nonfinite = int(input_points - len(values))
    if len(values) < 4:
        raise ValueError("At least four finite points are required to fit a surface.")

    coordinates, values, duplicates_merged = _collapse_duplicate_coordinates(
        coordinates, values, duplicate_reducer
    )
    unique_points = len(values)
    if unique_points < 4:
        raise ValueError("At least four unique (X, Y) locations are required.")
    centered = coordinates - np.mean(coordinates, axis=0)
    if np.linalg.matrix_rank(centered) < 2:
        raise ValueError(
            "The X/Y coordinates are collinear; a two-dimensional surface is undefined."
        )

    algorithm_parameters = default_surface_parameters(algorithm, unique_points)
    if parameters:
        algorithm_parameters.update(dict(parameters))
    minimum_points = _minimum_points(algorithm, algorithm_parameters)

    requested_cap = unique_points if max_fit_points is None else int(max_fit_points)
    algorithm_cap = surface_point_cap(algorithm, algorithm_parameters)
    effective_cap = min(requested_cap, algorithm_cap or requested_cap, unique_points)
    if effective_cap < minimum_points:
        raise ValueError(
            f"{SURFACE_ALGORITHMS[algorithm]} needs at least {minimum_points} fit points; "
            f"the current cap allows {effective_cap}."
        )
    if algorithm == "rbf" and algorithm_parameters.get("neighbors") not in (
        None,
        0,
    ):
        algorithm_parameters["neighbors"] = min(
            int(algorithm_parameters["neighbors"]), effective_cap
        )
    selected_indices = _subsample_indices(coordinates, effective_cap, int(random_state))
    fit_coordinates_raw = coordinates[selected_indices]
    fit_values = values[selected_indices]
    if np.linalg.matrix_rank(fit_coordinates_raw - np.mean(fit_coordinates_raw, axis=0)) < 2:
        raise ValueError("The sampled X/Y coordinates cannot define a two-dimensional surface.")

    warnings: list[str] = []
    if dropped_nonfinite:
        warnings.append(f"Dropped {dropped_nonfinite:,} non-finite point(s).")
    if duplicates_merged:
        warnings.append(
            f"Merged {duplicates_merged:,} duplicate X/Y location(s) using {duplicate_reducer}."
        )
    if effective_cap < unique_points:
        reason = (
            "the algorithm safety cap"
            if algorithm_cap is not None and algorithm_cap < min(requested_cap, unique_points)
            else "the selected fit-point cap"
        )
        warnings.append(
            f"Used {effective_cap:,} of {unique_points:,} unique points because of {reason}."
        )

    if normalize_coordinates:
        coordinate_center = np.mean(fit_coordinates_raw, axis=0)
        coordinate_scale = np.std(fit_coordinates_raw, axis=0)
        if np.any(coordinate_scale <= np.finfo(np.float64).eps):
            raise ValueError("X and Y must each vary to normalize the coordinates.")
    else:
        coordinate_center = np.zeros(2, dtype=np.float64)
        coordinate_scale = np.ones(2, dtype=np.float64)
    fit_coordinates = (fit_coordinates_raw - coordinate_center) / coordinate_scale

    validation_metrics: dict[str, float | int | None] = {
        "count": 0,
        "predicted_count": 0,
        "total_count": 0,
        "coverage": None,
        "rmse": None,
        "mae": None,
        "r2": None,
        "nrmse": None,
    }
    validation_seconds = 0.0
    if validation_fraction > 0:
        validation_minimum_points = minimum_points
        if algorithm == "rbf" and algorithm_parameters.get("neighbors") not in (
            None,
            0,
        ):
            validation_minimum_points = max(
                validation_minimum_points, int(algorithm_parameters["neighbors"])
            )
        validation_count = max(1, int(round(effective_cap * validation_fraction)))
        validation_count = min(validation_count, effective_cap - validation_minimum_points)
        if validation_count < 1:
            warnings.append("There are too few points for held-out validation at this model size.")
        else:
            training_indices, validation_indices = _validation_split_indices(
                fit_coordinates_raw, validation_count, int(random_state)
            )
            if len(validation_indices) < validation_count:
                warnings.append(
                    "Held-out validation kept coordinate extrema in training, reducing the "
                    f"validation set from {validation_count:,} to {len(validation_indices):,}."
                )
            try:
                validation_started = perf_counter()
                training_coordinates_raw = fit_coordinates_raw[training_indices]
                validation_coordinates_raw = fit_coordinates_raw[validation_indices]
                if (
                    np.linalg.matrix_rank(
                        training_coordinates_raw - np.mean(training_coordinates_raw, axis=0)
                    )
                    < 2
                ):
                    raise ValueError("the validation training coordinates are collinear")
                if normalize_coordinates:
                    validation_center = np.mean(training_coordinates_raw, axis=0)
                    validation_scale = np.std(training_coordinates_raw, axis=0)
                    if np.any(validation_scale <= np.finfo(np.float64).eps):
                        raise ValueError("the validation training coordinates cannot be normalized")
                else:
                    validation_center = np.zeros(2, dtype=np.float64)
                    validation_scale = np.ones(2, dtype=np.float64)
                validation_training_coordinates = (
                    training_coordinates_raw - validation_center
                ) / validation_scale
                validation_coordinates = (
                    validation_coordinates_raw - validation_center
                ) / validation_scale
                validation_model = _fit_model(
                    algorithm,
                    validation_training_coordinates,
                    fit_values[training_indices],
                    algorithm_parameters,
                    int(random_state),
                )
                validation_prediction, _ = _predict_model(validation_model, validation_coordinates)
                validation_seconds = perf_counter() - validation_started
                validation_metrics = _regression_metrics(
                    fit_values[validation_indices], validation_prediction
                )
                if validation_metrics["coverage"] != 1.0:
                    warnings.append(
                        "Some held-out points were outside the interpolation domain; compare "
                        "validation coverage as well as error."
                    )
            except (ArithmeticError, ValueError, np.linalg.LinAlgError, QhullError) as exc:
                validation_seconds = perf_counter() - validation_started
                warnings.append(f"Held-out validation was unavailable: {_short_error(exc)}")

    try:
        fit_started = perf_counter()
        model = _fit_model(
            algorithm,
            fit_coordinates,
            fit_values,
            algorithm_parameters,
            int(random_state),
        )
        fit_seconds = perf_counter() - fit_started
        fitted_model = SurfaceModel(
            algorithm=algorithm,
            parameters=algorithm_parameters.copy(),
            estimator=model,
            coordinate_center=coordinate_center.copy(),
            coordinate_scale=coordinate_scale.copy(),
            training_bounds=np.column_stack(
                [
                    np.min(fit_coordinates_raw, axis=0),
                    np.max(fit_coordinates_raw, axis=0),
                ]
            ),
            training_output_range=(float(np.min(fit_values)), float(np.max(fit_values))),
            input_features=(input_features[0], input_features[1]),
            output_feature=output_feature,
        )
        point_prediction = fitted_model.predict(fit_coordinates_raw)
    except (ArithmeticError, ValueError, np.linalg.LinAlgError, QhullError) as exc:
        raise ValueError(
            f"{SURFACE_ALGORITHMS[algorithm]} could not fit these points: {_short_error(exc)}"
        ) from exc

    train_metrics = _regression_metrics(fit_values, point_prediction)
    x_span = float(np.ptp(coordinates[:, 0]))
    y_span = float(np.ptp(coordinates[:, 1]))
    x_grid = np.linspace(
        float(np.min(coordinates[:, 0]) - padding_fraction * x_span),
        float(np.max(coordinates[:, 0]) + padding_fraction * x_span),
        int(grid_resolution),
    )
    y_grid = np.linspace(
        float(np.min(coordinates[:, 1]) - padding_fraction * y_span),
        float(np.max(coordinates[:, 1]) + padding_fraction * y_span),
        int(grid_resolution),
    )
    grid_x, grid_y = np.meshgrid(x_grid, y_grid)
    grid_coordinates_raw = np.column_stack([grid_x.ravel(), grid_y.ravel()])
    grid_coordinates = (grid_coordinates_raw - coordinate_center) / coordinate_scale

    prediction_started = perf_counter()
    if algorithm == "gaussian_process":
        grid_prediction, grid_uncertainty = fitted_model.predict(
            grid_coordinates_raw, return_std=True
        )
    else:
        grid_prediction = fitted_model.predict(grid_coordinates_raw)
        grid_uncertainty = None
    prediction_seconds = perf_counter() - prediction_started

    domain = _surface_domain_mask(
        fit_coordinates, grid_coordinates, domain_mask, float(density_factor)
    )
    grid_prediction[~domain] = np.nan
    if grid_uncertainty is not None:
        grid_uncertainty[~domain] = np.nan

    finite_supported = np.isfinite(grid_prediction)
    if finite_supported.any():
        observed_low = float(np.min(values))
        observed_high = float(np.max(values))
        overshoot = (grid_prediction[finite_supported] < observed_low) | (
            grid_prediction[finite_supported] > observed_high
        )
        overshoot_fraction = float(np.mean(overshoot))
    else:
        overshoot_fraction = None

    if clip_mode == "observed_range":
        clip_low, clip_high = np.min(values), np.max(values)
        grid_prediction = np.clip(grid_prediction, clip_low, clip_high)
    elif clip_mode == "robust_range":
        clip_low, clip_high = np.percentile(values, [1, 99])
        grid_prediction = np.clip(grid_prediction, clip_low, clip_high)

    metrics: dict[str, float | int | None] = {
        "input_points": int(input_points),
        "finite_points": int(input_points - dropped_nonfinite),
        "unique_points": int(unique_points),
        "fit_points": int(effective_cap),
        "dropped_nonfinite": dropped_nonfinite,
        "duplicates_merged": duplicates_merged,
        "fit_seconds": float(fit_seconds),
        "validation_seconds": float(validation_seconds),
        "prediction_seconds": float(prediction_seconds),
        "total_seconds": float(perf_counter() - total_started),
        "grid_coverage": float(np.isfinite(grid_prediction).mean()),
        "overshoot_fraction": overshoot_fraction,
    }
    metrics.update(_prefixed_metrics("train", train_metrics))
    metrics.update(_prefixed_metrics("validation", validation_metrics))

    return SurfaceFitResult(
        algorithm=algorithm,
        parameters=algorithm_parameters,
        model=fitted_model,
        grid_x=grid_x,
        grid_y=grid_y,
        grid_z=grid_prediction.reshape(grid_x.shape),
        grid_uncertainty=(
            None if grid_uncertainty is None else grid_uncertainty.reshape(grid_x.shape)
        ),
        point_x=fit_coordinates_raw[:, 0].copy(),
        point_y=fit_coordinates_raw[:, 1].copy(),
        point_z=fit_values.copy(),
        point_prediction=point_prediction,
        metrics=metrics,
        warnings=tuple(warnings),
    )
