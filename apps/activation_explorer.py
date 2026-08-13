"""Streamlit UI for exploring locally cached conversational activations."""

from __future__ import annotations

import gc
from hashlib import sha256
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from pandas.api.types import is_bool_dtype, is_numeric_dtype
from sklearn import __version__ as sklearn_version
from sklearn.cross_decomposition import PLSRegression

from temporal_manifolds.activations.extraction_policy import (
    CACHED_POSITION_INDEX,
    PROMPT_TOKEN_POSITION,
    TARGET_LAYER_COMPONENT,
)
from temporal_manifolds.geometry.extrusion.rms_spline_surface_transformer import (
    RMSSplineSurfaceEvaluationResult,
    RMSSplineSurfaceTransformer,
    add_surface_parameter_columns,
    evaluate_rms_spline_surface,
    serialize_rms_spline_surface,
)
from temporal_manifolds.viz.activation_explorer import (
    PCA_PROJECTION_FINGERPRINT_VERSION,
    SOURCE_FOLDER_FIELD,
    discover_activation_batch_paths,
    extract_activation_slice,
    fit_pca_projection,
    inspect_sources,
    load_pca_model,
    load_pls_model,
    metadata_filter_choices,
    metadata_filter_mask,
    pca_projection_fingerprint,
    prepare_analysis_data,
    projection_details_table,
    reconstruction_residual_rms,
    select_activation_batch_uploads,
    serialize_pca_model,
    serialize_pls_model,
    transform_pca_projection,
)
from temporal_manifolds.viz.curve_fitting import (
    CURVE_ALGORITHMS,
    CURVE_DESCRIPTIONS,
    CURVE_MODEL_ARTIFACT_VERSION,
    CurveModel,
    CurveDisplayResult,
    CurveEvaluationResult,
    CurveFitResult,
    evaluate_curve_model,
    fit_geometric_spline,
    load_curve_model,
    parse_spline_quantiles,
    project_onto_curve_parameter,
    serialize_curve_model,
)
from temporal_manifolds.viz.surface_fitting import (
    ALGORITHM_POINT_CAPS,
    SURFACE_ALGORITHMS,
    SURFACE_DESCRIPTIONS,
    SURFACE_MODEL_ARTIFACT_VERSION,
    SurfaceEvaluationResult,
    SurfaceFitResult,
    default_surface_parameters,
    evaluate_surface_model,
    fit_surface,
    load_surface_model,
    serialize_surface_model,
    surface_point_cap,
)

SurfaceDisplayResult = SurfaceFitResult | SurfaceEvaluationResult

st.set_page_config(
    page_title="Activation Atlas",
    page_icon=":material/scatter_plot:",
    layout="wide",
)
st.caption("TEMPORAL MANIFOLDS · LOCAL ANALYSIS")
st.title("Activation Atlas")
st.write(
    "Filter conversational activation batches, aggregate comparable prompts, and inspect "
    "PCA or log-time-horizon-supervised PLS directions. Your files stay in this local app "
    "session."
)

st.session_state.setdefault("sources", None)
st.session_state.setdefault("source_label", "")
st.session_state.setdefault("source_is_local", None)
st.session_state.setdefault("source_revision", 0)


def reset_loaded_data() -> None:
    for key in (
        "inspection",
        "slice_key",
        "activation_matrix",
        "activation_cache_path",
        "analysis_key",
        "prepared_key",
        "prepared_matrix",
        "prepared_row_offsets",
        "prepared_metadata",
        "prepared_details",
        "pca_key",
        "projection",
        "pca",
        "details",
        "surface_result",
        "surface_identity",
        "surface_benchmark",
        "surface_benchmark_identity",
        "loaded_surface_result",
        "loaded_surface_identity",
        "curve_result",
        "curve_identity",
        "loaded_curve_result",
        "loaded_curve_identity",
        "loaded_extruded_surface_result",
        "loaded_extruded_surface_identity",
    ):
        st.session_state.pop(key, None)


def use_sources(sources: list, *, label: str, is_local: bool) -> None:
    """Install a source selection and invalidate every derived result."""
    st.session_state.sources = sources
    st.session_state.source_label = label
    st.session_state.source_is_local = is_local
    st.session_state.source_revision += 1
    reset_loaded_data()


def clear_surface_fits() -> None:
    for key in (
        "surface_result",
        "surface_identity",
        "surface_benchmark",
        "surface_benchmark_identity",
        "loaded_surface_result",
        "loaded_surface_identity",
        "curve_result",
        "curve_identity",
        "loaded_curve_result",
        "loaded_curve_identity",
        "loaded_extruded_surface_result",
        "loaded_extruded_surface_identity",
    ):
        st.session_state.pop(key, None)


def clear_visual_filters() -> None:
    for key in list(st.session_state):
        if key == "visual_filter_fields" or key.startswith("visual_filter::"):
            st.session_state.pop(key, None)


def forget_loaded_pca() -> None:
    for key in (
        "pca_model_upload",
        "loaded_pca_digest",
        "loaded_pca_model",
        "loaded_pca_provenance",
        "loaded_pca_filename",
    ):
        st.session_state.pop(key, None)


def forget_loaded_pls() -> None:
    for key in (
        "pls_model_upload",
        "loaded_pls_digest",
        "loaded_pls_model",
        "loaded_pls_provenance",
        "loaded_pls_filename",
    ):
        st.session_state.pop(key, None)


def forget_loaded_surface() -> None:
    for key in (
        "surface_model_upload",
        "loaded_surface_digest",
        "loaded_surface_model",
        "loaded_surface_provenance",
        "loaded_surface_filename",
        "loaded_surface_result",
        "loaded_surface_identity",
    ):
        st.session_state.pop(key, None)


def forget_loaded_curve() -> None:
    for key in (
        "curve_model_upload",
        "loaded_curve_digest",
        "loaded_curve_model",
        "loaded_curve_provenance",
        "loaded_curve_filename",
        "loaded_curve_result",
        "loaded_curve_identity",
    ):
        st.session_state.pop(key, None)


def forget_loaded_extruded_surface() -> None:
    for key in (
        "extrusion_curve_upload",
        "extruded_surface_model_upload",
        "loaded_extruded_surface_digest",
        "loaded_extrusion_curve_model",
        "loaded_extruded_surface_model",
        "loaded_extruded_surface_provenance",
        "loaded_extruded_surface_filename",
        "loaded_extruded_surface_result",
        "loaded_extruded_surface_identity",
    ):
        st.session_state.pop(key, None)


@st.cache_data(max_entries=16, show_spinner=False)
def fit_surface_cached(
    x_values: np.ndarray,
    y_values: np.ndarray,
    z_values: np.ndarray,
    options: dict[str, Any],
) -> SurfaceFitResult:
    """Cache the reusable fitted model, preview grid, and diagnostics."""

    return fit_surface(x_values, y_values, z_values, **options)


@st.cache_data(max_entries=16, show_spinner=False)
def fit_geometric_spline_cached(
    coordinates: np.ndarray,
    options: dict[str, Any],
) -> CurveFitResult:
    """Cache a geometric principal-spline fit."""

    return fit_geometric_spline(coordinates, **options)


def fit_pls_projection(
    activation_matrix: np.ndarray,
    prepared_matrix: np.ndarray | None,
    row_offsets: np.ndarray,
    analysis_metadata: pd.DataFrame,
    *,
    n_components: int,
    scale: bool = True,
    max_iter: int = 500,
    tol: float = 1e-6,
    details: dict[str, Any],
) -> tuple[pd.DataFrame, PLSRegression, dict[str, Any]]:
    """Fit PLS directions supervised by log10 time horizon in months."""

    analysis_rows = len(analysis_metadata)
    feature_count = int(activation_matrix.shape[1])
    max_components = min(max(analysis_rows - 1, 0), feature_count)
    if not 1 <= n_components <= max_components:
        raise ValueError(
            f"PLS components must be between 1 and {max_components} for the prepared matrix."
        )
    if max_iter < 1:
        raise ValueError("PLS maximum iterations must be at least 1.")
    if not np.isfinite(tol) or tol <= 0:
        raise ValueError("PLS convergence tolerance must be positive and finite.")

    if "time_horizon_months" not in analysis_metadata:
        raise ValueError("PLS requires the derived time_horizon_months metadata field.")
    horizons = pd.to_numeric(analysis_metadata["time_horizon_months"], errors="coerce").to_numpy(
        dtype=np.float64
    )
    if not np.isfinite(horizons).all() or np.any(horizons <= 0):
        raise ValueError("PLS requires positive, finite time horizons.")
    target = np.log10(horizons)
    if np.ptp(target) <= np.finfo(np.float64).eps:
        raise ValueError("PLS requires at least two distinct log-time-horizon values.")

    if prepared_matrix is not None:
        values = np.asarray(prepared_matrix, dtype=np.float32)
    else:
        values = np.asarray(activation_matrix[row_offsets], dtype=np.float32)
    expected_shape = (analysis_rows, feature_count)
    if values.shape != expected_shape:
        raise ValueError(
            "Prepared activations and metadata are misaligned: "
            f"expected {expected_shape}, got {values.shape}."
        )

    pls = PLSRegression(
        n_components=n_components,
        scale=scale,
        max_iter=max_iter,
        tol=tol,
        copy=True,
    )
    scores, _ = pls.fit_transform(values, target.reshape(-1, 1))
    scores = np.asarray(scores)

    # Express each fitted rotation directly in raw activation coordinates. These aliases let
    # the existing coordinate fingerprint protect saved surface artifacts in PLS space too.
    x_scale = np.asarray(pls._x_std, dtype=np.float64)  # noqa: SLF001 - fitted sklearn state
    pls.components_ = np.asarray(pls.x_rotations_).T / x_scale[np.newaxis, :]
    pls.mean_ = np.asarray(pls._x_mean)  # noqa: SLF001 - fitted sklearn state

    scaled_total = np.sum(
        np.square((values - pls.mean_) / x_scale, dtype=np.float64), dtype=np.float64
    )
    component_energy = np.sum(np.square(scores), axis=0) * np.sum(
        np.square(pls.x_loadings_), axis=0
    )
    explained_x_variance = (
        np.clip(component_energy / scaled_total, 0.0, 1.0)
        if scaled_total > np.finfo(np.float64).eps
        else np.zeros(n_components, dtype=np.float64)
    )
    pls.explained_variance_ratio_ = explained_x_variance

    result = analysis_metadata.copy().reset_index(drop=True)
    for index in range(n_components):
        result[f"PLS{index + 1}"] = scores[:, index]
    result["log10_time_horizon_months"] = target
    projection_details = {
        **details,
        "explained_variance": explained_x_variance,
        "direction_method": "PLS",
        "direction_source": "fitted",
        "direction_solver": type(pls).__name__,
        "pls_target": "log10_time_horizon_months",
        "pls_scale": scale,
        "pls_max_iter": max_iter,
        "pls_tolerance": tol,
        "pls_iterations": list(pls.n_iter_),
    }
    return result, pls, projection_details


def transform_pls_projection(
    activation_matrix: np.ndarray,
    prepared_matrix: np.ndarray | None,
    row_offsets: np.ndarray,
    analysis_metadata: pd.DataFrame,
    *,
    pls: PLSRegression,
    details: dict[str, Any],
) -> tuple[pd.DataFrame, PLSRegression, dict[str, Any]]:
    """Transform prepared activations with a loaded PLS direction model."""

    analysis_rows = len(analysis_metadata)
    feature_count = int(activation_matrix.shape[1])
    if prepared_matrix is not None:
        values = np.asarray(prepared_matrix, dtype=np.float32)
    else:
        values = np.asarray(activation_matrix[row_offsets], dtype=np.float32)
    expected_shape = (analysis_rows, feature_count)
    if values.shape != expected_shape:
        raise ValueError(
            "Prepared activations and metadata are misaligned: "
            f"expected {expected_shape}, got {values.shape}."
        )

    expected_features = int(pls.components_.shape[1])
    if feature_count != expected_features:
        raise ValueError(
            f"The loaded PLS expects {expected_features:,} activation features, "
            f"but the selected component has {feature_count:,}."
        )
    scores = np.asarray(pls.transform(values))
    component_count = int(pls.components_.shape[0])
    if scores.shape != (analysis_rows, component_count):
        raise ValueError("The loaded PLS model returned an invalid projection shape.")

    result = analysis_metadata.copy().reset_index(drop=True)
    for index in range(component_count):
        result[f"PLS{index + 1}"] = scores[:, index]
    if "time_horizon_months" in result:
        horizons = pd.to_numeric(result["time_horizon_months"], errors="coerce").to_numpy(
            dtype=np.float64
        )
        if np.isfinite(horizons).all() and np.all(horizons > 0):
            result["log10_time_horizon_months"] = np.log10(horizons)

    projection_details = {
        **details,
        "explained_variance": np.asarray(pls.explained_variance_ratio_),
        "direction_method": "PLS",
        "direction_source": "loaded",
        "direction_solver": type(pls).__name__,
        "pls_target": "log10_time_horizon_months",
        "pls_scale": bool(pls.scale),
        "pls_max_iter": int(pls.max_iter),
        "pls_tolerance": float(pls.tol),
        "pls_iterations": list(pls.n_iter_),
    }
    return result, pls, projection_details


def surface_algorithm_controls(algorithm: str, point_count: int) -> dict[str, Any]:
    """Render algorithm-specific controls inside the surface-fit form."""

    prefix = f"surface_parameter::{algorithm}"
    defaults = default_surface_parameters(algorithm, point_count)

    if algorithm == "plane":
        estimator = st.selectbox(
            "Plane estimator",
            ["least_squares", "huber", "ransac"],
            format_func={
                "least_squares": "Least squares",
                "huber": "Huber robust loss",
                "ransac": "RANSAC outlier rejection",
            }.__getitem__,
            key=f"{prefix}::estimator",
            persist_state="page",
        )
        columns = st.columns(3)
        huber_epsilon = columns[0].slider(
            "Huber epsilon",
            1.01,
            3.0,
            1.35,
            0.05,
            key=f"{prefix}::huber_epsilon",
            persist_state="page",
        )
        huber_alpha = 10.0 ** columns[1].slider(
            "log10 Huber alpha",
            -8.0,
            1.0,
            -4.0,
            0.25,
            key=f"{prefix}::huber_alpha",
            persist_state="page",
        )
        ransac_min_samples = columns[2].slider(
            "RANSAC sample fraction",
            0.1,
            0.95,
            0.5,
            0.05,
            key=f"{prefix}::ransac_min_samples",
            persist_state="page",
        )
        ransac_columns = st.columns(2)
        automatic_threshold = ransac_columns[0].checkbox(
            "Automatic residual threshold",
            value=True,
            key=f"{prefix}::ransac_auto",
            persist_state="page",
        )
        residual_threshold = ransac_columns[1].number_input(
            "RANSAC residual threshold",
            min_value=0.0,
            value=1.0,
            step=0.1,
            key=f"{prefix}::ransac_threshold",
            help="Ignored while automatic thresholding is selected.",
            persist_state="page",
        )
        return {
            "estimator": estimator,
            "huber_epsilon": huber_epsilon,
            "huber_alpha": huber_alpha,
            "ransac_min_samples": ransac_min_samples,
            "ransac_residual_threshold": None if automatic_threshold else residual_threshold,
            "ransac_max_trials": 200,
        }

    if algorithm == "polynomial":
        columns = st.columns(3)
        degree = columns[0].slider(
            "Polynomial degree",
            1,
            6,
            int(defaults["degree"]),
            key=f"{prefix}::degree",
            persist_state="page",
        )
        alpha = 10.0 ** columns[1].slider(
            "log10 ridge alpha",
            -8.0,
            4.0,
            -2.0,
            0.25,
            key=f"{prefix}::alpha",
            persist_state="page",
        )
        interaction_only = columns[2].checkbox(
            "Interactions only",
            value=False,
            key=f"{prefix}::interactions",
            persist_state="page",
        )
        term_count = (
            (3 if degree == 1 else 4) if interaction_only else ((degree + 1) * (degree + 2) // 2)
        )
        st.caption(f"Degree {degree} uses {term_count:,} coefficients including the intercept.")
        return {
            "degree": degree,
            "alpha": alpha,
            "interaction_only": interaction_only,
        }

    if algorithm == "rbf":
        columns = st.columns(3)
        kernel = columns[0].selectbox(
            "RBF kernel",
            [
                "thin_plate_spline",
                "linear",
                "cubic",
                "quintic",
                "multiquadric",
                "inverse_multiquadric",
                "inverse_quadratic",
                "gaussian",
            ],
            format_func=lambda value: value.replace("_", " ").capitalize(),
            key=f"{prefix}::kernel",
            persist_state="page",
        )
        exact_fit = columns[1].checkbox(
            "Exact interpolation",
            value=False,
            key=f"{prefix}::exact",
            persist_state="page",
        )
        smoothing = 10.0 ** columns[2].slider(
            "log10 smoothing",
            -8.0,
            2.0,
            -2.0,
            0.25,
            key=f"{prefix}::smoothing",
            persist_state="page",
        )
        locality = st.columns(3)
        use_all_points = locality[0].checkbox(
            "Use all fit points",
            value=False,
            key=f"{prefix}::all_points",
            persist_state="page",
        )
        maximum_neighbors = max(3, min(point_count, 500))
        neighbor_default = min(maximum_neighbors, int(defaults["neighbors"]))
        neighbor_key = f"{prefix}::neighbors"
        clamp_integer_widget_state(
            neighbor_key,
            minimum=3,
            maximum=maximum_neighbors,
            default=max(3, neighbor_default),
        )
        neighbors = locality[1].slider(
            "Local neighbors",
            3,
            maximum_neighbors,
            max(3, neighbor_default),
            key=neighbor_key,
            help="Ignored while all fit points are selected.",
            persist_state="page",
        )
        epsilon = 10.0 ** locality[2].slider(
            "log10 epsilon",
            -2.0,
            2.0,
            0.0,
            0.25,
            key=f"{prefix}::epsilon",
            persist_state="page",
        )
        degree_label = st.selectbox(
            "Polynomial tail degree",
            ["Automatic", "None (-1)", "Constant (0)", "Linear (1)", "Quadratic (2)"],
            key=f"{prefix}::degree",
            help="Automatic follows SciPy's safe minimum for the selected kernel.",
            persist_state="page",
        )
        degree_map = {
            "Automatic": None,
            "None (-1)": -1,
            "Constant (0)": 0,
            "Linear (1)": 1,
            "Quadratic (2)": 2,
        }
        st.caption(
            "Epsilon affects multiquadric, inverse-multiquadric, inverse-quadratic, and "
            "Gaussian kernels. Global RBF fits are capped at 1,000 points; local-neighbor "
            "fits can use more."
        )
        return {
            "kernel": kernel,
            "smoothing": 0.0 if exact_fit else smoothing,
            "neighbors": None if use_all_points else neighbors,
            "epsilon": epsilon,
            "degree": degree_map[degree_label],
        }

    if algorithm == "svr":
        columns = st.columns(3)
        kernel = columns[0].selectbox(
            "SVR kernel",
            ["linear", "poly", "rbf"],
            key=f"{prefix}::kernel",
            persist_state="page",
        )
        c_value = 10.0 ** columns[1].slider(
            "log10 C",
            -3.0,
            4.0,
            1.0,
            0.25,
            key=f"{prefix}::c",
            persist_state="page",
        )
        epsilon = 10.0 ** columns[2].slider(
            "log10 epsilon",
            -5.0,
            1.0,
            -1.0,
            0.25,
            key=f"{prefix}::epsilon",
            persist_state="page",
        )
        gamma_columns = st.columns(3)
        gamma_mode = gamma_columns[0].selectbox(
            "Gamma mode",
            ["scale", "auto", "manual"],
            key=f"{prefix}::gamma_mode",
            persist_state="page",
        )
        gamma_value = 10.0 ** gamma_columns[1].slider(
            "log10 manual gamma",
            -4.0,
            3.0,
            0.0,
            0.25,
            key=f"{prefix}::gamma",
            help="Ignored unless gamma mode is Manual.",
            persist_state="page",
        )
        degree = gamma_columns[2].slider(
            "Polynomial degree",
            2,
            6,
            3,
            key=f"{prefix}::degree",
            help="Ignored unless the polynomial kernel is selected.",
            persist_state="page",
        )
        return {
            "kernel": kernel,
            "c": c_value,
            "epsilon": epsilon,
            "gamma": gamma_value if gamma_mode == "manual" else gamma_mode,
            "degree": degree,
        }

    if algorithm == "gaussian_process":
        columns = st.columns(3)
        kernel = columns[0].selectbox(
            "Covariance kernel",
            ["matern_1.5", "rbf", "matern_0.5", "matern_2.5", "rational_quadratic"],
            format_func=lambda value: value.replace("_", " ").title(),
            key=f"{prefix}::kernel",
            persist_state="page",
        )
        length_scale = 10.0 ** columns[1].slider(
            "log10 length scale",
            -3.0,
            3.0,
            0.0,
            0.25,
            key=f"{prefix}::length",
            persist_state="page",
        )
        noise_level = 10.0 ** columns[2].slider(
            "log10 noise level",
            -8.0,
            1.0,
            -2.0,
            0.25,
            key=f"{prefix}::noise",
            persist_state="page",
        )
        optimizer_columns = st.columns(2)
        optimize = optimizer_columns[0].checkbox(
            "Optimize kernel",
            value=True,
            key=f"{prefix}::optimize",
            persist_state="page",
        )
        restarts = optimizer_columns[1].slider(
            "Optimizer restarts",
            0,
            3,
            0,
            key=f"{prefix}::restarts",
            help="Ignored when kernel optimization is disabled.",
            persist_state="page",
        )
        st.caption("Gaussian-process fits are automatically capped at 1,000 points.")
        return {
            "kernel": kernel,
            "length_scale": length_scale,
            "noise_level": noise_level,
            "optimize": optimize,
            "optimizer_restarts": restarts,
        }

    raise ValueError(f"Unknown surface algorithm: {algorithm!r}.")


def _format_metric(value: float | int | None, *, percent: bool = False) -> str:
    if value is None or not np.isfinite(float(value)):
        return "—"
    if percent:
        return f"{float(value):.1%}"
    return f"{float(value):.4g}"


def clamp_integer_widget_state(key: str, *, minimum: int, maximum: int, default: int) -> None:
    """Keep persisted controls valid when filters shrink their dynamic bounds."""

    try:
        stored_value = int(st.session_state.get(key, default))
    except (TypeError, ValueError):
        stored_value = default
    clamped_value = max(minimum, min(maximum, stored_value))
    if st.session_state.get(key) != clamped_value:
        st.session_state[key] = clamped_value


def surface_appearance_controls(
    result: SurfaceDisplayResult,
) -> dict[str, Any]:
    """Render appearance controls shared by fitted and loaded surface previews."""

    with st.popover("Surface appearance", icon=":material/palette:"):
        color_options = ["Predicted height"]
        if result.grid_uncertainty is not None:
            color_options.append("Predictive uncertainty")
        if st.session_state.get("surface_color_mode") not in (None, *color_options):
            st.session_state.pop("surface_color_mode", None)
        surface_color = st.segmented_control(
            "Surface color",
            color_options,
            default=color_options[0],
            key="surface_color_mode",
            persist_state="page",
        )
        colorscale = st.selectbox(
            "Colorscale",
            [
                "Viridis",
                "Cividis",
                "Plasma",
                "Magma",
                "Inferno",
                "Turbo",
                "RdBu",
                "Spectral",
            ],
            key="surface_colorscale",
            persist_state="page",
        )
        opacity = st.slider(
            "Surface opacity",
            0.05,
            1.0,
            0.58,
            0.01,
            key="surface_opacity",
            persist_state="page",
        )
        show_colorbar = st.toggle(
            "Show surface colorbar",
            value=False,
            key="surface_colorbar",
            persist_state="page",
        )
        reverse_scale = st.toggle(
            "Reverse colorscale",
            value=False,
            key="surface_reverse",
            persist_state="page",
        )
        show_contours = st.toggle(
            "Show height contours",
            value=False,
            key="surface_contours",
            persist_state="page",
        )
        show_wireframe = st.toggle(
            "Show X/Y grid lines",
            value=False,
            key="surface_wireframe",
            persist_state="page",
        )
        show_residuals = st.toggle(
            "Show residual sticks",
            value=False,
            key="surface_residuals",
            persist_state="page",
        )
        residual_count = min(100, len(result.point_x))
        residual_selection = "Largest residuals"
        if show_residuals:
            residual_maximum = min(500, len(result.point_x))
            clamp_integer_widget_state(
                "surface_residual_count",
                minimum=1,
                maximum=residual_maximum,
                default=min(100, residual_maximum),
            )
            residual_count = st.slider(
                "Residual sticks",
                1,
                residual_maximum,
                min(100, residual_maximum),
                key="surface_residual_count",
                persist_state="page",
            )
            residual_selection = st.selectbox(
                "Residual selection",
                ["Largest residuals", "Even sample"],
                key="surface_residual_selection",
                persist_state="page",
            )
    return {
        "color_mode": surface_color,
        "colorscale": colorscale,
        "opacity": opacity,
        "show_colorbar": show_colorbar,
        "reverse_scale": reverse_scale,
        "show_contours": show_contours,
        "show_wireframe": show_wireframe,
        "show_residuals": show_residuals,
        "residual_count": residual_count,
        "residual_selection": residual_selection,
    }


def curve_appearance_controls(result: CurveDisplayResult) -> dict[str, Any]:
    """Render appearance controls shared by fitted and loaded curves."""

    with st.popover("Curve appearance", icon=":material/palette:"):
        color = st.selectbox(
            "Curve color",
            ["#d81b60", "#7b1fa2", "#1565c0", "#00897b", "#ef6c00", "#263238"],
            format_func={
                "#d81b60": "Rose",
                "#7b1fa2": "Purple",
                "#1565c0": "Blue",
                "#00897b": "Teal",
                "#ef6c00": "Orange",
                "#263238": "Charcoal",
            }.__getitem__,
            key="curve_color",
            persist_state="page",
        )
        line_width = st.slider(
            "Line width", 1, 14, 6, key="curve_line_width", persist_state="page"
        )
        opacity = st.slider(
            "Curve opacity",
            0.1,
            1.0,
            0.95,
            0.05,
            key="curve_opacity",
            persist_state="page",
        )
        show_samples = st.toggle(
            "Show sampled positions",
            value=False,
            key="curve_show_samples",
            persist_state="page",
        )
        show_residuals = st.toggle(
            "Show residual sticks",
            value=False,
            key="curve_show_residuals",
            persist_state="page",
        )
        residual_count = min(100, len(result.point_parameter))
        if show_residuals:
            residual_maximum = min(500, len(result.point_parameter))
            clamp_integer_widget_state(
                "curve_residual_count",
                minimum=1,
                maximum=residual_maximum,
                default=min(100, residual_maximum),
            )
            residual_count = st.slider(
                "Residual sticks",
                1,
                residual_maximum,
                min(100, residual_maximum),
                key="curve_residual_count",
                persist_state="page",
            )
    return {
        "color": color,
        "line_width": line_width,
        "opacity": opacity,
        "show_samples": show_samples,
        "show_residuals": show_residuals,
        "residual_count": residual_count,
    }


def extruded_surface_appearance_controls(
    result: RMSSplineSurfaceEvaluationResult,
) -> dict[str, Any]:
    """Render appearance controls for a loaded parametric surface."""

    with st.popover("Extruded surface appearance", icon=":material/palette:"):
        color_mode = st.segmented_control(
            "Surface color",
            ["Curve parameter", "Extrusion coordinate"],
            default="Curve parameter",
            key="extruded_surface_color_mode",
            persist_state="page",
        )
        colorscale = st.selectbox(
            "Colorscale",
            ["Viridis", "Cividis", "Plasma", "Magma", "Inferno", "Turbo", "RdBu"],
            key="extruded_surface_colorscale",
            persist_state="page",
        )
        opacity = st.slider(
            "Surface opacity",
            0.05,
            1.0,
            0.52,
            0.01,
            key="extruded_surface_opacity",
            persist_state="page",
        )
        show_colorbar = st.toggle(
            "Show surface colorbar",
            value=False,
            key="extruded_surface_colorbar",
            persist_state="page",
        )
        reverse_scale = st.toggle(
            "Reverse colorscale",
            value=False,
            key="extruded_surface_reverse",
            persist_state="page",
        )
        show_wireframe = st.toggle(
            "Show sampled grid lines",
            value=False,
            key="extruded_surface_wireframe",
            persist_state="page",
        )
        show_residuals = st.toggle(
            "Show projection residuals",
            value=False,
            key="extruded_surface_residuals",
            persist_state="page",
        )
        residual_count = min(100, len(result.point_xyz))
        if show_residuals:
            residual_maximum = min(500, len(result.point_xyz))
            clamp_integer_widget_state(
                "extruded_surface_residual_count",
                minimum=1,
                maximum=residual_maximum,
                default=min(100, residual_maximum),
            )
            residual_count = st.slider(
                "Projection residuals",
                1,
                residual_maximum,
                min(100, residual_maximum),
                key="extruded_surface_residual_count",
                persist_state="page",
            )
    return {
        "color_mode": color_mode,
        "colorscale": colorscale,
        "opacity": opacity,
        "show_colorbar": show_colorbar,
        "reverse_scale": reverse_scale,
        "show_wireframe": show_wireframe,
        "show_residuals": show_residuals,
        "residual_count": residual_count,
    }


def new_curve_controls(
    *,
    plot_data: pd.DataFrame,
    projection: pd.DataFrame,
    x_component: str,
    y_component: str,
    z_component: str,
    active_pca: Any,
    direction_method: str,
    layer_component: str,
    cached_position: Any,
    aggregation_fields: list[str],
) -> tuple[CurveFitResult | None, dict[str, Any]]:
    """Render controls for fitting a geometry-parameterized spline."""

    coordinate_features = (x_component, y_component, z_component)
    with st.container(border=True):
        st.subheader("Curve fitting")
        st.caption(
            "Fit a geometric principal spline **C(t) = (X(t), Y(t), Z(t))**. The dummy "
            "parameter **t ∈ [0, 1]** is inferred from the point-cloud geometry and refined "
            "to minimize nearest-curve distance."
        )
        algorithm = "spline"
        parameter_feature = "t"
        fit_scope = st.segmented_control(
            "Fit points",
            ["Visible", "All projected"],
            default="Visible",
            key="curve_scope",
            persist_state="page",
        )
        st.caption(CURVE_DESCRIPTIONS[algorithm])

        curve_source = plot_data if fit_scope == "Visible" else projection
        coordinates = curve_source[list(coordinate_features)].to_numpy(dtype=np.float64)
        finite = np.isfinite(coordinates).all(axis=1)
        unique_count = int(len(np.unique(coordinates[finite], axis=0)))
        display_coordinates = plot_data[list(coordinate_features)].to_numpy(dtype=np.float64)
        finite_display = display_coordinates[np.isfinite(display_coordinates).all(axis=1)]
        display_coordinate_bounds = np.column_stack(
            [finite_display.min(axis=0), finite_display.max(axis=0)]
        )
        st.caption(
            f"Current scope contains {len(curve_source):,} point(s) and {unique_count:,} "
            + "distinct finite 3D location(s)."
        )
        minimum_unique = 4
        if unique_count < minimum_unique:
            st.warning(
                f"At least {minimum_unique} distinct finite "
                + "3D locations"
                + " are required."
            )
            return None, {}

        with st.form(f"curve_fit_form::{algorithm}"):
            st.markdown("**Model controls**")
            model_columns = st.columns(2)
            maximum_degree = min(5, unique_count - 1)
            clamp_integer_widget_state(
                "curve_spline_degree",
                minimum=1,
                maximum=maximum_degree,
                default=min(3, maximum_degree),
            )
            spline_degree = model_columns[0].slider(
                "Spline degree",
                1,
                maximum_degree,
                min(3, maximum_degree),
                key="curve_spline_degree",
                persist_state="page",
            )
            smoothing = model_columns[1].slider(
                "Smoothing strength",
                0.0,
                2.0,
                0.15,
                0.01,
                key="curve_spline_smoothing",
                help="Zero follows the inferred centerline; larger values smooth more.",
                persist_state="page",
            )
            model_parameters = {"degree": spline_degree, "smoothing": smoothing}
            knot_error = None
            knot_mode = st.segmented_control(
                "Cubic spline knot specification",
                ["Quantiles", "Number of knots"],
                default="Number of knots",
                key="curve_spline_knot_mode",
                persist_state="page",
            )
            knot_quantile_text = st.text_input(
                "Knot quantiles (Python list)",
                value="[0.25, 0.5, 0.75]",
                key="curve_spline_knot_quantiles",
                help="Strictly increasing quantiles between 0 and 1 along inferred t.",
                disabled=knot_mode != "Quantiles",
                persist_state="page",
            )
            knot_count = int(
                st.number_input(
                    "Total number of knots",
                    min_value=0,
                    max_value=max(0, unique_count - 4),
                    value=0,
                    key="curve_spline_knot_count",
                    help=(
                        "This many interior positions are placed at evenly spaced quantiles "
                        "of inferred t; boundary knots are implicit."
                    ),
                    disabled=knot_mode != "Number of knots",
                    persist_state="page",
                )
            )
            if spline_degree != 3 and (knot_mode == "Quantiles" or knot_count > 0):
                knot_error = "Quantile-based knots require spline degree 3."
            elif knot_mode == "Quantiles":
                try:
                    knot_quantiles = parse_spline_quantiles(knot_quantile_text)
                except ValueError as exc:
                    knot_error = str(exc)
                else:
                    model_parameters["knot_quantiles"] = knot_quantiles
            elif knot_count:
                model_parameters["knot_count"] = knot_count

            st.markdown("**Sampling**")
            common_columns = st.columns(4)
            max_points_key = f"curve_common::{algorithm}::max_points"
            clamp_integer_widget_state(
                max_points_key,
                minimum=4,
                maximum=unique_count,
                default=min(unique_count, 3_000),
            )
            max_fit_points = int(
                common_columns[0].number_input(
                    "Maximum fit values",
                    min_value=4,
                    max_value=unique_count,
                    value=min(unique_count, 3_000),
                    key=max_points_key,
                    persist_state="page",
                )
            )
            random_state = int(
                common_columns[1].number_input(
                    "Random seed",
                    min_value=0,
                    max_value=2_147_483_647,
                    value=42,
                    key=f"curve_common::{algorithm}::seed",
                    persist_state="page",
                )
            )
            sample_count = common_columns[2].slider(
                "Curve samples",
                40,
                1_000,
                240,
                20,
                key=f"curve_common::{algorithm}::samples",
                persist_state="page",
            )
            padding_percent = common_columns[3].slider(
                "Curve padding",
                0,
                100,
                0,
                1,
                format="%d%%",
                key=f"curve_common::{algorithm}::padding",
                help=(
                    "At 0%, each curve end reaches the visible plot boundary. Increase this "
                    "to extrapolate beyond that box."
                ),
                persist_state="page",
            )
            fit_submitted = st.form_submit_button(
                "Fit / update curve",
                type="primary",
                icon=":material/gesture:",
                width="stretch",
            )

        fit_options = {
            "parameters": model_parameters,
            "sample_count": sample_count,
            "padding_fraction": padding_percent / 100.0,
            "display_coordinate_bounds": display_coordinate_bounds,
            "max_fit_points": max_fit_points,
            "random_state": random_state,
            "coordinate_features": coordinate_features,
        }
        data_digest = sha256(
            np.ascontiguousarray(coordinates).tobytes()
        ).hexdigest()
        curve_identity = (
            data_digest,
            parameter_feature,
            coordinate_features,
            fit_scope,
            algorithm,
            CURVE_MODEL_ARTIFACT_VERSION,
            knot_mode,
            knot_quantile_text,
            knot_count,
            repr(fit_options),
        )
        if fit_submitted:
            try:
                if knot_error is not None:
                    raise ValueError(knot_error)
                with st.spinner(f"Fitting {CURVE_ALGORITHMS[algorithm]}…"):
                    fitted_curve = fit_geometric_spline_cached(
                        coordinates, fit_options
                    )
            except (ValueError, MemoryError) as exc:
                st.session_state.pop("curve_result", None)
                st.session_state.pop("curve_identity", None)
                st.error(f"Curve fitting failed: {exc}")
            else:
                st.session_state.curve_result = fitted_curve
                st.session_state.curve_identity = curve_identity

        result = None
        if st.session_state.get("curve_identity") == curve_identity:
            result = st.session_state.get("curve_result")
        elif st.session_state.get("curve_result") is not None:
            st.info(
                "The points, axes, parameter, or controls changed. Click "
                "**Fit / update curve** to refresh the overlay."
            )
        else:
            st.info("Choose the controls above, then fit the first curve.")
        if result is None:
            return None, {}

        metrics = result.metrics
        diagnostic_columns = st.columns(4)
        diagnostic_columns[0].metric(
            "Geometric 3D RMSE", _format_metric(metrics["geometric_rmse_3d"])
        )
        diagnostic_columns[1].metric(
            "Geometric 3D MAE", _format_metric(metrics["geometric_mae_3d"])
        )
        diagnostic_columns[2].metric(
            "Training 3D RMSE", _format_metric(metrics["train_rmse_3d"])
        )
        diagnostic_columns[3].metric("Fit values", f"{int(metrics['fit_points']):,}")
        if result.warnings:
            st.warning(" ".join(result.warnings))

        metadata = {
            "parameter_feature": parameter_feature,
            "coordinate_features": coordinate_features,
            "fit_scope": fit_scope,
            "curve_data_sha256": data_digest,
            "pca_sha256": pca_projection_fingerprint(active_pca),
            "pca_fingerprint_version": PCA_PROJECTION_FINGERPRINT_VERSION,
            "direction_method": direction_method,
            "direction_target": (
                "log10_time_horizon_months" if direction_method == "PLS" else None
            ),
            "layer_component": layer_component,
            "cached_position": cached_position,
            "aggregation_fields": list(aggregation_fields),
            "random_state": random_state,
        }
        safe_name = "-".join(
            str(value).replace("/", "-").replace("\\", "-").replace(" ", "_")
            for value in (*coordinate_features, "by", parameter_feature)
        )
        st.download_button(
            "Download curve model",
            data=lambda: serialize_curve_model(result.model, metadata=metadata),
            file_name=f"activation_curve_{safe_name}_{algorithm}.joblib",
            mime="application/octet-stream",
            icon=":material/download:",
            on_click="ignore",
            help=(
                "Versioned joblib artifact containing the fitted coordinate predictors, "
                "parameter normalization, bounds, and projection provenance."
            ),
        )
        with st.expander("Use the saved curve"):
            st.code(
                "from temporal_manifolds.viz.curve_fitting import load_curve_model\n\n"
                'curve, provenance = load_curve_model("curve.joblib")\n'
                f"{x_component.lower()}, {y_component.lower()}, {z_component.lower()} = "
                f"curve.predict([{parameter_feature.lower()}])[0]",
                language="python",
            )
        appearance = curve_appearance_controls(result)
        return result, appearance


def loaded_curve_controls(
    *,
    plot_data: pd.DataFrame,
    projection: pd.DataFrame,
    x_component: str,
    y_component: str,
    z_component: str,
    active_pca: Any,
    direction_method: str,
    layer_component: str,
    cached_position: Any,
) -> tuple[CurveEvaluationResult | None, dict[str, Any]]:
    """Load a trusted curve artifact and preview it in compatible 3D coordinates."""

    load_succeeded = False
    with st.container(border=True):
        st.subheader("Saved curve")
        st.warning(
            "Only load curve files you trust. Joblib and pickle artifacts can execute code "
            "when opened."
        )
        model_upload = st.file_uploader(
            "Saved curve model",
            type=["joblib"],
            key="curve_model_upload",
            help="Choose a curve artifact downloaded from Activation Atlas.",
        )
        model_bytes = model_upload.getvalue() if model_upload is not None else None
        uploaded_digest = sha256(model_bytes).hexdigest() if model_bytes is not None else None
        already_loaded = (
            uploaded_digest is not None
            and uploaded_digest == st.session_state.get("loaded_curve_digest")
            and st.session_state.get("loaded_curve_model") is not None
        )
        if st.button(
            "Load uploaded curve",
            type="primary",
            icon=":material/upload_file:",
            width="stretch",
            disabled=model_bytes is None or already_loaded,
            key="load_curve_model_button",
        ):
            try:
                loaded_model, loaded_provenance = load_curve_model(model_bytes)
            except ValueError as exc:
                st.error(str(exc))
            else:
                st.session_state.loaded_curve_digest = uploaded_digest
                st.session_state.loaded_curve_model = loaded_model
                st.session_state.loaded_curve_provenance = loaded_provenance
                st.session_state.loaded_curve_filename = model_upload.name
                st.session_state.pop("loaded_curve_result", None)
                st.session_state.pop("loaded_curve_identity", None)
                load_succeeded = True

        loaded_model = st.session_state.get("loaded_curve_model")
        loaded_provenance = st.session_state.get("loaded_curve_provenance", {})
        loaded_digest = st.session_state.get("loaded_curve_digest")
        if loaded_model is None:
            st.info("Choose a saved curve and click **Load uploaded curve**.")
            return None, {}

        loaded_filename = st.session_state.get("loaded_curve_filename", "Saved curve")
        st.success(
            f"{loaded_filename} · {CURVE_ALGORITHMS[loaded_model.algorithm]} · "
            f"parameter: {loaded_model.parameter_feature}"
        )
        if uploaded_digest is not None and uploaded_digest != loaded_digest:
            st.info(
                "A different file is selected but has not been loaded. The named model above "
                "is still active."
            )
        st.button(
            "Forget loaded curve",
            icon=":material/delete:",
            on_click=forget_loaded_curve,
            key="forget_loaded_curve_button",
        )

        current_coordinates = (x_component, y_component, z_component)
        compatible = True
        if set(loaded_model.coordinate_features) != set(current_coordinates):
            st.error(
                "Select the saved curve coordinates on the chart: "
                + ", ".join(loaded_model.coordinate_features)
                + "."
            )
            compatible = False
        saved_direction_method = loaded_provenance.get("direction_method", "PCA")
        if saved_direction_method != direction_method:
            st.error(
                f"This curve was fitted in {saved_direction_method} coordinates, but the "
                f"current projection uses {direction_method}."
            )
            compatible = False
        current_pca_digest = pca_projection_fingerprint(active_pca)
        saved_pca_digest = loaded_provenance.get("pca_sha256")
        saved_fingerprint_version = loaded_provenance.get("pca_fingerprint_version")
        if saved_pca_digest is None:
            st.warning(
                "This artifact does not identify its coordinate basis, so compatibility cannot "
                "be fully verified."
            )
        elif saved_fingerprint_version == PCA_PROJECTION_FINGERPRINT_VERSION:
            if saved_pca_digest != current_pca_digest:
                st.error(
                    "This curve was fitted in a different coordinate system. Use the matching "
                    "direction model."
                )
                compatible = False
        elif saved_fingerprint_version in (None, 1):
            if saved_pca_digest != pca_projection_fingerprint(active_pca, version=1):
                st.error(
                    "This curve was fitted in a different coordinate system. Use the matching "
                    "direction model."
                )
                compatible = False
        else:
            st.error(
                "This curve uses an unsupported coordinate fingerprint version: "
                f"{saved_fingerprint_version!r}."
            )
            compatible = False

        saved_layer = loaded_provenance.get("layer_component")
        if saved_layer is not None and saved_layer != layer_component:
            st.warning(f"This curve was saved for {saved_layer!r}, not {layer_component!r}.")
        saved_position = loaded_provenance.get("cached_position")
        if saved_position is not None and saved_position != cached_position:
            st.warning(
                f"This curve was saved for token position {saved_position!r}, not "
                f"{cached_position!r}."
            )
        if not compatible:
            return None, {}

        with st.expander("Saved model details"):
            st.write(
                {
                    "Algorithm": CURVE_ALGORITHMS[loaded_model.algorithm],
                    "Parameter": loaded_model.parameter_feature,
                    "Parameter bounds": loaded_model.training_parameter_bounds.tolist(),
                    "Coordinates": loaded_model.coordinate_features,
                    "Model parameters": loaded_model.parameters,
                    "Artifact version": loaded_provenance.get("artifact_version"),
                }
            )

        preview_scope = st.segmented_control(
            "Preview points",
            ["Visible", "All projected"],
            default="Visible",
            key="loaded_curve_scope",
            persist_state="page",
        )
        preview_source = plot_data if preview_scope == "Visible" else projection
        preview_columns = st.columns(2)
        sample_count = preview_columns[0].slider(
            "Curve samples",
            40,
            1_000,
            240,
            20,
            key="loaded_curve_samples",
            persist_state="page",
        )
        padding_percent = preview_columns[1].slider(
            "Curve padding",
            0,
            100,
            0,
            1,
            format="%d%%",
            key="loaded_curve_padding",
            help=(
                "At 0%, each curve end reaches the visible plot boundary. Increase this to "
                "extrapolate beyond that box."
            ),
            persist_state="page",
        )
        preview_submitted = st.button(
            "Apply loaded curve",
            type="primary",
            icon=":material/preview:",
            width="stretch",
            key="apply_loaded_curve_button",
        )
        saved_coordinates = list(loaded_model.coordinate_features)
        coordinates = preview_source[saved_coordinates].to_numpy(dtype=np.float64)
        finite_coordinates = np.isfinite(coordinates).all(axis=1)
        parameter_values = np.full(len(coordinates), np.nan, dtype=np.float64)
        if np.any(finite_coordinates):
            parameter_values[finite_coordinates] = project_onto_curve_parameter(
                loaded_model, coordinates[finite_coordinates], grid_size=10_001
            )
        display_coordinates = plot_data[saved_coordinates].to_numpy(dtype=np.float64)
        finite_display = display_coordinates[np.isfinite(display_coordinates).all(axis=1)]
        display_coordinate_bounds = np.column_stack(
            [finite_display.min(axis=0), finite_display.max(axis=0)]
        )
        data_digest = sha256(
            np.ascontiguousarray(coordinates).tobytes()
        ).hexdigest()
        preview_identity = (
            loaded_digest,
            current_pca_digest,
            data_digest,
            preview_scope,
            sample_count,
            padding_percent,
        )
        if preview_submitted or load_succeeded:
            try:
                with st.spinner("Evaluating the saved curve…"):
                    result = evaluate_curve_model(
                        loaded_model,
                        parameter_values,
                        coordinates,
                        sample_count=sample_count,
                        padding_fraction=padding_percent / 100.0,
                        display_coordinate_bounds=display_coordinate_bounds,
                    )
            except (ValueError, MemoryError) as exc:
                st.session_state.pop("loaded_curve_result", None)
                st.session_state.pop("loaded_curve_identity", None)
                st.error(f"Saved-curve preview failed: {exc}")
            else:
                st.session_state.loaded_curve_result = result
                st.session_state.loaded_curve_identity = preview_identity

        result = None
        if st.session_state.get("loaded_curve_identity") == preview_identity:
            result = st.session_state.get("loaded_curve_result")
        elif st.session_state.get("loaded_curve_result") is not None:
            st.info("The preview points or controls changed. Click **Apply loaded curve**.")
        if result is None:
            return None, {}
        metrics = result.metrics
        diagnostic_columns = st.columns(3)
        diagnostic_columns[0].metric(
            "Current 3D RMSE", _format_metric(metrics["current_rmse_3d"])
        )
        diagnostic_columns[1].metric("Current R²", _format_metric(metrics["current_r2"]))
        diagnostic_columns[2].metric(
            "Outside fit range", f"{int(metrics['extrapolation_points']):,}"
        )
        if result.warnings:
            st.warning(" ".join(result.warnings))
        appearance = curve_appearance_controls(result)
        return result, appearance


def loaded_extruded_surface_controls(
    *,
    plot_data: pd.DataFrame,
    projection: pd.DataFrame,
    x_component: str,
    y_component: str,
    z_component: str,
    active_pca: Any,
    direction_method: str,
    layer_component: str,
    cached_position: Any,
) -> tuple[RMSSplineSurfaceEvaluationResult | None, dict[str, Any]]:
    """Fit or load an RMS-conditioned extrusion around a saved cubic spline."""

    load_succeeded = False
    with st.container(border=True):
        st.subheader("Extruded spline surface")
        st.caption(
            "Fit **S(t, r) = C(t) + E(u(r))** from a saved cubic spline, using "
            "reconstruction residual RMS to determine each point's extrusion offset."
        )
        st.warning(
            "Only load model files you trust. Joblib and pickle artifacts can "
            "execute code when opened."
        )
        source_mode = st.segmented_control(
            "Extrusion source",
            ["Fit from cubic spline", "Load saved surface"],
            default="Fit from cubic spline",
            key="extruded_surface_source_mode",
            persist_state="page",
        )
        fitting_direction = source_mode == "Fit from cubic spline"
        model_upload = st.file_uploader(
            "Cubic spline model" if fitting_direction else "Saved extruded surface",
            type=["joblib"],
            key=("extrusion_curve_upload" if fitting_direction else "extruded_surface_model_upload"),
            help=(
                "Choose a degree-3 smoothing-spline model downloaded from the curve overlay."
                if fitting_direction
                else "Choose an extruded-surface artifact downloaded from this overlay."
            ),
        )
        uploaded_digest = None
        model_bytes = None
        if model_upload is not None:
            model_bytes = model_upload.getvalue()
            uploaded_digest = sha256(model_bytes).hexdigest()
        already_loaded = (
            uploaded_digest is not None
            and uploaded_digest == st.session_state.get("loaded_extruded_surface_digest")
            and (
                st.session_state.get("loaded_extrusion_curve_model") is not None
                if fitting_direction
                else st.session_state.get("loaded_extruded_surface_model") is not None
            )
        )
        if st.button(
            "Load cubic spline" if fitting_direction else "Load extruded surface",
            type="primary",
            icon=":material/upload_file:",
            width="stretch",
            disabled=model_bytes is None or already_loaded,
            key="load_extruded_surface_button",
        ):
            try:
                if fitting_direction:
                    loaded_model, loaded_provenance = load_curve_model(model_bytes)
                    if (
                        loaded_model.algorithm != "spline"
                        or int(loaded_model.parameters.get("degree", 0)) != 3
                    ):
                        raise ValueError(
                            "The extrusion input must be a degree-3 cubic spline curve model."
                        )
                else:
                    loaded_model, loaded_provenance = (
                        RMSSplineSurfaceTransformer.load_artifact(model_bytes)
                    )
            except (TypeError, ValueError) as exc:
                st.error(str(exc))
            else:
                st.session_state.loaded_extruded_surface_digest = uploaded_digest
                if fitting_direction:
                    st.session_state.loaded_extrusion_curve_model = loaded_model
                    st.session_state.pop("loaded_extruded_surface_model", None)
                else:
                    st.session_state.loaded_extruded_surface_model = loaded_model
                    st.session_state.pop("loaded_extrusion_curve_model", None)
                st.session_state.loaded_extruded_surface_provenance = loaded_provenance
                st.session_state.loaded_extruded_surface_filename = model_upload.name
                st.session_state.pop("loaded_extruded_surface_result", None)
                st.session_state.pop("loaded_extruded_surface_identity", None)
                load_succeeded = True

        loaded_curve_model: CurveModel | None = st.session_state.get(
            "loaded_extrusion_curve_model"
        ) if fitting_direction else None
        loaded_surface_model = (
            st.session_state.get("loaded_extruded_surface_model")
            if not fitting_direction
            else None
        )
        loaded_model = loaded_curve_model if fitting_direction else loaded_surface_model
        loaded_provenance = st.session_state.get("loaded_extruded_surface_provenance", {})
        loaded_digest = st.session_state.get("loaded_extruded_surface_digest")
        if loaded_model is None:
            if fitting_direction:
                st.info("Choose a saved cubic spline and click **Load cubic spline**.")
            else:
                st.info("Choose a saved extruded surface and click **Load extruded surface**.")
            return None, {}

        loaded_filename = st.session_state.get(
            "loaded_extruded_surface_filename",
            "Cubic spline" if fitting_direction else "Extruded surface",
        )
        st.success(
            f"{loaded_filename} · {loaded_model.parameter_feature} · "
            f"{', '.join(loaded_model.coordinate_features)}"
        )
        if uploaded_digest is not None and uploaded_digest != loaded_digest:
            st.info(
                "A different file is selected but has not been loaded. The named model above "
                "is still active."
            )
        st.button(
            "Forget loaded model",
            icon=":material/delete:",
            on_click=forget_loaded_extruded_surface,
            key="forget_extruded_surface_button",
        )

        compatible = True
        chart_components = (x_component, y_component, z_component)
        if set(chart_components) != set(loaded_model.coordinate_features):
            st.error(
                "Select the saved model coordinates on the chart: "
                f"{', '.join(loaded_model.coordinate_features)}."
            )
            compatible = False

        saved_direction_method = loaded_provenance.get("direction_method")
        if saved_direction_method is not None and saved_direction_method != direction_method:
            st.error(
                f"This model was created in {saved_direction_method} coordinates, but the "
                f"current projection uses {direction_method}."
            )
            compatible = False
        current_pca_digest = pca_projection_fingerprint(active_pca)
        saved_pca_digest = loaded_provenance.get("pca_sha256")
        saved_fingerprint_version = loaded_provenance.get("pca_fingerprint_version")
        if saved_pca_digest is None:
            st.warning(
                "This artifact does not identify its fitted coordinate basis. "
                "Coordinate labels match, but basis compatibility cannot be verified."
            )
        elif saved_fingerprint_version == PCA_PROJECTION_FINGERPRINT_VERSION:
            if saved_pca_digest != current_pca_digest:
                st.error(
                    "This model was created in a different coordinate basis. Use the "
                    "matching direction model."
                )
                compatible = False
        elif saved_fingerprint_version in (None, 1):
            if saved_pca_digest != pca_projection_fingerprint(active_pca, version=1):
                st.error(
                    "This model was created in a different coordinate basis. Use the "
                    "matching direction model."
                )
                compatible = False
        else:
            st.error(
                "This model uses an unsupported coordinate fingerprint version: "
                f"{saved_fingerprint_version!r}."
            )
            compatible = False

        residual_feature = "reconstruction_residual_rms"
        residual_available = residual_feature in projection
        if not residual_available:
            st.error(
                "Reconstruction residual RMS is unavailable. Recompute the current "
                "projection before fitting or applying this surface."
            )
            compatible = False
        elif not is_numeric_dtype(projection[residual_feature]) or is_bool_dtype(
            projection[residual_feature]
        ):
            st.error("Reconstruction residual RMS must be numeric.")
            compatible = False
        saved_layer = loaded_provenance.get("layer_component")
        if saved_layer is not None and saved_layer != layer_component:
            st.warning(
                f"This model was saved for {saved_layer!r}, not {layer_component!r}."
            )
        saved_position = loaded_provenance.get("cached_position")
        if saved_position is not None and saved_position != cached_position:
            st.warning(
                f"This model was saved for token position {saved_position!r}, not "
                f"{cached_position!r}."
            )
        if not compatible:
            return None, {}

        with st.expander("Saved model details"):
            model_details = {
                "Parameter": loaded_model.parameter_feature,
                "Parameter bounds": (
                    loaded_model.training_parameter_bounds.tolist()
                    if fitting_direction
                    else loaded_model.projection_parameter_bounds.tolist()
                ),
                "Coordinates": loaded_model.coordinate_features,
                "Artifact version": loaded_provenance.get("artifact_version"),
            }
            if fitting_direction:
                model_details.update(
                    {
                        "Spline degree": loaded_model.parameters.get("degree"),
                        "Spline smoothing": loaded_model.parameters.get("smoothing"),
                    }
                )
            else:
                model_details.update(
                    {
                        "Extrusion degree": loaded_model.degree,
                        "Residual feature": loaded_model.residual_feature,
                        "Residual reference": loaded_model.residual_reference,
                        "Residual fit bounds": loaded_model.residual_bounds.tolist(),
                        "Log-residual scale": loaded_model.log_residual_scale,
                        "Coefficient rows": {
                            feature: row.tolist()
                            for feature, row in zip(
                                loaded_model.coordinate_features,
                                loaded_model.coefficients,
                                strict=True,
                            )
                        },
                    }
                )
            st.write(model_details)

        preview_scope = st.segmented_control(
            "Fit points" if fitting_direction else "Preview points",
            ["Visible", "All projected"],
            default="Visible",
            key="extruded_surface_scope",
            help=(
                "The selected points fit the RMS-conditioned polynomial offset and define "
                "the preview diagnostics."
                if fitting_direction
                else "The selected points determine the preview extent and diagnostics."
            ),
            persist_state="page",
        )
        preview_source = plot_data if preview_scope == "Visible" else projection
        saved_coordinates = list(loaded_model.coordinate_features)
        point_xyz = preview_source[saved_coordinates].to_numpy(dtype=np.float64)
        point_residual = pd.to_numeric(
            preview_source[residual_feature], errors="coerce"
        ).to_numpy(dtype=np.float64)
        if fitting_direction:
            finite_point_xyz = np.isfinite(point_xyz).all(axis=1)
            point_parameter = np.full(len(point_xyz), np.nan, dtype=np.float64)
            if np.any(finite_point_xyz):
                point_parameter[finite_point_xyz] = project_onto_curve_parameter(
                    loaded_model, point_xyz[finite_point_xyz], grid_size=10_001
                )
        else:
            point_parameter = None
        source_values = (
            preview_source[SOURCE_FOLDER_FIELD].fillna("<missing>").astype(str)
            if SOURCE_FOLDER_FIELD in preview_source
            else pd.Series("<current>", index=preview_source.index)
        )
        numeric_fit_data = np.column_stack(
            [point_xyz, point_residual, *([] if point_parameter is None else [point_parameter])]
        )
        data_digest = sha256(np.ascontiguousarray(numeric_fit_data).tobytes()).hexdigest()
        source_options = sorted(source_values.unique().tolist())
        default_base_source = "abst" if "abst" in source_options else source_options[0]
        preferred_fit_sources = [
            source for source in ("conv", "nof_conv") if source in source_options
        ]
        if not preferred_fit_sources:
            preferred_fit_sources = [
                source for source in source_options if source != default_base_source
            ] or [default_base_source]

        with st.form("extruded_surface_preview_form"):
            if fitting_direction:
                fit_columns = st.columns(2)
                base_source = fit_columns[0].selectbox(
                    "Reference source",
                    source_options,
                    index=source_options.index(default_base_source),
                    key="extruded_surface_base_source",
                    help="Its lowest positive reconstruction RMS anchors the saved spline edge.",
                    persist_state="page",
                )
                fit_sources = fit_columns[1].multiselect(
                    "Fit sources",
                    source_options,
                    default=preferred_fit_sources,
                    key="extruded_surface_fit_sources",
                    help="Rows from these sources fit the RMS-to-offset polynomial.",
                    persist_state="page",
                )
                extrusion_degree_label = st.segmented_control(
                    "Extrusion degree",
                    ["Linear", "Quadratic", "Cubic"],
                    default="Cubic",
                    key="extruded_surface_degree",
                    help=(
                        "Fits E(u) as a no-intercept ridge polynomial; cubic matches the "
                        "shared RMS spline-surface transformer default."
                    ),
                    persist_state="page",
                )
                extrusion_degree = {
                    "Linear": 1,
                    "Quadratic": 2,
                    "Cubic": 3,
                }[extrusion_degree_label]
                ridge_alpha = st.number_input(
                    "Ridge regularization",
                    min_value=0.0,
                    value=1e-3,
                    format="%.3g",
                    key="extruded_surface_ridge_alpha",
                    help="L2 penalty applied to the no-intercept polynomial coefficients.",
                    persist_state="page",
                )
            else:
                base_source = None
                fit_sources = []
                extrusion_degree = loaded_model.degree
                ridge_alpha = loaded_model.ridge_alpha
                st.caption(f"Saved extrusion degree: {extrusion_degree}")
            sample_columns = st.columns(4)
            parameter_samples = sample_columns[0].slider(
                "Curve samples",
                20,
                500,
                120,
                20,
                key="extruded_surface_parameter_samples",
                persist_state="page",
            )
            extrusion_samples = sample_columns[1].slider(
                "Extrusion samples",
                2,
                120,
                30,
                2,
                key="extruded_surface_extrusion_samples",
                persist_state="page",
            )
            parameter_padding = sample_columns[2].slider(
                "Curve padding",
                0,
                100,
                0,
                1,
                format="%d%%",
                key="extruded_surface_parameter_padding",
                persist_state="page",
            )
            extrusion_padding = sample_columns[3].slider(
                "Width padding",
                0,
                200,
                10,
                5,
                format="%d%%",
                key="extruded_surface_width_padding",
                help="Extends the sheet in log-RMS space beyond the current points.",
                persist_state="page",
            )
            clip_residual = st.toggle(
                "Clip residual RMS to fitted range",
                value=True,
                key="extruded_surface_clip_residual",
                help=(
                    "Prevents polynomial extrapolation beyond the residual range used to "
                    "fit the model."
                ),
                persist_state="page",
            )
            extent_mode = st.segmented_control(
                "Residual extent",
                ["Current points", "Manual"],
                default="Current points",
                key="extruded_surface_extent_mode",
                help="Current points derives the positive RMS range from the selected rows.",
                persist_state="page",
            )
            manual_bounds = None
            if extent_mode == "Manual":
                positive_residual = point_residual[
                    np.isfinite(point_residual) & (point_residual > 0)
                ]
                if len(positive_residual):
                    default_residual_low = float(np.min(positive_residual))
                    default_residual_high = float(np.max(positive_residual))
                elif not fitting_direction:
                    default_residual_low, default_residual_high = map(
                        float, loaded_model.residual_bounds
                    )
                else:
                    default_residual_low, default_residual_high = 0.1, 1.0
                if np.isclose(default_residual_low, default_residual_high):
                    default_residual_low *= 0.9
                    default_residual_high *= 1.1
                bound_columns = st.columns(2)
                extrusion_minimum = bound_columns[0].number_input(
                    "Minimum residual RMS",
                    min_value=np.finfo(float).tiny,
                    value=default_residual_low,
                    key="extruded_surface_minimum_u",
                    persist_state="page",
                )
                extrusion_maximum = bound_columns[1].number_input(
                    "Maximum residual RMS",
                    min_value=np.finfo(float).tiny,
                    value=default_residual_high,
                    key="extruded_surface_maximum_u",
                    persist_state="page",
                )
                manual_bounds = (extrusion_minimum, extrusion_maximum)
            preview_submitted = st.form_submit_button(
                "Fit / update extrusion" if fitting_direction else "Apply loaded surface",
                type="primary",
                icon=":material/fit_screen:",
                width="stretch",
            )

        preview_identity = (
            source_mode,
            base_source,
            tuple(fit_sources),
            extrusion_degree,
            ridge_alpha,
            loaded_digest,
            current_pca_digest,
            data_digest,
            preview_scope,
            parameter_samples,
            extrusion_samples,
            parameter_padding,
            extrusion_padding,
            clip_residual,
            extent_mode,
            manual_bounds,
        )
        if preview_submitted or load_succeeded:
            try:
                spinner_text = (
                    "Fitting the RMS-conditioned spline surface and sampling it…"
                    if fitting_direction
                    else "Evaluating the loaded RMS spline surface…"
                )
                with st.spinner(spinner_text):
                    if fitting_direction:
                        if not fit_sources:
                            raise ValueError("Select at least one fit source.")
                        fit_frame = preview_source.copy()
                        fit_frame.loc[:, saved_coordinates] = point_xyz
                        fit_frame.loc[:, residual_feature] = point_residual
                        fit_frame.loc[:, loaded_model.parameter_feature] = point_parameter
                        fit_frame.loc[:, SOURCE_FOLDER_FIELD] = source_values.to_numpy()
                        finite_fit = (
                            np.isfinite(point_xyz).all(axis=1)
                            & np.isfinite(point_residual)
                            & (point_residual > 0)
                            & np.isfinite(point_parameter)
                        )
                        fit_frame = fit_frame.loc[finite_fit].copy()
                        surface_model = RMSSplineSurfaceTransformer.fit(
                            fit_frame,
                            {"model": loaded_model},
                            base_source=base_source,
                            fit_sources=fit_sources,
                            source_feature=SOURCE_FOLDER_FIELD,
                            residual_feature=residual_feature,
                            degree=extrusion_degree,
                            ridge_alpha=float(ridge_alpha),
                        )
                    else:
                        surface_model = loaded_model
                    result = evaluate_rms_spline_surface(
                        surface_model,
                        point_xyz,
                        point_residual,
                        true_parameter=point_parameter,
                        parameter_samples=parameter_samples,
                        residual_samples=extrusion_samples,
                        parameter_padding_fraction=parameter_padding / 100.0,
                        residual_padding_fraction=extrusion_padding / 100.0,
                        residual_bounds=manual_bounds,
                        clip=clip_residual,
                    )
            except (ValueError, MemoryError) as exc:
                st.session_state.pop("loaded_extruded_surface_result", None)
                st.session_state.pop("loaded_extruded_surface_identity", None)
                action = "fit" if fitting_direction else "preview"
                st.error(f"Extruded-surface {action} failed: {exc}")
            else:
                st.session_state.loaded_extruded_surface_model = surface_model
                st.session_state.loaded_extruded_surface_result = result
                st.session_state.loaded_extruded_surface_identity = preview_identity

        result = None
        if st.session_state.get("loaded_extruded_surface_identity") == preview_identity:
            result = st.session_state.get("loaded_extruded_surface_result")
        elif st.session_state.get("loaded_extruded_surface_result") is not None:
            action_label = (
                "Fit / update extrusion" if fitting_direction else "Apply loaded surface"
            )
            st.info(f"The points or controls changed. Click **{action_label}**.")
        if result is None:
            return None, {}

        st.selectbox(
            "Point projection method",
            ["Nearest point on the curve (Euclidean distance)"],
            key="extruded_surface_projection_method",
            help=(
                "Dewarp each point by its RMS-derived extrusion offset, then choose the "
                "spline parameter whose curve point has the smallest Euclidean distance."
            ),
            persist_state="page",
        )

        metrics = result.metrics
        diagnostic_columns = st.columns(3)
        diagnostic_columns[0].metric(
            "Current surface RMSE", _format_metric(metrics["current_rmse_3d"])
        )
        diagnostic_columns[1].metric(
            "Parameter RMSE", _format_metric(metrics.get("parameter_rmse"))
        )
        diagnostic_columns[2].metric(
            "Clipped RMS points", f"{int(metrics['clipped_residual_points']):,}"
        )
        coefficient_text = {
            feature: [float(value) for value in row]
            for feature, row in zip(
                result.model.coordinate_features,
                result.model.coefficients,
                strict=True,
            )
        }
        st.caption(
            f"Degree {result.model.degree} · coefficients by coordinate: {coefficient_text} · "
            f"t ∈ [{metrics['display_parameter_bounds'][0]:.4g}, "
            f"{metrics['display_parameter_bounds'][1]:.4g}] · "
            f"RMS ∈ [{metrics['display_residual_bounds'][0]:.4g}, "
            f"{metrics['display_residual_bounds'][1]:.4g}]"
        )
        if result.warnings:
            st.warning(" ".join(result.warnings))
        artifact_metadata = {
            key: value
            for key, value in loaded_provenance.items()
            if key not in {"artifact_kind", "artifact_version"}
        }
        artifact_metadata.update(
            {
                "direction_method": direction_method,
                "pca_sha256": current_pca_digest,
                "pca_fingerprint_version": PCA_PROJECTION_FINGERPRINT_VERSION,
                "layer_component": layer_component,
                "cached_position": cached_position,
                "extrusion_degree": result.model.degree,
                "base_source": (
                    base_source if fitting_direction else artifact_metadata.get("base_source")
                ),
                "fit_sources": (
                    list(fit_sources)
                    if fitting_direction
                    else artifact_metadata.get("fit_sources")
                ),
                "fit_scope": (
                    preview_scope
                    if fitting_direction
                    else artifact_metadata.get("fit_scope")
                ),
                "fit_metrics": {
                    key: metrics[key]
                    for key in (
                        "current_rmse_3d",
                        "current_mae_3d",
                        "current_max_error_3d",
                        "parameter_rmse",
                        "parameter_mae",
                        "clipped_residual_points",
                    )
                    if key in metrics
                },
            }
        )
        safe_coordinates = "-".join(result.model.coordinate_features).replace("/", "-")
        st.download_button(
            "Download extruded surface",
            data=lambda: serialize_rms_spline_surface(
                result.model, metadata=artifact_metadata
            ),
            file_name=f"activation_rms_spline_surface_{safe_coordinates}.joblib",
            mime="application/octet-stream",
            icon=":material/download:",
            on_click="ignore",
            help=(
                "Download the cubic spline, polynomial extrusion coefficients, parameter "
                "bounds, coordinate labels, and projection provenance."
            ),
        )
        appearance = extruded_surface_appearance_controls(result)
        return result, appearance


def loaded_surface_controls(
    *,
    plot_data: pd.DataFrame,
    projection: pd.DataFrame,
    x_component: str,
    y_component: str,
    z_component: str,
    active_pca: Any,
    direction_method: str,
    layer_component: str,
    cached_position: Any,
) -> tuple[SurfaceEvaluationResult | None, dict[str, Any]]:
    """Load a trusted surface artifact and build a preview in the current projection space."""

    load_succeeded = False
    with st.container(border=True):
        st.subheader("Saved surface")
        st.warning(
            "Only load surface files you trust. Joblib and pickle artifacts can execute "
            "code when opened."
        )
        model_upload = st.file_uploader(
            "Saved surface model",
            type=["joblib"],
            key="surface_model_upload",
            help="Choose a surface artifact downloaded from Activation Atlas.",
        )
        uploaded_digest = None
        model_bytes = None
        if model_upload is not None:
            model_bytes = model_upload.getvalue()
            uploaded_digest = sha256(model_bytes).hexdigest()

        already_loaded = (
            uploaded_digest is not None
            and uploaded_digest == st.session_state.get("loaded_surface_digest")
            and st.session_state.get("loaded_surface_model") is not None
        )
        if st.button(
            "Load uploaded surface",
            type="primary",
            icon=":material/upload_file:",
            width="stretch",
            disabled=model_bytes is None or already_loaded,
            key="load_surface_model_button",
        ):
            try:
                loaded_model, loaded_provenance = load_surface_model(model_bytes)
            except ValueError as exc:
                st.error(str(exc))
            else:
                # Commit only after the entire artifact has loaded and validated successfully.
                st.session_state.loaded_surface_digest = uploaded_digest
                st.session_state.loaded_surface_model = loaded_model
                st.session_state.loaded_surface_provenance = loaded_provenance
                st.session_state.loaded_surface_filename = model_upload.name
                st.session_state.pop("loaded_surface_result", None)
                st.session_state.pop("loaded_surface_identity", None)
                load_succeeded = True

        loaded_model = st.session_state.get("loaded_surface_model")
        loaded_provenance = st.session_state.get("loaded_surface_provenance", {})
        loaded_digest = st.session_state.get("loaded_surface_digest")
        if loaded_model is None:
            st.info("Choose a saved surface and click **Load uploaded surface**.")
            return None, {}

        loaded_filename = st.session_state.get("loaded_surface_filename", "Saved surface")
        st.success(
            f"{loaded_filename} · {SURFACE_ALGORITHMS[loaded_model.algorithm]} · "
            f"{loaded_model.output_feature} = "
            f"f({loaded_model.input_features[0]}, {loaded_model.input_features[1]})"
        )
        if uploaded_digest is not None and uploaded_digest != loaded_digest:
            st.info(
                "A different file is selected but has not been loaded. The named model above "
                "is still active."
            )
        st.button(
            "Forget loaded surface",
            icon=":material/delete:",
            on_click=forget_loaded_surface,
            key="forget_loaded_surface_button",
        )

        compatible = True
        if (
            set(loaded_model.input_features) != {x_component, y_component}
            or len({x_component, y_component}) != 2
        ):
            st.error(
                "Select the saved input components on the chart: "
                f"X/Y must be {loaded_model.input_features[0]} and "
                f"{loaded_model.input_features[1]} (either order)."
            )
            compatible = False
        if loaded_model.output_feature != z_component:
            st.error(
                f"Select {loaded_model.output_feature} as the chart Z component to use this "
                "surface."
            )
            compatible = False

        saved_direction_method = loaded_provenance.get("direction_method", "PCA")
        if saved_direction_method != direction_method:
            st.error(
                f"This surface was fitted in {saved_direction_method} coordinates, but the "
                f"current projection uses {direction_method}."
            )
            compatible = False
        current_pca_digest = pca_projection_fingerprint(active_pca)
        saved_pca_digest = loaded_provenance.get("pca_sha256")
        saved_pca_fingerprint_version = loaded_provenance.get("pca_fingerprint_version")
        if saved_pca_digest is None:
            st.warning(
                "This artifact does not identify its coordinate basis. Component labels match, but "
                "coordinate compatibility cannot be verified."
            )
        elif saved_pca_fingerprint_version in (None, 1):
            legacy_pca_digest = pca_projection_fingerprint(active_pca, version=1)
            if saved_pca_digest != legacy_pca_digest:
                st.error(
                    "This surface was fitted in a different coordinate system. Use the matching "
                    "direction fit before applying the surface."
                )
                compatible = False
            elif bool(getattr(active_pca, "whiten", False)):
                st.warning(
                    "This legacy surface fingerprint predates full verification of whitened "
                    "PCA coordinates. The components and mean match, but the whitening scale "
                    "cannot be verified."
                )
        elif saved_pca_fingerprint_version == PCA_PROJECTION_FINGERPRINT_VERSION:
            if saved_pca_digest != current_pca_digest:
                st.error(
                    "This surface was fitted in a different coordinate system. Use the matching "
                    "direction fit before applying the surface."
                )
                compatible = False
        else:
            st.error(
                "This surface uses an unsupported coordinate fingerprint version: "
                f"{saved_pca_fingerprint_version!r}."
            )
            compatible = False

        saved_layer = loaded_provenance.get("layer_component")
        if saved_layer is not None and saved_layer != layer_component:
            st.warning(f"This surface was saved for {saved_layer!r}, not {layer_component!r}.")
        saved_position = loaded_provenance.get("cached_position")
        if saved_position is not None and saved_position != cached_position:
            st.warning(
                f"This surface was saved for token position {saved_position!r}, not "
                f"{cached_position!r}."
            )
        saved_sklearn_version = loaded_provenance.get("sklearn_version")
        if saved_sklearn_version and saved_sklearn_version != sklearn_version:
            st.warning(
                f"This surface was saved with scikit-learn {saved_sklearn_version}; "
                f"the app is running {sklearn_version}."
            )

        with st.expander("Saved model details"):
            bounds = {
                feature: {
                    "minimum": float(loaded_model.training_bounds[index, 0]),
                    "maximum": float(loaded_model.training_bounds[index, 1]),
                }
                for index, feature in enumerate(loaded_model.input_features)
            }
            st.write(
                {
                    "Algorithm": SURFACE_ALGORITHMS[loaded_model.algorithm],
                    "Input bounds": bounds,
                    "Training output range": loaded_model.training_output_range,
                    "Artifact version": loaded_provenance.get("artifact_version"),
                }
            )
            if loaded_model.equation is not None:
                st.markdown("**Exact fitted equation**")
                st.code(loaded_model.equation, language="text")

        if not compatible:
            return None, {}

        preview_scope = st.segmented_control(
            "Preview points",
            ["Visible", "All projected"],
            default="Visible",
            key="loaded_surface_scope",
            help="The selected points define the preview extent and current-data diagnostics.",
            persist_state="page",
        )
        surface_source = plot_data if preview_scope == "Visible" else projection
        surface_xyz = surface_source[[x_component, y_component, z_component]].to_numpy(
            dtype=np.float64
        )
        point_count = len(surface_xyz)
        st.caption(
            f"The preview will evaluate {point_count:,} current point(s). The saved model "
            "itself remains unchanged."
        )
        if point_count == 0:
            st.warning("At least one current point is required to preview the surface.")
            return None, {}

        with st.form("loaded_surface_preview_form"):
            st.markdown("**Preview grid**")
            preview_columns = st.columns(4)
            grid_resolution = preview_columns[0].slider(
                "Grid resolution",
                15,
                160,
                60,
                5,
                key="loaded_surface_resolution",
                help="Grid cost grows with the square of this value.",
                persist_state="page",
            )
            padding_percent = preview_columns[1].slider(
                "Grid padding",
                0,
                50,
                25,
                1,
                format="%d%%",
                key="loaded_surface_padding",
                help="Extends the grid beyond the current coordinate range for extrapolation.",
                persist_state="page",
            )
            domain_label = preview_columns[2].selectbox(
                "Displayed domain",
                ["Full rectangle", "Convex hull", "Dense support"],
                key="loaded_surface_domain",
                help="Full rectangle keeps extrapolated regions visible.",
                persist_state="page",
            )
            density_factor = preview_columns[3].slider(
                "Gap tolerance",
                1.0,
                8.0,
                3.0,
                0.25,
                key="loaded_surface_density",
                help="Used only for the Dense support domain.",
                persist_state="page",
            )
            clip_label = st.selectbox(
                "Display clipping",
                ["None", "Current Z range", "Robust current 1–99% Z range"],
                key="loaded_surface_clip",
                help="Clipping affects only this chart preview, not saved-model predictions.",
                persist_state="page",
            )
            preview_submitted = st.form_submit_button(
                "Apply loaded surface",
                type="primary",
                icon=":material/preview:",
                width="stretch",
            )

        domain_map = {
            "Full rectangle": "full_grid",
            "Convex hull": "convex_hull",
            "Dense support": "dense_support",
        }
        clip_map = {
            "None": "none",
            "Current Z range": "observed_range",
            "Robust current 1–99% Z range": "robust_range",
        }
        surface_data_digest = sha256(np.ascontiguousarray(surface_xyz).tobytes()).hexdigest()
        preview_identity = (
            loaded_digest,
            current_pca_digest,
            surface_data_digest,
            x_component,
            y_component,
            z_component,
            preview_scope,
            grid_resolution,
            padding_percent,
            domain_label,
            density_factor,
            clip_label,
        )
        if preview_submitted or load_succeeded:
            try:
                with st.spinner("Evaluating the saved surface…"):
                    loaded_result = evaluate_surface_model(
                        loaded_model,
                        surface_xyz[:, 0],
                        surface_xyz[:, 1],
                        surface_xyz[:, 2],
                        input_features=(x_component, y_component),
                        output_feature=z_component,
                        grid_resolution=grid_resolution,
                        padding_fraction=padding_percent / 100.0,
                        domain_mask=domain_map[domain_label],
                        density_factor=density_factor,
                        clip_mode=clip_map[clip_label],
                    )
            except ValueError as exc:
                st.session_state.pop("loaded_surface_result", None)
                st.session_state.pop("loaded_surface_identity", None)
                st.error(f"Saved-surface preview failed: {exc}")
            except MemoryError:
                st.session_state.pop("loaded_surface_result", None)
                st.session_state.pop("loaded_surface_identity", None)
                st.error(
                    "The saved-surface preview ran out of memory. Lower the grid resolution "
                    "and try again."
                )
            except Exception as exc:  # noqa: BLE001 - keep third-party predictor errors in the UI
                st.session_state.pop("loaded_surface_result", None)
                st.session_state.pop("loaded_surface_identity", None)
                st.error(f"Saved-surface preview failed: {exc}")
            else:
                st.session_state.loaded_surface_result = loaded_result
                st.session_state.loaded_surface_identity = preview_identity

        surface_result = None
        if st.session_state.get("loaded_surface_identity") == preview_identity:
            surface_result = st.session_state.get("loaded_surface_result")
        elif st.session_state.get("loaded_surface_result") is not None:
            st.info(
                "The preview points, axes, or controls changed. Click **Apply loaded surface** "
                "to refresh the overlay."
            )
        else:
            st.info("Apply the loaded model to build the chart preview.")

        if surface_result is None:
            return None, {}

        metrics = surface_result.metrics
        metric_columns = st.columns(4)
        metric_columns[0].metric("Current-data RMSE", _format_metric(metrics["evaluation_rmse"]))
        metric_columns[1].metric("Current-data R²", _format_metric(metrics["evaluation_r2"]))
        metric_columns[2].metric(
            "Outside saved range",
            _format_metric(metrics["point_extrapolation_fraction"], percent=True),
            help="Current points outside at least one saved training-coordinate interval.",
        )
        metric_columns[3].metric("Preview compute", f"{float(metrics['total_seconds']):.2f} s")
        st.caption(
            f"{_format_metric(metrics['evaluation_coverage'], percent=True)} current-point "
            f"coverage · {_format_metric(metrics['grid_coverage'], percent=True)} grid coverage · "
            f"{_format_metric(metrics['grid_extrapolation_fraction'], percent=True)} of the "
            "preview grid outside saved bounds"
        )
        if surface_result.warnings:
            st.warning(" ".join(surface_result.warnings))

        appearance = surface_appearance_controls(surface_result)
        with st.expander("Current-data diagnostics"):
            st.dataframe(
                pd.DataFrame(
                    [
                        {
                            "points": metrics["evaluation_total_count"],
                            "predicted points": metrics["evaluation_predicted_count"],
                            "coverage": metrics["evaluation_coverage"],
                            "RMSE": metrics["evaluation_rmse"],
                            "normalized RMSE": metrics["evaluation_nrmse"],
                            "MAE": metrics["evaluation_mae"],
                            "R²": metrics["evaluation_r2"],
                        }
                    ]
                ),
                hide_index=True,
                width="stretch",
                column_config={
                    "coverage": st.column_config.NumberColumn(format="percent"),
                    "RMSE": st.column_config.NumberColumn(format="%.5g"),
                    "normalized RMSE": st.column_config.NumberColumn(format="%.5g"),
                    "MAE": st.column_config.NumberColumn(format="%.5g"),
                    "R²": st.column_config.NumberColumn(format="%.4f"),
                },
            )
            st.write(
                {
                    "Current input points": metrics["input_points"],
                    "Finite current points": metrics["finite_points"],
                    "Unique current X/Y": metrics["unique_points"],
                    "Non-finite rows dropped": metrics["dropped_nonfinite"],
                    "Grid coverage": metrics["grid_coverage"],
                    "Unclipped overshoot fraction": metrics["overshoot_fraction"],
                    "Grid prediction seconds": metrics["prediction_seconds"],
                }
            )
        return surface_result, appearance


with st.sidebar:
    st.header("1 · Select data")
    source_mode = st.segmented_control(
        "Source",
        ["Local folders", "Upload folders"],
        default="Local folders",
        key="activation_source_mode",
    )
    if source_mode == "Upload folders":
        st.warning(
            "Folder uploads keep every file in browser/server memory. For multi-GB "
            "datasets, use **Local folders** instead."
        )
        uploads = st.file_uploader(
            "Activation folders",
            type=["pt"],
            accept_multiple_files="directory",
            key="activation_folder_uploads",
            help=(
                "Choose one folder, then use + to add more. Files with the same name "
                "in different folders are kept separately."
            ),
        )
        selected_uploads = select_activation_batch_uploads(uploads or [])
        if uploads:
            st.caption(f"{len(selected_uploads):,} activation batches selected.")
        if st.button("Load selected folders", type="primary", width="stretch"):
            sources = selected_uploads
            if not sources:
                st.error("No activations_batch_*.pt files were selected.")
            else:
                use_sources(
                    sources,
                    label=f"Uploaded folders · {len(sources):,} batches",
                    is_local=False,
                )
    else:
        folder_text = st.text_area(
            "Folder paths",
            placeholder="C:\\data\\run_a\nD:\\data\\run_b",
            help=(
                "Enter one folder per line. Same-named batch files in different folders "
                "are loaded independently."
            ),
            key="activation_folder_paths",
        )
        if st.button("Load local folders", type="primary", width="stretch"):
            try:
                folders = [line.strip() for line in folder_text.splitlines() if line.strip()]
                sources, resolved_folders = discover_activation_batch_paths(folders)
                if not sources:
                    raise ValueError("No activations_batch_*.pt files were found in those folders.")
                folder_label = "folder" if len(resolved_folders) == 1 else "folders"
                use_sources(
                    sources,
                    label=(
                        f"{len(resolved_folders):,} local {folder_label} · {len(sources):,} batches"
                    ),
                    is_local=True,
                )
            except ValueError as exc:
                st.error(str(exc))

sources = st.session_state.get("sources")
if not sources:
    st.info("Select one or more folders containing activation batches to begin.", icon="↖")
    st.stop()

# Migrate browser sessions created before source metadata was initialized centrally.
if not st.session_state.get("source_label"):
    sources_are_local = all(isinstance(source, (str, Path)) for source in sources)
    st.session_state["source_is_local"] = sources_are_local
    st.session_state["source_label"] = (
        str(Path(sources[0]).parent)
        if sources_are_local
        else f"Uploaded folders · {len(sources):,} batches"
    )
source_label = st.session_state.get("source_label", "Selected activation batches")

try:
    if "inspection" not in st.session_state:
        with st.spinner("Reading batch structure…"):
            st.session_state.inspection = inspect_sources(sources)
    inspection = st.session_state.inspection
except Exception as exc:  # noqa: BLE001 - surface invalid local batch errors in the UI
    st.error(f"Could not read the selected batches: {exc}")
    st.stop()

with st.sidebar:
    st.caption(source_label)
    st.success(f"{inspection['batch_count']:,} batches · {inspection['row_count']:,} source rows")
    st.divider()
    st.header("2 · Prepare analysis")
    component = TARGET_LAYER_COMPONENT
    position_index = CACHED_POSITION_INDEX
    st.caption("Fixed extraction slice")
    st.write(f"`{component}` · final prompt token (`{PROMPT_TOKEN_POSITION}`)")
    candidate_fields = inspection["metadata_fields"]
    metadata_index = inspection["metadata_index"]
    multiple_source_folders = (
        SOURCE_FOLDER_FIELD in metadata_index
        and metadata_index[SOURCE_FOLDER_FIELD].nunique(dropna=False) > 1
    )
    default_filter = []
    if "template_metadata.prompt_framing" in candidate_fields:
        default_filter.append("template_metadata.prompt_framing")
    if multiple_source_folders:
        default_filter.append(SOURCE_FOLDER_FIELD)
    filter_fields = st.multiselect("Filter fields", candidate_fields, default=default_filter)
    filter_options = {
        field: sorted(metadata_index[field].dropna().unique().tolist(), key=str)
        for field in filter_fields
    }
    filters = {}
    for field in filter_fields:
        values = filter_options[field]
        default_values = (
            ["task_available_time"]
            if field == "template_metadata.prompt_framing" and "task_available_time" in values
            else values
        )
        filters[field] = st.multiselect(
            f"Keep · {field}", values, default=default_values, key=f"filter::{field}"
        )
    aggregation_candidates = [*candidate_fields]
    if "time_horizon_months" not in aggregation_candidates:
        aggregation_candidates.append("time_horizon_months")
    default_aggregation_fields = ["time_horizon_months"]
    if multiple_source_folders:
        default_aggregation_fields.append(SOURCE_FOLDER_FIELD)
    aggregation_fields = st.multiselect(
        "Aggregate by",
        aggregation_candidates,
        default=default_aggregation_fields,
        help="Rows in each group are averaged before fitting the selected directions.",
    )
    max_samples_enabled = st.toggle("Limit source samples", value=False)
    max_samples = (
        int(st.number_input("Maximum samples", min_value=1, value=1000, step=100))
        if max_samples_enabled
        else None
    )
    direction_method = st.segmented_control(
        "Direction method",
        ["PCA", "PLS"],
        default="PCA",
        key="direction_method",
        help=(
            "PCA finds unsupervised variance directions. PLS finds directions supervised "
            "by log10(time_horizon_months)."
        ),
    )
    loaded_pca_model = None
    loaded_pca_provenance: dict = {}
    loaded_pls_model = None
    loaded_pls_provenance: dict = {}
    if direction_method == "PLS":
        pca_mode = "Fit new"
        pls_mode = st.segmented_control(
            "PLS model", ["Fit new", "Use saved"], default="Fit new", key="pls_mode"
        )
        if pls_mode == "Fit new":
            with st.form("pls_fit_configuration"):
                n_components = int(
                    st.number_input(
                        "PLS components",
                        min_value=2,
                        max_value=50,
                        value=3,
                        key="pls_components",
                        help="Number of supervised latent directions to fit.",
                        persist_state="page",
                    )
                )
                with st.expander("Advanced configuration"):
                    pls_scale = st.toggle(
                        "Scale activations and target",
                        value=True,
                        key="pls_scale",
                        help="Standardize X and log-time-horizon y before fitting.",
                        persist_state="page",
                    )
                    pls_max_iter = int(
                        st.number_input(
                            "Maximum iterations",
                            min_value=1,
                            max_value=100_000,
                            value=500,
                            step=100,
                            key="pls_max_iter",
                            help="Maximum NIPALS power-method iterations per component.",
                            persist_state="page",
                        )
                    )
                    pls_tolerance = float(
                        st.number_input(
                            "Convergence tolerance",
                            min_value=1e-12,
                            max_value=1e-1,
                            value=1e-6,
                            step=1e-6,
                            format="%.1e",
                            key="pls_tolerance",
                            help=(
                                "Stop when the squared change in the left singular vector "
                                "is smaller."
                            ),
                            persist_state="page",
                        )
                    )
                st.form_submit_button(
                    "Apply PLS configuration",
                    icon=":material/tune:",
                    width="stretch",
                )
            pca_identity = (
                "pls",
                "fit",
                n_components,
                pls_scale,
                pls_max_iter,
                pls_tolerance,
                "log10_time_horizon_months",
            )
            st.caption(
                "Target: `log10(time_horizon_months)` (fixed). PLS is refitted when the "
                "prepared data or any configuration value changes."
            )
        else:
            st.warning(
                "Only load PLS files you trust. Joblib and pickle files can execute code when "
                "opened."
            )
            model_upload = st.file_uploader(
                "Saved PLS model",
                type=["joblib"],
                key="pls_model_upload",
                help="Accepts PLS artifacts downloaded from Activation Atlas.",
            )
            uploaded_digest = None
            if model_upload is not None:
                model_bytes = model_upload.getvalue()
                uploaded_digest = sha256(model_bytes).hexdigest()
                if (
                    st.session_state.get("loaded_pls_digest") != uploaded_digest
                    or "loaded_pls_model" not in st.session_state
                ):
                    if st.button(
                        "Load uploaded PLS",
                        type="primary",
                        icon=":material/upload_file:",
                        width="stretch",
                    ):
                        try:
                            loaded_pls_model, loaded_pls_provenance = load_pls_model(model_bytes)
                        except ValueError as exc:
                            st.error(str(exc))
                            st.stop()
                        st.session_state.loaded_pls_digest = uploaded_digest
                        st.session_state.loaded_pls_model = loaded_pls_model
                        st.session_state.loaded_pls_provenance = loaded_pls_provenance
                        st.session_state.loaded_pls_filename = model_upload.name
                else:
                    loaded_pls_model = st.session_state.loaded_pls_model
                    loaded_pls_provenance = st.session_state.get("loaded_pls_provenance", {})
            elif "loaded_pls_model" in st.session_state:
                uploaded_digest = st.session_state.get("loaded_pls_digest")
                loaded_pls_model = st.session_state.loaded_pls_model
                loaded_pls_provenance = st.session_state.get("loaded_pls_provenance", {})

            if loaded_pls_model is None:
                st.info("Choose a saved model and click **Load uploaded PLS** to continue.")
                st.stop()
            n_components = int(loaded_pls_model.components_.shape[0])
            if n_components < 2:
                st.error("The loaded PLS needs at least two components for this explorer.")
                st.stop()
            pca_identity = ("pls", "loaded", uploaded_digest)
            st.success(
                f"{st.session_state.get('loaded_pls_filename', 'Saved PLS')} · "
                f"{n_components:,} components"
            )
            st.caption(
                f"Scale: {'on' if loaded_pls_model.scale else 'off'} · "
                f"Maximum iterations: {loaded_pls_model.max_iter:,} · "
                f"Tolerance: {loaded_pls_model.tol:.1e}"
            )
            st.button(
                "Forget loaded PLS",
                icon=":material/delete:",
                on_click=forget_loaded_pls,
                width="stretch",
            )
            saved_component = loaded_pls_provenance.get("layer_component")
            if saved_component is not None and saved_component != component:
                st.warning(f"This model was saved for {saved_component!r}, not {component!r}.")
            saved_position = loaded_pls_provenance.get("cached_position")
            current_position = inspection["positions"][position_index]
            if saved_position is not None and saved_position != current_position:
                st.warning(
                    f"This model was saved for token position {saved_position!r}, "
                    f"not {current_position!r}."
                )
            saved_sklearn_version = loaded_pls_provenance.get("sklearn_version")
            if saved_sklearn_version and saved_sklearn_version != sklearn_version:
                st.warning(
                    f"This model was saved with scikit-learn {saved_sklearn_version}; "
                    f"the app is running {sklearn_version}."
                )
            st.caption(
                "Preparation changes retransform the selected points without refitting the model."
            )
    else:
        pls_mode = "Fit new"
        pca_mode = st.segmented_control(
            "PCA model", ["Fit new", "Use saved"], default="Fit new", key="pca_mode"
        )
        if pca_mode == "Fit new":
            n_components = int(
                st.number_input("PCA components", min_value=2, max_value=50, value=3)
            )
            pca_identity = ("fit", n_components)
            st.caption(
                "Filter and aggregation changes rebuild the prepared data; component-count "
                "changes refit PCA only."
            )
        else:
            st.warning(
                "Only load PCA files you trust. Joblib and pickle files can execute code when "
                "opened."
            )
            model_upload = st.file_uploader(
                "Saved PCA model",
                type=["joblib", "pkl", "pickle"],
                key="pca_model_upload",
                help="Accepts Activation Atlas artifacts and raw fitted PCA/IncrementalPCA files.",
            )
            uploaded_digest = None
            if model_upload is not None:
                model_bytes = model_upload.getvalue()
                uploaded_digest = sha256(model_bytes).hexdigest()
                if (
                    st.session_state.get("loaded_pca_digest") != uploaded_digest
                    or "loaded_pca_model" not in st.session_state
                ):
                    if st.button(
                        "Load uploaded PCA",
                        type="primary",
                        icon=":material/upload_file:",
                        width="stretch",
                    ):
                        try:
                            loaded_pca_model, loaded_pca_provenance = load_pca_model(model_bytes)
                        except ValueError as exc:
                            st.error(str(exc))
                            st.stop()
                        st.session_state.loaded_pca_digest = uploaded_digest
                        st.session_state.loaded_pca_model = loaded_pca_model
                        st.session_state.loaded_pca_provenance = loaded_pca_provenance
                        st.session_state.loaded_pca_filename = model_upload.name
                else:
                    loaded_pca_model = st.session_state.loaded_pca_model
                    loaded_pca_provenance = st.session_state.get("loaded_pca_provenance", {})
            elif "loaded_pca_model" in st.session_state:
                uploaded_digest = st.session_state.get("loaded_pca_digest")
                loaded_pca_model = st.session_state.loaded_pca_model
                loaded_pca_provenance = st.session_state.get("loaded_pca_provenance", {})

            if loaded_pca_model is None:
                st.info("Choose a saved model and click **Load uploaded PCA** to continue.")
                st.stop()
            n_components = int(loaded_pca_model.components_.shape[0])
            if n_components < 2:
                st.error("The loaded PCA needs at least two components for this explorer.")
                st.stop()
            pca_identity = ("loaded", uploaded_digest)
            st.success(
                f"{st.session_state.get('loaded_pca_filename', 'Saved PCA')} · "
                f"{n_components:,} components"
            )
            st.button(
                "Forget loaded PCA",
                icon=":material/delete:",
                on_click=forget_loaded_pca,
                width="stretch",
            )
            saved_component = loaded_pca_provenance.get("layer_component")
            if saved_component is not None and saved_component != component:
                st.warning(f"This model was saved for {saved_component!r}, not {component!r}.")
            saved_position = loaded_pca_provenance.get("cached_position")
            current_position = inspection["positions"][position_index]
            if saved_position is not None and saved_position != current_position:
                st.warning(
                    f"This model was saved for token position {saved_position!r}, "
                    f"not {current_position!r}."
                )
            saved_sklearn_version = loaded_pca_provenance.get("sklearn_version")
            if saved_sklearn_version and saved_sklearn_version != sklearn_version:
                st.warning(
                    f"This model was saved with scikit-learn {saved_sklearn_version}; "
                    f"the app is running {sklearn_version}."
                )
            st.caption(
                "Preparation changes retransform the selected points without refitting the model."
            )

slice_key = (component, position_index, st.session_state.get("source_revision", 0))
if st.session_state.get("slice_key") != slice_key:
    try:
        progress_bar = st.progress(0, text="Preparing activation slice…")

        def update_extraction_progress(done: int, total: int) -> None:
            progress_bar.progress(
                done / total,
                text=f"Extracting selected activations — batch {done:,} of {total:,}",
            )

        activation_matrix, cache_path = extract_activation_slice(
            sources,
            layer_component=component,
            position_index=position_index,
            source_row_counts=inspection["source_row_counts"],
            cache_dir=(Path("data") / "activation_explorer_cache")
            if st.session_state.get("source_is_local")
            else None,
            progress=update_extraction_progress,
        )
        progress_bar.empty()
        st.session_state.activation_matrix = activation_matrix
        st.session_state.activation_cache_path = cache_path
        st.session_state.slice_key = slice_key
        st.session_state.pop("prepared_key", None)
        st.session_state.pop("pca_key", None)
        clear_surface_fits()
    except Exception as exc:  # noqa: BLE001 - surface local extraction errors in the UI
        st.error(f"Activation slice could not be prepared: {exc}")
        st.stop()

activation_matrix = st.session_state.activation_matrix
if st.session_state.get("activation_cache_path") is not None:
    st.caption(
        "Using a disk-backed activation slice; source batches stay closed during direction work."
    )
if loaded_pca_model is not None and int(loaded_pca_model.components_.shape[1]) != int(
    activation_matrix.shape[1]
):
    st.error(
        f"The loaded PCA expects {loaded_pca_model.components_.shape[1]:,} activation "
        f"features, but {component!r} has {activation_matrix.shape[1]:,}."
    )
    st.stop()
if loaded_pls_model is not None and int(loaded_pls_model.components_.shape[1]) != int(
    activation_matrix.shape[1]
):
    st.error(
        f"The loaded PLS expects {loaded_pls_model.components_.shape[1]:,} activation "
        f"features, but {component!r} has {activation_matrix.shape[1]:,}."
    )
    st.stop()

prepared_key = (
    tuple((field, tuple(filters[field])) for field in filter_fields),
    tuple(aggregation_fields),
    max_samples,
)
if st.session_state.get("prepared_key") != prepared_key:
    try:
        for key in (
            "prepared_matrix",
            "prepared_row_offsets",
            "prepared_metadata",
            "prepared_details",
            "projection",
            "pca",
            "details",
        ):
            st.session_state.pop(key, None)
        gc.collect()
        with st.spinner("Filtering and aggregating in bounded-memory batches…"):
            prepared_matrix, prepared_row_offsets, prepared_metadata, prepared_details = (
                prepare_analysis_data(
                    activation_matrix,
                    inspection["metadata_index"],
                    cached_position=inspection["positions"][position_index],
                    metadata_filters=filters,
                    aggregation_fields=aggregation_fields,
                    max_samples=max_samples,
                )
            )
        st.session_state.prepared_key = prepared_key
        st.session_state.prepared_matrix = prepared_matrix
        st.session_state.prepared_row_offsets = prepared_row_offsets
        st.session_state.prepared_metadata = prepared_metadata
        st.session_state.prepared_details = prepared_details
        st.session_state.pop("pca_key", None)
        clear_surface_fits()
    except Exception as exc:  # noqa: BLE001 - surface analysis/data errors in the UI
        st.error(f"Analysis data could not be prepared: {exc}")
        st.stop()

pca_key = (prepared_key, pca_identity)
if st.session_state.get("pca_key") != pca_key:
    try:
        for key in ("projection", "pca", "details"):
            st.session_state.pop(key, None)
        gc.collect()
        if direction_method == "PLS" and pls_mode == "Fit new":
            spinner_text = "Fitting PLS directions against log-time-horizon…"
        elif direction_method == "PLS":
            spinner_text = "Projecting with the loaded PLS model…"
        elif pca_mode == "Fit new":
            spinner_text = "Fitting a new bounded-memory PCA model…"
        else:
            spinner_text = "Projecting with the loaded PCA model…"
        with st.spinner(spinner_text):
            if direction_method == "PLS" and pls_mode == "Fit new":
                projection, pca, details = fit_pls_projection(
                    activation_matrix,
                    st.session_state.get("prepared_matrix"),
                    st.session_state["prepared_row_offsets"],
                    st.session_state["prepared_metadata"],
                    n_components=n_components,
                    scale=pls_scale,
                    max_iter=pls_max_iter,
                    tol=pls_tolerance,
                    details=st.session_state["prepared_details"],
                )
            elif direction_method == "PLS":
                projection, pca, details = transform_pls_projection(
                    activation_matrix,
                    st.session_state.get("prepared_matrix"),
                    st.session_state["prepared_row_offsets"],
                    st.session_state["prepared_metadata"],
                    pls=loaded_pls_model,
                    details=st.session_state["prepared_details"],
                )
            elif pca_mode == "Fit new":
                projection, pca, details = fit_pca_projection(
                    activation_matrix,
                    st.session_state.get("prepared_matrix"),
                    st.session_state["prepared_row_offsets"],
                    st.session_state["prepared_metadata"],
                    n_components=n_components,
                    details=st.session_state["prepared_details"],
                )
            else:
                projection, pca, details = transform_pca_projection(
                    activation_matrix,
                    st.session_state.get("prepared_matrix"),
                    st.session_state["prepared_row_offsets"],
                    st.session_state["prepared_metadata"],
                    pca=loaded_pca_model,
                    details=st.session_state["prepared_details"],
                )
            score_prefix = "PLS" if direction_method == "PLS" else "PC"
            score_fields = [
                f"{score_prefix}{index + 1}" for index in range(pca.components_.shape[0])
            ]
            projection["reconstruction_residual_rms"] = reconstruction_residual_rms(
                activation_matrix,
                st.session_state.get("prepared_matrix"),
                st.session_state["prepared_row_offsets"],
                projection[score_fields].to_numpy(),
                pca,
            )
        st.session_state.pca_key = pca_key
        st.session_state.projection = projection
        st.session_state.pca = pca
        st.session_state.details = details
        clear_surface_fits()
    except Exception as exc:  # noqa: BLE001 - surface analysis/data errors in the UI
        st.error(f"{direction_method} projection could not be prepared: {exc}")
        st.stop()

projection: pd.DataFrame = st.session_state.projection
details = st.session_state.details
active_pca = st.session_state.pca
component_count = int(active_pca.components_.shape[0])
direction_prefix = "PC" if direction_method == "PCA" else "PLS"
pc_fields = [f"{direction_prefix}{index}" for index in range(1, component_count + 1)]
axis_fields = [*pc_fields, "reconstruction_residual_rms"]
metadata_fields = sorted(
    column for column in projection if column not in {*pc_fields, "sample_index"}
)
color_fields = [
    "log10_time_horizon_months",
    *[field for field in metadata_fields if field != "log10_time_horizon_months"],
]

metric_columns = st.columns(4)
metric_columns[0].metric("Source samples", f"{details['loaded_samples']:,}")
metric_columns[1].metric("Projected points", f"{details['analysis_rows']:,}")
metric_columns[2].metric("Activation width", f"{details['feature_count']:,}")
variance_metric_label = (
    "Variance captured" if direction_method == "PCA" else "X variance represented"
)
metric_columns[3].metric(variance_metric_label, f"{sum(details['explained_variance']):.1%}")
if details.get("pca_source") == "loaded":
    st.caption("Explained variance describes the loaded model's original training data.")
st.subheader("Projection")
controls = st.columns([1.1, 1.25, 1.25, 2])
plot_mode = controls[0].segmented_control(
    "Plot", ["2D", "3D"], default="3D" if len(axis_fields) >= 3 else "2D"
)
required_axes = 3 if plot_mode == "3D" else 2
if len(axis_fields) < required_axes:
    st.warning(
        f"At least {required_axes} numeric projection fields are required for a {plot_mode} plot."
    )
    st.stop()
x_component = controls[1].selectbox("X axis", axis_fields, index=0)
y_choices = [field for field in axis_fields if field != x_component]
y_component = controls[2].selectbox("Y axis", y_choices, index=0)
z_component = None
if plot_mode == "3D":
    z_choices = [field for field in axis_fields if field not in {x_component, y_component}]
    z_component = controls[3].selectbox("Z axis", z_choices, index=0)
else:
    controls[3].caption("Choose any two projection fields for the plane.")

plot_controls = st.columns([1.1, 1, 1, 2])
color_field = plot_controls[0].selectbox("Color by", color_fields)
point_size = plot_controls[1].slider(
    "Point size",
    min_value=1,
    max_value=20,
    value=4 if plot_mode == "3D" else 7,
    key=f"point_size::{plot_mode}",
)
point_opacity = plot_controls[2].slider(
    "Point opacity",
    min_value=0.05,
    max_value=1.0,
    value=0.76,
    step=0.01,
)
tooltip_fields = plot_controls[3].multiselect(
    "Tooltip fields",
    ["sample_index", *metadata_fields],
    default=[field for field in ["sample_index", "time_horizon_months"] if field in projection],
)

stored_visual_fields = st.session_state.get("visual_filter_fields", [])
valid_visual_fields = [field for field in stored_visual_fields if field in metadata_fields]
if valid_visual_fields != stored_visual_fields:
    st.session_state.visual_filter_fields = valid_visual_fields

visual_filters = {}
with st.popover(
    "Filter visible points",
    icon=":material/filter_alt:",
    help="These filters change the chart only; they do not refit the selected directions.",
):
    st.caption(
        "Values within a field are combined with OR; fields are combined with AND. "
        "Leave a value selector empty to show all values."
    )
    selected_visual_fields = st.multiselect(
        "Filter plot by",
        metadata_fields,
        key="visual_filter_fields",
        placeholder="Choose metadata fields",
    )
    for field in selected_visual_fields:
        choices = metadata_filter_choices(projection[field])
        option_tokens = [token for token, _ in choices]
        option_labels = dict(choices)
        widget_key = f"visual_filter::{field}"
        stored_tokens = st.session_state.get(widget_key, [])
        valid_tokens = [
            token for token in stored_tokens if isinstance(token, tuple) and token in option_labels
        ]
        if valid_tokens != stored_tokens:
            st.session_state[widget_key] = valid_tokens
        visual_filters[field] = st.multiselect(
            f"Show only · {field}",
            option_tokens,
            key=widget_key,
            format_func=option_labels.__getitem__,
            placeholder="All values",
            help="Leave empty to include every value, including missing values.",
        )
    if selected_visual_fields:
        st.button(
            "Clear visual filters",
            icon=":material/filter_alt_off:",
            on_click=clear_visual_filters,
        )
    if details.get("aggregation_applied"):
        st.caption(
            "Filters use each projected aggregate's retained metadata. Include a field in "
            "Aggregate by when it must define the groups exactly."
        )

visual_mask = metadata_filter_mask(projection, visual_filters)
plot_data = projection.loc[visual_mask].copy()
st.caption(f"Visible {len(plot_data):,} of {len(projection):,} projected points.")
surface_result: SurfaceDisplayResult | None = None
surface_appearance: dict[str, Any] = {}
curve_result: CurveDisplayResult | None = None
curve_appearance: dict[str, Any] = {}
extruded_surface_result: RMSSplineSurfaceEvaluationResult | None = None
extruded_surface_appearance: dict[str, Any] = {}
if plot_data.empty:
    st.warning("No projected points match the visual filters.")
else:
    if plot_mode == "3D":
        overlay_controls = st.container(horizontal=True)
        surface_enabled = overlay_controls.toggle(
            "Surface overlay",
            value=False,
            key="surface_enabled",
            help=(
                f"Fit or load {z_component} = f({x_component}, {y_component}) and overlay "
                "the result."
            ),
            persist_state="page",
        )
        curve_enabled = overlay_controls.toggle(
            "Curve overlay",
            value=False,
            key="curve_enabled",
            help=(
                "Fit or load a parameterized trajectory through the three displayed "
                "coordinates."
            ),
            persist_state="page",
        )
        extruded_surface_enabled = overlay_controls.toggle(
            "Extruded surface",
            value=False,
            key="extruded_surface_enabled",
            help=(
                "Load a cubic spline and fit an RMS-conditioned polynomial extrusion "
                "for the selected displayed points."
            ),
            persist_state="page",
        )
        surface_mode = "Fit new"
        if surface_enabled:
            surface_mode = st.segmented_control(
                "Surface model",
                ["Fit new", "Use saved"],
                default="Fit new",
                key="surface_model_mode",
                persist_state="page",
            )
        if surface_enabled and surface_mode == "Fit new":
            with st.container(border=True):
                st.subheader("Surface fitting")
                st.caption(
                    f"Every method fits the axis-dependent height field **{z_component} = "
                    f"f({x_component}, {y_component})**. A folded or overhanging manifold "
                    "cannot be represented by one such surface. Hull-only interpolators, "
                    "nearest-neighbor smoothers, and tree models are omitted because they do "
                    "not extend a fitted trend beyond the observations."
                )
                fit_selectors = st.columns([1.4, 1.0])
                surface_algorithm = fit_selectors[0].selectbox(
                    "Algorithm",
                    list(SURFACE_ALGORITHMS),
                    index=list(SURFACE_ALGORITHMS).index("rbf"),
                    format_func=SURFACE_ALGORITHMS.__getitem__,
                    key="surface_algorithm",
                    persist_state="page",
                )
                fit_scope = fit_selectors[1].segmented_control(
                    "Fit points",
                    ["Visible", "All projected"],
                    default="Visible",
                    key="surface_scope",
                    help="Visible uses the chart's metadata filters; all projected ignores them.",
                    persist_state="page",
                )
                st.caption(SURFACE_DESCRIPTIONS[surface_algorithm])

                surface_source = plot_data if fit_scope == "Visible" else projection
                surface_xyz = surface_source[[x_component, y_component, z_component]].to_numpy(
                    dtype=np.float64
                )
                surface_data_digest = sha256(
                    np.ascontiguousarray(surface_xyz).tobytes()
                ).hexdigest()
                surface_data_identity = (
                    surface_data_digest,
                    x_component,
                    y_component,
                    z_component,
                    fit_scope,
                )
                surface_identity = (
                    *surface_data_identity,
                    surface_algorithm,
                    SURFACE_MODEL_ARTIFACT_VERSION,
                )
                point_count = len(surface_xyz)
                st.caption(f"Current fit scope contains {point_count:,} point(s).")

                if point_count < 4:
                    st.warning("At least four points are required to fit a surface.")
                else:
                    with st.form(f"surface_fit_form::{surface_algorithm}"):
                        st.markdown("**Model controls**")
                        surface_parameters = surface_algorithm_controls(
                            surface_algorithm, point_count
                        )
                        st.divider()
                        st.markdown("**Sampling and validation**")
                        sample_columns = st.columns(4)
                        safety_cap = surface_point_cap(surface_algorithm, surface_parameters)
                        default_point_cap = min(
                            point_count,
                            safety_cap if safety_cap is not None else 3_000,
                        )
                        maximum_point_cap = min(
                            point_count, safety_cap if safety_cap is not None else point_count
                        )
                        max_points_key = f"surface_common::{surface_algorithm}::max_points"
                        clamp_integer_widget_state(
                            max_points_key,
                            minimum=4,
                            maximum=maximum_point_cap,
                            default=max(4, default_point_cap),
                        )
                        max_fit_points = int(
                            sample_columns[0].number_input(
                                "Maximum fit points",
                                min_value=4,
                                max_value=maximum_point_cap,
                                value=max(4, default_point_cap),
                                step=max(1, min(100, point_count // 10)),
                                key=max_points_key,
                                persist_state="page",
                            )
                        )
                        validation_fraction = sample_columns[1].slider(
                            "Held-out fraction",
                            0.0,
                            0.4,
                            0.2,
                            0.05,
                            key=f"surface_common::{surface_algorithm}::validation",
                            persist_state="page",
                        )
                        random_state = int(
                            sample_columns[2].number_input(
                                "Random seed",
                                min_value=0,
                                max_value=2_147_483_647,
                                value=42,
                                key=f"surface_common::{surface_algorithm}::seed",
                                persist_state="page",
                            )
                        )
                        normalize_coordinates = sample_columns[3].checkbox(
                            "Normalize X/Y",
                            value=True,
                            key=f"surface_common::{surface_algorithm}::normalize",
                            help="Recommended for distance- and kernel-based methods.",
                            persist_state="page",
                        )

                        st.markdown("**Preview grid**")
                        grid_columns = st.columns(4)
                        grid_resolution = grid_columns[0].slider(
                            "Grid resolution",
                            15,
                            160,
                            60,
                            5,
                            key=f"surface_common::{surface_algorithm}::resolution",
                            help="Grid cost grows with the square of this value.",
                            persist_state="page",
                        )
                        padding_percent = grid_columns[1].slider(
                            "Grid padding",
                            0,
                            25,
                            0,
                            1,
                            format="%d%%",
                            key=f"surface_common::{surface_algorithm}::padding",
                            persist_state="page",
                        )
                        domain_label = grid_columns[2].selectbox(
                            "Displayed domain",
                            ["Convex hull", "Dense support", "Full rectangle"],
                            key=f"surface_common::{surface_algorithm}::domain",
                            persist_state="page",
                        )
                        density_factor = grid_columns[3].slider(
                            "Gap tolerance",
                            1.0,
                            8.0,
                            3.0,
                            0.25,
                            key=f"surface_common::{surface_algorithm}::density",
                            help="Used only for the Dense support domain.",
                            persist_state="page",
                        )
                        finishing_columns = st.columns(2)
                        clip_label = finishing_columns[0].selectbox(
                            "Display clipping",
                            ["None", "Observed Z range", "Robust 1–99% Z range"],
                            key=f"surface_common::{surface_algorithm}::clip",
                            persist_state="page",
                        )
                        duplicate_reducer = finishing_columns[1].selectbox(
                            "Duplicate X/Y reducer",
                            ["mean", "median"],
                            format_func=str.capitalize,
                            key=f"surface_common::{surface_algorithm}::duplicates",
                            persist_state="page",
                        )
                        st.caption(
                            "Grid masking and clipping affect only the chart preview. The saved "
                            "model returns unclipped predictions at any finite coordinate pair."
                        )
                        fit_submitted = st.form_submit_button(
                            "Fit / update surface",
                            type="primary",
                            icon=":material/architecture:",
                            width="stretch",
                        )

                    if fit_submitted:
                        domain_map = {
                            "Convex hull": "convex_hull",
                            "Dense support": "dense_support",
                            "Full rectangle": "full_grid",
                        }
                        clip_map = {
                            "None": "none",
                            "Observed Z range": "observed_range",
                            "Robust 1–99% Z range": "robust_range",
                        }
                        fit_options = {
                            "algorithm": surface_algorithm,
                            "parameters": surface_parameters,
                            "grid_resolution": grid_resolution,
                            "padding_fraction": padding_percent / 100.0,
                            "domain_mask": domain_map[domain_label],
                            "density_factor": density_factor,
                            "clip_mode": clip_map[clip_label],
                            "duplicate_reducer": duplicate_reducer,
                            "normalize_coordinates": normalize_coordinates,
                            "max_fit_points": max_fit_points,
                            "validation_fraction": validation_fraction,
                            "random_state": random_state,
                            "input_features": (x_component, y_component),
                            "output_feature": z_component,
                        }
                        try:
                            with st.spinner(f"Fitting {SURFACE_ALGORITHMS[surface_algorithm]}…"):
                                fitted_surface = fit_surface_cached(
                                    surface_xyz[:, 0],
                                    surface_xyz[:, 1],
                                    surface_xyz[:, 2],
                                    fit_options,
                                )
                            st.session_state.surface_result = fitted_surface
                            st.session_state.surface_identity = surface_identity
                        except ValueError as exc:
                            st.session_state.pop("surface_result", None)
                            st.session_state.pop("surface_identity", None)
                            st.error(f"Surface fitting failed: {exc}")

                    if st.session_state.get("surface_identity") == surface_identity:
                        surface_result = st.session_state.get("surface_result")
                    elif st.session_state.get("surface_result") is not None:
                        st.info(
                            "The points, axes, scope, or algorithm changed. Click "
                            "**Fit / update surface** to refresh the overlay."
                        )
                    else:
                        st.info("Choose the controls above, then fit the first surface.")

                    if surface_result is not None:
                        fit_metrics = surface_result.metrics
                        diagnostic_columns = st.columns(4)
                        diagnostic_columns[0].metric(
                            "Held-out RMSE",
                            _format_metric(fit_metrics["validation_rmse"]),
                            help="Root-mean-square error on points excluded from model fitting.",
                        )
                        diagnostic_columns[1].metric(
                            "Held-out R²", _format_metric(fit_metrics["validation_r2"])
                        )
                        diagnostic_columns[2].metric(
                            "Validation coverage",
                            _format_metric(fit_metrics["validation_coverage"], percent=True),
                            help="Fraction of held-out points for which the model could predict.",
                        )
                        diagnostic_columns[3].metric(
                            "Total compute", f"{float(fit_metrics['total_seconds']):.2f} s"
                        )
                        st.caption(
                            f"Training RMSE {_format_metric(fit_metrics['train_rmse'])} · "
                            f"{int(fit_metrics['fit_points']):,} fit points · "
                            f"{_format_metric(fit_metrics['grid_coverage'], percent=True)} grid "
                            "coverage"
                        )
                        if surface_result.warnings:
                            st.warning(" ".join(surface_result.warnings))

                        surface_artifact_metadata = {
                            "x_component": x_component,
                            "y_component": y_component,
                            "z_component": z_component,
                            "fit_scope": fit_scope,
                            "surface_data_sha256": surface_data_digest,
                            "pca_sha256": pca_projection_fingerprint(active_pca),
                            "pca_fingerprint_version": (PCA_PROJECTION_FINGERPRINT_VERSION),
                            "direction_method": direction_method,
                            "direction_target": (
                                "log10_time_horizon_months"
                                if direction_method == "PLS"
                                else None
                            ),
                            "layer_component": component,
                            "cached_position": inspection["positions"][position_index],
                            "aggregation_fields": list(aggregation_fields),
                            "fit_points": fit_metrics["fit_points"],
                            "input_points": fit_metrics["input_points"],
                            "random_state": random_state,
                            "maximum_fit_points": max_fit_points,
                            "duplicate_reducer": duplicate_reducer,
                        }
                        safe_surface_name = "-".join(
                            str(value).replace("/", "-").replace("\\", "-").replace(" ", "_")
                            for value in (z_component, "from", x_component, y_component)
                        )
                        fitted_equation = surface_result.model.equation
                        with st.container(horizontal=True):
                            st.download_button(
                                "Download surface model",
                                data=lambda: serialize_surface_model(
                                    surface_result.model,
                                    metadata=surface_artifact_metadata,
                                ),
                                file_name=(
                                    f"activation_surface_{safe_surface_name}_"
                                    f"{surface_result.algorithm}.joblib"
                                ),
                                mime="application/octet-stream",
                                icon=":material/download:",
                                on_click="ignore",
                                help=(
                                    "Versioned joblib artifact containing the fitted estimator, "
                                    "raw-coordinate normalization, axes, bounds, and provenance."
                                ),
                            )
                            if fitted_equation is not None:
                                st.download_button(
                                    "Download equation",
                                    data=fitted_equation.encode("utf-8"),
                                    file_name=f"activation_surface_{safe_surface_name}.txt",
                                    mime="text/plain",
                                    icon=":material/function:",
                                    on_click="ignore",
                                )

                        with st.expander("Use the saved model"):
                            st.caption(
                                "The predictor accepts raw coordinates in the saved axis order. "
                                "Its extrapolation mask identifies rows outside either fitted "
                                "coordinate interval. Only load joblib files you trust."
                            )
                            st.code(
                                "from temporal_manifolds.viz.surface_fitting import "
                                "load_surface_model\n\n"
                                'surface, provenance = load_surface_model("surface.joblib")\n'
                                f"{z_component.lower()} = surface.predict([[{x_component.lower()}, "
                                f"{y_component.lower()}]])[0]\n"
                                "is_extrapolation = surface.extrapolation_mask("
                                f"[[{x_component.lower()}, {y_component.lower()}]])[0]",
                                language="python",
                            )
                            if fitted_equation is not None:
                                st.markdown("**Exact fitted equation**")
                                st.code(fitted_equation, language="text")

                        surface_appearance = surface_appearance_controls(surface_result)

                        with st.expander("Surface diagnostics"):
                            diagnostic_table = pd.DataFrame(
                                [
                                    {
                                        "split": "Training",
                                        "points": fit_metrics["train_total_count"],
                                        "predicted points": fit_metrics["train_predicted_count"],
                                        "coverage": fit_metrics["train_coverage"],
                                        "RMSE": fit_metrics["train_rmse"],
                                        "normalized RMSE": fit_metrics["train_nrmse"],
                                        "MAE": fit_metrics["train_mae"],
                                        "R²": fit_metrics["train_r2"],
                                    },
                                    {
                                        "split": "Held-out validation",
                                        "points": fit_metrics["validation_total_count"],
                                        "predicted points": fit_metrics[
                                            "validation_predicted_count"
                                        ],
                                        "coverage": fit_metrics["validation_coverage"],
                                        "RMSE": fit_metrics["validation_rmse"],
                                        "normalized RMSE": fit_metrics["validation_nrmse"],
                                        "MAE": fit_metrics["validation_mae"],
                                        "R²": fit_metrics["validation_r2"],
                                    },
                                ]
                            )
                            st.dataframe(
                                diagnostic_table,
                                hide_index=True,
                                width="stretch",
                                column_config={
                                    "coverage": st.column_config.NumberColumn(format="percent"),
                                    "RMSE": st.column_config.NumberColumn(format="%.5g"),
                                    "normalized RMSE": st.column_config.NumberColumn(format="%.5g"),
                                    "MAE": st.column_config.NumberColumn(format="%.5g"),
                                    "R²": st.column_config.NumberColumn(format="%.4f"),
                                },
                            )
                            st.write(
                                {
                                    "Input points": fit_metrics["input_points"],
                                    "Unique X/Y": fit_metrics["unique_points"],
                                    "Fit points": fit_metrics["fit_points"],
                                    "Duplicates merged": fit_metrics["duplicates_merged"],
                                    "Non-finite rows dropped": fit_metrics["dropped_nonfinite"],
                                    "Grid coverage": fit_metrics["grid_coverage"],
                                    "Unclipped overshoot fraction": fit_metrics[
                                        "overshoot_fraction"
                                    ],
                                    "Grid prediction seconds": fit_metrics["prediction_seconds"],
                                }
                            )

                    comparison = st.expander("Quick algorithm comparison")
                    with comparison:
                        st.caption(
                            "Fits selected methods with conservative defaults on the same "
                            "held-out split. Lower selection score is better; the score is "
                            "normalized RMSE divided by coverage, plus a missing-coverage "
                            "penalty. Tune the winner above before drawing conclusions."
                        )
                        with st.form("surface_comparison_form"):
                            comparison_algorithms = st.multiselect(
                                "Methods",
                                list(SURFACE_ALGORITHMS),
                                default=["rbf", "polynomial", "plane", "svr"],
                                format_func=SURFACE_ALGORITHMS.__getitem__,
                                key="surface_comparison_algorithms",
                                persist_state="page",
                            )
                            comparison_columns = st.columns(3)
                            comparison_points_key = "surface_comparison_max_points"
                            clamp_integer_widget_state(
                                comparison_points_key,
                                minimum=4,
                                maximum=point_count,
                                default=min(point_count, 800),
                            )
                            comparison_max_points = int(
                                comparison_columns[0].number_input(
                                    "Maximum points per method",
                                    min_value=4,
                                    max_value=point_count,
                                    value=min(point_count, 800),
                                    step=max(1, min(100, point_count // 10)),
                                    key=comparison_points_key,
                                    persist_state="page",
                                )
                            )
                            comparison_validation = comparison_columns[1].slider(
                                "Held-out fraction",
                                0.1,
                                0.4,
                                0.2,
                                0.05,
                                key="surface_comparison_validation",
                                persist_state="page",
                            )
                            comparison_seed = int(
                                comparison_columns[2].number_input(
                                    "Random seed",
                                    min_value=0,
                                    max_value=2_147_483_647,
                                    value=42,
                                    key="surface_comparison_seed",
                                    persist_state="page",
                                )
                            )
                            comparison_submitted = st.form_submit_button(
                                "Compare methods",
                                icon=":material/compare_arrows:",
                                width="stretch",
                            )

                        if comparison_submitted:
                            if not comparison_algorithms:
                                st.warning("Select at least one method to compare.")
                            else:
                                comparison_rows = []
                                finite_z = surface_xyz[np.isfinite(surface_xyz[:, 2]), 2]
                                comparison_scale = (
                                    float(np.subtract(*np.percentile(finite_z, [75, 25])))
                                    if len(finite_z)
                                    else 1.0
                                )
                                if comparison_scale <= np.finfo(np.float64).eps:
                                    comparison_scale = float(np.std(finite_z))
                                if comparison_scale <= np.finfo(np.float64).eps:
                                    comparison_scale = float(np.ptp(finite_z))
                                if comparison_scale <= np.finfo(np.float64).eps:
                                    comparison_scale = 1.0
                                method_caps = [
                                    ALGORITHM_POINT_CAPS[candidate]
                                    for candidate in comparison_algorithms
                                    if ALGORITHM_POINT_CAPS[candidate] is not None
                                ]
                                comparison_common_cap = min(
                                    comparison_max_points,
                                    *(method_caps or [comparison_max_points]),
                                )
                                if comparison_common_cap < comparison_max_points:
                                    st.info(
                                        f"Using a common {comparison_common_cap:,}-point cap "
                                        "so every selected method sees the same sample."
                                    )
                                progress = st.progress(0, text="Comparing surface methods…")
                                for index, candidate in enumerate(comparison_algorithms):
                                    progress.progress(
                                        index / len(comparison_algorithms),
                                        text=f"Fitting {SURFACE_ALGORITHMS[candidate]}…",
                                    )
                                    candidate_parameters = default_surface_parameters(
                                        candidate, comparison_common_cap
                                    )
                                    if candidate == "gaussian_process":
                                        candidate_parameters["optimize"] = False
                                    candidate_options = {
                                        "algorithm": candidate,
                                        "parameters": candidate_parameters,
                                        "grid_resolution": 20,
                                        "padding_fraction": 0.0,
                                        "domain_mask": "convex_hull",
                                        "density_factor": 3.0,
                                        "clip_mode": "none",
                                        "duplicate_reducer": "mean",
                                        "normalize_coordinates": True,
                                        "max_fit_points": comparison_common_cap,
                                        "validation_fraction": comparison_validation,
                                        "random_state": comparison_seed,
                                        "input_features": (x_component, y_component),
                                        "output_feature": z_component,
                                    }
                                    try:
                                        candidate_result = fit_surface_cached(
                                            surface_xyz[:, 0],
                                            surface_xyz[:, 1],
                                            surface_xyz[:, 2],
                                            candidate_options,
                                        )
                                        candidate_metrics = candidate_result.metrics
                                        validation_rmse = candidate_metrics["validation_rmse"]
                                        validation_coverage = candidate_metrics[
                                            "validation_coverage"
                                        ]
                                        score = (
                                            float(validation_rmse)
                                            / comparison_scale
                                            / float(validation_coverage)
                                            + (1.0 - float(validation_coverage))
                                            if validation_rmse is not None
                                            and validation_coverage not in (None, 0)
                                            else None
                                        )
                                        comparison_rows.append(
                                            {
                                                "Algorithm": SURFACE_ALGORITHMS[candidate],
                                                "Selection score": score,
                                                "Validation RMSE": validation_rmse,
                                                "Validation MAE": candidate_metrics[
                                                    "validation_mae"
                                                ],
                                                "Validation R²": candidate_metrics["validation_r2"],
                                                "Coverage": validation_coverage,
                                                "Train RMSE": candidate_metrics["train_rmse"],
                                                "Total seconds": candidate_metrics["total_seconds"],
                                                "Status": " · ".join(candidate_result.warnings)
                                                or "OK",
                                            }
                                        )
                                    except ValueError as exc:
                                        comparison_rows.append(
                                            {
                                                "Algorithm": SURFACE_ALGORITHMS[candidate],
                                                "Selection score": None,
                                                "Validation RMSE": None,
                                                "Validation MAE": None,
                                                "Validation R²": None,
                                                "Coverage": None,
                                                "Train RMSE": None,
                                                "Total seconds": None,
                                                "Status": str(exc),
                                            }
                                        )
                                progress.progress(1.0, text="Comparison complete")
                                progress.empty()
                                comparison_table = pd.DataFrame(comparison_rows).sort_values(
                                    "Selection score", na_position="last"
                                )
                                st.session_state.surface_benchmark = comparison_table
                                st.session_state.surface_benchmark_identity = surface_data_identity

                        if (
                            st.session_state.get("surface_benchmark_identity")
                            == surface_data_identity
                            and st.session_state.get("surface_benchmark") is not None
                        ):
                            comparison_table = st.session_state.surface_benchmark
                            st.dataframe(
                                comparison_table,
                                hide_index=True,
                                width="stretch",
                                column_config={
                                    "Selection score": st.column_config.NumberColumn(format="%.4f"),
                                    "Validation RMSE": st.column_config.NumberColumn(format="%.5g"),
                                    "Validation MAE": st.column_config.NumberColumn(format="%.5g"),
                                    "Validation R²": st.column_config.NumberColumn(format="%.4f"),
                                    "Coverage": st.column_config.NumberColumn(format="percent"),
                                    "Train RMSE": st.column_config.NumberColumn(format="%.5g"),
                                    "Total seconds": st.column_config.NumberColumn(format="%.3f s"),
                                },
                            )
                            ranked = comparison_table.dropna(subset=["Selection score"])
                            if not ranked.empty:
                                st.success(
                                    f"Best default on this split: "
                                    f"**{ranked.iloc[0]['Algorithm']}**."
                                )
                        elif st.session_state.get("surface_benchmark") is not None:
                            st.info(
                                "The benchmark points or axes changed. Run the comparison again."
                            )
        elif surface_enabled:
            surface_result, surface_appearance = loaded_surface_controls(
                plot_data=plot_data,
                projection=projection,
                x_component=x_component,
                y_component=y_component,
                z_component=z_component,
                active_pca=active_pca,
                direction_method=direction_method,
                layer_component=component,
                cached_position=inspection["positions"][position_index],
            )
        curve_mode = "Fit new"
        if curve_enabled:
            curve_mode = st.segmented_control(
                "Curve model",
                ["Fit new", "Use saved"],
                default="Fit new",
                key="curve_model_mode",
                persist_state="page",
            )
        if curve_enabled and curve_mode == "Fit new":
            curve_result, curve_appearance = new_curve_controls(
                plot_data=plot_data,
                projection=projection,
                x_component=x_component,
                y_component=y_component,
                z_component=z_component,
                active_pca=active_pca,
                direction_method=direction_method,
                layer_component=component,
                cached_position=inspection["positions"][position_index],
                aggregation_fields=list(aggregation_fields),
            )
        elif curve_enabled:
            curve_result, curve_appearance = loaded_curve_controls(
                plot_data=plot_data,
                projection=projection,
                x_component=x_component,
                y_component=y_component,
                z_component=z_component,
                active_pca=active_pca,
                direction_method=direction_method,
                layer_component=component,
                cached_position=inspection["positions"][position_index],
            )
        if extruded_surface_enabled:
            extruded_surface_result, extruded_surface_appearance = (
                loaded_extruded_surface_controls(
                    plot_data=plot_data,
                    projection=projection,
                    x_component=x_component,
                    y_component=y_component,
                    z_component=z_component,
                    active_pca=active_pca,
                    direction_method=direction_method,
                    layer_component=component,
                    cached_position=inspection["positions"][position_index],
                )
            )
    else:
        st.caption(
            "Switch the plot to 3D to fit or load surface and curve overlays, including "
            "parametric extruded surfaces."
        )

    visibility_controls = st.container(horizontal=True)
    show_points = visibility_controls.toggle(
        "Show points",
        value=True,
        key="plot_show_points",
        help="Show or hide the projected activation points.",
        persist_state="page",
    )
    show_fitted_surface = True
    show_fitted_curve = True
    show_extruded_surface = True
    if plot_mode == "3D":
        show_fitted_surface = visibility_controls.toggle(
            "Show surface",
            value=True,
            key="plot_show_fitted_surface",
            help="Show or hide the fitted or loaded surface without discarding it.",
            disabled=surface_result is None,
            persist_state="page",
        )
        show_fitted_curve = visibility_controls.toggle(
            "Show curve",
            value=True,
            key="plot_show_fitted_curve",
            help="Show or hide the fitted or loaded curve without discarding it.",
            disabled=curve_result is None,
            persist_state="page",
        )
        show_extruded_surface = visibility_controls.toggle(
            "Show extrusion",
            value=True,
            key="plot_show_extruded_surface",
            help="Show or hide the loaded extruded surface without discarding it.",
            disabled=extruded_surface_result is None,
            persist_state="page",
        )

    numeric_color = is_numeric_dtype(plot_data[color_field]) and not is_bool_dtype(
        plot_data[color_field]
    )
    if not numeric_color:
        plot_data[color_field] = plot_data[color_field].astype("string").fillna("<missing>")
    common = {
        "data_frame": plot_data,
        "x": x_component,
        "y": y_component,
        "color": color_field,
        "color_continuous_scale": "Viridis" if numeric_color else None,
        "hover_data": tooltip_fields,
        "opacity": point_opacity,
    }
    curve_plot_bounds: np.ndarray | None = None
    if plot_mode == "3D":
        figure = px.scatter_3d(**common, z=z_component)
        figure.update_traces(marker={"size": point_size}, visible=show_points)
        if surface_result is not None and show_fitted_surface:
            use_uncertainty = (
                surface_appearance.get("color_mode") == "Predictive uncertainty"
                and surface_result.grid_uncertainty is not None
            )
            surface_color = (
                surface_result.grid_uncertainty if use_uncertainty else surface_result.grid_z
            )
            surface_color_hover = "<br>Uncertainty: %{surfacecolor:.4g}" if use_uncertainty else ""
            show_wireframe = bool(surface_appearance.get("show_wireframe"))
            figure.add_trace(
                go.Surface(
                    x=surface_result.grid_x,
                    y=surface_result.grid_y,
                    z=surface_result.grid_z,
                    surfacecolor=surface_color,
                    colorscale=surface_appearance.get("colorscale", "Viridis"),
                    reversescale=bool(surface_appearance.get("reverse_scale")),
                    opacity=float(surface_appearance.get("opacity", 0.58)),
                    showscale=bool(surface_appearance.get("show_colorbar")),
                    colorbar={
                        "title": "Uncertainty" if use_uncertainty else z_component,
                        "x": 1.12,
                    },
                    contours={
                        "x": {"show": show_wireframe, "color": "rgba(255,255,255,0.45)"},
                        "y": {"show": show_wireframe, "color": "rgba(255,255,255,0.45)"},
                        "z": {"show": bool(surface_appearance.get("show_contours"))},
                    },
                    connectgaps=False,
                    name=SURFACE_ALGORITHMS[surface_result.algorithm],
                    hovertemplate=(
                        f"{x_component}: %{{x:.4g}}<br>"
                        f"{y_component}: %{{y:.4g}}<br>"
                        f"{z_component}: %{{z:.4g}}"
                        f"{surface_color_hover}<extra>Surface</extra>"
                    ),
                )
            )

            if surface_appearance.get("show_residuals"):
                finite_residuals = np.isfinite(surface_result.point_prediction)
                available = np.flatnonzero(finite_residuals)
                residual_count = min(
                    int(surface_appearance.get("residual_count", 100)), len(available)
                )
                if surface_appearance.get("residual_selection") == "Largest residuals":
                    residual_size = np.abs(
                        surface_result.point_z[available]
                        - surface_result.point_prediction[available]
                    )
                    selected_residuals = available[np.argsort(residual_size)[-residual_count:]]
                else:
                    positions = np.linspace(0, len(available) - 1, residual_count).astype(int)
                    selected_residuals = available[positions]
                line_x: list[float | None] = []
                line_y: list[float | None] = []
                line_z: list[float | None] = []
                for residual_index in selected_residuals:
                    line_x.extend(
                        [
                            float(surface_result.point_x[residual_index]),
                            float(surface_result.point_x[residual_index]),
                            None,
                        ]
                    )
                    line_y.extend(
                        [
                            float(surface_result.point_y[residual_index]),
                            float(surface_result.point_y[residual_index]),
                            None,
                        ]
                    )
                    line_z.extend(
                        [
                            float(surface_result.point_z[residual_index]),
                            float(surface_result.point_prediction[residual_index]),
                            None,
                        ]
                    )
                figure.add_trace(
                    go.Scatter3d(
                        x=line_x,
                        y=line_y,
                        z=line_z,
                        mode="lines",
                        line={"color": "rgba(220, 38, 38, 0.62)", "width": 3},
                        name="Fit residuals",
                        hoverinfo="skip",
                    )
                )
        if extruded_surface_result is not None and show_extruded_surface:
            coordinate_indices = [
                extruded_surface_result.model.coordinate_features.index(component_name)
                for component_name in (x_component, y_component, z_component)
            ]
            surface_grid = extruded_surface_result.grid_xyz[..., coordinate_indices]
            use_extrusion_color = (
                extruded_surface_appearance.get("color_mode") == "Extrusion coordinate"
            )
            surface_color = (
                extruded_surface_result.grid_extrusion
                if use_extrusion_color
                else extruded_surface_result.grid_parameter
            )
            color_title = (
                "u" if use_extrusion_color else extruded_surface_result.model.parameter_feature
            )
            show_wireframe = bool(extruded_surface_appearance.get("show_wireframe"))
            figure.add_trace(
                go.Surface(
                    x=surface_grid[..., 0],
                    y=surface_grid[..., 1],
                    z=surface_grid[..., 2],
                    surfacecolor=surface_color,
                    customdata=np.stack(
                        [
                            extruded_surface_result.grid_parameter,
                            extruded_surface_result.grid_extrusion,
                            extruded_surface_result.grid_residual,
                        ],
                        axis=-1,
                    ),
                    colorscale=extruded_surface_appearance.get("colorscale", "Viridis"),
                    reversescale=bool(extruded_surface_appearance.get("reverse_scale")),
                    opacity=float(extruded_surface_appearance.get("opacity", 0.52)),
                    showscale=bool(extruded_surface_appearance.get("show_colorbar")),
                    colorbar={"title": color_title, "x": 1.24},
                    contours={
                        "x": {"show": show_wireframe, "color": "rgba(255,255,255,0.45)"},
                        "y": {"show": show_wireframe, "color": "rgba(255,255,255,0.45)"},
                    },
                    connectgaps=False,
                    name="Extruded spline surface",
                    hovertemplate=(
                        f"{x_component}: %{{x:.4g}}<br>"
                        f"{y_component}: %{{y:.4g}}<br>"
                        f"{z_component}: %{{z:.4g}}<br>"
                        f"{extruded_surface_result.model.parameter_feature}: "
                        "%{customdata[0]:.4g}<br>u: %{customdata[1]:.4g}<br>"
                        "Residual RMS: %{customdata[2]:.4g}"
                        "<extra>Extruded surface</extra>"
                    ),
                )
            )
            if extruded_surface_appearance.get("show_residuals"):
                point_xyz = extruded_surface_result.point_xyz[:, coordinate_indices]
                point_projection = extruded_surface_result.point_projection[:, coordinate_indices]
                residual_size = np.linalg.norm(point_xyz - point_projection, axis=1)
                residual_count = min(
                    int(extruded_surface_appearance.get("residual_count", 100)), len(point_xyz)
                )
                selected = np.argsort(residual_size)[-residual_count:]
                residual_x: list[float | None] = []
                residual_y: list[float | None] = []
                residual_z: list[float | None] = []
                for residual_index in selected:
                    residual_x.extend(
                        [point_xyz[residual_index, 0], point_projection[residual_index, 0], None]
                    )
                    residual_y.extend(
                        [point_xyz[residual_index, 1], point_projection[residual_index, 1], None]
                    )
                    residual_z.extend(
                        [point_xyz[residual_index, 2], point_projection[residual_index, 2], None]
                    )
                figure.add_trace(
                    go.Scatter3d(
                        x=residual_x,
                        y=residual_y,
                        z=residual_z,
                        mode="lines",
                        line={"color": "rgba(0, 137, 123, 0.72)", "width": 3},
                        name="Surface projection residuals",
                        hoverinfo="skip",
                    )
                )
        if curve_result is not None and show_fitted_curve:
            coordinate_indices = [
                curve_result.model.coordinate_features.index(component_name)
                for component_name in (x_component, y_component, z_component)
            ]
            saved_display_bounds = np.asarray(
                curve_result.metrics["display_coordinate_bounds"], dtype=np.float64
            )
            curve_plot_bounds = saved_display_bounds[coordinate_indices]
            curve_xyz = curve_result.curve_xyz[:, coordinate_indices]
            curve_mode = (
                "lines+markers" if curve_appearance.get("show_samples") else "lines"
            )
            curve_color = curve_appearance.get("color", "#d81b60")
            figure.add_trace(
                go.Scatter3d(
                    x=curve_xyz[:, 0],
                    y=curve_xyz[:, 1],
                    z=curve_xyz[:, 2],
                    customdata=curve_result.curve_parameter,
                    mode=curve_mode,
                    line={
                        "color": curve_color,
                        "width": int(curve_appearance.get("line_width", 6)),
                    },
                    marker={"color": curve_color, "size": 3},
                    opacity=float(curve_appearance.get("opacity", 0.95)),
                    name=CURVE_ALGORITHMS[curve_result.algorithm],
                    hovertemplate=(
                        f"{x_component}: %{{x:.4g}}<br>"
                        f"{y_component}: %{{y:.4g}}<br>"
                        f"{z_component}: %{{z:.4g}}<br>"
                        f"{curve_result.model.parameter_feature}: %{{customdata:.4g}}"
                        "<extra>Curve</extra>"
                    ),
                )
            )
            if curve_appearance.get("show_residuals"):
                point_xyz = curve_result.point_xyz[:, coordinate_indices]
                point_prediction = curve_result.point_prediction[:, coordinate_indices]
                residual_size = np.linalg.norm(point_xyz - point_prediction, axis=1)
                residual_count = min(
                    int(curve_appearance.get("residual_count", 100)), len(point_xyz)
                )
                selected = np.argsort(residual_size)[-residual_count:]
                residual_x: list[float | None] = []
                residual_y: list[float | None] = []
                residual_z: list[float | None] = []
                for residual_index in selected:
                    residual_x.extend(
                        [point_xyz[residual_index, 0], point_prediction[residual_index, 0], None]
                    )
                    residual_y.extend(
                        [point_xyz[residual_index, 1], point_prediction[residual_index, 1], None]
                    )
                    residual_z.extend(
                        [point_xyz[residual_index, 2], point_prediction[residual_index, 2], None]
                    )
                figure.add_trace(
                    go.Scatter3d(
                        x=residual_x,
                        y=residual_y,
                        z=residual_z,
                        mode="lines",
                        line={"color": curve_color, "width": 2},
                        opacity=0.45,
                        name="Curve residuals",
                        hoverinfo="skip",
                    )
                )
    else:
        figure = px.scatter(**common)
        figure.update_traces(
            marker={"size": point_size, "line": {"width": 0.35, "color": "white"}},
            visible=show_points,
        )

    surface_colorbar_visible = bool(
        surface_result is not None
        and show_fitted_surface
        and surface_appearance.get("show_colorbar")
    )
    extruded_colorbar_visible = bool(
        extruded_surface_result is not None
        and show_extruded_surface
        and extruded_surface_appearance.get("show_colorbar")
    )
    if surface_colorbar_visible and numeric_color:
        figure.update_layout(coloraxis_colorbar={"x": 1.0})
    if curve_plot_bounds is not None:
        figure.update_layout(
            scene={
                "xaxis": {"range": curve_plot_bounds[0].tolist()},
                "yaxis": {"range": curve_plot_bounds[1].tolist()},
                "zaxis": {"range": curve_plot_bounds[2].tolist()},
            }
        )
    figure.update_layout(
        height=680,
        margin={
            "l": 12,
            "r": 220 if surface_colorbar_visible or extruded_colorbar_visible else 12,
            "t": 28,
            "b": 12,
        },
        paper_bgcolor="white",
        plot_bgcolor="white",
        font={"family": "Inter, ui-sans-serif, system-ui", "color": "#17201d"},
        legend_title_text=color_field,
    )
    st.plotly_chart(figure, width="stretch", config={"displaylogo": False})

with st.expander(f"{direction_method} details and downloads"):
    variance_column = (
        "explained_variance" if direction_method == "PCA" else "represented_x_variance"
    )
    variance = pd.DataFrame(
        {
            "component": pc_fields,
            variance_column: details["explained_variance"],
            "cumulative_variance": details["explained_variance"].cumsum(),
        }
    )
    st.dataframe(variance, hide_index=True, width="stretch")
    if direction_method == "PLS":
        st.caption(
            "PLS directions are supervised only by `log10(time_horizon_months)`. Variance "
            "values describe the activation (X) variance represented by each latent score."
        )
    st.dataframe(projection_details_table(projection), hide_index=True, width="stretch")
    st.caption("Downloads contain the full projection, independent of visual filters.")
    safe_component = component.replace("/", "-").replace("\\", "-")
    with st.container(horizontal=True):
        if direction_method == "PCA":
            if pca_mode == "Fit new":
                model_metadata = {
                    "layer_component": component,
                    "cached_position": inspection["positions"][position_index],
                    "pca_solver": details.get("pca_solver"),
                    "aggregation_fields": list(aggregation_fields),
                }
            else:
                artifact_fields = {
                    "artifact_kind",
                    "artifact_version",
                    "sklearn_version",
                    "component_count",
                    "feature_count",
                }
                model_metadata = {
                    key: value
                    for key, value in loaded_pca_provenance.items()
                    if key not in artifact_fields
                }
            st.download_button(
                "Download PCA model",
                data=lambda: serialize_pca_model(active_pca, metadata=model_metadata),
                file_name=f"activation_pca_{safe_component}.joblib",
                mime="application/octet-stream",
                icon=":material/download:",
                on_click="ignore",
            )
        else:
            if pls_mode == "Fit new":
                model_metadata = {
                    "layer_component": component,
                    "cached_position": inspection["positions"][position_index],
                    "pls_target": "log10_time_horizon_months",
                    "aggregation_fields": list(aggregation_fields),
                }
            else:
                artifact_fields = {
                    "artifact_kind",
                    "artifact_version",
                    "sklearn_version",
                    "component_count",
                    "feature_count",
                }
                model_metadata = {
                    key: value
                    for key, value in loaded_pls_provenance.items()
                    if key not in artifact_fields
                }
            st.download_button(
                "Download PLS model",
                data=lambda: serialize_pls_model(active_pca, metadata=model_metadata),
                file_name=f"activation_pls_{safe_component}.joblib",
                mime="application/octet-stream",
                icon=":material/download:",
                on_click="ignore",
            )
        active_extruded_surface_model = st.session_state.get(
            "loaded_extruded_surface_model"
        )
        if active_extruded_surface_model is not None:
            clip_surface_residual = bool(
                st.session_state.get("extruded_surface_clip_residual", True)
            )

            def transformed_csv_data() -> bytes:
                return add_surface_parameter_columns(
                    projection,
                    active_extruded_surface_model,
                    clip=clip_surface_residual,
                ).to_csv(index=False).encode("utf-8")

            csv_data = transformed_csv_data
        else:
            csv_data = projection.to_csv(index=False).encode("utf-8")
        st.download_button(
            "Download projected CSV",
            csv_data,
            file_name=f"activation_{direction_method.lower()}_projection.csv",
            mime="text/csv",
            icon=":material/download:",
            on_click="ignore",
            help=(
                "Includes t (spline parameter) and u (extrusion parameter) when an "
                "extruded surface is loaded."
            ),
        )
