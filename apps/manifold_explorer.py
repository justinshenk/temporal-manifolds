"""Streamlit UI for exploring Kernel PCA, PCA, and PLS activation embeddings.

Every method embeds the same fixed activation slice, optionally after saved Fisher LDA
deflation, and is scored against the fixed target ``log10_time_horizon_months``. PLS
additionally uses that target during fitting.
"""

from __future__ import annotations

import gc
from dataclasses import dataclass, replace
from hashlib import sha256
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
from pandas.api.types import is_bool_dtype, is_numeric_dtype
from sklearn.cross_decomposition import PLSRegression
from sklearn.decomposition import PCA

from temporal_manifolds.activations.extraction_policy import (
    CACHED_POSITION_INDEX,
    NOT_APPLICABLE,
    PROMPT_TOKEN_POSITION,
    TARGET_LAYER_COMPONENT,
)
from temporal_manifolds.viz.activation_explorer import (
    SOURCE_FOLDER_FIELD,
    discover_activation_batch_paths,
    extract_activation_slice,
    inspect_sources,
    log10_time_horizon,
    load_pca_model,
    load_pls_model,
    metadata_filter_choices,
    metadata_filter_mask,
    prepare_analysis_data,
    projection_details_table,
    select_activation_batch_uploads,
    serialize_pca_model,
    serialize_pls_model,
    subtract_unconstrained_baseline,
)
from temporal_manifolds.viz.manifold_explorer import (
    ALGORITHM_DESCRIPTION,
    ALGORITHM_LABEL,
    KERNELS,
    MANIFOLD_MODEL_ARTIFACT_VERSION,
    TARGET_FEATURE,
    ManifoldFitResult,
    default_manifold_parameters,
    eigenvalue_spectrum_table,
    embedding_fields,
    explained_variance_table,
    fit_manifold,
    load_manifold_model,
    manifold_projection_frame,
    serialize_manifold_model,
    target_alignment_metrics,
    _structure_metrics,
)
from temporal_manifolds.viz.manifold_regression import (
    GROUP_FEATURE,
    RegressionFitResult,
    evaluate_regression_model,
    fit_manifold_regression,
    load_regression_model,
    regression_scores_table,
    serialize_regression_model,
)

st.set_page_config(
    page_title="Manifold Atlas",
    page_icon=":material/blur_on:",
    layout="wide",
)
st.caption("TEMPORAL MANIFOLDS · LOCAL NONLINEAR ANALYSIS")
st.title("Manifold Atlas")
st.write(
    "Filter conversational activation batches, aggregate comparable prompts, and embed them "
    "with Kernel PCA, PCA, or PLS, optionally after Fisher LDA deflation. Every embedding is "
    f"scored against `{TARGET_FEATURE}`. Your files stay in this local app session."
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
        "applied_preparation_settings",
        "prepared_key",
        "prepared_matrix",
        "prepared_row_offsets",
        "prepared_metadata",
        "prepared_details",
        "deflation_preprocessing_key",
        "deflation_preprocessed_values",
        "refit_after_deflation_change",
        "manifold_key",
        "manifold_result",
        "projection",
        "details",
    ):
        st.session_state.pop(key, None)


def use_sources(sources: list, *, label: str, is_local: bool) -> None:
    """Install a source selection and invalidate every derived result."""
    st.session_state.sources = sources
    st.session_state.source_label = label
    st.session_state.source_is_local = is_local
    st.session_state.source_revision += 1
    reset_loaded_data()


def forget_loaded_manifold() -> None:
    for key in (
        "manifold_model_upload",
        "loaded_manifold_digest",
        "loaded_manifold_model",
        "loaded_manifold_provenance",
        "loaded_manifold_filename",
        "manifold_result",
        "manifold_key",
        "projection",
        "details",
    ):
        st.session_state.pop(key, None)


def forget_loaded_linear_embedding() -> None:
    for key in (
        "linear_model_upload",
        "loaded_linear_digest",
        "loaded_linear_model",
        "loaded_linear_provenance",
        "loaded_linear_filename",
        "loaded_linear_method",
        "manifold_result",
        "manifold_key",
        "projection",
        "details",
    ):
        st.session_state.pop(key, None)


def forget_loaded_regression() -> None:
    for key in (
        "regression_model_upload",
        "loaded_regression_digest",
        "loaded_regression_model",
        "loaded_regression_provenance",
        "loaded_regression_filename",
    ):
        st.session_state.pop(key, None)


def clear_regression() -> None:
    for key in ("regression_result", "regression_key"):
        st.session_state.pop(key, None)


def handle_deflation_iteration_change() -> None:
    """Invalidate every result downstream of the deflation preprocessing choice."""

    had_embedding = st.session_state.get("manifold_result") is not None
    for key in (
        "deflation_preprocessing_key",
        "deflation_preprocessed_values",
        "manifold_key",
        "manifold_result",
        "projection",
        "details",
        "regression_key",
        "regression_result",
    ):
        st.session_state.pop(key, None)
    st.session_state["refit_after_deflation_change"] = bool(
        had_embedding and st.session_state.get("manifold_model_mode", "Fit new") == "Fit new"
    )


def reset_prompt_framing_filter() -> None:
    """Re-derive the framing filter so toggling subtraction re-applies its default."""
    st.session_state.pop("filter::template_metadata.prompt_framing", None)


def clear_visual_filters() -> None:
    for key in list(st.session_state):
        if key == "visual_filter_fields" or key.startswith("visual_filter::"):
            st.session_state.pop(key, None)


def clamp_integer_widget_state(key: str, *, minimum: int, maximum: int, default: int) -> None:
    """Keep persisted controls valid when filters shrink their dynamic bounds."""

    try:
        stored_value = int(st.session_state.get(key, default))
    except (TypeError, ValueError):
        stored_value = default
    clamped_value = max(minimum, min(maximum, stored_value))
    if st.session_state.get(key) != clamped_value:
        st.session_state[key] = clamped_value


def _format_metric(value: float | int | None, *, percent: bool = False) -> str:
    if value is None or not np.isfinite(float(value)):
        return "—"
    if percent:
        return f"{float(value):.1%}"
    return f"{float(value):.4g}"


FOLDER_BALANCE_SEED = 20240517
EMBEDDING_METHODS = [ALGORITHM_LABEL, "PCA", "PLS"]
REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
DEFLATION_ARTIFACTS_DIR = REPOSITORY_ROOT / "results" / "deflation_artifacts"
DEFLATION_ARTIFACT_FILENAME = "lda_pipeline.joblib"
DEFLATION_UPDATE_CHUNK_ROWS = 2_048


def discover_deflation_artifacts(
    artifacts_dir: Path = DEFLATION_ARTIFACTS_DIR,
) -> dict[int, Path]:
    """Return the available iteration bundles keyed by completed deflation count."""

    artifacts: dict[int, Path] = {}
    if not artifacts_dir.is_dir():
        return artifacts
    for iteration_dir in artifacts_dir.glob("iter_*"):
        try:
            iteration = int(iteration_dir.name.removeprefix("iter_"))
        except ValueError:
            continue
        artifact_path = iteration_dir / DEFLATION_ARTIFACT_FILENAME
        if iteration >= 0 and artifact_path.is_file():
            artifacts[iteration] = artifact_path
    return dict(sorted(artifacts.items()))


@st.cache_resource(max_entries=12, show_spinner=False)
def load_deflation_artifact(artifact_path: str, modified_time_ns: int) -> dict[str, Any]:
    """Load one trusted local bundle, invalidating the cache when it is overwritten."""

    del modified_time_ns  # Used only as part of the Streamlit cache key.
    artifact = joblib.load(artifact_path)
    if not isinstance(artifact, dict):
        raise ValueError("The deflation artifact must contain a dictionary bundle.")
    return artifact


def apply_lda_deflation_preprocessing(
    values: np.ndarray,
    artifact: dict[str, Any],
    *,
    iterations: int,
    expected_component: str,
    expected_position_index: int,
) -> np.ndarray:
    """Deflate in the artifact's PC space and reconstruct activation-space vectors."""

    prepared = np.asarray(values)
    if prepared.ndim != 2:
        raise ValueError(f"Expected a 2D activation matrix, got shape {prepared.shape}.")
    if iterations == 0:
        return np.asarray(prepared, dtype=np.float64)

    required_fields = {"pca", "completed_deflations", "deflations_completed"}
    missing_fields = sorted(required_fields - set(artifact))
    if missing_fields:
        raise ValueError(
            "The deflation artifact is missing required fields: " + ", ".join(missing_fields)
        )
    saved_iterations = int(artifact["deflations_completed"])
    if saved_iterations != iterations:
        raise ValueError(
            f"Selected {iterations} deflations, but the artifact represents {saved_iterations}."
        )
    steps = artifact["completed_deflations"]
    if not isinstance(steps, (list, tuple)) or len(steps) != iterations:
        raise ValueError(
            f"The iteration-{iterations} artifact must contain {iterations} completed steps."
        )

    saved_component = artifact.get("activation_component")
    if saved_component is not None and saved_component != expected_component:
        raise ValueError(
            f"The artifact was fitted for {saved_component!r}, not {expected_component!r}."
        )
    saved_position = artifact.get("position_index")
    if saved_position is not None and int(saved_position) != expected_position_index:
        raise ValueError(
            f"The artifact was fitted for cached position {saved_position}, not "
            f"{expected_position_index}."
        )

    pca = artifact["pca"]
    expected_width = int(
        artifact.get("input_feature_count", getattr(pca, "n_features_in_", -1))
    )
    if prepared.shape[1] != expected_width:
        raise ValueError(
            f"The deflation PCA expects {expected_width:,} activation features, but the "
            f"prepared slice has {prepared.shape[1]:,}."
        )
    if not hasattr(pca, "transform") or not hasattr(pca, "inverse_transform"):
        raise ValueError("The deflation artifact does not contain a fitted, invertible PCA model.")

    component_dtype = np.asarray(pca.components_).dtype
    if not np.issubdtype(component_dtype, np.floating):
        component_dtype = np.dtype(np.float64)
    pc_values = np.asarray(pca.transform(prepared), dtype=component_dtype)
    if pc_values.ndim != 2 or not np.isfinite(pc_values).all():
        raise ValueError("PCA projection produced invalid deflation coordinates.")

    for step_index, step in enumerate(steps, start=1):
        if not isinstance(step, dict):
            raise ValueError(f"Deflation step {step_index} is not a dictionary.")
        center = np.asarray(step.get("center"), dtype=component_dtype)
        direction = np.asarray(step.get("direction"), dtype=component_dtype)
        alpha = float(step.get("alpha", artifact.get("deflation_alpha", 1.0)))
        expected_shape = (pc_values.shape[1],)
        if center.shape != expected_shape or direction.shape != expected_shape:
            raise ValueError(
                f"Deflation step {step_index} has incompatible center/direction shapes."
            )
        if not np.isfinite(center).all() or not np.isfinite(direction).all() or not np.isfinite(alpha):
            raise ValueError(f"Deflation step {step_index} contains non-finite values.")

        scores = pc_values @ direction
        scores -= np.dot(center, direction)
        for start in range(0, len(pc_values), DEFLATION_UPDATE_CHUNK_ROWS):
            stop = min(start + DEFLATION_UPDATE_CHUNK_ROWS, len(pc_values))
            pc_values[start:stop] -= (
                alpha * scores[start:stop, None] * direction[None, :]
            )

    reconstructed = np.asarray(pca.inverse_transform(pc_values), dtype=np.float64)
    if reconstructed.shape != prepared.shape or not np.isfinite(reconstructed).all():
        raise ValueError("Inverse PCA produced invalid activation-space vectors.")
    return reconstructed


@dataclass(frozen=True)
class LinearEmbeddingModel:
    """PCA/PLS adapter exposing the model contract used by the shared explorer UI."""

    estimator: PCA | PLSRegression
    method: str
    component_count: int
    feature_count: int
    parameters: dict[str, Any]
    training_embedding: np.ndarray
    training_target: np.ndarray
    explained_variance_ratio: np.ndarray
    metrics: dict[str, Any]
    retained_variance_fraction: float | None
    scaler: None = None
    warnings: tuple[str, ...] = ()

    def transform(self, values: np.ndarray) -> np.ndarray:
        prepared = np.asarray(values, dtype=np.float64)
        if prepared.ndim != 2 or prepared.shape[1] != self.feature_count:
            raise ValueError(
                f"Expected activations with {self.feature_count:,} features, got {prepared.shape}."
            )
        return np.asarray(self.estimator.transform(prepared), dtype=np.float64)


def fit_linear_embedding(
    values: np.ndarray,
    target: np.ndarray,
    *,
    method: str,
    n_components: int,
    random_state: int,
    quality_neighbors: int,
    fit_mask: np.ndarray | None = None,
    pls_scale: bool = True,
    pls_max_iter: int = 500,
    pls_tolerance: float = 1e-6,
) -> ManifoldFitResult:
    """Fit PCA or horizon-supervised PLS and return the common embedding contract."""

    values = np.asarray(values, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    mask = (
        np.ones(len(values), dtype=bool)
        if fit_mask is None
        else np.asarray(fit_mask, dtype=bool).copy()
    )
    if method == "PLS" and not np.isfinite(target[mask]).all():
        raise ValueError(
            "PLS requires positive, finite time horizons. Exclude horizon-free prompts or "
            "subtract their unconstrained baselines before fitting."
        )
    training_values = values[mask]
    training_target = target[mask]
    maximum = min(max(len(training_values) - 1, 0), values.shape[1])
    if not 1 <= n_components <= maximum:
        raise ValueError(f"{method} components must be between 1 and {maximum}.")

    if method == "PCA":
        estimator: PCA | PLSRegression = PCA(
            n_components=n_components,
            svd_solver="auto",
            random_state=random_state,
        )
        training_embedding = np.asarray(
            estimator.fit_transform(training_values), dtype=np.float64
        )
        explained_variance = np.asarray(estimator.explained_variance_ratio_, dtype=np.float64)
        parameters = {"method": "PCA", "kernel": "linear", "supervised": False}
    elif method == "PLS":
        if len(training_values) < 3 or np.ptp(training_target) <= np.finfo(np.float64).eps:
            raise ValueError("PLS requires at least three points with two distinct time horizons.")
        estimator = PLSRegression(
            n_components=n_components,
            scale=pls_scale,
            max_iter=pls_max_iter,
            tol=pls_tolerance,
            copy=True,
        )
        training_embedding, _ = estimator.fit_transform(
            training_values, training_target.reshape(-1, 1)
        )
        training_embedding = np.asarray(training_embedding, dtype=np.float64)
        x_scale = np.asarray(estimator._x_std, dtype=np.float64)  # noqa: SLF001
        estimator.components_ = np.asarray(estimator.x_rotations_).T / x_scale[np.newaxis, :]
        estimator.mean_ = np.asarray(estimator._x_mean)  # noqa: SLF001
        scaled_total = np.sum(
            np.square((training_values - estimator.mean_) / x_scale, dtype=np.float64),
            dtype=np.float64,
        )
        component_energy = np.sum(np.square(training_embedding), axis=0) * np.sum(
            np.square(estimator.x_loadings_), axis=0
        )
        explained_variance = (
            np.clip(component_energy / scaled_total, 0.0, 1.0)
            if scaled_total > np.finfo(np.float64).eps
            else np.zeros(n_components, dtype=np.float64)
        )
        estimator.explained_variance_ratio_ = explained_variance
        parameters = {
            "method": "PLS",
            "kernel": "linear",
            "supervised": True,
            "scale": pls_scale,
            "max_iter": pls_max_iter,
            "tolerance": pls_tolerance,
            "target": TARGET_FEATURE,
        }
    else:
        raise ValueError(f"Unknown linear embedding method: {method!r}.")

    embedding = (
        training_embedding
        if fit_mask is None and method != "PLS"
        else np.asarray(estimator.transform(values), dtype=np.float64)
    )
    metrics = target_alignment_metrics(
        training_embedding,
        training_target,
        n_neighbors=min(quality_neighbors, max(len(training_values) - 1, 1)),
    )
    metrics.update(
        _structure_metrics(
            training_values,
            training_embedding,
            quality_neighbors=quality_neighbors,
            max_quality_points=2_000,
            random_state=random_state,
        )
    )
    metrics.update(
        {
            "algorithm": method.lower(),
            "input_points": len(training_values),
            "fit_points": len(training_values),
            "projected_points": len(values),
            "held_out_points": len(values) - len(training_values),
            "feature_count": values.shape[1],
            "component_count": n_components,
            "standardized": bool(method == "PLS" and pls_scale),
            "explained_variance_ratio": explained_variance,
            "cumulative_variance_ratio": np.cumsum(explained_variance),
            "retained_variance_fraction": float(np.sum(explained_variance)),
            "full_variance_spectrum": method == "PCA",
        }
    )
    model = LinearEmbeddingModel(
        estimator=estimator,
        method=method,
        component_count=n_components,
        feature_count=values.shape[1],
        parameters=parameters,
        training_embedding=training_embedding,
        training_target=training_target,
        explained_variance_ratio=explained_variance,
        retained_variance_fraction=float(np.sum(explained_variance)),
        metrics=metrics,
    )
    return ManifoldFitResult(
        model=model,  # type: ignore[arg-type]
        embedding=embedding,
        target=target,
        metrics=metrics,
    )


def balanced_folder_mask(
    folders: pd.Series, base_mask: np.ndarray | None, *, seed: int = FOLDER_BALANCE_SEED
) -> np.ndarray | None:
    """Return a fit mask that keeps the same number of rows from every source folder.

    Kernel PCA has no sample-weight parameter: the geometry is the centered kernel matrix of
    the rows it is given, so a folder contributing ten times as many rows pulls the leading
    components ten times as hard. Equalizing the row counts is therefore how each folder is
    given equal weight. Every folder is cut down to the smallest folder's row count with a
    deterministic random subsample; the dropped rows are still projected into the fitted
    coordinates and plotted, exactly like held-out folders.

    ``None`` is returned when balancing would change nothing (a single folder, or folders that
    are already the same size).
    """

    labels = folders.astype(str).to_numpy()
    if base_mask is None:
        base_mask = np.ones(len(labels), dtype=bool)
    else:
        base_mask = np.asarray(base_mask, dtype=bool)
    eligible = np.flatnonzero(base_mask)
    if len(eligible) == 0:
        return base_mask
    groups = {}
    for row in eligible:
        groups.setdefault(labels[row], []).append(row)
    if len(groups) < 2:
        return None
    quota = min(len(rows) for rows in groups.values())
    if all(len(rows) == quota for rows in groups.values()):
        return None
    rng = np.random.default_rng(seed)
    balanced = np.zeros(len(labels), dtype=bool)
    for folder in sorted(groups):
        rows = np.asarray(groups[folder])
        if len(rows) > quota:
            rows = rows[rng.choice(len(rows), size=quota, replace=False)]
        balanced[rows] = True
    return balanced


@st.cache_data(max_entries=8, show_spinner=False)
def fit_manifold_cached(
    values: np.ndarray,
    target: np.ndarray,
    options: dict[str, Any],
    fit_mask: np.ndarray | None = None,
) -> ManifoldFitResult:
    """Cache one fitted embedding keyed by its data and every control value.

    When ``fit_mask`` is given, the coordinate system is fitted on the masked rows only and
    every prepared row is then projected into it. The returned embedding therefore still
    covers all points, so the plot shows every source folder while the manifold's geometry is
    defined by the chosen ones alone.
    """

    method = str(options.get("method", ALGORITHM_LABEL))
    if method in {"PCA", "PLS"}:
        linear_options = dict(options)
        linear_options.pop("method", None)
        return fit_linear_embedding(
            values,
            target,
            method=method,
            fit_mask=fit_mask,
            **linear_options,
        )

    kernel_options = dict(options)
    kernel_options.pop("method", None)
    if fit_mask is None:
        return fit_manifold(values, target, **kernel_options)

    mask = np.asarray(fit_mask, dtype=bool)
    fitted = fit_manifold(values[mask], target[mask], **kernel_options)
    embedding = fitted.model.transform(values)
    metrics = {
        **fitted.metrics,
        # ``input_points`` counts the rows the kernel saw; the plot carries more than that.
        "fit_points": int(mask.sum()),
        "projected_points": int(len(values)),
        "held_out_points": int(len(values) - mask.sum()),
    }
    return replace(fitted, embedding=embedding, target=target, metrics=metrics)


def kernel_pca_controls() -> dict[str, Any]:
    """Render the Kernel PCA controls inside the embedding form."""

    prefix = "manifold_parameter::kernel_pca"
    defaults = default_manifold_parameters()

    columns = st.columns(3)
    kernel = columns[0].selectbox(
        "Kernel",
        list(KERNELS),
        format_func=KERNELS.__getitem__,
        key=f"{prefix}::kernel",
        help=(
            "The kernel defines the implicit feature space. Linear reproduces ordinary PCA "
            "and is useful as a sanity check."
        ),
        persist_state="page",
    )
    automatic_gamma = columns[1].checkbox(
        "Automatic gamma",
        value=False,
        key=f"{prefix}::auto_gamma",
        help="Uses 1 / n_features, scikit-learn's default kernel width.",
        persist_state="page",
    )
    gamma = 10.0 ** columns[2].slider(
        "log10 gamma",
        -6.0,
        2.0,
        -3.0,
        0.25,
        key=f"{prefix}::gamma",
        help="Ignored while automatic gamma is selected. Affects RBF, poly, and sigmoid.",
        persist_state="page",
    )
    detail_columns = st.columns(3)
    degree = detail_columns[0].slider(
        "Polynomial degree",
        2,
        6,
        int(defaults["degree"]),
        key=f"{prefix}::degree",
        help="Ignored unless the polynomial kernel is selected.",
        persist_state="page",
    )
    coef0 = detail_columns[1].slider(
        "Kernel coef0",
        0.0,
        10.0,
        float(defaults["coef0"]),
        0.5,
        key=f"{prefix}::coef0",
        help="Independent term for the polynomial and sigmoid kernels.",
        persist_state="page",
    )
    alpha = 10.0 ** detail_columns[2].slider(
        "log10 alpha",
        -6.0,
        2.0,
        0.0,
        0.25,
        key=f"{prefix}::alpha",
        help="Ridge regularization used by the kernel eigen-decomposition.",
        persist_state="page",
    )
    return {
        "kernel": kernel,
        "gamma": None if automatic_gamma else gamma,
        "degree": degree,
        "coef0": coef0,
        "alpha": alpha,
    }


with st.sidebar:
    st.header("1 · Choose activations")
    source_mode = st.segmented_control(
        "Source",
        ["Local folders", "Upload folders"],
        default="Local folders",
        key="source_mode",
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
activation_cache_dir = (
    Path("data") / "activation_explorer_cache" if st.session_state.get("source_is_local") else None
)

try:
    if "inspection" not in st.session_state:
        with st.spinner("Reading batch structure…"):
            st.session_state.inspection = inspect_sources(
                sources,
                cache_dir=activation_cache_dir,
            )
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
    filter_fields = st.multiselect("Filter fields", candidate_fields, default=[])
    filter_options = {
        field: sorted(metadata_index[field].dropna().unique().tolist(), key=str)
        for field in filter_fields
    }
    draft_filters = {}
    # The subtraction toggle is rendered below, so its previous value is read from session
    # state. Unconstrained prompts carry the NOT_APPLICABLE framing, and dropping them by
    # default would leave subtraction with no baselines to compute.
    baseline_framings = (
        [NOT_APPLICABLE] if st.session_state.get("subtract_unconstrained_baseline") else []
    )
    for field in filter_fields:
        values = filter_options[field]
        if field == "template_metadata.prompt_framing" and "task_available_time" in values:
            default_values = [
                framing
                for framing in ("task_available_time", *baseline_framings)
                if framing in values
            ]
        elif field == "base_unit":
            default_values = [
                value for value in values if str(value).casefold() not in {"millennia", "seconds"}
            ]
        else:
            default_values = values
        draft_filters[field] = st.multiselect(
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
        help="Rows in each group are averaged before the embedding is fitted.",
    )
    subtract_baseline = st.toggle(
        "Subtract unconstrained activations",
        value=False,
        key="subtract_unconstrained_baseline",
        on_change=reset_prompt_framing_filter,
        help=(
            "Average the unconstrained prompts of each task and subtract that baseline from "
            "the task's constrained points. Constrained points whose task has no "
            "unconstrained counterpart are dropped."
        ),
    )
    if subtract_baseline:
        st.caption(
            "Load the unconstrained batches (`.acts/no_constraints`) alongside the "
            "constrained folders, and keep their rows in the filters below. Unconstrained "
            "rows share one missing horizon, so add `task` to **Aggregate by** to keep each "
            "task's baseline separate."
        )
    max_samples_enabled = st.toggle("Limit source samples", value=False)
    max_samples = (
        int(st.number_input("Maximum samples", min_value=1, value=1000, step=100))
        if max_samples_enabled
        else None
    )

    draft_preparation_settings = {
        "filters": {
            field: list(draft_filters[field])
            for field in filter_fields
        },
        "aggregation_fields": list(aggregation_fields),
        "max_samples": max_samples,
        "subtract_baseline": bool(subtract_baseline),
    }
    applied_preparation_settings = st.session_state.get("applied_preparation_settings")
    if applied_preparation_settings is None:
        applied_preparation_settings = draft_preparation_settings
        st.session_state.applied_preparation_settings = applied_preparation_settings
    filters_pending = applied_preparation_settings != draft_preparation_settings
    apply_filters = st.button(
        "Apply filters",
        type="primary",
        icon=":material/filter_alt:",
        width="stretch",
        disabled=not filters_pending,
        help="Apply the current filters and rebuild the prepared analysis data.",
    )
    if apply_filters:
        applied_preparation_settings = draft_preparation_settings
        st.session_state.applied_preparation_settings = applied_preparation_settings
    elif filters_pending:
        st.caption("Filter changes are pending. The current analysis remains active.")

    st.divider()
    st.header("3 · Embedding method")
    embedding_method = st.segmented_control(
        "Embedding method",
        EMBEDDING_METHODS,
        default=ALGORITHM_LABEL,
        key="embedding_method",
        help=(
            "Kernel PCA learns nonlinear coordinates. PCA finds unsupervised linear variance "
            "directions. PLS finds linear directions supervised by log10(time_horizon_months)."
        ),
        persist_state="page",
    )
    if embedding_method == ALGORITHM_LABEL:
        st.caption(ALGORITHM_DESCRIPTION)
    elif embedding_method == "PCA":
        st.caption("Unsupervised linear directions ordered by activation variance.")
    else:
        st.caption(f"Supervised linear directions fitted against `{TARGET_FEATURE}`.")
    st.markdown("**Preprocessing**")
    deflation_artifacts = discover_deflation_artifacts()
    deflation_iteration_options = sorted({0, *deflation_artifacts})
    if st.session_state.get("lda_deflation_iterations", 0) not in deflation_iteration_options:
        st.session_state.lda_deflation_iterations = 0
    deflation_iterations = int(
        st.selectbox(
            "LDA deflation iterations",
            deflation_iteration_options,
            index=0,
            format_func=lambda count: (
                "None" if count == 0 else f"{count} iteration{'s' if count != 1 else ''}"
            ),
            key="lda_deflation_iterations",
            on_change=handle_deflation_iteration_change,
            help=(
                "Before fitting the selected embedding, project activations through the saved "
                "Fisher LDA PCA, replay this many deflation updates, then reconstruct activation "
                "vectors with inverse PCA. Zero leaves the prepared activations unchanged."
            ),
            persist_state="page",
        )
    )
    if len(deflation_iteration_options) == 1:
        st.caption(
            "No deflation bundles were found under "
            "`results/deflation_artifacts/iter_*`; run the Fisher LDA notebook to create them."
        )
    elif deflation_iterations:
        selected_path = deflation_artifacts[deflation_iterations]
        st.caption(f"Using `{selected_path.parent.name}/{selected_path.name}`.")
    st.info(f"Optimization target: `{TARGET_FEATURE}` (fixed).", icon=":material/target:")

applied_preparation_settings = st.session_state["applied_preparation_settings"]
filters = {
    field: list(values)
    for field, values in applied_preparation_settings["filters"].items()
}
analysis_aggregation_fields = list(applied_preparation_settings["aggregation_fields"])
max_samples = applied_preparation_settings["max_samples"]
subtract_baseline = bool(applied_preparation_settings["subtract_baseline"])
if subtract_baseline and "task" not in candidate_fields:
    st.error("Subtracting unconstrained activations requires the 'task' metadata field.")
    st.stop()

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
            cache_dir=activation_cache_dir,
            progress=update_extraction_progress,
        )
        progress_bar.empty()
        st.session_state.activation_matrix = activation_matrix
        st.session_state.activation_cache_path = cache_path
        st.session_state.slice_key = slice_key
        st.session_state.pop("prepared_key", None)
        st.session_state.pop("manifold_key", None)
    except Exception as exc:  # noqa: BLE001 - surface local extraction errors in the UI
        st.error(f"Activation slice could not be prepared: {exc}")
        st.stop()

activation_matrix = st.session_state.activation_matrix
if st.session_state.get("activation_cache_path") is not None:
    st.caption(
        "Using a disk-backed activation slice; source batches stay closed during embedding work."
    )

prepared_key = (
    tuple((field, tuple(values)) for field, values in filters.items()),
    tuple(analysis_aggregation_fields),
    max_samples,
    subtract_baseline,
)
if st.session_state.get("prepared_key") != prepared_key:
    try:
        # Drop the stale preparation key first. If preparation below fails, the popped data
        # and the key stay gone together, so returning to a previously successful selection
        # re-runs preparation instead of matching a key whose data no longer exists.
        for key in (
            "prepared_key",
            "prepared_matrix",
            "prepared_row_offsets",
            "prepared_metadata",
            "prepared_details",
            "deflation_preprocessing_key",
            "deflation_preprocessed_values",
            "projection",
            "details",
            "manifold_result",
            "manifold_key",
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
                    aggregation_fields=analysis_aggregation_fields,
                    max_samples=max_samples,
                )
            )
            if subtract_baseline:
                (
                    prepared_matrix,
                    prepared_row_offsets,
                    prepared_metadata,
                    baseline_details,
                ) = subtract_unconstrained_baseline(
                    prepared_matrix,
                    prepared_row_offsets,
                    prepared_metadata,
                    activation_matrix,
                )
                prepared_details = {
                    **prepared_details,
                    **baseline_details,
                    "analysis_rows": len(prepared_metadata),
                    "baseline_subtracted": True,
                }
        st.session_state.prepared_key = prepared_key
        st.session_state.prepared_matrix = prepared_matrix
        st.session_state.prepared_row_offsets = prepared_row_offsets
        st.session_state.prepared_metadata = prepared_metadata
        st.session_state.prepared_details = prepared_details
        st.session_state.pop("manifold_key", None)
    except Exception as exc:  # noqa: BLE001 - surface analysis/data errors in the UI
        st.error(f"Analysis data could not be prepared: {exc}")
        st.stop()

prepared_metadata: pd.DataFrame = st.session_state["prepared_metadata"]
prepared_matrix = st.session_state.get("prepared_matrix")
prepared_row_offsets = st.session_state["prepared_row_offsets"]
prepared_details = st.session_state["prepared_details"]
point_count = len(prepared_metadata)

# Source folders available to restrict the fit to. The multiselect below lives inside the fit
# form, so its committed value is read from session state here and used consistently for the
# row mask, the control clamps, and the fit key.
if SOURCE_FOLDER_FIELD in prepared_metadata:
    fit_folder_values = sorted(
        prepared_metadata[SOURCE_FOLDER_FIELD].dropna().astype(str).unique().tolist()
    )
else:
    fit_folder_values = []
committed_fit_folders = st.session_state.get("manifold_fit_folders")
selected_fit_folders = [
    folder
    for folder in fit_folder_values
    if committed_fit_folders is None or folder in set(committed_fit_folders)
]
restrict_fit_folders = bool(fit_folder_values) and len(selected_fit_folders) != len(
    fit_folder_values
)
if restrict_fit_folders:
    fit_row_mask = (
        prepared_metadata[SOURCE_FOLDER_FIELD]
        .astype(str)
        .isin(selected_fit_folders)
        .to_numpy(dtype=bool)
    )
else:
    fit_row_mask = None

# Folder balancing lives in the fit form as well, so its committed value is read back here for
# the same reason the folder selection is: the mask, the captions, and the fit key must agree.
balance_fit_folders = bool(st.session_state.get("manifold_balance_folders", False)) and (
    len(fit_folder_values) > 1
)
if balance_fit_folders:
    balanced_mask = balanced_folder_mask(
        prepared_metadata[SOURCE_FOLDER_FIELD], fit_row_mask
    )
    if balanced_mask is None:
        # Nothing to equalize; keep the unbalanced mask so the fit key stays stable.
        balance_fit_folders = False
    else:
        fit_row_mask = balanced_mask
fit_point_count = int(fit_row_mask.sum()) if fit_row_mask is not None else point_count

# Materialize the prepared rows once. Every method here is a whole-matrix batch algorithm, so
# unlike incremental PCA there is no streaming path to preserve. Optional LDA deflation happens
# in the saved PCA space, then inverse PCA reconstructs vectors at the original activation width
# before any of the three embedding methods sees them.
if prepared_matrix is not None:
    raw_embedding_values = np.asarray(prepared_matrix)
else:
    raw_embedding_values = np.asarray(activation_matrix[prepared_row_offsets])

selected_deflation_artifact = deflation_artifacts.get(deflation_iterations)
deflation_artifact_version: int | None = None
if deflation_iterations:
    if selected_deflation_artifact is None:
        st.error(
            f"No saved artifact is available for {deflation_iterations} deflation iterations."
        )
        st.stop()
    try:
        artifact_stat = selected_deflation_artifact.stat()
        deflation_preprocessing_key = (
            slice_key,
            prepared_key,
            deflation_iterations,
            str(selected_deflation_artifact.resolve()),
            artifact_stat.st_mtime_ns,
        )
        deflation_artifact = load_deflation_artifact(
            str(selected_deflation_artifact.resolve()), artifact_stat.st_mtime_ns
        )
        deflation_artifact_version = int(deflation_artifact.get("artifact_version", 0))
        if st.session_state.get("deflation_preprocessing_key") != deflation_preprocessing_key:
            st.session_state.pop("deflation_preprocessed_values", None)
            gc.collect()
            with st.spinner(
                f"Applying {deflation_iterations} LDA deflation "
                f"iteration{'s' if deflation_iterations != 1 else ''} and inverse PCA…"
            ):
                st.session_state.deflation_preprocessed_values = (
                    apply_lda_deflation_preprocessing(
                        raw_embedding_values,
                        deflation_artifact,
                        iterations=deflation_iterations,
                        expected_component=component,
                        expected_position_index=position_index,
                    )
                )
                st.session_state.deflation_preprocessing_key = deflation_preprocessing_key
        embedding_values = st.session_state["deflation_preprocessed_values"]
    except Exception as exc:  # noqa: BLE001 - surface invalid local artifact errors in the UI
        st.session_state.pop("deflation_preprocessing_key", None)
        st.session_state.pop("deflation_preprocessed_values", None)
        st.error(f"LDA deflation preprocessing failed: {exc}")
        st.stop()
else:
    st.session_state.pop("deflation_preprocessing_key", None)
    st.session_state.pop("deflation_preprocessed_values", None)
    deflation_preprocessing_key = ("none",)
    embedding_values = np.asarray(raw_embedding_values, dtype=np.float64)

embedding_preparation_details = {
    **prepared_details,
    "lda_deflation_iterations": deflation_iterations,
    "lda_deflation_artifact": (
        selected_deflation_artifact.relative_to(REPOSITORY_ROOT).as_posix()
        if selected_deflation_artifact is not None and deflation_iterations
        else None
    ),
    "lda_deflation_artifact_version": deflation_artifact_version,
    "deflation_projected_back_to_input_space": bool(deflation_iterations),
}

if "time_horizon_months" not in prepared_metadata:
    st.error("The prepared metadata has no `time_horizon_months` field to build the target from.")
    st.stop()
embedding_target = log10_time_horizon(prepared_metadata["time_horizon_months"])
finite_target_count = int(np.isfinite(embedding_target).sum())

st.subheader("Embedding")
with st.container(border=True):
    st.markdown(f"**{embedding_method}**")
    deflation_summary = (
        "no LDA deflation"
        if deflation_iterations == 0
        else f"{deflation_iterations} LDA deflation iteration"
        f"{'s' if deflation_iterations != 1 else ''} + inverse PCA"
    )
    st.caption(
        f"{point_count:,} prepared point(s) · {embedding_values.shape[1]:,} activation "
        f"features · {deflation_summary} · {finite_target_count:,} point(s) carry a finite "
        f"`{TARGET_FEATURE}`."
    )
    if point_count < 3:
        st.warning("At least three prepared points are required to fit an embedding.")
        st.stop()
    if finite_target_count < 3:
        st.warning(
            f"Fewer than three points carry a finite `{TARGET_FEATURE}`, so target-alignment "
            "scores will be unavailable. Unconstrained prompts state no horizon."
        )
    embedding_mode = st.segmented_control(
        "Embedding model",
        ["Fit new", "Use saved"],
        default="Fit new",
        key="manifold_model_mode",
        persist_state="page",
    )

manifold_result: ManifoldFitResult | None = None
if embedding_method in {"PCA", "PLS"} and embedding_mode == "Use saved":
    with st.container(border=True):
        st.subheader(f"Saved {embedding_method} embedding")
        st.warning(
            "Only load model files you trust. Joblib and pickle artifacts can execute code "
            "when opened."
        )
        upload = st.file_uploader(
            f"Saved {embedding_method} model",
            type=["joblib", "pkl", "pickle"],
            key="linear_model_upload",
            help=f"Choose an {embedding_method} artifact downloaded from Activation Atlas.",
        )
        upload_bytes = upload.getvalue() if upload is not None else None
        upload_digest = sha256(upload_bytes).hexdigest() if upload_bytes is not None else None
        already_loaded = (
            upload_digest is not None
            and upload_digest == st.session_state.get("loaded_linear_digest")
            and embedding_method == st.session_state.get("loaded_linear_method")
        )
        if st.button(
            f"Load uploaded {embedding_method}",
            type="primary",
            icon=":material/upload_file:",
            width="stretch",
            disabled=upload_bytes is None or already_loaded,
            key="load_linear_model_button",
        ):
            try:
                loader = load_pca_model if embedding_method == "PCA" else load_pls_model
                loaded_model, loaded_provenance = loader(upload_bytes)
            except ValueError as exc:
                st.error(str(exc))
            else:
                st.session_state.loaded_linear_digest = upload_digest
                st.session_state.loaded_linear_model = loaded_model
                st.session_state.loaded_linear_provenance = loaded_provenance
                st.session_state.loaded_linear_filename = upload.name
                st.session_state.loaded_linear_method = embedding_method

        loaded_estimator = st.session_state.get("loaded_linear_model")
        if loaded_estimator is None or st.session_state.get("loaded_linear_method") != embedding_method:
            st.info(f"Choose a saved model and click **Load uploaded {embedding_method}**.")
            st.stop()
        loaded_provenance = st.session_state.get("loaded_linear_provenance", {})
        saved_deflation_iterations = loaded_provenance.get("lda_deflation_iterations")
        if (
            saved_deflation_iterations is not None
            and int(saved_deflation_iterations) != deflation_iterations
        ):
            st.error(
                f"This saved {embedding_method} model was fitted after "
                f"{int(saved_deflation_iterations)} LDA deflation iterations. Select the same "
                "preprocessing count before projecting with it."
            )
            st.stop()
        n_components = int(loaded_estimator.components_.shape[0])
        feature_count = int(loaded_estimator.components_.shape[1])
        if feature_count != embedding_values.shape[1]:
            st.error(
                f"The loaded {embedding_method} expects {feature_count:,} activation features, "
                f"but the prepared slice has {embedding_values.shape[1]:,}."
            )
            st.stop()
        embedding = np.asarray(loaded_estimator.transform(embedding_values), dtype=np.float64)
        explained_variance = np.asarray(
            loaded_estimator.explained_variance_ratio_, dtype=np.float64
        )
        current_metrics = target_alignment_metrics(embedding, embedding_target)
        current_metrics.update(
            _structure_metrics(
                embedding_values,
                embedding,
                quality_neighbors=10,
                max_quality_points=2_000,
                random_state=42,
            )
        )
        current_metrics.update(
            {
                "feature_count": feature_count,
                "component_count": n_components,
                "standardized": bool(
                    embedding_method == "PLS" and getattr(loaded_estimator, "scale", False)
                ),
                "explained_variance_ratio": explained_variance,
                "cumulative_variance_ratio": np.cumsum(explained_variance),
                "retained_variance_fraction": float(np.sum(explained_variance)),
                "direction_source": "loaded",
            }
        )
        linear_model = LinearEmbeddingModel(
            estimator=loaded_estimator,
            method=embedding_method,
            component_count=n_components,
            feature_count=feature_count,
            parameters={
                "method": embedding_method,
                "kernel": "linear",
                "supervised": embedding_method == "PLS",
                "target": TARGET_FEATURE if embedding_method == "PLS" else None,
            },
            training_embedding=embedding,
            training_target=embedding_target,
            explained_variance_ratio=explained_variance,
            retained_variance_fraction=float(np.sum(explained_variance)),
            metrics=current_metrics,
        )
        manifold_result = ManifoldFitResult(
            model=linear_model,  # type: ignore[arg-type]
            embedding=embedding,
            target=embedding_target,
            metrics=current_metrics,
        )
        random_state = int(loaded_provenance.get("random_state", 42))
        standardize = current_metrics["standardized"]
        st.success(
            f"{st.session_state.get('loaded_linear_filename', f'Saved {embedding_method}')} · "
            f"{n_components:,} components"
        )
        st.button(
            f"Forget loaded {embedding_method}",
            icon=":material/delete:",
            on_click=forget_loaded_linear_embedding,
        )
    st.session_state.manifold_result = manifold_result
    st.session_state.manifold_key = (
        "loaded",
        embedding_method,
        upload_digest,
        prepared_key,
        deflation_preprocessing_key,
    )
elif embedding_mode == "Use saved":
    with st.container(border=True):
        st.subheader("Saved embedding")
        st.warning(
            "Only load model files you trust. Joblib and pickle artifacts can execute code "
            "when opened."
        )
        upload = st.file_uploader(
            "Saved Kernel PCA model",
            type=["joblib"],
            key="manifold_model_upload",
            help="Choose a manifold artifact downloaded from this app.",
        )
        upload_bytes = upload.getvalue() if upload is not None else None
        upload_digest = sha256(upload_bytes).hexdigest() if upload_bytes is not None else None
        already_loaded = (
            upload_digest is not None
            and upload_digest == st.session_state.get("loaded_manifold_digest")
            and st.session_state.get("loaded_manifold_model") is not None
        )
        if st.button(
            "Load uploaded embedding",
            type="primary",
            icon=":material/upload_file:",
            width="stretch",
            disabled=upload_bytes is None or already_loaded,
            key="load_manifold_model_button",
        ):
            try:
                loaded_model, loaded_provenance = load_manifold_model(upload_bytes)
            except ValueError as exc:
                st.error(str(exc))
            else:
                st.session_state.loaded_manifold_digest = upload_digest
                st.session_state.loaded_manifold_model = loaded_model
                st.session_state.loaded_manifold_provenance = loaded_provenance
                st.session_state.loaded_manifold_filename = upload.name

        loaded_manifold = st.session_state.get("loaded_manifold_model")
        if loaded_manifold is None:
            st.info("Choose a saved embedding and click **Load uploaded embedding**.")
            st.stop()

        loaded_provenance = st.session_state.get("loaded_manifold_provenance", {})
        saved_deflation_iterations = loaded_provenance.get("lda_deflation_iterations")
        if (
            saved_deflation_iterations is not None
            and int(saved_deflation_iterations) != deflation_iterations
        ):
            st.error(
                "This saved Kernel PCA model was fitted after "
                f"{int(saved_deflation_iterations)} LDA deflation iterations. Select the same "
                "preprocessing count before projecting with it."
            )
            st.stop()
        st.success(
            f"{st.session_state.get('loaded_manifold_filename', 'Saved embedding')} · "
            f"{loaded_manifold.component_count:,} components · "
            f"{loaded_manifold.parameters['kernel']} kernel"
        )
        st.button(
            "Forget loaded embedding",
            icon=":material/delete:",
            on_click=forget_loaded_manifold,
            key="forget_manifold_button",
        )
        if loaded_manifold.feature_count != int(embedding_values.shape[1]):
            st.error(
                f"The loaded embedding expects {loaded_manifold.feature_count:,} activation "
                f"features, but the prepared slice has {embedding_values.shape[1]:,}."
            )
            st.stop()
        saved_layer = loaded_provenance.get("layer_component")
        if saved_layer is not None and saved_layer != component:
            st.warning(f"This model was saved for {saved_layer!r}, not {component!r}.")

        n_components = int(loaded_manifold.component_count)
        standardize = bool(loaded_manifold.scaler is not None)
        random_state = int(loaded_provenance.get("random_state", 42))
        try:
            with st.spinner("Projecting with the loaded embedding…"):
                embedding = loaded_manifold.transform(embedding_values)
        except ValueError as exc:
            st.error(f"The loaded embedding could not project these points: {exc}")
            st.stop()
        # A loaded model carries its own training diagnostics. Only the coordinates transfer
        # to the current points, so its stored metrics describe the original training data.
        manifold_result = ManifoldFitResult(
            model=loaded_manifold,
            embedding=embedding,
            target=embedding_target,
            metrics={**loaded_manifold.metrics, "direction_source": "loaded"},
            warnings=[],
        )
        st.caption(
            "Explained variance and structure scores describe the loaded model's original "
            "training data, not the points shown here."
        )
    st.session_state.manifold_result = manifold_result
    st.session_state.manifold_key = (
        "loaded",
        upload_digest,
        prepared_key,
        deflation_preprocessing_key,
    )
else:
    with st.container(border=True):
        with st.form("manifold_fit_form"):
            st.markdown("**Model controls**")
            manifold_parameters = (
                kernel_pca_controls() if embedding_method == ALGORITHM_LABEL else {}
            )
            if embedding_method == "PCA":
                st.caption("PCA centers activations and orders linear directions by variance.")
            elif embedding_method == "PLS":
                st.caption(f"PLS is supervised by `{TARGET_FEATURE}`.")
            st.divider()
            st.markdown("**Embedding and scoring**")
            common_columns = st.columns(4)
            eligible_component_points = fit_point_count
            if embedding_method == "PLS":
                eligible_mask = np.isfinite(embedding_target)
                if fit_row_mask is not None:
                    eligible_mask &= fit_row_mask
                eligible_component_points = int(eligible_mask.sum())
            maximum_components = max(
                1, min(eligible_component_points - 1, embedding_values.shape[1])
            )
            component_default = min(3, maximum_components)
            clamp_integer_widget_state(
                "manifold_components",
                minimum=1,
                maximum=maximum_components,
                default=component_default,
            )
            n_components = int(
                common_columns[0].number_input(
                    "Components",
                    min_value=1,
                    max_value=maximum_components,
                    value=component_default,
                    key="manifold_components",
                    help="Number of embedding coordinates to compute.",
                    persist_state="page",
                )
            )
            if embedding_method == ALGORITHM_LABEL:
                standardize = common_columns[1].checkbox(
                    "Standardize features",
                    value=True,
                    key="manifold_standardize",
                    help=(
                        "Z-score each activation dimension first. Recommended: kernels are "
                        "distance-based, so unscaled features let a few dimensions dominate."
                    ),
                    persist_state="page",
                )
                pls_scale = True
            elif embedding_method == "PLS":
                pls_scale = common_columns[1].checkbox(
                    "Scale activations and target",
                    value=False,
                    key="manifold_pls_scale",
                    persist_state="page",
                )
                standardize = pls_scale
            else:
                common_columns[1].caption("PCA centers features automatically.")
                standardize = False
                pls_scale = True
            quality_neighbors = int(
                common_columns[2].number_input(
                    "Quality neighbors",
                    min_value=2,
                    max_value=max(2, min(100, fit_point_count - 2)),
                    value=min(10, max(2, fit_point_count - 2)),
                    key="manifold_quality_neighbors",
                    help=(
                        "Neighborhood size used for trustworthiness, continuity, and the "
                        "neighborhood target error."
                    ),
                    persist_state="page",
                )
            )
            random_state = int(
                common_columns[3].number_input(
                    "Random seed",
                    min_value=0,
                    max_value=2_147_483_647,
                    value=42,
                    key="manifold_seed",
                    persist_state="page",
                )
            )
            full_variance_spectrum = False
            if embedding_method == ALGORITHM_LABEL:
                full_variance_spectrum = st.checkbox(
                    "Report variance against the full spectrum",
                    value=False,
                    key="manifold_full_spectrum",
                    help=(
                        "By default the ratios are shares among retained components. This "
                        "computes every eigenvalue to report the retained share of total kernel "
                        "variance, at the cost of a full N×N decomposition."
                    ),
                    persist_state="page",
                )
            pls_max_iter = 500
            pls_tolerance = 1e-6
            if embedding_method == "PLS":
                advanced_columns = st.columns(2)
                pls_max_iter = int(
                    advanced_columns[0].number_input(
                        "Maximum iterations",
                        min_value=1,
                        max_value=100_000,
                        value=500,
                        step=100,
                        key="manifold_pls_max_iter",
                        persist_state="page",
                    )
                )
                pls_tolerance = float(
                    advanced_columns[1].number_input(
                        "Convergence tolerance",
                        min_value=1e-12,
                        max_value=1e-1,
                        value=1e-6,
                        format="%.1e",
                        key="manifold_pls_tolerance",
                        persist_state="page",
                    )
                )
            if len(fit_folder_values) > 1:
                st.divider()
                st.markdown("**Fit scope**")
                st.multiselect(
                    "Fit on source folders",
                    fit_folder_values,
                    default=fit_folder_values,
                    key="manifold_fit_folders",
                    help=(
                        "Restrict which folders define the coordinate system. Every prepared "
                        "point is still projected and plotted; the excluded folders are simply "
                        "held out of the fit and land as out-of-sample points."
                    ),
                )
                st.caption(
                    "Clearing a folder here does not remove its points from the plot. Use the "
                    "sidebar filters to drop rows entirely."
                )
                st.checkbox(
                    "Enable folder-balanced subsampling",
                    value=False,
                    key="manifold_balance_folders",
                    help=(
                        "When enabled, deterministically subsample every fitted folder to the "
                        "smallest folder's point count before fitting. When disabled, "
                        "fit on every eligible point, so larger folders contribute more strongly. "
                        "Subsampled rows are still projected and plotted."
                    ),
                    persist_state="page",
                )
                if "<averaged>" in fit_folder_values:
                    st.warning(
                        f"Some aggregated points span several folders and are labelled "
                        f"`<averaged>`. Add `{SOURCE_FOLDER_FIELD}` to **Aggregate by** in the "
                        "sidebar to keep each folder's points separate.",
                        icon=":material/warning:",
                    )
            fit_submitted = st.form_submit_button(
                "Fit / update embedding",
                type="primary",
                icon=":material/blur_on:",
                width="stretch",
            )

        if embedding_method == ALGORITHM_LABEL:
            fit_options = {
                "method": embedding_method,
                "n_components": n_components,
                "parameters": manifold_parameters,
                "standardize": standardize,
                "random_state": random_state,
                "quality_neighbors": quality_neighbors,
                "full_variance_spectrum": full_variance_spectrum,
            }
        else:
            fit_options = {
                "method": embedding_method,
                "n_components": n_components,
                "random_state": random_state,
                "quality_neighbors": quality_neighbors,
                "pls_scale": pls_scale,
                "pls_max_iter": pls_max_iter,
                "pls_tolerance": pls_tolerance,
            }
        data_digest = sha256(np.ascontiguousarray(embedding_values).tobytes()).hexdigest()
        manifold_key = (
            prepared_key,
            data_digest,
            MANIFOLD_MODEL_ARTIFACT_VERSION,
            repr(sorted(fit_options.items(), key=lambda item: item[0])),
            tuple(selected_fit_folders) if restrict_fit_folders else None,
            balance_fit_folders,
        )
        if balance_fit_folders:
            st.caption(
                f"Fitting on {fit_point_count:,} folder-balanced point(s) · every fitted folder "
                f"contributes {fit_point_count // max(len(selected_fit_folders), 1):,} point(s) "
                "and the rest are projected into the fitted coordinates."
            )
        elif restrict_fit_folders:
            st.caption(
                f"Fitting on {fit_point_count:,} point(s) from "
                f"{len(selected_fit_folders):,} of {len(fit_folder_values):,} source folders · "
                f"{point_count - fit_point_count:,} held-out point(s) will be projected into "
                "the fitted coordinates."
            )
        refit_after_deflation_change = bool(
            st.session_state.pop("refit_after_deflation_change", False)
        )
        fit_requested = fit_submitted or refit_after_deflation_change
        if fit_requested and restrict_fit_folders and fit_point_count < 3:
            st.error(
                "At least three points are required to fit an embedding. The selected source "
                "folders contain "
                f"{fit_point_count:,}."
            )
            fit_requested = False
        if fit_requested:
            try:
                spinner_text = (
                    f"Recomputing {embedding_method} after LDA deflation change…"
                    if refit_after_deflation_change
                    else f"Fitting {embedding_method}…"
                )
                with st.spinner(spinner_text):
                    fitted = fit_manifold_cached(
                        embedding_values, embedding_target, fit_options, fit_row_mask
                    )
            except (ValueError, MemoryError, RuntimeError) as exc:
                st.session_state.pop("manifold_result", None)
                st.session_state.pop("manifold_key", None)
                st.error(f"Embedding failed: {exc}")
            else:
                st.session_state.manifold_result = fitted
                st.session_state.manifold_key = manifold_key

        manifold_result: ManifoldFitResult | None = None
        if st.session_state.get("manifold_key") == manifold_key:
            manifold_result = st.session_state.get("manifold_result")
        else:
            # The displayed projection belongs to the superseded fit, so withdraw it rather than
            # leaving coordinates behind that no longer describe the current selection.
            st.session_state.pop("projection", None)
            st.session_state.pop("details", None)
            if st.session_state.get("manifold_result") is not None:
                st.info(
                    "The prepared points, method, or controls changed. Click "
                    "**Fit / update embedding** to refresh the projection."
                )
            else:
                st.info("Choose the controls above, then fit the first embedding.")

if manifold_result is None:
    st.stop()

try:
    if embedding_method == ALGORITHM_LABEL:
        projection, details = manifold_projection_frame(
            prepared_metadata, manifold_result, details=embedding_preparation_details
        )
    else:
        projection = prepared_metadata.copy().reset_index(drop=True)
        direction_prefix = "PC" if embedding_method == "PCA" else "PLS"
        for index in range(manifold_result.model.component_count):
            projection[f"{direction_prefix}{index + 1}"] = manifold_result.embedding[:, index]
        projection[TARGET_FEATURE] = manifold_result.target
        details = {
            **embedding_preparation_details,
            **manifold_result.metrics,
            "direction_method": embedding_method,
            "direction_source": manifold_result.metrics.get("direction_source", "fitted"),
            "direction_target": TARGET_FEATURE if embedding_method == "PLS" else None,
            "manifold_parameters": dict(manifold_result.model.parameters),
        }
except ValueError as exc:
    st.error(f"The embedding could not be aligned with the prepared metadata: {exc}")
    st.stop()
if fit_row_mask is not None and len(fit_row_mask) == len(projection):
    # Exposed as an ordinary metadata column so it can be used to color or hover the plot.
    projection["manifold_fit_role"] = np.where(fit_row_mask, "fitted", "projected")
st.session_state.projection = projection
st.session_state.details = details

cumulative_variance = np.asarray(details["cumulative_variance_ratio"], dtype=np.float64)
retained_fraction = details.get("retained_variance_fraction")
captured_variance = (
    float(retained_fraction)
    if retained_fraction is not None
    else (float(cumulative_variance[-1]) if len(cumulative_variance) else float("nan"))
)

metric_columns = st.columns(5)
metric_columns[0].metric("Source samples", f"{details['loaded_samples']:,}")
metric_columns[1].metric("Embedded points", f"{details['analysis_rows']:,}")
metric_columns[2].metric("Activation width", f"{details['feature_count']:,}")
metric_columns[3].metric(
    "X variance represented" if embedding_method == "PLS" else "Variance captured",
    _format_metric(captured_variance, percent=True),
    help=(
        f"Cumulative variance represented by the retained {embedding_method} components. "
        + (
            "Measured against total activation variance."
            if retained_fraction is not None
            else "Measured among the retained components only, so it sums to 100%. Enable "
            "**Report variance against the full spectrum** for the share of total variance."
        )
    ),
)
metric_columns[4].metric(
    "Target R²",
    _format_metric(details["target_linear_r2"]),
    help=(
        f"Fraction of `{TARGET_FEATURE}` explained by a linear readout of the embedding "
        "coordinates. It asks whether the manifold laid the time horizon out in a directly "
        "usable way. Fitted and scored on the same points, so it is not a predictive score."
    ),
)
if retained_fraction is None and embedding_method == ALGORITHM_LABEL:
    st.caption(
        "Explained variance is reported among the retained components. Kernel-space variance "
        "is not activation-space variance: it describes how the kernel's geometry is "
        "distributed, not how much of the original signal was kept."
    )

quality_columns = st.columns(4)
quality_columns[0].metric(
    "Target rank correlation",
    _format_metric(details["target_spearman"]),
    help=(
        "Strongest single-axis Spearman correlation with the target. Catches monotone but "
        "curved layouts that the linear R² understates."
    ),
)
quality_columns[1].metric(
    "Neighborhood target error",
    _format_metric(details["target_neighborhood_error"]),
    help=(
        "Mean absolute target difference among embedded neighbors, in target standard "
        "deviations. Lower is better: neighbors should share a similar horizon."
    ),
)
quality_columns[2].metric(
    "Trustworthiness",
    _format_metric(details["trustworthiness"]),
    help=(
        "Fraction of embedded neighbors that were genuine activation-space neighbors. "
        "Low values mean the embedding invented proximity."
    ),
)
quality_columns[3].metric(
    "Continuity",
    _format_metric(details["continuity"]),
    help="Whether true activation-space neighbors stayed together in the embedding.",
)
if balance_fit_folders:
    st.info(
        f"The coordinate system was fitted on {fit_point_count:,} point(s) drawn in equal "
        "numbers from each fitted source folder, so no folder dominates the components. All "
        f"{point_count:,} prepared point(s) are plotted, and the variance, structure, and "
        "target scores above describe the balanced fit sample only. Color by "
        "`manifold_fit_role` to separate fitted from projected points.",
        icon=":material/balance:",
    )
elif restrict_fit_folders:
    st.info(
        f"The coordinate system was fitted on {fit_point_count:,} point(s) from "
        f"{', '.join(f'`{folder}`' for folder in selected_fit_folders)}. All "
        f"{point_count:,} prepared point(s) are plotted, and the variance, structure, and "
        "target scores above describe the fitted folders only. Color by "
        "`manifold_fit_role` to separate fitted from held-out points.",
        icon=":material/filter_alt:",
    )
if details.get("quality_subsampled"):
    st.caption(
        f"Structure scores use a {details['quality_points']:,}-point random subsample; "
        "the pairwise-distance matrices they need grow with the square of the point count."
    )
if details.get("baseline_subtracted"):
    dropped = details.get("baseline_dropped_rows", 0)
    st.info(
        f"Unconstrained baselines subtracted · {details['baseline_groups']:,} "
        f"`{details['baseline_field']}` groups from {details['baseline_rows']:,} unconstrained "
        + (
            f"point(s) · {dropped:,} point(s) dropped without a matching baseline."
            if dropped
            else "point(s) · every point had a matching baseline."
        ),
        icon=":material/exposure_neg_1:",
    )
if manifold_result.warnings:
    st.warning(" ".join(manifold_result.warnings))
if manifold_result.model.parameters.get("supervised"):
    st.warning(
        f"This embedding was supervised by `{TARGET_FEATURE}`, so its target scores are "
        "optimistic by construction and are not evidence that the horizon is recoverable "
        "from the activations alone.",
        icon=":material/warning:",
    )

st.subheader(
    "Feature-space eigenvalue spectrum"
    if embedding_method == ALGORITHM_LABEL
    else f"{embedding_method} component spectrum"
)
with st.container(border=True):
    if embedding_method == ALGORITHM_LABEL:
        spectrum = eigenvalue_spectrum_table(manifold_result)
    else:
        ratios = np.asarray(
            manifold_result.model.explained_variance_ratio, dtype=np.float64
        )
        if embedding_method == "PCA":
            spectrum_values = np.asarray(
                manifold_result.model.estimator.explained_variance_, dtype=np.float64
            )
            prefix = "PC"
        else:
            spectrum_values = ratios
            prefix = "PLS"
        spectrum = pd.DataFrame(
            {
                "component": [f"{prefix}{index + 1}" for index in range(len(ratios))],
                "rank": np.arange(1, len(ratios) + 1),
                "eigenvalue": spectrum_values,
                "explained_variance": ratios,
                "cumulative_variance": np.cumsum(ratios),
            }
        )
    spectrum_controls = st.columns([1, 1, 2])
    log_scale = spectrum_controls[0].toggle(
        "Log scale",
        value=True,
        key="spectrum_log_scale",
        help="Eigenvalues usually decay by orders of magnitude, so a log axis keeps the tail readable.",
    )
    show_cumulative = spectrum_controls[1].toggle(
        "Cumulative share",
        value=True,
        key="spectrum_cumulative",
        help="Overlay the cumulative share of retained kernel-space variance.",
    )
    spectrum_figure = px.bar(
        spectrum,
        x="component",
        y="eigenvalue",
        hover_data=["rank", "explained_variance", "cumulative_variance"],
        color_discrete_sequence=["#2f6f5e"],
    )
    spectrum_figure.update_yaxes(
        title_text=(
            "Kernel eigenvalue"
            if embedding_method == ALGORITHM_LABEL
            else "Component variance"
        ),
        type="log" if log_scale else "linear",
    )
    if show_cumulative:
        spectrum_figure.add_scatter(
            x=spectrum["component"],
            y=spectrum["cumulative_variance"],
            mode="lines+markers",
            name="Cumulative share",
            yaxis="y2",
            line={"color": "#c2703d", "width": 2},
        )
        spectrum_figure.update_layout(
            yaxis2={
                "title": "Cumulative variance share",
                "overlaying": "y",
                "side": "right",
                "range": [0, 1.02],
                "showgrid": False,
            }
        )
    spectrum_figure.update_layout(
        height=380,
        margin={"l": 12, "r": 12, "t": 28, "b": 12},
        paper_bgcolor="white",
        plot_bgcolor="white",
        font={"family": "Inter, ui-sans-serif, system-ui", "color": "#17201d"},
        xaxis_title="Component",
        showlegend=show_cumulative,
    )
    st.plotly_chart(spectrum_figure, width="stretch", config={"displaylogo": False})
    if embedding_method == ALGORITHM_LABEL:
        st.caption(
            "Eigenvalues of the centered kernel matrix for the current fit. Only retained "
            "components are shown."
        )
    elif embedding_method == "PLS":
        st.caption(
            "PLS is supervised by log-time horizon; bars show activation-space variance "
            "represented by each latent score."
        )
    else:
        st.caption("PCA eigenvalues and their share of total activation-space variance.")

if embedding_method == ALGORITHM_LABEL:
    coordinate_fields = embedding_fields(n_components)
elif embedding_method == "PCA":
    coordinate_fields = [f"PC{index + 1}" for index in range(n_components)]
else:
    coordinate_fields = [f"PLS{index + 1}" for index in range(n_components)]
axis_fields = [*coordinate_fields]
metadata_fields = sorted(
    column for column in projection if column not in {*coordinate_fields, "sample_index"}
)
color_fields = [
    TARGET_FEATURE,
    *[field for field in metadata_fields if field != TARGET_FEATURE],
]

st.subheader("Projection")
controls = st.columns([1.1, 1.25, 1.25, 2])
plot_mode = controls[0].segmented_control(
    "Plot", ["2D", "3D"], default="3D" if len(axis_fields) >= 3 else "2D"
)
required_axes = 3 if plot_mode == "3D" else 2
if len(axis_fields) < required_axes:
    st.warning(
        f"At least {required_axes} embedding coordinates are required for a {plot_mode} plot. "
        "Increase **Components** and refit."
    )
    st.stop()
if st.session_state.get("manifold_x_axis") not in axis_fields:
    st.session_state["manifold_x_axis"] = axis_fields[0]
x_component = controls[1].selectbox(
    "X axis",
    axis_fields,
    key="manifold_x_axis",
    persist_state="page",
)
y_choices = [field for field in axis_fields if field != x_component]
if st.session_state.get("manifold_y_axis") not in y_choices:
    st.session_state["manifold_y_axis"] = y_choices[0]
y_component = controls[2].selectbox(
    "Y axis",
    y_choices,
    key="manifold_y_axis",
    persist_state="page",
)
z_component = None
if plot_mode == "3D":
    z_choices = [field for field in axis_fields if field not in {x_component, y_component}]
    if st.session_state.get("manifold_z_axis") not in z_choices:
        st.session_state["manifold_z_axis"] = z_choices[0]
    z_component = controls[3].selectbox(
        "Z axis",
        z_choices,
        key="manifold_z_axis",
        persist_state="page",
    )
else:
    controls[3].caption("Choose any two embedding coordinates for the plane.")

plot_controls = st.columns([1.1, 1, 1, 2])
color_field = plot_controls[0].selectbox("Color by", color_fields)
point_size = plot_controls[1].slider(
    "Point size",
    min_value=1,
    max_value=20,
    value=2,
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
    default=[
        field
        for field in ["sample_index", "time_horizon_months", "task"]
        if field in projection
    ],
)

stored_visual_fields = st.session_state.get("visual_filter_fields", [])
valid_visual_fields = [field for field in stored_visual_fields if field in metadata_fields]
if valid_visual_fields != stored_visual_fields:
    st.session_state.visual_filter_fields = valid_visual_fields

visual_filters = {}
with st.popover(
    "Filter visible points",
    icon=":material/filter_alt:",
    help="These filters change the chart only; they do not refit the embedding.",
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
            "Filters use each embedded aggregate's retained metadata. Include a field in "
            "Aggregate by when it must define the groups exactly."
        )

target_range = st.container()
with target_range:
    finite_target = embedding_target[np.isfinite(embedding_target)]
    restrict_target = st.toggle(
        "Restrict visible time horizons",
        value=False,
        key="manifold_target_filter",
        help=(
            f"Show only points whose `{TARGET_FEATURE}` falls inside a range. Points with no "
            "horizon are controlled separately."
        ),
        persist_state="page",
    )
    target_bounds: tuple[float, float] | None = None
    include_missing_target = True
    if restrict_target and len(finite_target):
        low = float(np.min(finite_target))
        high = float(np.max(finite_target))
        if np.isclose(low, high):
            low, high = low - 0.5, high + 0.5
        range_columns = st.columns([3, 1])
        target_bounds = range_columns[0].slider(
            f"Visible {TARGET_FEATURE}",
            low,
            high,
            (low, high),
            step=max((high - low) / 100.0, 1e-6),
            key="manifold_target_range",
            persist_state="page",
        )
        include_missing_target = range_columns[1].checkbox(
            "Keep horizon-free points",
            value=True,
            key="manifold_target_keep_missing",
            help="Unconstrained prompts state no horizon and fall outside any range.",
            persist_state="page",
        )
    elif restrict_target:
        st.caption("No point carries a finite time horizon, so there is no range to restrict.")

visual_mask = metadata_filter_mask(projection, visual_filters)
if target_bounds is not None:
    projected_target = projection[TARGET_FEATURE].to_numpy(dtype=np.float64)
    finite_projected = np.isfinite(projected_target)
    inside = finite_projected & (projected_target >= target_bounds[0])
    inside &= projected_target <= target_bounds[1]
    if include_missing_target:
        inside |= ~finite_projected
    visual_mask = visual_mask & inside
plot_data = projection.loc[visual_mask].copy()
st.caption(f"Visible {len(plot_data):,} of {len(projection):,} embedded points.")

if plot_data.empty:
    st.warning("No embedded points match the visual filters.")
else:
    show_points = st.toggle(
        "Show points",
        value=True,
        key="plot_show_points",
        help="Show or hide the embedded activation points.",
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
    if plot_mode == "3D":
        figure = px.scatter_3d(**common, z=z_component)
        figure.update_traces(marker={"size": point_size}, visible=show_points)
    else:
        figure = px.scatter(**common)
        figure.update_traces(
            marker={"size": point_size, "line": {"width": 0.35, "color": "white"}},
            visible=show_points,
        )
    figure.update_layout(
        height=680,
        margin={"l": 12, "r": 12, "t": 28, "b": 12},
        paper_bgcolor="white",
        plot_bgcolor="white",
        font={"family": "Inter, ui-sans-serif, system-ui", "color": "#17201d"},
        legend_title_text=color_field,
    )
    st.plotly_chart(figure, width="stretch", config={"displaylogo": False})

st.subheader("Horizon regression")
with st.container(border=True):
    st.caption(
        f"Fit a polynomial ridge model predicting `{TARGET_FEATURE}` from the embedding "
        f"coordinates. The train/test split is **disjoint by `{GROUP_FEATURE}`**: every "
        "point sharing a task lands wholly in one side. Each task recurs at many horizons, "
        "so a random split would leave near-duplicates of each test point in training."
    )
    regression_available = GROUP_FEATURE in projection
    if not regression_available:
        st.error(
            f"The prepared metadata has no `{GROUP_FEATURE}` field, so a task-disjoint "
            "split cannot be built."
        )
    else:
        regression_groups = projection[GROUP_FEATURE].astype(str)
        distinct_tasks = int(regression_groups.nunique())
        finite_rows = int(np.isfinite(projection[TARGET_FEATURE].to_numpy(dtype=float)).sum())
        st.caption(
            f"{distinct_tasks:,} distinct task(s) · {finite_rows:,} point(s) with a finite "
            f"`{TARGET_FEATURE}`."
        )
        if GROUP_FEATURE not in analysis_aggregation_fields:
            st.warning(
                f"`{GROUP_FEATURE}` is not one of the aggregation fields, so an aggregated "
                "point can mix tasks and carries only its first row's task label. Add "
                f"`{GROUP_FEATURE}` to **Aggregate by** for a clean task-disjoint split.",
                icon=":material/warning:",
            )
        if distinct_tasks < 2:
            st.warning(
                "A task-disjoint split needs at least two distinct tasks. Widen the metadata "
                "filters to include more."
            )
        else:
            regression_mode = st.segmented_control(
                "Regression model",
                ["Fit new", "Use saved"],
                default="Fit new",
                key="regression_model_mode",
                persist_state="page",
            )
            # How many leading embedding components feed the regression. The embedding may
            # carry more coordinates than the model should consume, and a saved embedding
            # fixes its own width, so this stays adjustable in both modes.
            clamp_integer_widget_state(
                "regression_feature_count",
                minimum=1,
                maximum=len(coordinate_fields),
                default=len(coordinate_fields),
            )
            regression_feature_count = int(
                st.number_input(
                    "Feature components",
                    min_value=1,
                    max_value=len(coordinate_fields),
                    value=len(coordinate_fields),
                    key="regression_feature_count",
                    help=(
                        "Number of leading embedding coordinates the regression consumes. "
                        f"The current embedding provides {len(coordinate_fields)}."
                    ),
                    persist_state="page",
                )
            )
            regression_features = list(coordinate_fields[:regression_feature_count])
            st.caption(f"Regressing on {', '.join(regression_features)}.")
            regression_coordinates = projection[regression_features].to_numpy(dtype=np.float64)
            regression_target = projection[TARGET_FEATURE].to_numpy(dtype=np.float64)

            if regression_mode == "Use saved":
                st.warning(
                    "Only load model files you trust. Joblib and pickle artifacts can "
                    "execute code when opened."
                )
                regression_upload = st.file_uploader(
                    "Saved regression model",
                    type=["joblib"],
                    key="regression_model_upload",
                    help="Choose a regression artifact downloaded from this app.",
                )
                regression_bytes = (
                    regression_upload.getvalue() if regression_upload is not None else None
                )
                regression_digest = (
                    sha256(regression_bytes).hexdigest()
                    if regression_bytes is not None
                    else None
                )
                regression_loaded = (
                    regression_digest is not None
                    and regression_digest == st.session_state.get("loaded_regression_digest")
                    and st.session_state.get("loaded_regression_model") is not None
                )
                if st.button(
                    "Load uploaded regression",
                    type="primary",
                    icon=":material/upload_file:",
                    width="stretch",
                    disabled=regression_bytes is None or regression_loaded,
                    key="load_regression_button",
                ):
                    try:
                        loaded_regression, regression_provenance = load_regression_model(
                            regression_bytes
                        )
                    except ValueError as exc:
                        st.error(str(exc))
                    else:
                        st.session_state.loaded_regression_digest = regression_digest
                        st.session_state.loaded_regression_model = loaded_regression
                        st.session_state.loaded_regression_provenance = regression_provenance
                        st.session_state.loaded_regression_filename = regression_upload.name

                loaded_regression = st.session_state.get("loaded_regression_model")
                if loaded_regression is None:
                    st.info(
                        "Choose a saved regression and click **Load uploaded regression**."
                    )
                else:
                    regression_provenance = st.session_state.get(
                        "loaded_regression_provenance", {}
                    )
                    st.success(
                        f"{st.session_state.get('loaded_regression_filename', 'Saved model')}"
                        f" · degree {loaded_regression.degree} · alpha "
                        f"{loaded_regression.alpha:g}"
                    )
                    st.button(
                        "Forget loaded regression",
                        icon=":material/delete:",
                        on_click=forget_loaded_regression,
                        key="forget_regression_button",
                    )
                    saved_coordinates = list(loaded_regression.coordinate_features)
                    if saved_coordinates != regression_features:
                        st.error(
                            "This model consumes "
                            f"{', '.join(saved_coordinates)}, but the current selection "
                            f"provides {', '.join(regression_features)}. Set **Feature "
                            f"components** to {len(saved_coordinates)} to match it."
                        )
                    else:
                        try:
                            loaded_scores = evaluate_regression_model(
                                loaded_regression, regression_coordinates, regression_target
                            )
                        except ValueError as exc:
                            st.error(f"The loaded regression could not be scored: {exc}")
                        else:
                            score_columns = st.columns(3)
                            score_columns[0].metric(
                                "R² on current points", _format_metric(loaded_scores["r2"])
                            )
                            score_columns[1].metric(
                                "RMSE on current points",
                                _format_metric(loaded_scores["rmse"]),
                            )
                            score_columns[2].metric(
                                "Scored points", f"{loaded_scores['points']:,}"
                            )
                            st.warning(
                                "These points may overlap the model's original training "
                                "data, which this app cannot verify. Treat the scores as "
                                "held-out only if the current selection is genuinely "
                                "unseen.",
                                icon=":material/warning:",
                            )
            else:
                with st.form("regression_fit_form"):
                    st.markdown("**Model controls**")
                    model_columns = st.columns(3)
                    degree = int(
                        model_columns[0].number_input(
                            "Polynomial degree",
                            min_value=1,
                            max_value=6,
                            value=2,
                            key="regression_degree",
                            help=(
                                "Degree 1 is plain ridge on the coordinates. Higher degrees "
                                "add curvature but grow the term count quickly."
                            ),
                            persist_state="page",
                        )
                    )
                    alpha = 10.0 ** model_columns[1].slider(
                        "log10 ridge alpha",
                        -6.0,
                        6.0,
                        0.0,
                        0.25,
                        key="regression_alpha",
                        help="L2 penalty. Larger values shrink the polynomial coefficients.",
                        persist_state="page",
                    )
                    interaction_only = model_columns[2].checkbox(
                        "Interactions only",
                        value=False,
                        key="regression_interactions",
                        help="Drop pure powers and keep only cross-terms.",
                        persist_state="page",
                    )
                    split_columns = st.columns(3)
                    test_fraction = split_columns[0].slider(
                        "Test fraction (by task)",
                        0.1,
                        0.5,
                        0.25,
                        0.05,
                        key="regression_test_fraction",
                        help="Approximate share of *tasks* held out, not of points.",
                        persist_state="page",
                    )
                    maximum_folds = max(2, min(10, distinct_tasks))
                    cv_folds = int(
                        split_columns[1].number_input(
                            "Cross-validation folds",
                            min_value=0,
                            max_value=maximum_folds,
                            value=0,
                            key="regression_cv_folds",
                            help=(
                                "0 disables it. Folds are grouped by task, so each held-out "
                                "fold contains only unseen tasks."
                            ),
                            persist_state="page",
                        )
                    )
                    regression_seed = int(
                        split_columns[2].number_input(
                            "Random seed",
                            min_value=0,
                            max_value=2_147_483_647,
                            value=42,
                            key="regression_seed",
                            persist_state="page",
                        )
                    )
                    regression_standardize = st.checkbox(
                        "Standardize coordinates",
                        value=True,
                        key="regression_standardize",
                        help=(
                            "Z-score the coordinates before the polynomial expansion so the "
                            "ridge penalty applies evenly across terms."
                        ),
                        persist_state="page",
                    )
                    regression_submitted = st.form_submit_button(
                        "Fit / update regression",
                        type="primary",
                        icon=":material/timeline:",
                        width="stretch",
                    )

                regression_options = {
                    "coordinate_features": tuple(regression_features),
                    "degree": degree,
                    "alpha": alpha,
                    "interaction_only": interaction_only,
                    "standardize": regression_standardize,
                    "test_fraction": test_fraction,
                    "random_state": regression_seed,
                    "cross_validation_folds": cv_folds,
                }
                regression_key = (
                    st.session_state.get("manifold_key"),
                    sha256(
                        np.ascontiguousarray(regression_coordinates).tobytes()
                    ).hexdigest(),
                    repr(sorted(regression_options.items(), key=lambda item: item[0])),
                )
                if regression_submitted:
                    try:
                        with st.spinner("Fitting the polynomial ridge model…"):
                            regression_result = fit_manifold_regression(
                                regression_coordinates,
                                regression_target,
                                regression_groups.tolist(),
                                **regression_options,
                            )
                    except (ValueError, MemoryError) as exc:
                        clear_regression()
                        st.error(f"Regression failed: {exc}")
                    else:
                        st.session_state.regression_result = regression_result
                        st.session_state.regression_key = regression_key

                regression_result: RegressionFitResult | None = None
                if st.session_state.get("regression_key") == regression_key:
                    regression_result = st.session_state.get("regression_result")
                else:
                    clear_regression()
                    st.info(
                        "Choose the controls above, then fit the regression. It refits when "
                        "the embedding or any control changes."
                    )

                if regression_result is not None:
                    metrics = regression_result.metrics
                    score_columns = st.columns(4)
                    score_columns[0].metric(
                        "Train R²",
                        _format_metric(metrics["train_r2"]),
                        help="Fit quality on the tasks the model trained on.",
                    )
                    score_columns[1].metric(
                        "Train RMSE",
                        _format_metric(metrics["train_rmse"]),
                        help=f"In `{TARGET_FEATURE}` units, so 1.0 is one decade.",
                    )
                    score_columns[2].metric(
                        "Test R²",
                        _format_metric(metrics["test_r2"]),
                        help=(
                            "Measured on held-out tasks. Negative means the model does worse "
                            "than predicting the test split's mean horizon."
                        ),
                    )
                    score_columns[3].metric(
                        "Test RMSE",
                        _format_metric(metrics["test_rmse"]),
                        help=f"In `{TARGET_FEATURE}` units, so 1.0 is one decade.",
                    )
                    st.dataframe(
                        regression_scores_table(metrics),
                        hide_index=True,
                        width="stretch",
                        column_config={
                            "R²": st.column_config.NumberColumn(format="%.4f"),
                            "RMSE": st.column_config.NumberColumn(format="%.5g"),
                        },
                    )
                    st.caption(
                        f"{metrics['train_tasks']:,} training task(s) and "
                        f"{metrics['test_tasks']:,} held-out task(s) share no overlap · "
                        f"{metrics['polynomial_terms']:,} polynomial term(s)."
                    )
                    if regression_result.warnings:
                        st.warning(" ".join(regression_result.warnings))

                    predicted_frame = pd.DataFrame(
                        {
                            "actual": np.concatenate(
                                [
                                    regression_result.train_actual,
                                    regression_result.test_actual,
                                ]
                            ),
                            "predicted": np.concatenate(
                                [
                                    regression_result.train_prediction,
                                    regression_result.test_prediction,
                                ]
                            ),
                            "split": ["Train"] * len(regression_result.train_actual)
                            + ["Test"] * len(regression_result.test_actual),
                        }
                    )
                    parity_figure = px.scatter(
                        predicted_frame,
                        x="actual",
                        y="predicted",
                        color="split",
                        opacity=0.75,
                        labels={
                            "actual": f"Actual {TARGET_FEATURE}",
                            "predicted": f"Predicted {TARGET_FEATURE}",
                        },
                        color_discrete_map={"Train": "#7b9acc", "Test": "#d81b60"},
                    )
                    parity_span = [
                        float(predicted_frame[["actual", "predicted"]].to_numpy().min()),
                        float(predicted_frame[["actual", "predicted"]].to_numpy().max()),
                    ]
                    parity_figure.add_shape(
                        type="line",
                        x0=parity_span[0],
                        y0=parity_span[0],
                        x1=parity_span[1],
                        y1=parity_span[1],
                        line={"color": "rgba(120,120,120,0.7)", "dash": "dash"},
                    )
                    parity_figure.update_layout(
                        height=420,
                        margin={"l": 12, "r": 12, "t": 28, "b": 12},
                        paper_bgcolor="white",
                        plot_bgcolor="white",
                        font={
                            "family": "Inter, ui-sans-serif, system-ui",
                            "color": "#17201d",
                        },
                    )
                    st.plotly_chart(
                        parity_figure, width="stretch", config={"displaylogo": False}
                    )
                    st.download_button(
                        "Download regression model",
                        data=lambda: serialize_regression_model(
                            regression_result.model,
                            metadata={
                                "layer_component": component,
                                "cached_position": inspection["positions"][position_index],
                                "aggregation_fields": list(analysis_aggregation_fields),
                                "embedding_method": embedding_method,
                                "embedding_components": n_components,
                                "lda_deflation_iterations": deflation_iterations,
                                "lda_deflation_artifact": embedding_preparation_details[
                                    "lda_deflation_artifact"
                                ],
                            },
                        ),
                        file_name="activation_horizon_regression.joblib",
                        mime="application/octet-stream",
                        icon=":material/download:",
                        on_click="ignore",
                        help=(
                            "Versioned joblib artifact containing the fitted pipeline, the "
                            "coordinate names it consumes, and its split provenance."
                        ),
                    )

with st.expander(f"{embedding_method} details and downloads"):
    if embedding_method == ALGORITHM_LABEL:
        variance = explained_variance_table(manifold_result)
    else:
        variance = pd.DataFrame(
            {
                "component": coordinate_fields,
                "explained_variance": manifold_result.model.explained_variance_ratio,
                "cumulative_variance": np.cumsum(
                    manifold_result.model.explained_variance_ratio
                ),
            }
        )
    st.dataframe(
        variance,
        hide_index=True,
        width="stretch",
        column_config={
            "explained_variance": st.column_config.NumberColumn(format="%.5g"),
            "cumulative_variance": st.column_config.NumberColumn(format="%.5g"),
        },
    )
    if embedding_method == ALGORITHM_LABEL:
        st.caption(
            "Ratios are shares of kernel-space variance rather than activation-space variance."
        )
    elif embedding_method == "PLS":
        st.caption("Ratios describe activation (X) variance represented by each PLS score.")
    else:
        st.caption("Ratios are shares of total activation-space variance explained by PCA.")
    st.write(
        {
            "Method": embedding_method,
            "Target": TARGET_FEATURE if embedding_method != "PCA" else None,
            "Parameters": manifold_result.model.parameters,
            "LDA deflation iterations": deflation_iterations,
            "Standardized features": details["standardized"],
            "Quality neighbors": details["quality_neighbors"],
        }
    )
    axis_correlations = details.get("target_axis_correlations") or []
    if axis_correlations:
        st.dataframe(
            pd.DataFrame(
                {
                    "coordinate": coordinate_fields,
                    "target_rank_correlation": axis_correlations,
                }
            ),
            hide_index=True,
            width="stretch",
            column_config={
                "target_rank_correlation": st.column_config.NumberColumn(format="%.4f")
            },
        )
    st.dataframe(projection_details_table(projection), hide_index=True, width="stretch")
    st.caption("Downloads contain the full embedding, independent of visual filters.")
    safe_component = component.replace("/", "-").replace("\\", "-")
    model_metadata = {
        "layer_component": component,
        "fit_source_folders": (list(selected_fit_folders) if restrict_fit_folders else "all"),
        "fit_folders_balanced": balance_fit_folders,
        "cached_position": inspection["positions"][position_index],
        "aggregation_fields": list(analysis_aggregation_fields),
        "lda_deflation_iterations": deflation_iterations,
        "lda_deflation_artifact": embedding_preparation_details["lda_deflation_artifact"],
        "lda_deflation_artifact_version": deflation_artifact_version,
        "deflation_projected_back_to_input_space": bool(deflation_iterations),
        "standardized": details["standardized"],
        "random_state": random_state,
    }
    if embedding_method == ALGORITHM_LABEL:
        model_data = lambda: serialize_manifold_model(  # noqa: E731
            manifold_result.model, metadata=model_metadata
        )
        model_filename = f"activation_manifold_kernel_pca_{safe_component}.joblib"
    elif embedding_method == "PCA":
        model_data = lambda: serialize_pca_model(  # noqa: E731
            manifold_result.model.estimator, metadata=model_metadata
        )
        model_filename = f"activation_pca_{safe_component}.joblib"
    else:
        model_data = lambda: serialize_pls_model(  # noqa: E731
            manifold_result.model.estimator, metadata=model_metadata
        )
        model_filename = f"activation_pls_{safe_component}.joblib"
    with st.container(horizontal=True):
        st.download_button(
            f"Download {embedding_method} model",
            data=model_data,
            file_name=model_filename,
            mime="application/octet-stream",
            icon=":material/download:",
            on_click="ignore",
            help=(
                "Versioned joblib artifact containing the fitted estimator, feature scaler, "
                "training embedding, quality metrics, and provenance."
            ),
        )
        st.download_button(
            "Download embedded CSV",
            projection.to_csv(index=False).encode("utf-8"),
            file_name=f"activation_{embedding_method.lower().replace(' ', '_')}_embedding.csv",
            mime="text/csv",
            icon=":material/download:",
            on_click="ignore",
            help="Includes every embedding coordinate and the retained prompt metadata.",
        )
    st.warning("Only load model files you trust; joblib and pickle can execute code.")
