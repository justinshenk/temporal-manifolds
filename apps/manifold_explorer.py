"""Streamlit UI for exploring Kernel PCA embeddings of cached activations.

The linear explorer (``apps/activation_explorer.py``) fits PCA and log-time-horizon
supervised PLS. This app is its nonlinear counterpart: it embeds the same fixed activation
slice with Kernel PCA and scores every embedding against the fixed target
``log10_time_horizon_months``.
"""

from __future__ import annotations

import gc
from hashlib import sha256
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
from pandas.api.types import is_bool_dtype, is_numeric_dtype

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
    metadata_filter_choices,
    metadata_filter_mask,
    prepare_analysis_data,
    projection_details_table,
    select_activation_batch_uploads,
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
    embedding_fields,
    explained_variance_table,
    fit_manifold,
    load_manifold_model,
    manifold_projection_frame,
    serialize_manifold_model,
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
    f"with Kernel PCA. Every embedding is scored against `{TARGET_FEATURE}`. Your files stay "
    "in this local app session."
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
        "prepared_key",
        "prepared_matrix",
        "prepared_row_offsets",
        "prepared_metadata",
        "prepared_details",
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


@st.cache_data(max_entries=8, show_spinner=False)
def fit_manifold_cached(
    values: np.ndarray,
    target: np.ndarray,
    options: dict[str, Any],
) -> ManifoldFitResult:
    """Cache one fitted embedding keyed by its data and every control value."""

    return fit_manifold(values, target, **options)


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
        value=True,
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
    # The subtraction toggle is rendered below, so its previous value is read from session
    # state. Unconstrained prompts carry the NOT_APPLICABLE framing, and dropping them by
    # default would leave subtraction with no baselines to compute.
    baseline_framings = (
        [NOT_APPLICABLE] if st.session_state.get("subtract_unconstrained_baseline") else []
    )
    for field in filter_fields:
        values = filter_options[field]
        default_values = (
            [
                framing
                for framing in ("task_available_time", *baseline_framings)
                if framing in values
            ]
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

    st.divider()
    st.header("3 · Embedding method")
    st.markdown(f"**{ALGORITHM_LABEL}**")
    st.caption(ALGORITHM_DESCRIPTION)
    st.info(f"Optimization target: `{TARGET_FEATURE}` (fixed).", icon=":material/target:")

analysis_aggregation_fields = list(aggregation_fields)
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
    tuple((field, tuple(filters[field])) for field in filter_fields),
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

# Materialize the prepared rows once. Every method here is a whole-matrix batch algorithm, so
# unlike incremental PCA there is no streaming path to preserve.
if prepared_matrix is not None:
    embedding_values = np.asarray(prepared_matrix, dtype=np.float64)
else:
    embedding_values = np.asarray(activation_matrix[prepared_row_offsets], dtype=np.float64)

if "time_horizon_months" not in prepared_metadata:
    st.error("The prepared metadata has no `time_horizon_months` field to build the target from.")
    st.stop()
embedding_target = log10_time_horizon(prepared_metadata["time_horizon_months"])
finite_target_count = int(np.isfinite(embedding_target).sum())

st.subheader("Embedding")
with st.container(border=True):
    st.markdown(f"**{ALGORITHM_LABEL}**")
    st.caption(
        f"{point_count:,} prepared point(s) · {embedding_values.shape[1]:,} activation "
        f"features · {finite_target_count:,} point(s) carry a finite `{TARGET_FEATURE}`."
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
if embedding_mode == "Use saved":
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
    st.session_state.manifold_key = ("loaded", upload_digest, prepared_key)
else:
    with st.container(border=True):
        with st.form("manifold_fit_form"):
            st.markdown("**Model controls**")
            manifold_parameters = kernel_pca_controls()
            st.divider()
            st.markdown("**Embedding and scoring**")
            common_columns = st.columns(4)
            maximum_components = max(1, min(point_count - 1, embedding_values.shape[1]))
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
            quality_neighbors = int(
                common_columns[2].number_input(
                    "Quality neighbors",
                    min_value=2,
                    max_value=max(2, min(100, point_count - 2)),
                    value=min(10, max(2, point_count - 2)),
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
            full_variance_spectrum = st.checkbox(
                "Report variance against the full spectrum",
                value=False,
                key="manifold_full_spectrum",
                help=(
                    "By default the ratios are shares among the retained components, so they sum "
                    "to 100%. This computes every eigenvalue to report the retained share of "
                    "total kernel variance instead, at the cost of a full N×N decomposition."
                ),
                persist_state="page",
            )
            fit_submitted = st.form_submit_button(
                "Fit / update embedding",
                type="primary",
                icon=":material/blur_on:",
                width="stretch",
            )

        fit_options = {
            "n_components": n_components,
            "parameters": manifold_parameters,
            "standardize": standardize,
            "random_state": random_state,
            "quality_neighbors": quality_neighbors,
            "full_variance_spectrum": full_variance_spectrum,
        }
        data_digest = sha256(np.ascontiguousarray(embedding_values).tobytes()).hexdigest()
        manifold_key = (
            prepared_key,
            data_digest,
            MANIFOLD_MODEL_ARTIFACT_VERSION,
            repr(sorted(fit_options.items(), key=lambda item: item[0])),
        )
        if fit_submitted:
            try:
                with st.spinner(f"Fitting {ALGORITHM_LABEL}…"):
                    fitted = fit_manifold_cached(embedding_values, embedding_target, fit_options)
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
    projection, details = manifold_projection_frame(
        prepared_metadata, manifold_result, details=prepared_details
    )
except ValueError as exc:
    st.error(f"The embedding could not be aligned with the prepared metadata: {exc}")
    st.stop()
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
    "Variance captured",
    _format_metric(captured_variance, percent=True),
    help=(
        "Cumulative share of kernel-space variance carried by the retained components. "
        + (
            "Measured against the full eigenvalue spectrum."
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
if retained_fraction is None:
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

coordinate_fields = embedding_fields(n_components)
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
                                "kernel": manifold_result.model.parameters["kernel"],
                                "embedding_components": n_components,
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

with st.expander(f"{ALGORITHM_LABEL} details and downloads"):
    variance = explained_variance_table(manifold_result)
    st.dataframe(
        variance,
        hide_index=True,
        width="stretch",
        column_config={
            "explained_variance": st.column_config.NumberColumn(format="%.5g"),
            "cumulative_variance": st.column_config.NumberColumn(format="%.5g"),
        },
    )
    st.caption(
        "Ratios are shares of kernel-space variance, the same quantity PCA reports but "
        "measured in the kernel's implicit feature space rather than activation space. "
        + (
            "They are measured against the full eigenvalue spectrum."
            if retained_fraction is not None
            else "They are measured among the retained components, so the cumulative column "
            "reaches 1.0 by construction."
        )
    )
    st.write(
        {
            "Method": ALGORITHM_LABEL,
            "Target": TARGET_FEATURE,
            "Parameters": manifold_result.model.parameters,
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
        "cached_position": inspection["positions"][position_index],
        "aggregation_fields": list(analysis_aggregation_fields),
        "standardized": details["standardized"],
        "random_state": random_state,
    }
    with st.container(horizontal=True):
        st.download_button(
            "Download manifold model",
            data=lambda: serialize_manifold_model(
                manifold_result.model, metadata=model_metadata
            ),
            file_name=f"activation_manifold_kernel_pca_{safe_component}.joblib",
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
            file_name="activation_kernel_pca_embedding.csv",
            mime="text/csv",
            icon=":material/download:",
            on_click="ignore",
            help="Includes every embedding coordinate and the retained prompt metadata.",
        )
    st.warning(
        "Only load manifold files you trust. Joblib and pickle artifacts can execute code "
        "when opened."
    )
    st.code(
        "from temporal_manifolds.viz.manifold_explorer import load_manifold_model\n\n"
        'model, provenance = load_manifold_model("manifold.joblib")\n'
        "coordinates = model.transform(activation_rows)",
        language="python",
    )
