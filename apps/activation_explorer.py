"""Streamlit UI for exploring locally cached conversational activations."""

from __future__ import annotations

import gc
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st
from pandas.api.types import is_bool_dtype, is_numeric_dtype

from temporal_manifolds.viz.activation_explorer import (
    activation_batch_basename,
    extract_activation_slice,
    fit_pca_projection,
    inspect_sources,
    prepare_analysis_data,
)

st.set_page_config(
    page_title="Activation Atlas",
    page_icon=":material/scatter_plot:",
    layout="wide",
)
st.caption("TEMPORAL MANIFOLDS · LOCAL ANALYSIS")
st.title("Activation Atlas")
st.write(
    "Filter conversational activation batches, aggregate comparable prompts, and inspect "
    "a freshly fitted PCA projection. Your files stay in this local app session."
)

st.session_state.setdefault("sources", None)
st.session_state.setdefault("source_label", "")
st.session_state.setdefault("source_is_local", None)


def discover_paths(folder: str) -> list[str]:
    path = Path(folder).expanduser()
    if not path.is_dir():
        raise ValueError("That folder does not exist or is not accessible.")
    return [str(item) for item in sorted(path.rglob("activations_batch_*.pt"))]


def uploaded_sources(files) -> list:
    return [
        file
        for file in files
        if activation_batch_basename(file.name) is not None
    ]


def reset_loaded_data() -> None:
    for key in (
        "inspection", "slice_key", "activation_matrix", "activation_cache_path",
        "analysis_key",
        "prepared_key", "prepared_matrix", "prepared_row_offsets",
        "prepared_metadata", "prepared_details", "pca_key", "projection", "pca",
        "details",
    ):
        st.session_state.pop(key, None)


with st.sidebar:
    st.header("1 · Select data")
    source_mode = st.segmented_control(
        "Source", ["Local path", "Folder upload"], default="Local path"
    )
    if source_mode == "Folder upload":
        st.warning(
            "Folder upload keeps every file in browser/server memory. For multi-GB "
            "datasets, use **Local path** instead."
        )
        uploads = st.file_uploader(
            "Activation folder", type=["pt"], accept_multiple_files="directory",
            help="Choose the folder containing activations_batch_*.pt files.",
        )
        if st.button("Load selected folder", type="primary", width="stretch"):
            sources = uploaded_sources(uploads or [])
            if not sources:
                st.error("No activations_batch_*.pt files were selected.")
            else:
                st.session_state.sources = sources
                st.session_state.source_label = f"Selected folder · {len(sources):,} batches"
                st.session_state.source_is_local = False
                reset_loaded_data()
    else:
        folder = st.text_input("Folder path", placeholder=r"C:\data\selected_acts")
        if st.button("Load local path", type="primary", width="stretch"):
            try:
                sources = discover_paths(folder)
                if not sources:
                    raise ValueError("No activations_batch_*.pt files were found in that folder.")
                st.session_state.sources = sources
                st.session_state.source_label = str(Path(folder).expanduser())
                st.session_state.source_is_local = True
                reset_loaded_data()
            except ValueError as exc:
                st.error(str(exc))

sources = st.session_state.get("sources")
if not sources:
    st.info("Select the local folder that contains your activation batches to begin.", icon="↖")
    st.stop()

# Migrate browser sessions created before source metadata was initialized centrally.
if not st.session_state.get("source_label"):
    sources_are_local = all(isinstance(source, (str, Path)) for source in sources)
    st.session_state["source_is_local"] = sources_are_local
    st.session_state["source_label"] = (
        str(Path(sources[0]).parent)
        if sources_are_local
        else f"Selected folder · {len(sources):,} batches"
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
    default_component = (
        "layer_out/21" if "layer_out/21" in inspection["components"]
        else inspection["components"][0]
    )
    component = st.selectbox(
        "Layer / component", inspection["components"],
        index=inspection["components"].index(default_component),
    )
    position_index = st.selectbox(
        "Cached position", range(len(inspection["positions"])),
        format_func=lambda index: f"Index {index} · token {inspection['positions'][index]}",
    )
    candidate_fields = inspection["metadata_fields"]
    default_filter = (
        ["template_metadata.prompt_framing"]
        if "template_metadata.prompt_framing" in candidate_fields else []
    )
    filter_fields = st.multiselect("Filter fields", candidate_fields, default=default_filter)
    metadata_index = inspection["metadata_index"]
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
    aggregation_fields = st.multiselect(
        "Aggregate by", aggregation_candidates, default=["time_horizon_months"],
        help="Rows in each group are averaged before PCA, matching the notebook.",
    )
    n_components = st.number_input("PCA components", min_value=2, max_value=50, value=3)
    max_samples_enabled = st.toggle("Limit source samples", value=False)
    max_samples = (
        int(st.number_input("Maximum samples", min_value=1, value=1000, step=100))
        if max_samples_enabled else None
    )
    st.caption(
        "Filter and aggregation changes rebuild the prepared data; component-count "
        "changes refit PCA only."
    )

slice_key = (component, position_index, source_label)
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
            if st.session_state.get("source_is_local") else None,
            progress=update_extraction_progress,
        )
        progress_bar.empty()
        st.session_state.activation_matrix = activation_matrix
        st.session_state.activation_cache_path = cache_path
        st.session_state.slice_key = slice_key
        st.session_state.pop("prepared_key", None)
        st.session_state.pop("pca_key", None)
    except Exception as exc:  # noqa: BLE001 - surface local extraction errors in the UI
        st.error(f"Activation slice could not be prepared: {exc}")
        st.stop()

activation_matrix = st.session_state.activation_matrix
if st.session_state.get("activation_cache_path") is not None:
    st.caption("Using a disk-backed activation slice; source batches stay closed during refits.")

prepared_key = (
    tuple((field, tuple(filters[field])) for field in filter_fields),
    tuple(aggregation_fields), max_samples,
)
if st.session_state.get("prepared_key") != prepared_key:
    try:
        for key in (
            "prepared_matrix", "prepared_row_offsets", "prepared_metadata",
            "prepared_details", "projection", "pca", "details",
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
    except Exception as exc:  # noqa: BLE001 - surface analysis/data errors in the UI
        st.error(f"Analysis data could not be prepared: {exc}")
        st.stop()

pca_key = (prepared_key, int(n_components))
if st.session_state.get("pca_key") != pca_key:
    try:
        for key in ("projection", "pca", "details"):
            st.session_state.pop(key, None)
        gc.collect()
        with st.spinner("Fitting a new bounded-memory PCA model…"):
            projection, pca, details = fit_pca_projection(
                activation_matrix,
                st.session_state.get("prepared_matrix"),
                st.session_state["prepared_row_offsets"],
                st.session_state["prepared_metadata"],
                n_components=int(n_components),
                details=st.session_state["prepared_details"],
            )
        st.session_state.pca_key = pca_key
        st.session_state.projection = projection
        st.session_state.pca = pca
        st.session_state.details = details
    except Exception as exc:  # noqa: BLE001 - surface analysis/data errors in the UI
        st.error(f"PCA could not be fitted: {exc}")
        st.stop()

projection: pd.DataFrame = st.session_state.projection
details = st.session_state.details
pc_fields = [f"PC{index}" for index in range(1, int(n_components) + 1)]
metadata_fields = sorted(column for column in projection if column not in {*pc_fields, "sample_index"})
color_fields = ["log10_time_horizon_months", *[
    field for field in metadata_fields if field != "log10_time_horizon_months"
]]

metric_columns = st.columns(4)
metric_columns[0].metric("Source samples", f"{details['loaded_samples']:,}")
metric_columns[1].metric("Projected points", f"{details['analysis_rows']:,}")
metric_columns[2].metric("Activation width", f"{details['feature_count']:,}")
metric_columns[3].metric("Variance captured", f"{sum(details['explained_variance']):.1%}")

st.subheader("Projection")
controls = st.columns([1.1, 1.25, 1.25, 2])
plot_mode = controls[0].segmented_control("Plot", ["2D", "3D"], default="3D")
required_axes = 3 if plot_mode == "3D" else 2
if len(pc_fields) < required_axes:
    st.warning(f"Fit at least {required_axes} PCA components for a {plot_mode} plot.")
    st.stop()
x_component = controls[1].selectbox("X component", pc_fields, index=0)
y_choices = [field for field in pc_fields if field != x_component]
y_component = controls[2].selectbox("Y component", y_choices, index=0)
z_component = None
if plot_mode == "3D":
    z_choices = [field for field in pc_fields if field not in {x_component, y_component}]
    z_component = controls[3].selectbox("Z component", z_choices, index=0)
else:
    controls[3].caption("Choose any two fitted components for the plane.")

plot_controls = st.columns([1, 2])
color_field = plot_controls[0].selectbox("Color by", color_fields)
tooltip_fields = plot_controls[1].multiselect(
    "Tooltip fields", ["sample_index", *metadata_fields],
    default=[field for field in ["sample_index", "time_horizon_months"] if field in projection],
)

plot_data = projection.copy()
numeric_color = is_numeric_dtype(plot_data[color_field]) and not is_bool_dtype(plot_data[color_field])
if not numeric_color:
    plot_data[color_field] = plot_data[color_field].astype("string").fillna("<missing>")
common = {
    "data_frame": plot_data,
    "x": x_component,
    "y": y_component,
    "color": color_field,
    "color_continuous_scale": "Viridis" if numeric_color else None,
    "hover_data": tooltip_fields,
    "opacity": 0.76,
}
if plot_mode == "3D":
    figure = px.scatter_3d(**common, z=z_component)
    figure.update_traces(marker={"size": 4})
else:
    figure = px.scatter(**common)
    figure.update_traces(marker={"size": 7, "line": {"width": 0.35, "color": "white"}})
figure.update_layout(
    height=680, margin={"l": 12, "r": 12, "t": 28, "b": 12},
    paper_bgcolor="white", plot_bgcolor="white",
    font={"family": "Inter, ui-sans-serif, system-ui", "color": "#17201d"},
    legend_title_text=color_field,
)
st.plotly_chart(figure, width="stretch", config={"displaylogo": False})

with st.expander("PCA details and projected data"):
    variance = pd.DataFrame({
        "component": pc_fields,
        "explained_variance": details["explained_variance"],
        "cumulative_variance": details["explained_variance"].cumsum(),
    })
    st.dataframe(variance, hide_index=True, width="stretch")
    st.dataframe(projection, hide_index=True, width="stretch")
    st.download_button(
        "Download projected CSV", projection.to_csv(index=False).encode("utf-8"),
        file_name="activation_pca_projection.csv", mime="text/csv",
    )
