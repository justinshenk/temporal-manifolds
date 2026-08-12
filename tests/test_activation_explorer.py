from __future__ import annotations

from copy import deepcopy
import io
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pytest
import torch
from sklearn.cross_decomposition import PLSRegression
from sklearn.decomposition import PCA
from streamlit.testing.v1 import AppTest

from temporal_manifolds.activations.extraction_policy import (
    CACHED_POSITION_INDEX,
    PROMPT_TOKEN_POSITION,
    TARGET_LAYER_COMPONENT,
)
from temporal_manifolds.viz.activation_explorer import (
    PCA_PROJECTION_FINGERPRINT_VERSION,
    SOURCE_FOLDER_FIELD,
    activation_batch_basename,
    activation_source_folder_name,
    activation_source_folder_names,
    discover_activation_batch_paths,
    extract_activation_slice,
    fit_pca_projection,
    flatten_scalar_metadata,
    inspect_sources,
    load_pca_model,
    load_pls_model,
    metadata_filter_choices,
    metadata_filter_mask,
    metadata_value_token,
    pca_projection_fingerprint,
    prepare_analysis_data,
    prepare_projection,
    prepare_projection_from_matrix,
    projection_details_table,
    select_activation_batch_uploads,
    serialize_pca_model,
    serialize_pls_model,
    transform_pca_projection,
)
from temporal_manifolds.viz.curve_fitting import fit_curve, serialize_curve_model
from temporal_manifolds.viz.extruded_spline_surface import serialize_extruded_surface


def test_pca_projection_fingerprint_includes_whitening_scale() -> None:
    values = np.arange(60, dtype=np.float64).reshape(10, 6)
    plain = PCA(n_components=3, whiten=False).fit(values)
    whitened = deepcopy(plain)
    whitened.whiten = True
    rescaled = deepcopy(whitened)
    rescaled.explained_variance_ = rescaled.explained_variance_ * 2.0

    assert pca_projection_fingerprint(plain, version=1) == pca_projection_fingerprint(
        whitened, version=1
    )
    assert pca_projection_fingerprint(plain) != pca_projection_fingerprint(whitened)
    assert pca_projection_fingerprint(whitened) != pca_projection_fingerprint(rescaled)
    assert PCA_PROJECTION_FINGERPRINT_VERSION == 2
    assert not np.allclose(plain.transform(values), whitened.transform(values))


def test_activation_batch_basename_accepts_directory_upload_paths() -> None:
    assert (
        activation_batch_basename("batches/run_a/activations_batch_00000.pt")
        == "activations_batch_00000.pt"
    )
    assert (
        activation_batch_basename(r"batches\run_a\activations_batch_00001.pt")
        == "activations_batch_00001.pt"
    )
    assert activation_batch_basename("batches/metadata.json") is None


def test_activation_source_folder_name_supports_local_and_uploaded_paths() -> None:
    class Upload:
        name = r"collection\run_b\activations_batch_00001.pt"

    assert activation_source_folder_name(Path("collection/run_a/activations_batch_00000.pt")) == (
        "run_a"
    )
    assert activation_source_folder_name(Upload()) == "run_b"
    root_upload = type("RootUpload", (), {"name": "activations_batch_00002.pt"})()
    assert activation_source_folder_name(root_upload) == "<root>"
    assert activation_source_folder_name(io.BytesIO()) == "<unknown>"

    assert activation_source_folder_names(
        [
            Path("collection_a/run/activations_batch_000.pt"),
            Path("collection_b/run/activations_batch_000.pt"),
        ]
    ) == ["collection_a/run", "collection_b/run"]
    assert activation_source_folder_names(
        [
            type("Upload", (), {"name": "collection_a/run/activations_batch_000.pt"})(),
            type("Upload", (), {"name": "collection_b/run/activations_batch_000.pt"})(),
        ]
    ) == ["collection_a/run", "collection_b/run"]


def test_upload_selection_preserves_same_named_files() -> None:
    class Upload:
        def __init__(self, name: str, marker: str) -> None:
            self.name = name
            self.marker = marker

    first = Upload("run_a/activations_batch_00000.pt", "first")
    second = Upload("run_b/activations_batch_00000.pt", "second")
    same_relative_name = Upload("run_a/activations_batch_00000.pt", "third")
    ignored = Upload("run_c/metadata.pt", "ignored")

    selected = select_activation_batch_uploads([first, second, same_relative_name, ignored])

    assert selected == [first, second, same_relative_name]


def test_app_exposes_multiple_local_and_uploaded_folder_controls() -> None:
    app_path = Path(__file__).resolve().parents[1] / "apps" / "activation_explorer.py"
    app = AppTest.from_file(app_path).run(timeout=30)

    assert not app.exception
    assert app.segmented_control[0].options == ["Local folders", "Upload folders"]
    assert app.text_area[0].label == "Folder paths"

    app.segmented_control[0].set_value("Upload folders").run(timeout=30)

    assert not app.exception
    assert app.file_uploader[0].label == "Activation folders"
    assert app.file_uploader[0].accept_directory
    assert app.file_uploader[0].multiple_files
    assert "Load selected folders" in [button.label for button in app.button]


def test_curve_overlay_is_available_only_for_three_dimensional_plots(tmp_path: Path) -> None:
    path = tmp_path / "activations_batch_000.pt"
    _batch(path)
    app_path = Path(__file__).resolve().parents[1] / "apps" / "activation_explorer.py"
    app = AppTest.from_file(app_path)
    app.session_state["sources"] = [str(path)]
    app.session_state["source_label"] = str(tmp_path)
    app.session_state["source_is_local"] = True
    app.session_state["source_revision"] = 1

    app.run(timeout=30)

    assert not app.exception
    assert "Curve overlay" in [toggle.label for toggle in app.toggle]
    assert "Extruded surface" in [toggle.label for toggle in app.toggle]
    curve_toggle = next(toggle for toggle in app.toggle if toggle.label == "Curve overlay")
    curve_toggle.set_value(True).run(timeout=30)

    assert not app.exception
    assert "Curve parameter" in [selectbox.label for selectbox in app.selectbox]
    assert "Curve padding" in [slider.label for slider in app.slider]
    fit_curve_button = next(
        button for button in app.button if button.label == "Fit / update curve"
    )
    fit_curve_button.click().run(timeout=30)

    assert not app.exception
    assert app.session_state["curve_result"] is not None
    displayed_curve = app.session_state["curve_result"]
    cubic_parameter = np.linspace(
        *displayed_curve.model.training_parameter_bounds,
        4,
    )
    cubic_spline = fit_curve(
        cubic_parameter,
        displayed_curve.model.predict(cubic_parameter),
        algorithm="spline",
        parameter_feature=displayed_curve.model.parameter_feature,
        coordinate_features=displayed_curve.model.coordinate_features,
        parameters={"degree": 3, "smoothing": 0.0},
        validation_fraction=0.0,
    )
    cubic_spline_bytes = serialize_curve_model(cubic_spline.model)
    curve_mode = next(
        control for control in app.segmented_control if control.label == "Curve model"
    )
    curve_mode.set_value("Use saved").run(timeout=30)

    assert not app.exception
    assert "Saved curve model" in [uploader.label for uploader in app.file_uploader]

    extruded_toggle = next(
        toggle for toggle in app.toggle if toggle.label == "Extruded surface"
    )
    extruded_toggle.set_value(True).run(timeout=30)

    assert not app.exception
    assert "Cubic spline model" in [uploader.label for uploader in app.file_uploader]
    assert "Load cubic spline" in [button.label for button in app.button]
    cubic_spline_uploader = next(
        uploader for uploader in app.file_uploader if uploader.label == "Cubic spline model"
    )
    cubic_spline_uploader.set_value(
        ("cubic_spline.joblib", cubic_spline_bytes, "application/octet-stream")
    ).run(timeout=30)
    load_spline = next(button for button in app.button if button.label == "Load cubic spline")
    load_spline.click().run(timeout=30)

    assert not app.exception
    extrusion_degree = next(
        control for control in app.segmented_control if control.label == "Extrusion degree"
    )
    assert extrusion_degree.options == ["Linear", "Quadratic"]
    extrusion_degree.set_value("Quadratic").run(timeout=30)
    update_extrusion = next(
        button for button in app.button if button.label == "Fit / update extrusion"
    )
    update_extrusion.click().run(timeout=30)

    assert not app.exception
    assert "loaded_extruded_surface_result" in app.session_state, [
        error.value for error in app.error
    ]
    assert app.session_state["loaded_extruded_surface_result"] is not None
    assert app.session_state["loaded_extruded_surface_model"].extrusion_degree == 2
    assert "Download extruded surface" in [button.label for button in app.download_button]
    extruded_surface_bytes = serialize_extruded_surface(
        app.session_state["loaded_extruded_surface_model"]
    )
    extrusion_source = next(
        control for control in app.segmented_control if control.label == "Extrusion source"
    )
    extrusion_source.set_value("Load saved surface").run(timeout=30)

    assert not app.exception
    assert "Saved extruded surface" in [uploader.label for uploader in app.file_uploader]
    assert "Load extruded surface" in [button.label for button in app.button]
    surface_uploader = next(
        uploader for uploader in app.file_uploader if uploader.label == "Saved extruded surface"
    )
    surface_uploader.set_value(
        ("extruded_surface.joblib", extruded_surface_bytes, "application/octet-stream")
    ).run(timeout=30)
    load_surface = next(
        button for button in app.button if button.label == "Load extruded surface"
    )
    load_surface.click().run(timeout=30)

    assert not app.exception
    assert app.session_state["loaded_extruded_surface_result"] is not None
    assert app.session_state["loaded_extruded_surface_model"].extrusion_degree == 2
    assert "Download extruded surface" in [button.label for button in app.download_button]
    plot_control = next(control for control in app.segmented_control if control.label == "Plot")
    plot_control.set_value("2D").run(timeout=30)

    assert not app.exception
    assert "Curve overlay" not in [toggle.label for toggle in app.toggle]
    assert any(
        "surface and curve overlays" in caption.value for caption in app.caption
    )


def _batch(path: Path) -> None:
    metadata = [
        {
            "base_value": value,
            "base_unit": unit,
            "template_metadata": {"prompt_framing": framing, "output_format": "steps"},
        }
        for value, unit, framing in [
            (1, "day", "task_available_time"),
            (1, "month", "task_available_time"),
            (1, "year", "other"),
            (2, "years", "task_available_time"),
        ]
    ]
    tensor = torch.arange(4 * 1 * 6, dtype=torch.float32).reshape(4, 1, 6)
    torch.save(
        {
            "sample_indices": [10, 11, 12, 13],
            "prompts": ["a", "b", "c", "d"],
            "prompt_metadata": metadata,
            "layer_component": TARGET_LAYER_COMPONENT,
            "positions": [PROMPT_TOKEN_POSITION],
            "activations": {TARGET_LAYER_COMPONENT: tensor},
        },
        path,
    )


def test_distinct_local_folders_keep_same_named_batches(tmp_path: Path) -> None:
    first_folder = tmp_path / "run_a"
    second_folder = tmp_path / "run_b"
    first_folder.mkdir()
    second_folder.mkdir()
    first_batch = first_folder / "activations_batch_000.pt"
    second_batch = second_folder / "activations_batch_000.pt"
    _batch(first_batch)
    _batch(second_batch)

    sources, roots = discover_activation_batch_paths([first_folder, second_folder])
    inspection = inspect_sources(sources)
    matrix, _ = extract_activation_slice(
        sources,
        layer_component=TARGET_LAYER_COMPONENT,
        position_index=CACHED_POSITION_INDEX,
        source_row_counts=inspection["source_row_counts"],
    )

    assert roots == [str(first_folder.resolve()), str(second_folder.resolve())]
    assert sources == [str(first_batch.resolve()), str(second_batch.resolve())]
    assert inspection["batch_count"] == 2
    assert inspection["source_row_counts"] == [4, 4]
    assert SOURCE_FOLDER_FIELD in inspection["metadata_fields"]
    assert inspection["metadata_index"][SOURCE_FOLDER_FIELD].tolist() == [
        *(["run_a"] * 4),
        *(["run_b"] * 4),
    ]
    assert matrix.shape == (8, 6)

    _, row_offsets, metadata, details = prepare_analysis_data(
        matrix,
        inspection["metadata_index"],
        cached_position=PROMPT_TOKEN_POSITION,
        metadata_filters={SOURCE_FOLDER_FIELD: ["run_b"]},
        aggregation_fields=[],
    )
    assert row_offsets.tolist() == [4, 5, 6, 7]
    assert metadata[SOURCE_FOLDER_FIELD].tolist() == ["run_b"] * 4
    assert details["loaded_samples"] == 4

    _, _, aggregated_metadata, _ = prepare_analysis_data(
        matrix,
        inspection["metadata_index"],
        cached_position=PROMPT_TOKEN_POSITION,
        metadata_filters=None,
        aggregation_fields=["time_horizon_months"],
    )
    assert aggregated_metadata[SOURCE_FOLDER_FIELD].tolist() == ["<averaged>"] * 4


def test_uploaded_directory_names_become_source_folder_metadata(tmp_path: Path) -> None:
    path = tmp_path / "activations_batch_000.pt"
    _batch(path)
    payload = path.read_bytes()

    class Upload(io.BytesIO):
        def __init__(self, name: str) -> None:
            super().__init__(payload)
            self.name = name

    inspection = inspect_sources(
        [
            Upload("collection/run_a/activations_batch_000.pt"),
            Upload(r"collection\run_b\activations_batch_000.pt"),
        ]
    )

    assert inspection["metadata_index"][SOURCE_FOLDER_FIELD].tolist() == [
        *(["run_a"] * 4),
        *(["run_b"] * 4),
    ]


def test_flatten_scalar_metadata_omits_collections() -> None:
    assert flatten_scalar_metadata({"a": {"b": 2}, "skip": [1]}) == {"a.b": 2}


def test_inspect_and_prepare_projection(tmp_path: Path) -> None:
    path = tmp_path / "activations_batch_000.pt"
    _batch(path)
    inspection = inspect_sources([path])
    assert inspection["components"] == [TARGET_LAYER_COMPONENT]
    assert inspection["positions"] == [PROMPT_TOKEN_POSITION]
    assert inspection["source_row_counts"] == [4]
    assert list(inspection["metadata_index"]["sample_index"]) == [10, 11, 12, 13]
    assert inspection["metadata_index"][SOURCE_FOLDER_FIELD].tolist() == [tmp_path.name] * 4

    result, pca, details = prepare_projection(
        [path],
        layer_component=TARGET_LAYER_COMPONENT,
        position_index=CACHED_POSITION_INDEX,
        metadata_filters={
            "template_metadata.prompt_framing": ["task_available_time"],
            SOURCE_FOLDER_FIELD: [tmp_path.name],
        },
        aggregation_fields=[],
        n_components=2,
    )
    assert list(result["sample_index"]) == [10, 11, 13]
    assert result[SOURCE_FOLDER_FIELD].tolist() == [tmp_path.name] * 3
    assert np.allclose(result["time_horizon_months"], [1 / 30.4375, 1, 24])
    assert [column for column in result if column.startswith("PC")] == ["PC1", "PC2"]
    assert pca.n_components == 2
    assert details["analysis_rows"] == 3


def test_derived_source_folder_overrides_prompt_metadata_collision(tmp_path: Path) -> None:
    folder = tmp_path / "actual_run"
    folder.mkdir()
    path = folder / "activations_batch_000.pt"
    _batch(path)
    payload = torch.load(path, map_location="cpu", weights_only=True)
    for metadata in payload["prompt_metadata"]:
        metadata[SOURCE_FOLDER_FIELD] = "spoofed"
    torch.save(payload, path)

    inspection = inspect_sources([path])

    assert inspection["metadata_index"][SOURCE_FOLDER_FIELD].tolist() == ["actual_run"] * 4


@pytest.mark.parametrize(
    ("mutation", "error_match"),
    [
        ("wrong_layer", "must declare layer_component='layer_out/21'"),
        ("wrong_position", r"must contain positions=\[-1\]"),
        ("extra_component", "must contain only 'layer_out/21'"),
        ("extra_position_axis", "batch x 1 cached position x hidden size"),
    ],
)
def test_inspect_rejects_batches_outside_extraction_contract(
    tmp_path: Path,
    mutation: str,
    error_match: str,
) -> None:
    path = tmp_path / "activations_batch_000.pt"
    _batch(path)
    payload = torch.load(path, map_location="cpu", weights_only=True)

    if mutation == "wrong_layer":
        payload["layer_component"] = "layer_out/20"
    elif mutation == "wrong_position":
        payload["positions"] = [0]
    elif mutation == "extra_component":
        payload["activations"]["layer_out/20"] = payload["activations"][
            TARGET_LAYER_COMPONENT
        ].clone()
    elif mutation == "extra_position_axis":
        payload["activations"][TARGET_LAYER_COMPONENT] = torch.zeros(4, 2, 6)
    else:  # pragma: no cover - the parametrization is exhaustive
        raise AssertionError(f"Unknown mutation: {mutation}")
    torch.save(payload, path)

    with pytest.raises(ValueError, match=error_match):
        inspect_sources([path])


@pytest.mark.parametrize(
    ("layer_component", "position_index", "error_match"),
    [
        ("layer_out/20", CACHED_POSITION_INDEX, "restricted to 'layer_out/21'"),
        (TARGET_LAYER_COMPONENT, 1, "restricted to cached position index 0"),
    ],
)
def test_extract_rejects_requests_outside_extraction_contract(
    tmp_path: Path,
    layer_component: str,
    position_index: int,
    error_match: str,
) -> None:
    path = tmp_path / "activations_batch_000.pt"
    _batch(path)

    with pytest.raises(ValueError, match=error_match):
        extract_activation_slice(
            [path],
            layer_component=layer_component,
            position_index=position_index,
            source_row_counts=[4],
        )


def test_prepare_rejects_non_final_prompt_position() -> None:
    with pytest.raises(ValueError, match="restricted to prompt token position -1"):
        prepare_analysis_data(
            np.zeros((1, 2), dtype=np.float32),
            pd.DataFrame({"sample_index": [0]}),
            cached_position=0,
            metadata_filters=None,
            aggregation_fields=None,
        )


def test_disk_backed_slice_is_reused_for_projection(tmp_path: Path) -> None:
    path = tmp_path / "activations_batch_000.pt"
    _batch(path)
    inspection = inspect_sources([path])
    cache_dir = tmp_path / "cache"
    matrix, cache_path = extract_activation_slice(
        [path],
        layer_component=TARGET_LAYER_COMPONENT,
        position_index=CACHED_POSITION_INDEX,
        source_row_counts=inspection["source_row_counts"],
        cache_dir=cache_dir,
    )
    assert isinstance(matrix, np.memmap)
    assert cache_path is not None and cache_path.exists()
    assert np.allclose(matrix[0], np.arange(6))

    reused, reused_path = extract_activation_slice(
        [path],
        layer_component=TARGET_LAYER_COMPONENT,
        position_index=CACHED_POSITION_INDEX,
        source_row_counts=inspection["source_row_counts"],
        cache_dir=cache_dir,
    )
    assert reused_path == cache_path
    result, _, details = prepare_projection_from_matrix(
        reused,
        inspection["metadata_index"],
        cached_position=PROMPT_TOKEN_POSITION,
        metadata_filters={"template_metadata.prompt_framing": ["task_available_time"]},
        aggregation_fields=[],
        n_components=2,
    )
    assert list(result["sample_index"]) == [10, 11, 13]
    assert details["loaded_samples"] == 3
    assert details["pca_solver"] == "incremental"


def test_aggregation_happens_before_pca(tmp_path: Path) -> None:
    path = tmp_path / "activations_batch_000.pt"
    _batch(path)
    result, _, details = prepare_projection(
        [path],
        layer_component=TARGET_LAYER_COMPONENT,
        position_index=CACHED_POSITION_INDEX,
        metadata_filters=None,
        aggregation_fields=["template_metadata.prompt_framing"],
        n_components=2,
    )
    assert len(result) == 2
    assert sorted(result["source_sample_count"].tolist()) == [1, 3]
    grouped = result.set_index("template_metadata.prompt_framing")
    assert grouped.loc["task_available_time", "base_value"] == "<averaged>"
    assert grouped.loc["other", "base_value"] == 1
    assert details["analysis_rows"] == 2


def test_projection_details_table_serializes_averaged_base_value_as_text() -> None:
    projection = pd.DataFrame(
        {
            "base_value": pd.Series(["<averaged>", np.int64(1)], dtype=object),
            "PC1": [0.25, -0.25],
        }
    )

    table = projection_details_table(projection)

    assert projection["base_value"].tolist() == ["<averaged>", np.int64(1)]
    assert table["base_value"].tolist() == ["<averaged>", "1"]
    arrow_table = pa.Table.from_pandas(table, preserve_index=False)
    assert arrow_table.schema.field("base_value").type in {pa.string(), pa.large_string()}


@pytest.mark.parametrize(
    ("aggregation_fields", "expected_solver"),
    [([], "incremental"), (["template_metadata.prompt_framing"], "randomized")],
)
def test_saved_pca_round_trip_reproduces_projection(
    tmp_path: Path,
    aggregation_fields: list[str],
    expected_solver: str,
) -> None:
    path = tmp_path / "activations_batch_000.pt"
    _batch(path)
    inspection = inspect_sources([path])
    matrix, _ = extract_activation_slice(
        [path],
        layer_component=TARGET_LAYER_COMPONENT,
        position_index=CACHED_POSITION_INDEX,
        source_row_counts=inspection["source_row_counts"],
    )
    prepared = prepare_analysis_data(
        matrix,
        inspection["metadata_index"],
        cached_position=PROMPT_TOKEN_POSITION,
        metadata_filters=None,
        aggregation_fields=aggregation_fields,
    )
    original, pca, fit_details = fit_pca_projection(
        matrix,
        *prepared[:3],
        n_components=2,
        details=prepared[3],
        batch_size=2,
    )
    artifact = serialize_pca_model(
        pca,
        metadata={
            "layer_component": TARGET_LAYER_COMPONENT,
            "cached_position": PROMPT_TOKEN_POSITION,
        },
    )
    restored, provenance = load_pca_model(artifact)
    transformed, _, transform_details = transform_pca_projection(
        matrix,
        *prepared[:3],
        pca=restored,
        details=prepared[3],
        batch_size=2,
    )

    assert np.allclose(
        transformed[["PC1", "PC2"]],
        original[["PC1", "PC2"]],
        atol=2e-6,
    )
    assert provenance["layer_component"] == TARGET_LAYER_COMPONENT
    assert provenance["component_count"] == 2
    assert provenance["feature_count"] == matrix.shape[1]
    assert fit_details["pca_solver"] == expected_solver
    assert fit_details["pca_source"] == "fitted"
    assert transform_details["pca_source"] == "loaded"


def test_load_pca_model_accepts_raw_pickle_and_rejects_invalid_models() -> None:
    fitted = PCA(n_components=2).fit(np.arange(24, dtype=np.float32).reshape(4, 6))
    restored, provenance = load_pca_model(pickle.dumps(fitted))
    assert provenance == {}
    assert np.allclose(restored.components_, fitted.components_)

    with pytest.raises(ValueError, match="has not been fitted"):
        load_pca_model(pickle.dumps(PCA(n_components=2)))
    with pytest.raises(ValueError, match="not a supported PCA model artifact"):
        load_pca_model(pickle.dumps({"model": fitted}))


def test_saved_pls_round_trip_reproduces_projection() -> None:
    rng = np.random.default_rng(42)
    values = rng.normal(size=(24, 6))
    target = values[:, 0] - 0.5 * values[:, 2] + rng.normal(scale=0.05, size=24)
    fitted = PLSRegression(
        n_components=3,
        scale=False,
        max_iter=750,
        tol=1e-7,
    ).fit(values, target)
    fitted.components_ = np.asarray(fitted.x_rotations_).T / np.asarray(fitted._x_std)[  # noqa: SLF001
        np.newaxis, :
    ]
    fitted.mean_ = np.asarray(fitted._x_mean)  # noqa: SLF001
    fitted.explained_variance_ratio_ = np.array([0.5, 0.25, 0.1])

    artifact = serialize_pls_model(
        fitted,
        metadata={
            "layer_component": TARGET_LAYER_COMPONENT,
            "cached_position": PROMPT_TOKEN_POSITION,
            "pls_target": "log10_time_horizon_months",
        },
    )
    restored, provenance = load_pls_model(artifact)

    assert np.allclose(restored.transform(values), fitted.transform(values))
    assert provenance["layer_component"] == TARGET_LAYER_COMPONENT
    assert provenance["component_count"] == 3
    assert provenance["feature_count"] == values.shape[1]
    assert provenance["pls_target"] == "log10_time_horizon_months"
    assert provenance["pls_scale"] is False
    assert provenance["pls_max_iter"] == 750
    assert provenance["pls_tolerance"] == pytest.approx(1e-7)

    with pytest.raises(ValueError, match="not a supported PLS model artifact"):
        load_pls_model(pickle.dumps(fitted))


def test_loaded_pca_rejects_incompatible_activation_width(tmp_path: Path) -> None:
    path = tmp_path / "activations_batch_000.pt"
    _batch(path)
    inspection = inspect_sources([path])
    matrix, _ = extract_activation_slice(
        [path],
        layer_component=TARGET_LAYER_COMPONENT,
        position_index=CACHED_POSITION_INDEX,
        source_row_counts=inspection["source_row_counts"],
    )
    prepared = prepare_analysis_data(
        matrix,
        inspection["metadata_index"],
        cached_position=PROMPT_TOKEN_POSITION,
        metadata_filters=None,
        aggregation_fields=[],
    )
    _, pca, _ = fit_pca_projection(
        matrix,
        *prepared[:3],
        n_components=2,
        details=prepared[3],
        batch_size=2,
    )

    with pytest.raises(ValueError, match="expects 6 activation features"):
        transform_pca_projection(
            matrix[:, :-1],
            *prepared[:3],
            pca=pca,
            details=prepared[3],
            batch_size=2,
        )


def test_visual_metadata_filters_are_typed_and_support_missing_values() -> None:
    dataframe = pd.DataFrame(
        {
            "kind": pd.Series(
                [1, "1", True, None, "<missing>", [1, 2], np.array([1, 2])], dtype=object
            ),
            "group": ["x", "x", "y", "y", "y", "x", "y"],
        }
    )
    choices = dict(metadata_filter_choices(dataframe["kind"]))
    assert metadata_value_token(1) in choices
    assert metadata_value_token("1") in choices
    assert metadata_value_token(True) in choices
    assert (
        len({metadata_value_token(1), metadata_value_token("1"), metadata_value_token(True)}) == 3
    )
    assert metadata_value_token(None) == metadata_value_token(np.nan)
    assert metadata_value_token(None) != metadata_value_token("<missing>")
    assert metadata_value_token([1, 2]) != metadata_value_token(np.array([1, 2]))

    original = dataframe.copy(deep=True)
    mask = metadata_filter_mask(
        dataframe,
        {
            "kind": [metadata_value_token(1), metadata_value_token(True)],
            "group": [metadata_value_token("y")],
        },
    )
    assert mask.tolist() == [False, False, True, False, False, False, False]
    assert metadata_filter_mask(dataframe, {"kind": []}).all()
    pd.testing.assert_frame_equal(dataframe, original)
