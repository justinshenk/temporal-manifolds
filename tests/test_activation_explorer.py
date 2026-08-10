from __future__ import annotations

from copy import deepcopy
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch
from sklearn.decomposition import PCA
from streamlit.testing.v1 import AppTest

from temporal_manifolds.viz.activation_explorer import (
    PCA_PROJECTION_FINGERPRINT_VERSION,
    activation_batch_basename,
    discover_activation_batch_paths,
    extract_activation_slice,
    fit_pca_projection,
    flatten_scalar_metadata,
    inspect_sources,
    load_pca_model,
    metadata_filter_choices,
    metadata_filter_mask,
    metadata_value_token,
    pca_projection_fingerprint,
    prepare_analysis_data,
    prepare_projection,
    prepare_projection_from_matrix,
    select_activation_batch_uploads,
    serialize_pca_model,
    transform_pca_projection,
)


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
    tensor = torch.arange(4 * 2 * 6, dtype=torch.float32).reshape(4, 2, 6)
    torch.save(
        {
            "sample_indices": [10, 11, 12, 13],
            "prompts": ["a", "b", "c", "d"],
            "prompt_metadata": metadata,
            "positions": [-1, -2],
            "activations": {"layer_out/21": tensor},
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
        layer_component="layer_out/21",
        position_index=0,
        source_row_counts=inspection["source_row_counts"],
    )

    assert roots == [str(first_folder.resolve()), str(second_folder.resolve())]
    assert sources == [str(first_batch.resolve()), str(second_batch.resolve())]
    assert inspection["batch_count"] == 2
    assert inspection["source_row_counts"] == [4, 4]
    assert matrix.shape == (8, 6)


def test_flatten_scalar_metadata_omits_collections() -> None:
    assert flatten_scalar_metadata({"a": {"b": 2}, "skip": [1]}) == {"a.b": 2}


def test_inspect_and_prepare_projection(tmp_path: Path) -> None:
    path = tmp_path / "activations_batch_000.pt"
    _batch(path)
    inspection = inspect_sources([path])
    assert inspection["components"] == ["layer_out/21"]
    assert inspection["positions"] == [-1, -2]
    assert inspection["source_row_counts"] == [4]
    assert list(inspection["metadata_index"]["sample_index"]) == [10, 11, 12, 13]

    result, pca, details = prepare_projection(
        [path],
        layer_component="layer_out/21",
        position_index=0,
        metadata_filters={"template_metadata.prompt_framing": ["task_available_time"]},
        aggregation_fields=[],
        n_components=2,
    )
    assert list(result["sample_index"]) == [10, 11, 13]
    assert np.allclose(result["time_horizon_months"], [1 / 30.4375, 1, 24])
    assert [column for column in result if column.startswith("PC")] == ["PC1", "PC2"]
    assert pca.n_components == 2
    assert details["analysis_rows"] == 3


def test_disk_backed_slice_is_reused_for_projection(tmp_path: Path) -> None:
    path = tmp_path / "activations_batch_000.pt"
    _batch(path)
    inspection = inspect_sources([path])
    cache_dir = tmp_path / "cache"
    matrix, cache_path = extract_activation_slice(
        [path],
        layer_component="layer_out/21",
        position_index=0,
        source_row_counts=inspection["source_row_counts"],
        cache_dir=cache_dir,
    )
    assert isinstance(matrix, np.memmap)
    assert cache_path is not None and cache_path.exists()
    assert np.allclose(matrix[0], np.arange(6))

    reused, reused_path = extract_activation_slice(
        [path],
        layer_component="layer_out/21",
        position_index=0,
        source_row_counts=inspection["source_row_counts"],
        cache_dir=cache_dir,
    )
    assert reused_path == cache_path
    result, _, details = prepare_projection_from_matrix(
        reused,
        inspection["metadata_index"],
        cached_position=-1,
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
        layer_component="layer_out/21",
        position_index=0,
        metadata_filters=None,
        aggregation_fields=["template_metadata.prompt_framing"],
        n_components=2,
    )
    assert len(result) == 2
    assert sorted(result["source_sample_count"].tolist()) == [1, 3]
    assert details["analysis_rows"] == 2


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
        layer_component="layer_out/21",
        position_index=0,
        source_row_counts=inspection["source_row_counts"],
    )
    prepared = prepare_analysis_data(
        matrix,
        inspection["metadata_index"],
        cached_position=-1,
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
        metadata={"layer_component": "layer_out/21", "cached_position": -1},
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
    assert provenance["layer_component"] == "layer_out/21"
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


def test_loaded_pca_rejects_incompatible_activation_width(tmp_path: Path) -> None:
    path = tmp_path / "activations_batch_000.pt"
    _batch(path)
    inspection = inspect_sources([path])
    matrix, _ = extract_activation_slice(
        [path],
        layer_component="layer_out/21",
        position_index=0,
        source_row_counts=inspection["source_row_counts"],
    )
    prepared = prepare_analysis_data(
        matrix,
        inspection["metadata_index"],
        cached_position=-1,
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
