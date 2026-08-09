from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from temporal_manifolds.viz.activation_explorer import (
    activation_batch_basename,
    extract_activation_slice,
    flatten_scalar_metadata,
    inspect_sources,
    prepare_projection,
    prepare_projection_from_matrix,
)


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
        [path], layer_component="layer_out/21", position_index=0,
        metadata_filters={"template_metadata.prompt_framing": ["task_available_time"]},
        aggregation_fields=[], n_components=2,
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
        [path], layer_component="layer_out/21", position_index=0,
        source_row_counts=inspection["source_row_counts"], cache_dir=cache_dir,
    )
    assert isinstance(matrix, np.memmap)
    assert cache_path is not None and cache_path.exists()
    assert np.allclose(matrix[0], np.arange(6))

    reused, reused_path = extract_activation_slice(
        [path], layer_component="layer_out/21", position_index=0,
        source_row_counts=inspection["source_row_counts"], cache_dir=cache_dir,
    )
    assert reused_path == cache_path
    result, _, details = prepare_projection_from_matrix(
        reused, inspection["metadata_index"], cached_position=-1,
        metadata_filters={"template_metadata.prompt_framing": ["task_available_time"]},
        aggregation_fields=[], n_components=2,
    )
    assert list(result["sample_index"]) == [10, 11, 13]
    assert details["loaded_samples"] == 3
    assert details["pca_solver"] == "incremental"


def test_aggregation_happens_before_pca(tmp_path: Path) -> None:
    path = tmp_path / "activations_batch_000.pt"
    _batch(path)
    result, _, details = prepare_projection(
        [path], layer_component="layer_out/21", position_index=0,
        metadata_filters=None,
        aggregation_fields=["template_metadata.prompt_framing"], n_components=2,
    )
    assert len(result) == 2
    assert sorted(result["source_sample_count"].tolist()) == [1, 3]
    assert details["analysis_rows"] == 2
