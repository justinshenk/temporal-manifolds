from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import torch

from temporal_manifolds.horizon.cache import build_cache, load_cache, time_horizon_months
from temporal_manifolds.horizon.evaluation import (
    balanced_group_folds,
    outlier_groups,
    regression_metrics,
)
from temporal_manifolds.horizon.pipeline import ModelConfig, ViewConfig, cross_validate
from temporal_manifolds.horizon.report import build_scorecards, task_folder_matrix
from temporal_manifolds.horizon.targets import (
    CANONICAL_UNITS,
    build_targets,
    canonical_unit,
    compose_from_parts,
    unit_offsets,
)

UNITS = ["seconds", "minutes", "hours", "days", "weeks", "months", "years", "decades"]


N_TASKS = 6
VALUES = (1, 2, 3, 5, 9)


def _synthetic_rows(seed: int):
    """Enumerate a task x unit x value grid the way the real prompt sets are constructed.

    Each task covers its own contiguous window of units, mirroring the real cache where
    "answer a yes-or-no question" never appears with centuries. Task, unit and value vary
    independently within that window, so a probe cannot recover the horizon from task identity.
    """
    generator = np.random.default_rng(seed)
    directions = generator.normal(size=(3, 16))
    for task_index in range(N_TASKS):
        window = range(task_index % 3, task_index % 3 + 6)
        for unit_index in window:
            unit = UNITS[unit_index % len(UNITS)]
            for value in VALUES:
                horizon = np.log10(time_horizon_months(value, unit))
                vector = (
                    horizon * directions[0]
                    + np.log10(value) * directions[1]
                    + task_index * directions[2]
                    + generator.normal(scale=0.02, size=16)
                )
                yield task_index, unit, value, vector


def _write_batches(root, folder: str, batch_size: int = 30, seed: int = 0):
    """Write the synthetic grid into activation batches shaped like the real ``.pt`` files."""
    samples = list(_synthetic_rows(seed))
    directory = root / folder
    directory.mkdir(parents=True, exist_ok=True)
    for batch, start in enumerate(range(0, len(samples), batch_size)):
        chunk = samples[start : start + batch_size]
        prompts, metadata, activations = [], [], []
        for task_index, unit, value, vector in chunk:
            activations.append(vector)
            prompts.append(f"task {task_index} in {value} {unit}")
            metadata.append(
                {
                    "template_id": f"template_{value % 2}",
                    "task": f"task_{task_index}",
                    "task_metadata": {"task_family": f"family_{task_index % 2}", "domain": "test"},
                    "template_metadata": {"prompt_framing": "N/A", "output_format": "N/A"},
                    "base_value": value,
                    "base_unit": unit,
                    "unit_variant": "original",
                    "number_format": "numeric",
                    "value": value,
                    "value_text": str(value),
                    "unit": unit,
                    "quantity": "N/A",
                    "quantity_text": "N/A",
                }
            )
        torch.save(
            {
                "dataset": folder,
                "model_name": "test-model",
                "batch_index": batch,
                "sample_indices": list(range(start, start + len(chunk))),
                "prompts": prompts,
                "prompt_metadata": metadata,
                "layer_component": "layer_out/21",
                "positions": [-1],
                "activations": {
                    "layer_out/21": torch.tensor(
                        np.asarray(activations)[:, None, :], dtype=torch.float32
                    )
                },
            },
            directory / f"activations_batch_{batch:05d}.pt",
        )


@pytest.fixture
def cache(tmp_path):
    acts = tmp_path / "acts"
    _write_batches(acts, "plain", seed=0)
    _write_batches(acts, "abst", seed=1)
    _write_batches(acts, "conv", seed=2)
    return build_cache(acts, tmp_path / "cache")


def test_unit_conversion_covers_the_dataset_vocabulary():
    assert time_horizon_months(1, "months") == pytest.approx(1.0)
    assert time_horizon_months(2, "years") == pytest.approx(24.0)
    assert time_horizon_months(1, "centuries") == pytest.approx(1200.0)
    assert np.isnan(time_horizon_months("N/A", "N/A"))
    with pytest.raises(ValueError, match="Cannot convert"):
        time_horizon_months(1, "fortnights")


def test_build_cache_excludes_the_conversational_outlier_folder(cache):
    folders = set(cache.metadata["source_folder"])
    assert folders == {"plain", "abst"}
    assert cache.n_rows == len(cache.metadata)
    assert cache.activations.dtype == np.float16


def test_cache_rows_read_back_as_float32_in_order(cache):
    positions = np.array([5, 1, 9])
    block = cache.rows(positions, chunk_size=2)
    assert block.dtype == np.float32
    np.testing.assert_allclose(block[1], np.asarray(cache.activations[1], dtype=np.float32))


def test_canonical_unit_and_offsets_agree_with_the_conversion_table():
    assert canonical_unit("centuries") == "century"
    assert canonical_unit("Days") == "day"
    offsets = unit_offsets()
    assert offsets[CANONICAL_UNITS.index("month")] == pytest.approx(0.0)
    assert offsets[CANONICAL_UNITS.index("year")] == pytest.approx(np.log10(12.0))
    with pytest.raises(ValueError, match="Unrecognised"):
        canonical_unit("fortnight")


def test_targets_decompose_the_horizon_exactly(cache):
    targets, names = build_targets(cache.metadata, auxiliary=True)
    horizon = targets[:, names.index("log10_time_horizon_months")]
    parts = targets[:, names.index("log10_value")] + targets[:, names.index("log10_unit_months")]
    np.testing.assert_allclose(horizon, parts, atol=1e-9)


def test_compose_from_parts_snaps_onto_the_unit_grid(cache):
    targets, names = build_targets(cache.metadata, auxiliary=True)
    # Feeding the true targets back in must reconstruct the horizon exactly.
    np.testing.assert_allclose(
        compose_from_parts(targets, names, snap_units=True), targets[:, 0], atol=1e-9
    )


def test_balanced_group_folds_keep_tasks_whole_and_spread_horizons(cache):
    metadata = cache.metadata
    folds = balanced_group_folds(metadata, n_splits=2)
    per_task = metadata.assign(fold=folds).groupby("task")["fold"].nunique()
    assert (per_task == 1).all()
    assert len(set(folds)) == 2


def test_balanced_group_folds_rejects_too_few_groups(cache):
    with pytest.raises(ValueError, match="groups available"):
        balanced_group_folds(cache.metadata, n_splits=9)


def test_cross_validate_recovers_a_planted_horizon_direction(cache):
    metadata = cache.metadata
    rows = metadata["row"].to_numpy(dtype=np.int64)
    folds = balanced_group_folds(metadata, n_splits=2)
    targets, names = build_targets(metadata, auxiliary=False)
    config = ModelConfig(
        views=(ViewConfig("raw", 3, 2),), poly_degree=2, alpha=1e-4, n_splits=2
    )
    predictions, artifacts, details = cross_validate(
        cache, rows, metadata, folds, targets, config
    )

    assert not np.isnan(predictions).any(), "every row must receive an out-of-fold prediction"
    assert details["views"][0]["n_supervised"] == 3
    assert len(artifacts) == 2
    metrics = regression_metrics(targets[:, 0], predictions[:, 0])
    assert metrics["r2"] > 0.95


def test_degree_four_over_many_features_is_refused_rather_than_thrashing(cache):
    metadata = cache.metadata
    rows = metadata["row"].to_numpy(dtype=np.int64)
    folds = balanced_group_folds(metadata, n_splits=2)
    targets, _ = build_targets(metadata, auxiliary=False)
    config = ModelConfig(
        views=(ViewConfig("raw", 8, 8),),
        poly_degree=4,
        alpha=1e-4,
        n_splits=2,
        max_expanded_features=100,
    )
    with pytest.raises(ValueError, match="above the 100 cap"):
        cross_validate(cache, rows, metadata, folds, targets, config)


def test_task_centered_view_removes_the_task_offset(cache):
    """Centring on task means is label-free and must leave no between-task mean structure."""
    from temporal_manifolds.horizon.pipeline import ViewReader

    metadata = cache.metadata
    rows = metadata["row"].to_numpy(dtype=np.int64)
    reader = ViewReader(cache, rows, metadata["task"].to_numpy(), "task_centered")
    block = reader.block(0, rows.size)
    per_task = pd.DataFrame(block).groupby(metadata["task"].to_numpy()).mean()
    assert np.abs(per_task.to_numpy()).max() < 1e-3


def test_scorecards_slice_by_task_and_folder(cache):
    metadata = cache.metadata
    rows = metadata["row"].to_numpy(dtype=np.int64)
    folds = balanced_group_folds(metadata, n_splits=2)
    targets, _ = build_targets(metadata, auxiliary=False)
    config = ModelConfig(views=(ViewConfig("raw", 3, 2),), poly_degree=2, alpha=1e-4, n_splits=2)
    predictions, _, _ = cross_validate(cache, rows, metadata, folds, targets, config)

    cards = build_scorecards(metadata, targets[:, 0], predictions[:, 0], folds)
    assert set(cards.by_folder["source_folder"]) == {"plain", "abst"}
    assert len(cards.by_task) == metadata["task"].nunique()
    assert cards.by_task["rmse"].is_monotonic_decreasing
    assert cards.by_fold["n"].sum() == len(metadata)
    assert task_folder_matrix(cards).shape[1] == 2
    assert "OVERALL" in cards.summary()


def test_outlier_groups_flags_only_the_extreme_scorecard_row():
    scorecard = pd.DataFrame(
        {"task": list("abcde"), "rmse": [0.20, 0.21, 0.19, 0.22, 5.00]}
    )
    assert outlier_groups(scorecard, key="task") == ["e"]
    assert outlier_groups(scorecard.head(4), key="task") == []
