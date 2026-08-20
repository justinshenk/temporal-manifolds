from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from temporal_manifolds.horizon.classification import (
    assign_classes,
    band_labels,
    build_report,
    fit_bands,
    one_hot,
    optimal_edges,
    unit_aligned_edges,
    within_range,
)
from temporal_manifolds.horizon.evaluation import balanced_group_folds
from temporal_manifolds.horizon.pipeline import ModelConfig, ViewConfig, compute_view_scores
from temporal_manifolds.horizon.targets import CANONICAL_UNITS, build_targets
from tests.test_horizon_pipeline import cache  # noqa: F401  (pytest fixture)


def test_assign_classes_clips_into_the_end_bands():
    edges = np.array([-np.inf, -1.0, 1.0, np.inf])
    values = np.array([-99.0, -1.5, 0.0, 1.5, 99.0])
    np.testing.assert_array_equal(assign_classes(values, edges), [0, 0, 1, 2, 2])


def test_assign_classes_puts_an_edge_value_in_the_upper_band():
    edges = np.array([-np.inf, 0.0, np.inf])
    np.testing.assert_array_equal(assign_classes(np.array([-1e-9, 0.0, 1e-9]), edges), [0, 1, 1])


def test_within_range_is_inclusive():
    values = np.array([-2.0, -1.0, 0.0, 1.0, 2.0])
    np.testing.assert_array_equal(within_range(values, -1.0, 1.0), [False, True, True, True, False])


def test_unit_aligned_edges_give_one_band_per_unit():
    edges = unit_aligned_edges()
    assert len(edges) - 1 == len(CANONICAL_UNITS)
    assert np.isneginf(edges[0]) and np.isposinf(edges[-1])
    assert np.all(np.diff(edges[1:-1]) > 0)
    # Each unit's own offset must land in its own band.
    from temporal_manifolds.horizon.targets import unit_offsets

    np.testing.assert_array_equal(assign_classes(unit_offsets(), edges), np.arange(len(CANONICAL_UNITS)))


def test_band_labels_read_as_durations():
    labels = band_labels(np.array([-np.inf, 0.0, np.inf]))
    assert labels == ["-inf..1mon", "1mon..inf"]


def test_optimal_edges_splits_perfect_predictions_into_usable_bands():
    values = np.linspace(-6.0, 6.0, 1200)
    edges = optimal_edges(values, values.copy(), 4)
    bands = assign_classes(values, edges)

    assert len(edges) - 1 == 4
    assert np.unique(bands).size == 4
    # Perfect predictions must be perfectly banded, whatever the cuts are.
    assert (assign_classes(values, edges) == bands).all()
    counts = np.bincount(bands, minlength=4)
    assert counts.min() >= 0.5 * len(values) / 4


def test_optimal_edges_refuses_the_degenerate_single_band_solution():
    """Unconstrained accuracy is maximised by never splitting; min_share must prevent that."""
    values = np.linspace(-6.0, 6.0, 1000)
    noisy = values + np.random.default_rng(0).normal(scale=2.0, size=values.size)
    edges = optimal_edges(values, noisy, 3, min_share=0.5)
    counts = np.bincount(assign_classes(values, edges), minlength=3)
    assert counts.min() >= 0.5 * len(values) / 3


def test_optimal_edges_reports_when_no_partition_fits():
    values = np.zeros(100)
    with pytest.raises(ValueError, match="No partition into 5 bands"):
        optimal_edges(values, values, 5, min_share=1.0)


def test_optimal_edges_rejects_bad_arguments():
    values = np.linspace(-1.0, 1.0, 50)
    with pytest.raises(ValueError, match="At least two bands"):
        optimal_edges(values, values, 1)
    with pytest.raises(ValueError, match="min_share"):
        optimal_edges(values, values, 2, min_share=0.0)


def test_one_hot_is_a_single_indicator_per_row():
    indicators = one_hot(np.array([0, 2, 1, 2]), 3)
    assert indicators.shape == (4, 3)
    np.testing.assert_array_equal(indicators.sum(axis=1), np.ones(4))
    np.testing.assert_array_equal(indicators[:, 2], [0, 1, 0, 1])


def test_build_report_scores_bands_tasks_and_confusion():
    metadata = pd.DataFrame(
        {
            "task": ["a", "a", "b", "b"],
            "source_folder": ["plain", "plain", "abst", "abst"],
        }
    )
    actual = np.array([0, 1, 1, 2])
    predicted = np.array([0, 1, 2, 2])
    edges = np.array([-np.inf, -1.0, 1.0, np.inf])
    report = build_report(metadata, actual, predicted, edges, np.zeros(4, dtype=int))

    assert report.accuracy == pytest.approx(0.75)
    assert report.within_one_band == pytest.approx(1.0)
    assert report.confusion.shape == (3, 3)
    assert report.per_band.loc[report.per_band["band"] == 1, "recall"].item() == pytest.approx(0.5)
    assert report.by_task.iloc[0]["accuracy"] == pytest.approx(0.5)  # sorted worst first
    assert set(report.by_folder["source_folder"]) == {"plain", "abst"}
    assert "accuracy=" in report.summary()


def test_fit_bands_recovers_planted_bands_under_grouped_cv(cache):  # noqa: F811
    metadata = cache.metadata
    rows = metadata["row"].to_numpy(dtype=np.int64)
    folds = balanced_group_folds(metadata, n_splits=2)
    targets, _ = build_targets(metadata, auxiliary=False)
    scores, _ = compute_view_scores(
        cache, rows, metadata, folds, targets, ViewConfig("raw", 4, 3), 2
    )
    edges = np.array([-np.inf, -2.0, 1.0, np.inf])
    actual_band = assign_classes(targets[:, 0], edges)

    predicted, band_scores = fit_bands(
        [scores], actual_band, folds, ModelConfig(poly_degree=3, alpha=1e-3, n_splits=2), 3
    )
    assert band_scores.shape == (len(metadata), 3)
    accuracy = (predicted == actual_band).mean()
    majority = np.bincount(actual_band).max() / actual_band.size
    # Six synthetic tasks split two ways is a deliberately hard grouped setting; the classifier
    # only has to clear the majority-class baseline by a wide margin.
    assert accuracy > 0.8
    assert accuracy > majority + 0.2
