from __future__ import annotations

import numpy as np
import pytest

from temporal_manifolds.horizon.experiment import DEFAULT_CONFIG, run_experiment, select_rows
from temporal_manifolds.horizon.pipeline import ModelConfig, ViewConfig
from temporal_manifolds.horizon.readout import (
    bin_edges,
    bin_indicators,
    blend,
    expectation_readout,
)
from tests.test_horizon_pipeline import cache  # noqa: F401  (pytest fixture)

SMALL_CONFIG = ModelConfig(
    views=(ViewConfig("raw", 4, 3),), poly_degree=3, poly_head=4, alpha=1e-3, n_splits=2
)


def test_bin_indicators_are_one_hot_and_cover_the_range():
    values = np.linspace(-6.0, 6.0, 200)
    edges = bin_edges(values, 12)
    indicators, centres = bin_indicators(values, edges)

    assert centres.size == 12
    assert indicators.sum() == len(values)
    np.testing.assert_array_equal(indicators.sum(axis=1), np.ones(len(values)))
    assert edges[0] < values.min() and edges[-1] > values.max()


def test_expectation_readout_recovers_the_bin_a_one_hot_points_at():
    values = np.array([-4.0, 0.0, 3.5])
    edges = bin_edges(np.linspace(-6, 6, 100), 24)
    indicators, centres = bin_indicators(values, edges)

    recovered = expectation_readout(indicators, centres)
    assert np.abs(recovered - values).max() < (centres[1] - centres[0])


def test_expectation_readout_falls_back_when_every_bin_score_is_negative():
    centres = np.array([-1.0, 0.0, 1.0])
    scores = np.array([[-3.0, -2.0, -5.0]])
    # Nothing usable in the row, so the readout must stay finite rather than divide by zero.
    assert expectation_readout(scores, centres)[0] == pytest.approx(0.0)


def test_softmax_readout_sharpens_towards_the_leading_bin():
    centres = np.array([-2.0, 0.0, 2.0])
    scores = np.array([[0.2, 0.1, 0.7]])
    sharp = expectation_readout(scores, centres, temperature=20.0)[0]
    soft = expectation_readout(scores, centres, temperature=1.0)[0]
    assert sharp > soft
    assert sharp == pytest.approx(2.0, abs=0.05)


def test_blend_averages_and_respects_weights():
    first, second = np.array([0.0, 2.0]), np.array([2.0, 4.0])
    np.testing.assert_allclose(blend(first, second), [1.0, 3.0])
    np.testing.assert_allclose(blend(first, second, weights=[3.0, 1.0]), [0.5, 2.5])


def test_select_rows_drops_unconstrained_prompts_and_named_tasks(cache):  # noqa: F811
    everything = select_rows(cache)
    assert everything["log10_time_horizon_months"].notna().all()

    reduced = select_rows(cache, drop_tasks=("task_0", "task_1"))
    assert not {"task_0", "task_1"} & set(reduced["task"])
    assert len(reduced) < len(everything)


def test_select_rows_subsamples_deterministically(cache):  # noqa: F811
    first = select_rows(cache, subsample=40)
    second = select_rows(cache, subsample=40)
    assert len(first) == 40
    np.testing.assert_array_equal(first["row"], second["row"])


def test_run_experiment_reports_every_readout_and_slice(cache):  # noqa: F811
    result = run_experiment(cache, config=SMALL_CONFIG)

    assert set(result.readouts) == {"direct", "unit_snapped", "distributional", "blended"}
    for estimate in result.readouts.values():
        assert estimate.shape == (len(result.metadata),)
        assert np.isfinite(estimate).all()
    assert set(result.details["readout_rmse"]) == set(result.readouts)
    assert result.scorecards.overall["n"] == len(result.metadata)
    assert len(result.scorecards.by_task) == result.metadata["task"].nunique()
    assert result.scorecards.by_folder["n"].sum() == len(result.metadata)


def test_run_experiment_beats_predicting_the_mean(cache):  # noqa: F811
    """A two-fold grouped split over six synthetic tasks is a hard, small setting; the model
    only has to clear the constant-prediction baseline by a wide margin."""
    result = run_experiment(cache, config=SMALL_CONFIG)
    assert result.scorecards.overall["r2"] > 0.85
    assert result.scorecards.overall["rmse"] < 0.5 * np.std(
        result.scorecards.predictions["actual"]
    )


def test_default_config_stays_within_the_degree_budget():
    assert DEFAULT_CONFIG.poly_degree <= 4
