from __future__ import annotations

import numpy as np
import pytest
from sklearn.cross_decomposition import PLSRegression
from sklearn.linear_model import Ridge

from temporal_manifolds.horizon.projection import build_projection, simpls
from temporal_manifolds.horizon.streaming import (
    SufficientStatistics,
    load_statistics,
    ridge_solution,
    save_statistics,
)


def _sample(n_rows: int = 400, n_features: int = 12, n_targets: int = 2, seed: int = 0):
    generator = np.random.default_rng(seed)
    features = generator.normal(size=(n_rows, n_features))
    weights = generator.normal(size=(n_features, n_targets))
    targets = features @ weights + generator.normal(scale=0.1, size=(n_rows, n_targets))
    return features, targets


def _accumulate(features: np.ndarray, targets: np.ndarray, chunk: int = 37):
    stats = SufficientStatistics.zeros(features.shape[1], targets.shape[1])
    for start in range(0, len(features), chunk):
        stats.partial_fit(features[start : start + chunk], targets[start : start + chunk])
    return stats


def test_partial_fit_matches_a_single_shot_fit():
    features, targets = _sample()
    chunked = _accumulate(features, targets, chunk=17)
    whole = _accumulate(features, targets, chunk=len(features))

    assert chunked.n == pytest.approx(whole.n)
    np.testing.assert_allclose(chunked.xtx, whole.xtx, rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(chunked.xty, whole.xty, rtol=1e-5, atol=1e-5)


def test_statistics_subtract_to_leave_out_a_block():
    """Fold training statistics are formed as total-minus-fold, so the algebra must hold."""
    features, targets = _sample()
    held_out = slice(0, 90)
    total = _accumulate(features, targets)
    block = _accumulate(features[held_out], targets[held_out])
    remainder = total - block
    direct = _accumulate(features[90:], targets[90:])

    assert remainder.n == pytest.approx(direct.n)
    np.testing.assert_allclose(remainder.xtx, direct.xtx, rtol=1e-4, atol=1e-4)
    np.testing.assert_allclose(remainder.mean_x, direct.mean_x, rtol=1e-4, atol=1e-4)


def test_centered_covariance_matches_numpy():
    features, targets = _sample()
    stats = _accumulate(features, targets)
    expected = np.cov(features, rowvar=False, bias=True)
    np.testing.assert_allclose(stats.centered_covariance(), expected, rtol=1e-4, atol=1e-5)


def test_ridge_solution_matches_scikit_learn():
    features, targets = _sample(n_rows=500, n_features=8, n_targets=1)
    stats = _accumulate(features, targets)
    alpha = 0.05
    coefficients, intercepts = ridge_solution(stats, alpha)

    # ridge_solution penalises the mean-scaled normal equations, so the equivalent
    # scikit-learn penalty is alpha * n_samples.
    reference = Ridge(alpha=alpha * len(features)).fit(features, targets)
    np.testing.assert_allclose(
        coefficients[:, 0], np.ravel(reference.coef_), rtol=1e-4, atol=1e-5
    )
    np.testing.assert_allclose(
        intercepts, np.ravel(reference.intercept_), rtol=1e-4, atol=1e-5
    )


def test_simpls_reproduces_scikit_learn_scores():
    features, targets = _sample(n_rows=300, n_features=10, n_targets=1)
    stats = _accumulate(features, targets)
    weights, loadings, _ = simpls(
        stats.centered_covariance(), stats.centered_cross_covariance(), 3
    )
    scores = (features - features.mean(axis=0)) @ weights

    reference = PLSRegression(n_components=3, scale=False).fit(features, targets)
    correlations = [
        abs(np.corrcoef(scores[:, index], reference.x_scores_[:, index])[0, 1])
        for index in range(3)
    ]
    assert min(correlations) > 0.999
    # SIMPLS normalises scores to unit variance and defines loadings as C @ weights.
    np.testing.assert_allclose(np.var(scores, axis=0), np.ones(3), rtol=1e-4)
    np.testing.assert_allclose(loadings, stats.centered_covariance() @ weights, atol=1e-8)


def test_projection_residual_directions_are_uncorrelated_with_pls_scores():
    features, targets = _sample(n_rows=600, n_features=15, n_targets=1)
    stats = _accumulate(features, targets)
    projection = build_projection(stats, n_supervised=3, n_residual=4)
    scores = projection.transform(features)

    assert projection.n_components == 7
    correlation = np.corrcoef(scores, rowvar=False)
    supervised_vs_residual = correlation[:3, 3:]
    assert np.abs(supervised_vs_residual).max() < 1e-3


def test_statistics_round_trip_through_disk(tmp_path):
    features, targets = _sample()
    blocks = [_accumulate(features[:200], targets[:200]), _accumulate(features[200:], targets[200:])]
    path = tmp_path / "stats.npz"
    save_statistics(path, blocks)
    restored = load_statistics(path)

    assert len(restored) == 2
    for original, loaded in zip(blocks, restored, strict=True):
        assert loaded.n == pytest.approx(original.n)
        np.testing.assert_allclose(loaded.xty, original.xty, rtol=1e-6)
        np.testing.assert_allclose(loaded.xtx, original.xtx, rtol=1e-5, atol=1e-3)
