"""Supervised and residual projections derived purely from second-moment statistics.

The previous best model built three PLS directions, took the PCA of what those directions left
behind, and fed the union to a polynomial fit. That recipe is generalised here: SIMPLS supplies
the supervised directions, the residual covariance is formed in closed form as ``C - P P'``,
and its leading eigenvectors supply the unsupervised directions. Because both stages read only
``X'X`` and ``X'Y``, the whole feature construction is a by-product of the streaming pass and
never touches the activation matrix again.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.utils.extmath import randomized_svd

from .streaming import SufficientStatistics


@dataclass(frozen=True)
class LatentProjection:
    """A single affine map from activations to model features.

    ``scores = (x - center) / feature_scale @ weights``, then standardized by
    ``(scores - score_mean) / score_scale``.
    """

    center: np.ndarray  # (d,)
    feature_scale: np.ndarray  # (d,)
    weights: np.ndarray  # (d, k)
    score_mean: np.ndarray  # (k,)
    score_scale: np.ndarray  # (k,)
    n_supervised: int
    n_residual: int
    explained_variance_ratio: np.ndarray

    @property
    def n_components(self) -> int:
        return int(self.weights.shape[1])

    def transform(self, x: np.ndarray) -> np.ndarray:
        scores = ((np.asarray(x, dtype=np.float32) - self.center) / self.feature_scale) @ (
            self.weights
        )
        return (scores - self.score_mean) / self.score_scale


def simpls(
    covariance: np.ndarray, cross_covariance: np.ndarray, n_components: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Run SIMPLS on covariance matrices instead of on the data.

    Returns ``(weights, loadings, y_loadings)`` where ``scores = X_centered @ weights`` are
    mutually orthogonal with unit variance under ``covariance``, and ``loadings = C @ weights``.
    """
    dimension = covariance.shape[0]
    cross = np.array(cross_covariance, dtype=np.float64, copy=True)
    if cross.ndim == 1:
        cross = cross[:, None]
    if n_components > dimension:
        raise ValueError(f"Cannot extract {n_components} components from {dimension} features.")

    weights = np.zeros((dimension, n_components))
    loadings = np.zeros((dimension, n_components))
    y_loadings = np.zeros((cross.shape[1], n_components))
    basis = np.zeros((dimension, n_components))

    for index in range(n_components):
        if cross.shape[1] == 1:
            direction = cross[:, 0].copy()
        else:
            left, _, _ = np.linalg.svd(cross, full_matrices=False)
            direction = cross @ (cross.T @ left[:, 0])
        norm = np.linalg.norm(direction)
        if norm <= 1e-12:
            # The supervised signal is exhausted; the remaining components would be noise.
            weights = weights[:, :index]
            loadings = loadings[:, :index]
            y_loadings = y_loadings[:, :index]
            break
        direction /= norm
        score_variance = float(direction @ covariance @ direction)
        if score_variance <= 1e-18:
            weights, loadings, y_loadings = (
                weights[:, :index],
                loadings[:, :index],
                y_loadings[:, :index],
            )
            break
        direction /= np.sqrt(score_variance)

        loading = covariance @ direction
        weights[:, index] = direction
        loadings[:, index] = loading
        y_loadings[:, index] = cross.T @ direction

        # Deflate the cross-covariance against the space already spanned by the loadings.
        deflator = loading.copy()
        if index:
            active = basis[:, :index]
            deflator -= active @ (active.T @ deflator)
            deflator -= active @ (active.T @ deflator)
        deflator_norm = np.linalg.norm(deflator)
        if deflator_norm <= 1e-12:
            weights, loadings, y_loadings = (
                weights[:, : index + 1],
                loadings[:, : index + 1],
                y_loadings[:, : index + 1],
            )
            break
        basis[:, index] = deflator / deflator_norm
        cross -= np.outer(basis[:, index], basis[:, index] @ cross)

    return weights, loadings, y_loadings


def residual_eigenvectors(
    covariance: np.ndarray,
    weights: np.ndarray,
    loadings: np.ndarray,
    n_components: int,
    *,
    random_state: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    """Return the leading eigenvectors of the covariance left over after PLS deflation.

    With SIMPLS scores orthonormalised under ``covariance``, the residual covariance of
    ``X - T P'`` is exactly ``C - P P'``, so the residual PCA needs no second data pass.
    """
    if n_components <= 0:
        return np.zeros((covariance.shape[0], 0)), np.zeros(0)
    residual = covariance - loadings @ loadings.T
    residual = (residual + residual.T) * 0.5
    vectors, singular_values, _ = randomized_svd(
        residual, n_components=n_components, n_iter=7, random_state=random_state
    )
    total = float(np.trace(residual))
    ratio = singular_values / total if total > 0 else np.zeros_like(singular_values)
    # Re-express the eigenvectors so they act on X directly rather than on the deflated X.
    projected = vectors - weights @ (loadings.T @ vectors)
    return projected, ratio


def build_projection(
    stats: SufficientStatistics,
    *,
    n_supervised: int,
    n_residual: int,
    standardize_features: bool = True,
    random_state: int = 0,
) -> LatentProjection:
    """Fit the supervised + residual projection for one training split."""
    scale = stats.feature_scale() if standardize_features else np.ones_like(stats.mean_x)
    covariance = stats.centered_covariance() / np.outer(scale, scale)
    cross = stats.centered_cross_covariance() / scale[:, None]
    # Auxiliary targets (log10 value, log10 unit, unit indicators) live on wildly different
    # scales, and SIMPLS weights them by raw covariance -- standardise so none dominates.
    target_scale = np.sqrt(np.maximum(np.diag(stats.centered_target_covariance()), 0.0)) + 1e-12
    cross = cross / target_scale

    weights, loadings, _ = simpls(covariance, cross, n_supervised)
    residual, ratio = residual_eigenvectors(
        covariance, weights, loadings, n_residual, random_state=random_state
    )
    combined = np.hstack([weights, residual])

    # Score statistics follow from the same covariance, so standardization is also free.
    score_variance = np.einsum("ij,jk,ki->i", combined.T, covariance, combined)
    score_scale = np.sqrt(np.maximum(score_variance, 0.0)) + 1e-12
    return LatentProjection(
        center=stats.mean_x.astype(np.float32),
        feature_scale=scale.astype(np.float32),
        weights=combined.astype(np.float32),
        score_mean=np.zeros(combined.shape[1], dtype=np.float32),
        score_scale=score_scale.astype(np.float32),
        n_supervised=int(weights.shape[1]),
        n_residual=int(residual.shape[1]),
        explained_variance_ratio=ratio,
    )
