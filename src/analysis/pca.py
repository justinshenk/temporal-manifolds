"""Minimal PCA on activation matrices (numpy SVD; no sklearn dependency)."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class PCAResult:
    projected: np.ndarray  # [n, k]
    components: np.ndarray  # [k, d]
    explained_variance_ratio: np.ndarray  # [k]
    mean: np.ndarray  # [d]


def fit_pca(X: np.ndarray, n_components: int = 3) -> PCAResult:
    if X.ndim != 2 or X.shape[0] < 2:
        raise ValueError(f"PCA needs [n>=2, d] matrix, got {X.shape}")
    k = min(n_components, X.shape[0] - 1, X.shape[1])
    mean = X.mean(axis=0)
    Xc = (X - mean).astype(np.float64)
    U, S, Vt = np.linalg.svd(Xc, full_matrices=False)
    var = (S**2) / (X.shape[0] - 1)
    ratio = var / var.sum() if var.sum() > 0 else var
    return PCAResult(
        projected=(Xc @ Vt[:k].T).astype(np.float32),
        components=Vt[:k].astype(np.float32),
        explained_variance_ratio=ratio[:k].astype(np.float32),
        mean=mean.astype(np.float32),
    )
