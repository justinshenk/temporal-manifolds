"""Streaming sufficient statistics for exact linear algebra over out-of-core activations.

Every model here is a linear map followed by a small nonlinear expansion, so a single pass
that accumulates ``n``, ``sum(X)``, ``X'X``, ``X'Y``, ``sum(Y)`` and ``sum(Y*Y)`` is enough to
recover PLS, PCA and ridge solutions *exactly* -- no approximation, and peak memory stays at
one chunk plus a 2560x2560 matrix.

The accumulators are additive, which is what makes k-fold cheap: statistics are gathered once
per fold-of-origin during a single pass, and a fold's training statistics are then the total
minus that fold's own block. Five folds therefore cost one pass over the data, not five.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np


@dataclass
class SufficientStatistics:
    """Additive second-moment statistics for a feature block and its targets."""

    n: float
    sum_x: np.ndarray  # (d,)
    sum_y: np.ndarray  # (m,)
    xtx: np.ndarray  # (d, d)
    xty: np.ndarray  # (d, m)
    yty: np.ndarray  # (m, m)

    @classmethod
    def zeros(cls, n_features: int, n_targets: int) -> SufficientStatistics:
        return cls(
            n=0.0,
            sum_x=np.zeros(n_features, dtype=np.float64),
            sum_y=np.zeros(n_targets, dtype=np.float64),
            xtx=np.zeros((n_features, n_features), dtype=np.float64),
            xty=np.zeros((n_features, n_targets), dtype=np.float64),
            yty=np.zeros((n_targets, n_targets), dtype=np.float64),
        )

    def partial_fit(self, x: np.ndarray, y: np.ndarray) -> SufficientStatistics:
        """Fold one chunk into the running totals.

        The chunk products are formed in float32 (BLAS-fast, and the inputs are float16-backed
        activations) and accumulated in float64 so that a quarter-million-row sum stays exact
        enough for the downstream Cholesky solves.
        """
        x = np.ascontiguousarray(x, dtype=np.float32)
        y = np.atleast_2d(np.asarray(y, dtype=np.float64))
        if y.shape[0] != x.shape[0]:
            y = y.T
        if y.shape[0] != x.shape[0]:
            raise ValueError(f"Chunk has {x.shape[0]} rows but {y.shape[0]} targets.")
        y32 = np.ascontiguousarray(y, dtype=np.float32)

        self.n += float(x.shape[0])
        self.sum_x += x.sum(axis=0, dtype=np.float64)
        self.sum_y += y.sum(axis=0)
        self.xtx += (x.T @ x).astype(np.float64)
        self.xty += (x.T @ y32).astype(np.float64)
        self.yty += y.T @ y
        return self

    def __add__(self, other: SufficientStatistics) -> SufficientStatistics:
        return SufficientStatistics(
            n=self.n + other.n,
            sum_x=self.sum_x + other.sum_x,
            sum_y=self.sum_y + other.sum_y,
            xtx=self.xtx + other.xtx,
            xty=self.xty + other.xty,
            yty=self.yty + other.yty,
        )

    def __sub__(self, other: SufficientStatistics) -> SufficientStatistics:
        return SufficientStatistics(
            n=self.n - other.n,
            sum_x=self.sum_x - other.sum_x,
            sum_y=self.sum_y - other.sum_y,
            xtx=self.xtx - other.xtx,
            xty=self.xty - other.xty,
            yty=self.yty - other.yty,
        )

    def copy(self) -> SufficientStatistics:
        return replace(
            self,
            sum_x=self.sum_x.copy(),
            sum_y=self.sum_y.copy(),
            xtx=self.xtx.copy(),
            xty=self.xty.copy(),
            yty=self.yty.copy(),
        )

    @property
    def mean_x(self) -> np.ndarray:
        return self.sum_x / self.n

    @property
    def mean_y(self) -> np.ndarray:
        return self.sum_y / self.n

    def centered_covariance(self) -> np.ndarray:
        """Return ``X'X / n`` after centering, symmetrised against round-off drift."""
        mean = self.mean_x
        cov = self.xtx / self.n - np.outer(mean, mean)
        return (cov + cov.T) * 0.5

    def centered_cross_covariance(self) -> np.ndarray:
        """Return ``X'Y / n`` after centering both blocks."""
        return self.xty / self.n - np.outer(self.mean_x, self.mean_y)

    def centered_target_covariance(self) -> np.ndarray:
        mean = self.mean_y
        cov = self.yty / self.n - np.outer(mean, mean)
        return (cov + cov.T) * 0.5

    def feature_scale(self, *, epsilon: float = 1e-8) -> np.ndarray:
        """Per-feature standard deviation, floored so constant features stay finite."""
        variance = np.diag(self.centered_covariance())
        return np.sqrt(np.maximum(variance, 0.0)) + epsilon


def save_statistics(path, blocks: list[SufficientStatistics]) -> None:
    """Persist per-fold statistics so component sweeps never re-read the activations."""
    np.savez(
        path,
        n=np.array([block.n for block in blocks]),
        sum_x=np.stack([block.sum_x for block in blocks]),
        sum_y=np.stack([block.sum_y for block in blocks]),
        xtx=np.stack([block.xtx for block in blocks]).astype(np.float32),
        xty=np.stack([block.xty for block in blocks]),
        yty=np.stack([block.yty for block in blocks]),
    )


def load_statistics(path) -> list[SufficientStatistics]:
    """Restore statistics written by :func:`save_statistics`."""
    payload = np.load(path)
    return [
        SufficientStatistics(
            n=float(payload["n"][index]),
            sum_x=payload["sum_x"][index],
            sum_y=payload["sum_y"][index],
            xtx=payload["xtx"][index].astype(np.float64),
            xty=payload["xty"][index],
            yty=payload["yty"][index],
        )
        for index in range(len(payload["n"]))
    ]


def ridge_solution(
    stats: SufficientStatistics, alpha: float, *, scale: np.ndarray | None = None
) -> tuple[np.ndarray, np.ndarray]:
    """Solve a centered ridge problem from sufficient statistics.

    Returns ``(coefficients, intercepts)`` for ``y ~ X @ coef + intercept``. When ``scale`` is
    given the penalty is applied in standardized coordinates, which keeps a single ``alpha``
    meaningful across features whose variances differ by orders of magnitude.
    """
    cov = stats.centered_covariance()
    cross = stats.centered_cross_covariance()
    if scale is None:
        scale = np.ones(cov.shape[0], dtype=np.float64)
    scaled_cov = cov / np.outer(scale, scale)
    scaled_cross = cross / scale[:, None]
    penalised = scaled_cov + alpha * np.eye(scaled_cov.shape[0])
    coefficients = np.linalg.solve(penalised, scaled_cross) / scale[:, None]
    intercepts = stats.mean_y - stats.mean_x @ coefficients
    return coefficients, intercepts
