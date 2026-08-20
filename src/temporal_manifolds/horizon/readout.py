"""Readouts that turn a multi-target head into one horizon estimate.

A least-squares scalar head is forced to answer with a conditional *mean*, which is the wrong
shape for this target: the prompts state a horizon on a discrete grid (10 units x ~30 numerals),
so the conditional distribution over log10 months is multi-modal, and the mean of two plausible
units lands in the gap between them. Predicting a distribution over bins and reading back its
expectation keeps the estimator continuous while letting the head express "either hours or
days, not something in between".
"""

from __future__ import annotations

import numpy as np


def bin_edges(values: np.ndarray, n_bins: int) -> np.ndarray:
    """Equal-width edges spanning the observed horizon range."""
    low, high = float(np.min(values)), float(np.max(values))
    margin = (high - low) / (2 * n_bins)
    return np.linspace(low - margin, high + margin, n_bins + 1)


def bin_indicators(values: np.ndarray, edges: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return the one-hot bin membership matrix and the bin centres."""
    centres = 0.5 * (edges[:-1] + edges[1:])
    index = np.clip(np.digitize(values, edges) - 1, 0, len(centres) - 1)
    indicators = np.zeros((values.size, centres.size))
    indicators[np.arange(values.size), index] = 1.0
    return indicators, centres


def expectation_readout(
    scores: np.ndarray, centres: np.ndarray, *, temperature: float | None = None
) -> np.ndarray:
    """Collapse per-bin head outputs into a scalar horizon.

    With ``temperature`` the bin scores are treated as logits and softmaxed; without it they are
    treated as (noisy) probabilities, clipped at zero and renormalised. The softmax form is the
    sharper of the two -- it can concentrate almost all mass on one unit -- while the clipped
    form degrades gracefully when the head is uncertain.
    """
    if temperature is None:
        weights = np.clip(scores, 0.0, None)
        total = weights.sum(axis=1, keepdims=True)
        # A row the head pushed entirely negative carries no usable shape; fall back to uniform.
        weights = np.where(total > 1e-9, weights, 1.0)
        total = weights.sum(axis=1, keepdims=True)
        return (weights @ centres) / total[:, 0]
    shifted = scores * temperature
    shifted -= shifted.max(axis=1, keepdims=True)
    weights = np.exp(shifted)
    return (weights @ centres) / weights.sum(axis=1)


def blend(*predictions: np.ndarray, weights: np.ndarray | None = None) -> np.ndarray:
    """Average several readouts of the same head.

    The readouts are deterministic functions of one fitted head, so averaging them adds no
    parameters and cannot leak: it is a fixed choice of estimator, not a second fit.
    """
    stacked = np.stack(predictions)
    if weights is None:
        return stacked.mean(axis=0)
    weights = np.asarray(weights, dtype=np.float64)
    return np.tensordot(weights / weights.sum(), stacked, axes=(0, 0))
