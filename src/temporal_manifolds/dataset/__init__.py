"""Parametric prompt-dataset generation.

A *scenario* defines a prompt template parameterised by a planning horizon
(in months). For every horizon we emit one prompt per *phrasing* in the
equivalent-phrasing group ("4 weeks" ≡ "1 month" ≡ "28 days"), so the resulting
dataset has columns:

    scenario, horizon_months, phrasing_group, phrasing, prompt, split

Train/test split is by phrasing-group, not by row, so held-out phrasings
exercise the question we actually care about: do equivalent phrasings collapse
onto the same point on the manifold?
"""

from temporal_manifolds.dataset.generate import generate_dataset

__all__ = ["generate_dataset"]
