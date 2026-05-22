"""Predictor models over residual activations.

Components:
    - linear baseline (linear regression: activations → planning horizon)
    - predictive head over concatenated multi-position activations,
      optionally trained with an auxiliary phrasing-equivalence objective
"""
