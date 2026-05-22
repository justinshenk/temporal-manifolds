# 07 — Predictive Model

> **Category**: Geometric Modeling (predictive head) · **Status**: Not started · **Track**: Conference
> **Output**: A model that takes concatenated node activations and returns the planning horizon.
> **Comments**: One option — use the distance between equivalent phrasings as part of the objective.

## TODO
- [ ] Define the head architecture (MLP over concatenated multi-position activations).
- [ ] Train with primary MSE (or cross-entropy over log-horizon bins) + optional phrasing-equivalence auxiliary loss.
- [ ] Hand off to experiment 08 for held-out eval.
