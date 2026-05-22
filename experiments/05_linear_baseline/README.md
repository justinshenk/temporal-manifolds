# 05 — Linear Baseline

> **Category**: Linear Baseline · **Status**: Not started · **Track**: Conference
> **Output**: A linear regression model that predicts planning horizon from residual-stream activations.
> **Comments**: Serves as a baseline for comparing our results.

## TODO
- [ ] Fit `sklearn.linear_model.Ridge` per cached position; report R² on train + test.
- [ ] Save coefficients alongside the cached activation tensors.
