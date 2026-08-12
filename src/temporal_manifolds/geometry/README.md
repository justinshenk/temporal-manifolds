# Residual trajectory scaler

This is a deliberately simple single-point correction for the supplied
3-dimensional PLS space. At inference it uses only:

1. one raw 2,560-dimensional activation, and
2. its three PLS scores.

It does **not** use `task`, `source_folder`, the target/time horizon, a group
centroid, or any other point from the trajectory.

## Method

The reconstruction residual is computed in the PLS model's standardized input
space:

```text
X_standardized = (X - x_mean) / x_std
R = X_standardized - PLS_scores @ x_loadings.T
rho = sqrt(mean(R ** 2))
```

The fitted one-feature model is:

```text
predicted_scale = exp(1.0745485649 - 0.7182994833 * rho)
```

The corrected PLS point is:

```text
PLS_corrected = center + (PLS - center) / predicted_scale
```

where `center = [7.10112485, -7.26315247, -7.56882737]` is fixed. The fitted
object clips `rho` to the observed training range by default so an out-of-range
single point cannot produce an extreme extrapolated scale.

The total arc length is not forced to match. Scale was defined relative to a
shared curve as a function of `log10_time_horizon_months`, so a `conv`
trajectory covering a shorter time range remains a shorter arc after correction.

## Usage

```python
from residual_trajectory_scaler import load_fitted_scaler

scaler = load_fitted_scaler()

# One point: raw_activation.shape == (2560,), pls_scores.shape == (3,)
corrected_pls = scaler.transform(raw_activation, pls_scores)

# Optional diagnostics for that same point
corrected_pls, predicted_scale, residual_rms = (
    scaler.transform_with_diagnostics(raw_activation, pls_scores)
)

# Batches also work: (n, 2560) and (n, 3) -> (n, 3)
corrected_batch = scaler.transform(raw_batch, pls_batch)
```

## Validation on the attached data

- Samples: 4,768
- Trajectories (`task` + `source_folder`): 84
- Validation: 5-fold grouped cross-validation holding out entire tasks
- Held-out log-scale prediction R²: 0.84
- Median absolute single-point scale error: 15.5%
- Translation-only common-curve R² before correction: 0.738
- Translation-only common-curve R² after held-out-task correction: 0.887
- Flexible per-trajectory scale + translation fit R²: 0.948
- Standard deviation of log trajectory scales: 0.864 before, 0.323 after

The overall R² is driven substantially by the large `conv` versus non-`conv`
scale difference. Within a source folder, this one-scalar model leaves more
task-to-task variation; that is the intended tradeoff for simplicity and robust
single-point use.
