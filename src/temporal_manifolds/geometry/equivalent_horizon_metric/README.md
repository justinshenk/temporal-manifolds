# Equivalent-horizon metric transform

This replaces trajectory scaling with a supervised three-dimensional metric
whose training objective directly contracts points having the same
`log10_time_horizon_months`.

All `conv` rows are removed **before** fitting, validation, centering, or CSV
transformation. The fitted data contains:

- `abst`: 2,820 rows
- `nof_conv`: 974 rows
- `plain`: 974 rows

## Transformation

For each point, construct

```text
x = [PLS1, PLS2, PLS3, rho, rho^2, rho^3, rho^4]
```

where `rho = reconstruction_residual_rms`. After standardizing these features,
the fit calculates equal-horizon-weighted within-horizon scatter `S_w` and
between-horizon scatter `S_b`, then solves

```text
S_b v = lambda (S_w + 1e-5 I) v
```

The leading three directions form the learned metric. A similarity-Procrustes
alignment maps the result back to the original PLS units, orientation, and
centroid. This produces `PLS1_metric`, `PLS2_metric`, and `PLS3_metric`.

Fitting uses horizon labels. Transforming a new point uses only its three PLS
scores and residual RMS; it does not use its horizon, task, source folder, or
other points from its trajectory.

## Task-held-out validation

Five-fold `GroupKFold` holds out complete tasks, including all source variants
of each held-out task.

| Metric | Before | After | Change |
|---|---:|---:|---:|
| Equal-horizon pairwise RMS distance | 19.948 | 13.365 | -33.0% |
| Normalized equal-horizon RMS | 0.557 | 0.392 | -29.5% |
| Horizon-centroid RMS spread | 36.457 | 33.997 | -6.7% |
| Median within-trajectory distance correlation | 1.000 | 0.966 | -3.4% |

The normalized metric divides equal-horizon RMS distance by horizon-centroid
spread. It therefore cannot be improved merely by shrinking all coordinates.

Using combined out-of-fold predictions, equal-horizon RMS improves separately
within every included source:

| Source | Before | After | Change |
|---|---:|---:|---:|
| `abst` | 16.837 | 9.851 | -41.5% |
| `nof_conv` | 28.407 | 21.011 | -26.0% |
| `plain` | 20.947 | 15.513 | -25.9% |

Cross-source distance between equal-horizon folder centroids falls from 23.910
to 20.558 (-14.0%).

## Usage

Fit and create transformed coordinates:

```bash
python equivalent_horizon_metric_transform.py fit input.csv \
  --model-out equivalent_horizon_metric.joblib \
  --transformed-out transformed_coordinates.csv \
  --validation-out validation.json
```

Transform another CSV:

```bash
python equivalent_horizon_metric_transform.py transform \
  equivalent_horizon_metric.joblib input.csv output.csv
```

Transform one unseen point:

```python
from equivalent_horizon_metric_transform import load_metric_transform

transform, metadata = load_metric_transform("equivalent_horizon_metric.joblib")
new_pls = transform.transform(
    [pls1, pls2, pls3],
    reconstruction_residual_rms,
)
```

By default, residual RMS is clipped to the range observed during training to
prevent unstable polynomial extrapolation.

## Limitations

- The transform deliberately changes the metric and compresses nuisance-heavy
  directions. Median local trajectory-step length is about 61% of the original,
  although relative within-trajectory geometry remains highly correlated.
- Validation covers unseen tasks drawn from the included `abst`, `nof_conv`,
  and `plain` sources. It should be refitted before use on a genuinely new
  source distribution.
- `PLS*_metric` values are aligned to the original PLS frame but are learned
  coordinates, not the original PLS scores.

