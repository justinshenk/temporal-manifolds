# RMS-conditioned spline-surface transformer

`rms_spline_surface_transformer.py` fits a two-dimensional surface in the
three-dimensional PLS space. It uses the fitted `abst` spline as the reference
curve and uses `residual_rms` to model the displacement of `conv` and
`nof_conv` away from that curve.

This is the PLS-coordinate method. It does **not** use raw activation
coordinates. After fitting, a single unseen point requires only:

- `PLS1`, `PLS2`, and `PLS3`;
- its `residual_rms` value; and
- the saved surface model.

It does not require the point's task, source-folder label, or true time horizon.

> The supplied CSV uses the name `nof_conv`; this is the dataset previously
> referred to as `of_conv`.

## Surface model

Let

- \(C(t)\in\mathbb{R}^3\) be the supplied `abst` spline;
- \(t=\log_{10}(\text{time horizon in months})\);
- \(r\) be `residual_rms`; and
- \(Z=(\mathrm{PLS1},\mathrm{PLS2},\mathrm{PLS3})\).

The fitted surface is

\[
S(t,r)=C(t)+E(u(r)),
\]

where

\[
u(r)=\frac{\log_{10}(r)-\log_{10}(r_{\mathrm{ref}})}{q_{\mathrm{scale}}}
\]

and the extrusion is a cubic 3D curve:

\[
E(u)=B_1u+B_2u^2+B_3u^3.
\]

Each \(B_k\) is a three-element vector, so the extrusion can curve jointly in
the PLS1, PLS2, and PLS3 directions.

There is deliberately no polynomial intercept. Consequently,
\(E(0)=0\), and the original `abst` spline is an exact edge of the surface:

\[
S(t,r_{\mathrm{ref}})=C(t).
\]

The fitted reference is the lower observed `abst` RMS boundary,
`r_ref = 0.4234717358`.

## What fitting does

For every `conv` and `nof_conv` training row, the script:

1. Reads its known training horizon \(t_i\).
2. Evaluates the reference spline at that horizon, \(C(t_i)\).
3. Calculates its displacement from the reference spline:

   \[
   D_i=Z_i-C(t_i).
   \]

4. Converts `residual_rms` to the normalized log-RMS coordinate \(u_i\).
5. Fits the no-intercept ridge regression

   \[
   D_i\approx B_1u_i+B_2u_i^2+B_3u_i^3.
   \]

`source_folder` is used to select the fitting rows and report per-dataset
metrics. It is not an input to the polynomial and is not needed at inference.

## Installation

Python 3.10 or newer is recommended.

```bash
pip install numpy pandas scipy scikit-learn joblib matplotlib
```

Keep these files together:

```text
rms_spline_surface_transformer.py
rms_spline_surface_model.joblib
```

## Expected fitting inputs

### CSV

The fitting CSV must contain:

| Column | Purpose |
|---|---|
| `PLS1`, `PLS2`, `PLS3` | Coordinates to model |
| `log10_time_horizon_months` | Known spline parameter used during fitting |
| `residual_rms` | Extrusion predictor |
| `source_folder` | Identifies `abst`, `conv`, and `nof_conv` rows |
| `task` | Groups complete trajectories during cross-validation |

`residual_rms` must be finite and strictly positive because the model uses its
base-10 logarithm.

### Spline artifact

The curve file must contain the fitted three-coordinate `abst` spline and its
parameter normalization. The loader includes a compatibility path for the
supplied `temporal_manifolds` artifact, so that package is not required merely
to load the spline.

## Fit from the command line

```bash
python rms_spline_surface_transformer.py \
  --csv all_rescaled_activation_pls_projection.csv \
  --curve final_activation_curve_PLS1-PLS2-PLS3-by-log10_time_horizon_months_spline.joblib \
  --output-dir surface_results
```

The command-line pipeline:

1. evaluates polynomial degrees 1–4 with eight-fold, task-grouped validation;
2. fits the final cubic extrusion to all `conv` and `nof_conv` rows;
3. saves the reusable transformer;
4. projects all CSV rows onto the surface;
5. writes diagnostics and augmented coordinates; and
6. produces the 3D figures.

The final cubic degree is currently fixed in `main()`. Change the
`degree=3` argument there if a different final degree is required.

## Use the fitted model on one unseen point

```python
import numpy as np

from rms_spline_surface_transformer import RMSSplineSurfaceTransformer

model = RMSSplineSurfaceTransformer.load(
    "rms_spline_surface_model.joblib"
)

# Shape: (n_points, 3). A single point is still represented as one row.
z = np.array([[pls1, pls2, pls3]], dtype=float)
r = np.array([residual_rms], dtype=float)

result = model.project(z, r)

log10_months_hat = result["parameter"][0]
z_on_surface = result["surface_points"][0]
distance_to_surface = result["surface_distance"][0]
```

The prediction procedure is:

1. predict the RMS-dependent offset \(E(u(r))\);
2. remove it from the observed point, producing
   \(Z_{\mathrm{dewarped}}=Z-E(u(r))\);
3. find the closest point on the `abst` spline to the dewarped point; and
4. add the extrusion back to obtain the corresponding surface point.

The closest-spline search first uses a dense grid and a KD-tree, then refines
the selected parameter with bounded scalar minimization.

## Batch inference

The same API accepts multiple points:

```python
z = dataframe[["PLS1", "PLS2", "PLS3"]].to_numpy()
r = dataframe["residual_rms"].to_numpy()

result = model.project(z, r)
dataframe["log10_months_hat"] = result["parameter"]
dataframe[["surface_PLS1", "surface_PLS2", "surface_PLS3"]] = (
    result["surface_points"]
)
```

## Main API

### `RMSSplineSurfaceTransformer.load(path)`

Loads a previously fitted `rms_spline_surface_model.joblib` artifact.

### `RMSSplineSurfaceTransformer.fit(frame, curve_artifact, ...)`

Fits the extrusion from a DataFrame and the reference spline. The default fit
sources are `conv` and `nof_conv`, the default base source is `abst`, and the
default polynomial degree is three.

### `model.extrusion_coordinate(residual_rms)`

Converts residual RMS into the normalized scalar extrusion coordinate \(u\).

### `model.extrusion_offset(residual_rms)`

Returns the learned 3D offset \(E(u)\), with one PLS1/PLS2/PLS3 offset per
input point.

### `model.surface(parameter, residual_rms)`

Evaluates \(S(t,r)\) directly when both the spline parameter and residual RMS
are known.

### `model.dewarp(coordinates, residual_rms)`

Removes the learned RMS extrusion:

\[
Z_{\mathrm{dewarped}}=Z-E(u(r)).
\]

This moves a point into the reference-spline coordinate frame.

### `model.rewarp(dewarped_coordinates, residual_rms)`

Adds the extrusion back. Apart from floating-point error,
`rewarp(dewarp(Z, r), r)` equals `Z`.

### `model.predict_parameter(coordinates, residual_rms)`

Dewarps each point and returns the nearest spline parameter, interpreted here
as predicted `log10_time_horizon_months`.

### `model.project(coordinates, residual_rms)`

Runs the complete projection. It returns:

| Key | Meaning |
|---|---|
| `parameter` | Estimated `log10_time_horizon_months` |
| `extrusion_coordinate` | Normalized log-RMS coordinate \(u\) |
| `surface_points` | Corresponding 3D points on the fitted surface |
| `dewarped_coordinates` | Input coordinates after removing the extrusion |
| `residual_vectors` | Input point minus its surface projection |
| `surface_distance` | Euclidean norm of each residual vector |
| `signed_normal_distance` | Signed displacement along the local surface normal |
| `surface_normals` | Estimated unit normal vectors |

## Augmented CSV columns

`all_pls_with_rms_surface_projection.csv` retains the original columns and
adds:

| Columns | Meaning |
|---|---|
| `extrusion_u` | Normalized log-RMS coordinate |
| `extrusion_offset_PLS1/2/3` | Fitted RMS-dependent 3D offset |
| `dewarped_PLS1/2/3` | Original coordinates with the extrusion removed |
| `surface_PLS1/2/3` | Coordinates of the closest fitted-surface point |
| `surface_parameter_hat` | Estimated log-horizon |
| `surface_parameter_error` | Estimated minus known training log-horizon |
| `surface_distance` | Euclidean point-to-surface distance |
| `surface_signed_normal_distance` | Signed local-normal distance |
| `flat_parameter` | Position along the reference spline |
| `flat_extrusion` | Position across the RMS extrusion |
| `flat_normal` | Displacement perpendicular to the surface |

In the straightened coordinate system, the fitted surface is simply
`flat_normal = 0`.

## Task-grouped validation result

Eight-fold validation held out complete tasks rather than randomly splitting
individual points.

| Dataset | Displacement R2 | Projected-surface RMSE per PLS coordinate | Horizon R2 | Horizon RMSE (log10 months) |
|---|---:|---:|---:|---:|
| Combined | 0.991 | 5.05 | 0.848 | 0.954 |
| `conv` | 0.994 | 5.99 | 0.833 | 1.003 |
| `nof_conv` | 0.899 | 3.89 | 0.864 | 0.904 |

`Displacement R2` measures how much of the squared displacement from the base
spline is explained by the RMS-conditioned extrusion. The horizon metrics are
computed after estimating the spline parameter from PLS coordinates and RMS.

## Generated files

| File | Contents |
|---|---|
| `rms_spline_surface_model.joblib` | Fitted reusable model |
| `all_pls_with_rms_surface_projection.csv` | Original and transformed coordinates |
| `spline_surface_diagnostics.json` | Coefficients, degree comparison, and validation metrics |
| `spline_surface_final_3d.png` | Original, surface-projected, and dewarped 3D views |
| `spline_surface_straightened_3d.png` | Straightened-coordinate 3D views |

## Limitations and extrapolation

- The model assumes an additive, separable surface `C(t) + E(r)`: the shape of
  the RMS extrusion does not change with horizon. This is intentionally simple
  and was sufficient for the current data.
- By default, RMS values outside the fitted interval
  `[0.4234717358, 3.0711044]` are clipped. Pass `clip=False` only if cubic
  extrapolation is intentional.
- The supplied spline reports an upper training bound of `5.07918` log10
  months, while 20 fitted-target rows extend to `6.07918`. Those rows use the
  spline's native extrapolation.
- To restrict prediction to the original spline training range, use:

  ```python
  bounds = tuple(model.curve_parameter_bounds)
  result = model.project(z, r, parameter_bounds=bounds)
  ```
- A low surface distance means the point is geometrically compatible with the
  learned surface. It does not by itself establish that the inferred horizon
  is causally represented by the model.
