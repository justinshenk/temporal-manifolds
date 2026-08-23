# temporal-manifolds

Experiments on the geometric structure of LLM residual-stream activations under
temporal and planning-horizon phrasings.

The core question is whether prompts that differ only in how a planning horizon
is expressed (for example, "4 weeks", "1 month", and "28 days") map onto a
low-dimensional temporal manifold.

## Fixed extraction contract

Every model-hook extraction in this repository is restricted to one activation:

- Component: `layer_out`
- Zero-based transformer layer: `21`
- Serialized key: `layer_out/21`
- Token position: `-1`, the final non-padding token of the fully formatted prompt

Batches are left-padded so position `-1` is the final prompt position for every
sample. The layer, component, and token position are constants rather than CLI
options, so callers cannot widen the extraction scope accidentally.

Cache conversational prompts with:

```bash
bash scripts/run_activation_caching_conversational_selected_acts.sh
```

Cache plain-English prompts to the isolated `plain_english_selected_acts` GCS
prefix with the same extraction settings:

```bash
bash scripts/run_activation_caching_plain_english_selected_acts.sh
```

Cache task-only prompts, which state the task with no horizon at all, to the
isolated `task_only_selected_acts` GCS prefix:

```bash
bash scripts/run_activation_caching_task_only_selected_acts.sh
```

Cache abstract prompts with:

```bash
bash scripts/run_activation_caching_abstract_selected_acts.sh
```

Generate the plain-English dataset, which uses the same tasks and horizons as
the conversational dataset without labelled fields or a required answer shape:

```bash
uv run python -m temporal_manifolds.dataset.generate --dataset plain_english \
  --randomize-template --output-path data/plain_english_prompts.json
```

Generate the task-only dataset, the time-free control condition. It reuses the
conversational tasks but never mentions a horizon, so it produces one prompt per
template and task rather than a product over the time grid:

```bash
uv run python -m temporal_manifolds.dataset.generate --dataset task_only \
  --output-path data/task_only_prompts.json
```

Generate the matched conversational time-free dataset, which has three explicit
task/goal/objective templates with no duration or deadline wording:

```bash
uv run python -m temporal_manifolds.dataset.generate \
  --dataset conversational_no_time \
  --output-path data/conversational_no_time_prompts.json
```

Generate the indirect-horizon dataset, whose prompts never write the horizon out
as a duration. Each one states it indirectly -- two wall-clock times,
two calendar dates, a fraction of a larger allowance, a count of fixed-length
passes -- so the horizon has to be derived rather than read off. It reuses the
conversational tasks and time grid, and renders every task and horizon at least
ten different ways:

```bash
uv run python -m temporal_manifolds.dataset.generate --dataset indirect_horizon   --output-path data/indirect_horizon_prompts.json
```

Cache indirect-horizon prompts to the isolated `indirect_horizon_selected_acts`
GCS prefix:

```bash
bash scripts/run_activation_caching_indirect_horizon_selected_acts.sh
```

Generate the event-anchored dataset, whose prompts express the planning horizon
through milestones, handoffs, reviews, transitions, and lifecycle boundaries.
It contains two explicit event prompts for every supported base unit and crosses
each one only with tasks for which that latent horizon is plausible:

```bash
uv run python -m temporal_manifolds.dataset.generate --dataset event_anchored \
  --output-path data/event_anchored_prompts.json
```

The dataset builds its own prompts instead of formatting one template string, so
`--randomize-template` and `--remove-time-constraints` are rejected rather than
silently ignored;
`task_only` and `conversational_no_time` provide explicit horizon-free controls.

## Pipeline

1. Generate parametric temporal prompts across equivalent horizon phrasings.
2. Cache `layer_out/21` at the prompt's final position.
3. Filter and aggregate comparable prompt metadata.
4. Fit PCA projections and optional reusable surface models.
5. Compare geometric structure across datasets and prompt variants.

## Quickstart

```bash
uv sync
cp .env.example .env
uv run pytest -q
```

## Local activation explorer

Launch the browser-based PCA explorer, then choose one or more folders containing
`activations_batch_*.pt` files:

```bash
uv run streamlit run apps/activation_explorer.py
```

The explorer accepts only batches whose sole activation is `layer_out/21` and
whose sole cached position is `-1`. It rejects broader or inconsistent caches.
Filtering, aggregation, PCA fitting, PCA reuse, and surface fitting all operate
on that fixed slice. **Aggregate by** starts empty of anything you did not pick:
the app never adds a grouping field on your behalf.

Every projection carries the reconstruction residual — what the retained
components could not rebuild — as `reconstruction_residual_rms` plus its leading
three PCA coordinates, `reconstruction_residual_PC1` through
`reconstruction_residual_PC3`. All of them are in the downloadable projection
CSV by default.

For large datasets, use **Local folders** instead of **Upload folders** and enter
one path per line. The app indexes metadata once and streams the fixed activation
slice into a disk-backed cache under `data/activation_explorer_cache/`.

## Nonlinear manifold explorer

The linear explorer fits PCA and log-time-horizon-supervised PLS. Its nonlinear
counterpart embeds the same fixed activation slice with Kernel PCA, which
applies PCA in an implicit feature space defined by a kernel and so can unfold
curvature that a linear subspace flattens:

```bash
uv run streamlit run apps/manifold_explorer.py
```

Kernel PCA was selected after comparing it against Isomap and UMAP on this data;
it outperformed both by a clear margin, so it is the only method offered.

The optimization target is fixed at `log10_time_horizon_months`. Kernel PCA is
unsupervised, so the target never influences the fit — it drives the reported
metrics only:

- **Variance captured** — cumulative share of kernel-space variance carried by
  the retained components, with a per-component table mirroring the linear
  explorer's.
- **Target R²** — how much of the horizon a linear readout of the embedding
  coordinates explains. Fitted and scored on the same points, so it measures
  geometric organization rather than predictive accuracy.
- **Target rank correlation** — the strongest single-axis Spearman correlation,
  which catches monotone but curved layouts.
- **Neighborhood target error** — mean absolute horizon difference among
  embedded neighbors, in target standard deviations. Lower is better.
- **Trustworthiness and continuity** — whether the embedding preserved genuine
  activation-space neighborhoods rather than inventing proximity.

Two caveats on explained variance. It is measured in the kernel's implicit
feature space, which is not activation-space variance: it describes how the
kernel's geometry is distributed, not how much of the original signal was kept.
And by default the ratios are shares *among the retained components*, so the
cumulative column reaches 100% by construction. Enable **Report variance against
the full spectrum** for the retained share of total kernel variance, which costs
a full N×N eigendecomposition. With a linear kernel the full-spectrum ratios
reproduce ordinary PCA's `explained_variance_ratio_` exactly, which the test
suite pins.

Source selection, metadata filtering, aggregation, and unconstrained-baseline
subtraction behave as in the linear explorer. **Aggregate by** is chosen by you
and defaults to `time_horizon_months`, plus `source_folder` when several folders
are loaded; nothing is added to the grouping behind your back. Adding `task`
is what makes each point belong to exactly one task, which the horizon
regression's split relies on — the app warns when it is absent rather than
forcing it.

### Horizon regression

The **Horizon regression** section fits a polynomial ridge model predicting
`log10_time_horizon_months` from the embedding coordinates, and reports R² and
RMSE on both the training and held-out splits. Degree, ridge alpha,
interaction-only terms, test fraction, and optional grouped cross-validation are
all configurable.

The train/test split is **disjoint by `task`**: every point sharing a task value
lands wholly in one side. This matters because each task recurs at many
horizons, so a random split would leave near-duplicate siblings of every test
point in the training set. The split reads whatever `task` label each point
carries; when `task` is not an aggregation field a point can mix tasks, and the
app says so rather than re-grouping the data. On a synthetic fixture with per-task offsets, a
random split reports R²=0.92 where the task-disjoint split reports 0.59 — the
gap is leakage, not skill. Cross-validation folds are grouped the same way.

Because `task` defines the split, the number of distinct tasks bounds what the
split can do: at least two are required, and the fold count is capped at the
task count. A negative test R² is meaningful rather than a bug — it says the
model does worse on unseen tasks than predicting their mean horizon.

Both the Kernel PCA embedding and the fitted regression can be downloaded as
versioned joblib artifacts and re-uploaded later via **Use saved**. A loaded
embedding re-projects the current points through its stored kernel; a loaded
regression is scored on whatever points are currently prepared, which the app
flags may overlap its original training data.

Only load PCA, manifold, regression, or surface-model joblib artifacts that you
trust.

```python
from temporal_manifolds.viz.surface_fitting import load_surface_model

surface, provenance = load_surface_model("activation_surface_PC3-from-PC1-PC2.joblib")
pc3 = surface.predict([[new_pc1, new_pc2]])[0]
outside_fit_range = surface.extrapolation_mask([[new_pc1, new_pc2]])[0]
```

## Layout

```text
apps/                      local Streamlit explorers (linear and nonlinear)
notebooks/                 fixed-slice analysis notebooks
scripts/                   fixed-contract caching and GCS wrappers
src/temporal_manifolds/    datasets, extraction policy, utilities, visualization
tests/                     contract, dataset, explorer, and surface tests
data/, results/            gitignored local artifacts
```
