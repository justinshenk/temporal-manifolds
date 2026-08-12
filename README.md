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

The no-output-format conversational variant uses the same fixed extraction
contract and writes to an isolated artifact namespace:

```bash
bash scripts/run_activation_caching_nof_conversational_selected_acts.sh
```

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
on that fixed slice.

For large datasets, use **Local folders** instead of **Upload folders** and enter
one path per line. The app indexes metadata once and streams the fixed activation
slice into a disk-backed cache under `data/activation_explorer_cache/`.

Only load PCA or surface-model joblib artifacts that you trust.

```python
from temporal_manifolds.viz.surface_fitting import load_surface_model

surface, provenance = load_surface_model("activation_surface_PC3-from-PC1-PC2.joblib")
pc3 = surface.predict([[new_pc1, new_pc2]])[0]
outside_fit_range = surface.extrapolation_mask([[new_pc1, new_pc2]])[0]
```

## Layout

```text
apps/                      local Streamlit explorer
notebooks/                 fixed-slice analysis notebooks
scripts/                   fixed-contract caching and GCS wrappers
src/temporal_manifolds/    datasets, extraction policy, utilities, visualization
tests/                     contract, dataset, explorer, and surface tests
data/, results/            gitignored local artifacts
```
