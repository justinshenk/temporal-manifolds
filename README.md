# temporal-manifolds

Experiments on the geometric structure of LLM residual-stream activations under
temporal / planning-horizon phrasings. Targeting a **BlackboxNLP 2026** workshop
submission (deadline **2026-07-17**).

Project ledger: [Temporal Manifolds (Google Sheet)](https://docs.google.com/spreadsheets/d/1hyTdfxwrIedeaYX2Ri5xhSj5VCqW0T4EW90J2H6s6p0/edit?gid=0#gid=0).
The same ledger is mirrored in [`ROADMAP.md`](ROADMAP.md) and used to organize
the `experiments/` directory.

## What we're studying

Given prompts that differ only in how a planning horizon is expressed
("4 weeks" vs. "1 month" vs. "28 days"), do the residual-stream activations
collapse onto a low-dimensional **temporal manifold**? And how does that
manifold vary by scenario?

Pipeline:

1. **Parametric dataset** — generate prompts across multiple scenarios, each
   with a set of equivalent phrasings for the same time horizon.
2. **Activation caching** — cache residuals at 5 key token positions
   (identified in prior work) and publish the tensors to the Hugging Face Hub.
3. **Geometric modeling** — PCA per scenario, manifold alignment across
   scenarios, replication of plots from `arxiv:2605.05115`.
4. **Constraint phrasing** — a metric for whether equivalent phrasings map to
   the same point.
5. **Linear baseline** — residual → planning-horizon regression as a baseline.
6. **Predictive model** — a learned head over concatenated activations.
7. **Evaluation** — held-out accuracy + comparisons to existing evals.
8. **Cross-model generalization** — repeat on other models.

## Quickstart

```bash
uv sync                                  # install deps (Python 3.12+)
cp .env.example .env                     # fill in HF_TOKEN, etc.
uv run pytest -q                         # smoke tests

# Generate a tiny example dataset
uv run python -m experiments.01_parametric_dataset.run \
    --config configs/scenarios/example.yaml \
    --out data/example/
```

## Layout

```
configs/scenarios/        scenario YAMLs (templates, phrasing groups, splits)
src/temporal_manifolds/   shared library (dataset, activations, geometry, ...)
experiments/NN_<name>/    one folder per ledger row; entry point + README
notebooks/                exploratory work (clean before committing)
scripts/                  thin CLI wrappers for batch / cluster runs
tests/                    pytest
data/, results/           gitignored
```

## Related prior work in this org

- [`temporal-awareness`](../temporal-awareness/) — closest predecessor; we
  reuse its prompt-pair patterns and activation-extraction style.
- [`temporal/latents`](../temporal/latents/) — CAA framework; reused for
  hook-based residual extraction patterns.
- External: `goodfire-ai/causalab` branch `manifold_steering` — referenced for
  replicating manifold plots.

## Status

Local repo, no remote yet. Add a remote when collaboration starts:

```bash
gh repo create temporal-manifolds --private --source=. --remote=origin
```
