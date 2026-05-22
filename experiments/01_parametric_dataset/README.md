# 01 — Parametric Dataset

> **Category**: Parametric Dataset · **Member**: Shantanu · **Status**: In progress
> **Dates**: 2026-05-20 → 2026-05-23
> **Output**: Code for generating datasets algorithmically. Scenario specified through config file.
> **Comments**: Multiple phrasings for the same time horizon are necessary (e.g. 4 weeks ≡ 1 month). A fraction is held out for the test set when used to train the predictive model.

## Entry point

```bash
uv run python -m experiments.01_parametric_dataset.run \
    --config configs/scenarios/example.yaml \
    --out data/example/
```

Writes:
- `data/example/dataset.jsonl` — one row per (scenario, phrasing) pair
- `data/example/config.snapshot.yaml` — config used for the run
