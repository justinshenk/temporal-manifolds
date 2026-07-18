# experiments/

Research scripts that use `src/`.

## run_experiment.py — generate_llm_responses

Runs multi-stage planning conversations against a model and captures
residual-stream activations at boundary tokens.

```bash
uv run python experiments/run_experiment.py --config experiments/configs/main_qwen14b_nothink.json
# overrides: --model <id> --backend hf|mlx --max-samples N --force
```

What happens per prompt (task × target horizon × phrasing):

1. The chat markup for the model family is detected and **verified against the
   tokenizer** (single-token round-trips + rendered-template structure).
2. The conversation is driven turn-by-turn: overview → "Continue." → one step
   per turn → "Plan Completed". Token sequence is built incrementally so every
   turn keeps its full boundary anatomy (incl. empty `<think></think>` blocks
   in no-thinking mode on Qwen3 hybrids).
3. Boundary tokens (`turn_end`, `post_turn_end_nl`, `turn_start`, `role`,
   `post_role_nl`, `think_open`, `think_close`) are located in the final
   sequence, decode-checked, and attributed to turns/steps.
4. One capture forward pass records `resid_post` at 40/60/80% depth; vectors
   at boundary positions are saved.

Output (resumable; completed samples skipped):

```
output/runs/<run_id>/                 # fingerprint(prompt texts, protocol, depths)
  run_config.json  prompt_dataset.json
  <model>/manifest.json
  <model>/samples/<sample_uid>/
    conversation.json   # turns, parsed Step/Time horizon:, completion
    boundaries.json     # boundary map + depth→layer + marker token ids
    activations.npz     # "L{layer}_p{abs_pos}" -> float32 [d_model]
```

Retrieval anywhere:

```python
from src.capture.store import ResponseStore
store = ResponseStore("<run_id>")
X, rows = store.query("Qwen/Qwen3-14B", layer=25, kind="turn_end", role="assistant")
# rows[i]: prompt_id, target_horizon_years, step_index, step_horizon_years, ...
```

## analyze_geometry.py

PCA per (depth × boundary kind) slice + pooled assistant boundaries; renders
2D and 3D scatters colored by `target_horizon`, `step_horizon`, `prompt_id`,
`task_id`, `turn_index`. Figures + `figures_manifest.json` land in
`output/runs/<run_id>/<model>/figures/`.

```bash
uv run python experiments/analyze_geometry.py --run <run_id> --model Qwen/Qwen3-14B
```

## configs/

- `smoke_small.json` — SmolLM2-135M end-to-end sanity (fast)
- `smoke_qwen06.json` — Qwen3-0.6B, exercises empty-think-block capture
- `smoke_qwen06_thinkcap.json` — capped-thinking code path check
- `main_qwen14b_nothink.json` — main experiment: Qwen3-14B, 4 tasks ×
  5 target horizons (1 day … 50 years), no-thinking, greedy
