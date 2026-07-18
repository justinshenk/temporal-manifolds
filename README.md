# temporal-manifolds

Experiments on the geometric structure of LLM residual-stream activations
under temporal / planning-horizon framings.

**Current project: temporal manifolds in multi-stage conversations.** Building
on [arXiv:2606.05194](https://arxiv.org/pdf/2606.05194) (temporal preference is
localizable; the change-of-turn token sequence is where the horizon manifold
collapses into committed preference), we track how time-horizon geometry
**evolves across the turns of a planning conversation**: the model plans a task
under a target horizon, is stepped through the plan with "Continue.", and we
capture residual-stream activations at 40/60/80% layer depth at change-of-turn
tokens and thinking delimiters (never inside chain-of-thought).

See [`DESIGN.md`](DESIGN.md) for the architecture and
[`experiments/README.md`](experiments/README.md) for how to run.

## Quickstart

```bash
uv sync                                  # install deps (Python 3.12+)
uv run pytest -q                         # tests (incl. live tokenizer checks)

# end-to-end smoke on a tiny model
uv run python experiments/run_experiment.py --config experiments/configs/smoke_small.json

# main run (Qwen3-14B, no-thinking) + analysis
uv run python experiments/run_experiment.py --config experiments/configs/main_qwen14b_nothink.json
uv run python experiments/analyze_geometry.py --run <run_id> --model Qwen/Qwen3-14B
```

## Layout

```
src/core/           schemas + deterministic ids, TimeValue, file/device/path utils
src/chat_markup/    per-family × per-generation chat token anatomy
                    (turn tokens, think delimiters), VERIFIED against real
                    tokenizers at engine init and in tests
src/datasets/       planning prompt datasets: tasks (crossed families ×
                    natural time scales) × target horizons × ecologically
                    valid horizon phrasings
src/engine/         model engines: HF transformers (MPS/CUDA/CPU) and MLX
                    (Apple Silicon, big quantized models); targeted
                    resid_post capture; depth→layer convention
src/conversation/   plan-then-Continue protocol driver, thinking policies
                    (disabled / capped / natural), response parsing, records
src/capture/        boundary-token location (decode-checked), activation
                    extraction, content-addressed ResponseStore + query API
src/analysis/       numpy PCA + 2D/3D scatter figures
src/temporal_manifolds/   legacy shared library (untouched; previous project)
experiments/        run_experiment.py (generate_llm_responses),
                    analyze_geometry.py, configs/
streams/            research stream notes
tests/              pytest (49 tests incl. live-tokenizer markup verification)
output/             gitignored; content-addressed runs:
                    output/runs/<run_id>/<model>/samples/<sample_uid>/
```

The pre-refactor prototype (binary-choice/intertemporal pipeline: inference
backends, interventions, token trees, preference datasets, entropy/diversity
math) is preserved in git history at commit `43b9cc9`.

## Related prior work in this org

- [arXiv:2606.05194](https://arxiv.org/pdf/2606.05194) — Temporal Preference
  Concepts and their Functions in a Large Language Model (the predecessor
  paper this project extends).
- [`temporal-awareness`](../temporal-awareness/) — prompt-pair patterns and
  activation-extraction style.
- [`temporal/latents`](../temporal/latents/) — CAA framework; hook-based
  residual extraction patterns.
