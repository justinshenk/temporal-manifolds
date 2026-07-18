# Temporal Manifolds in Multi-Stage Planning Conversations — Architecture

Goal: track how time-horizon manifolds evolve across multi-stage planning
conversations. A model is asked to plan a task under a target time horizon,
then stepped through the plan turn-by-turn ("Continue"); we capture residual
stream activations at 40/60/80% layer depth at **change-of-turn tokens and
thinking delimiters only** (never inside chain-of-thought), and look for
geometric structure (PCA) organized by target horizon, step horizon, prompt.

Background: arXiv:2606.05194 found (in Qwen3-4B-Instruct-2507) that time
horizon is encoded on non-linear manifolds in the residual stream and that the
user→assistant change-of-turn token sequence (`<|im_end|> → \n → <|im_start|>
→ assistant`) is where the continuous horizon manifold collapses into a
committed preference. This project asks how that geometry **evolves over the
turns of a planning conversation**.

## Refactored `src/` layout

`src/temporal_manifolds/` is legacy (untouched). Everything else is refactored
into focused packages:

```
src/
  core/           schema.py (BaseSchema + deterministic ids), time_value.py,
                  file_io.py, paths.py, device.py, logging.py
  chat_markup/    markup.py (ChatMarkup spec), registry.py (per family ×
                  generation: turn tokens, think delimiters, verified against
                  the real tokenizer at load time)
  datasets/       schema.py (PlanningTask, HorizonPhrasing, PlanningPrompt),
                  tasks.py (curated tasks w/ natural time scales + metadata),
                  phrasings.py (ecologically valid horizon wordings),
                  generator.py (tasks × target horizons × phrasings -> dataset)
  engine/         base.py (Engine interface), hf_engine.py (transformers,
                  MPS/CUDA/CPU), mlx_engine.py (Apple Silicon, big models),
                  depths.py (depth fraction -> layer index)
  conversation/   protocol.py (plan-then-Continue protocol, completion
                  detection), thinking.py (ThinkingPolicy: disabled / capped /
                  natural), driver.py (runs a full conversation vs an Engine),
                  records.py (ConversationRecord)
  capture/        boundaries.py (find boundary-token positions in final token
                  sequence given ChatMarkup), extractor.py (single forward
                  pass w/ hidden-state capture at chosen layers, slice
                  boundary positions), store.py (ResponseStore: content-
                  addressed output/ layout + retrieval API)
  analysis/       loader.py (assemble activation matrices + metadata),
                  pca.py (numpy PCA), viz.py (2D/3D scatters colored by
                  target horizon / step horizon / prompt)
experiments/
  configs/        run configs (JSON)
  run_experiment.py        generate_llm_responses: dataset -> conversations ->
                           activations, resumable
  analyze_geometry.py      PCA + figures from a stored run
  smoke_test.py            end-to-end small-model check
```

## Key design decisions

1. **Capture happens post-hoc, not during generation.** The conversation is
   driven turn-by-turn with ordinary KV-cached generation; when the
   conversation completes we re-tokenize the full transcript, locate boundary
   tokens, and do ONE forward pass capturing hidden states at the 3 target
   layers, saving only boundary positions. Simpler, backend-agnostic, exact.
   (Positions are verified: decoded token at each boundary position must match
   the expected marker string.)

2. **ChatMarkup registry, verified per model.** Family × generation table:
   - ChatML/Qwen3: `<|im_start|>`(151644), `<|im_end|>`(151645), role words,
     `\n`; think `<think>`(151667) `</think>`(151668). Qwen3-Instruct-2507:
     same turn tokens, no thinking behavior. Qwen3 hybrid: thinking by
     default, disable via `enable_thinking=False` (template inserts empty
     think block).
   - SmolLM2: ChatML turn tokens (ids 1, 2), NO think tokens, auto system msg.
   - Llama-3.x: `<|eot_id|>`(128009), `<|start_header_id|>`(128006), role,
     `<|end_header_id|>`(128007), `\n\n`. No think tokens.
   - Gemma-2/3: `<end_of_turn>`(106), `\n`, `<start_of_turn>`(105), role
     (`user`/`model`), `\n`. No think tokens.
   At engine init, the registry entry is validated against the tokenizer:
   every marker must round-trip to the expected single token id, and a
   rendered 3-message conversation must contain the expected boundary
   sequence. Hard error otherwise (never silently wrong).

3. **Thinking policy.** All experiments run `disabled` (empty think block via
   the family's mechanism: template `enable_thinking=False` for Qwen3 hybrid;
   nothing needed for non-reasoning families). `capped(n)`: during
   generation, if `<think>` opened and not closed within n tokens, force-
   inject `</think>` and continue (for small models that never stop);
   `natural`: let the model close it. Capped/natural exist and are
   smoke-tested only.

4. **Boundary tokens captured per turn** (`kind` labels):
   `turn_end` (`<|im_end|>`/`<|eot_id|>`/`<end_of_turn>`), `post_turn_end_nl`,
   `turn_start`, `role`, `post_role_nl`, `think_open`, `think_close`.
   Every captured position stores: abs position, kind, turn index, role,
   assistant-step index (which plan step the turn belongs to), token id/str.

5. **Layer depths.** `layer_index = round(depth * n_layers) - 1` on 0-indexed
   blocks; capture is `resid_post` (output of that block). Depths 0.4/0.6/0.8.

6. **Storage** (`output/runs/`): content-addressed like the old DataManager —
   `run_id = fingerprint(dataset cfg, protocol cfg, gen cfg, seed)`, model one
   level below; per-sample dir has `conversation.json` (turns, parsed steps,
   step horizons), `boundaries.json`, `activations.npz` (keys
   `L{layer}_p{abs_pos}`), sample uid = deterministic hash (prompt_id,
   model_id, target_horizon, protocol, seed). `manifest.json` enables resume.
   `ResponseStore.query(prompt_id=, model_id=, kind=, layer=, turn=...)`
   returns (matrix, metadata rows).

7. **Prompt scheme.** Prompt = task template with `{target_time_horizon}`
   slot filled by an ecologically valid phrasing at generation time; response
   format mirrors the protocol (overview then `Step: #` / `Time horizon:` /
   details; final turn answers `Plan Completed`). Tasks are chosen so the SAME
   task admits several target horizons (from conversational.py's crossed task
   families).

8. **Models.** Dev: SmolLM2-135M-Instruct, Qwen3-0.6B. Main: largest that
   fits 48GB M4 Max — Qwen3-14B bf16 on MPS via HF (preferred; falls in the
   Qwen3 dense family adjacent to the 20-40B target the lab will scale to on
   bigger hardware); optionally Qwen3-32B 4-bit via MLX engine.
