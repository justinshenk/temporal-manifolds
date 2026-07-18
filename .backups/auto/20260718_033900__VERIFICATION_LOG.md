# Verification Log

## 2026-07-17 — Multi-stage conversation pipeline (refactor + validation)

- **ChatMarkup registry facts** (turn/think tokens for ChatML/Qwen3/2507/SmolLM2/
  Llama-3.1/Gemma-3): established by running real tokenizers (apply_chat_template +
  single-token encode) and re-proved on every `pytest` run by
  tests/test_chat_markup.py (5 families, live tokenizers). — VERIFIED
- **Test suite** after refactor: `uv run pytest` → 49 passed, 0 failed (run twice,
  second run after protocol hardening). — VERIFIED
- **SmolLM2-135M smoke run** (run f95ed9ae8962, 4 conversations): opened
  conversation.json (10 alternating turns), boundaries.json (59 boundaries, kinds
  correct), activations.npz (177 vectors = 59 boundaries × 3 layers, all finite,
  key set exactly matches expectation, depth→layer 11/17/23 for 30 layers). — VERIFIED
- **Qwen3-0.6B smoke run** (run d409e0e629e6 after protocol fix): 4/4 conversations,
  full 5 assistant turns, empty-think-block delimiters captured per assistant turn;
  first replies read directly (role-play failure of tiny model documented, protocol
  hardened + completion detection restricted to standalone short reply). — VERIFIED
- **Capped-thinking smoke** (run 90488e0b1443): opened stored record — all 3
  assistant turns show think_forced_closed=True, think_open/think_close captured
  3× each, CoT interior not captured. — VERIFIED
- **Analysis stage** on d409e0e629e6: 150 figures written;
  d80_L21_turn_end__target_horizon__2d.png viewed WITH IMAGE TOKENS (axes,
  log colorbar, points render correctly). Other 149 figures NOT individually
  viewed. — VERIFIED (one exemplar); remainder UNVERIFIED individually
- **Main Qwen3-14B run**: in progress. Sample 668c38b9b4574c59 opened directly:
  18 turns, 8 steps expanded with parseable "Time horizon:" headers, 107
  boundaries, layers 15/23/31 (n_layers=40). — VERIFIED (1 of N; run ongoing)
- **Incremental analysis @ 8-9 samples** (run 23f1cdc1ecb8): raw-slice figures
  d80_L31_turn_end__target_horizon__2d.png and d80_L31_think_close__target_
  horizon__2d.png viewed WITH IMAGE TOKENS (clusters structural, horizon mixed
  — motivated task-centered analysis). horizon_geometry.py results:
  d80_L31_turn_end_all vs step_horizon Spearman rho=0.91 p=1.4e-9 n=76;
  d40_L15_role_first vs target_horizon rho=0.96 p=4.9e-5 n=9. Both figures
  viewed WITH IMAGE TOKENS — gradients visible along PC1 across both tasks.
  — VERIFIED (exemplars; full-run figures pending)
- **Main Qwen3-14B run COMPLETE** (run 23f1cdc1ecb8): 20/20 samples, 0 failed,
  manifest status complete (log tail + manifest read directly). Independent
  verifier agent re-checked: 20 sample dirs each with the 3 artifact files;
  3 randomly picked samples fully validated (alternating turns, 107 boundaries
  each all in-range with correct kinds/token ids, 321 npz arrays = 107×3
  layers, all (5120,) float32 finite). — VERIFIED
- **Headline statistic** (target horizon vs PC1 at first-assistant-turn `role`
  token, L15 = 40% depth, task-centered): Spearman rho=-0.963, p=1.2e-11,
  n=20 — INDEPENDENTLY RECOMPUTED by verifier agent from raw stored
  activations, matching horizon_results.json to full precision. — VERIFIED
- **Final figures**: d40_L15_role_first__target_horizon.png and
  d40_L15_turn_end_trajectories.png viewed WITH IMAGE TOKENS by me AND by the
  verifier agent (real plots, gradients/trajectories as described). Remaining
  ~215 figures generated but not individually viewed. — VERIFIED (exemplars);
  bulk figures UNVERIFIED individually

## 2026-07-18 — Phase 2: cross-family + scale-up

- **MLX engine numerical validation**: Qwen3-0.6B same tokens through HFEngine
  vs MLXEngine — cosine ≥ 0.9996, rel-err ≤ 2.8% at layers 11/16/22 (computed
  directly, printed). — VERIFIED
- **Qwen3-14B probes** (run 23f1cdc1ecb8): grouped-CV R²=0.965/ρ=0.97 (target,
  L23 role), LOTO R²=0.835 (think_close L23); step R²=0.588/ρ=0.80 n=60.
  Probe table read directly from script output. — VERIFIED
- **Llama-3.1-8B run** (17c2a9edea10): 20/20 done 0 failed (log read). Probes:
  target grouped-CV R²=0.776/ρ=0.88 (L18 turn_start), step R²=0.670/ρ=0.83
  n=74, LOTO R²=0.191. Figure d60_L18_turn_start_first__target_horizon.png
  viewed WITH IMAGE TOKENS (gradient present, diagonal, one outlier). Other
  Llama figures NOT individually viewed. — VERIFIED (exemplar)

## 2026-07-17 — Merge of origin/dev into ian-prototyping (commits 43b9cc9 → ede8ed1)

- **WIP commit 43b9cc9** (local prototype restructure, 160 files): staged file list reviewed
  before commit; `.backups/` and `.DS_Store` confirmed excluded via `git status --short`. — VERIFIED
- **Merge commit 24ffe60** (origin/dev, 26 commits): `git log HEAD..origin/dev` and
  `HEAD..origin/main` both empty — no commits lost. Independent verifier agent re-checked. — VERIFIED
- **Conflict markers**: `git grep -nE '^(<<<<<<<|=======$|>>>>>>>)'` → zero hits (checked by me
  and independently by verifier agent). — VERIFIED
- **pyproject.toml**: parsed with tomllib by verifier; contains merged dep set (local
  transformers-5 stack + dev's google-cloud-storage/google-auth/einops), restored `api` extra,
  and Linux-gated cu128 torch index. — VERIFIED
- **uv.lock**: regenerated via `uv lock` (not hand-merged); `uv lock --check` → "Resolved 181
  packages", exit 0. — VERIFIED
- **Dual layout**: `git diff origin/dev HEAD -- src/temporal_manifolds/` empty after ede8ed1
  (legacy package identical to dev); `src/datasets/other/` keeps local generate/phrasings/
  templates only. — VERIFIED
- **Test suite**: `uv run pytest` → 37 passed, 1 failed. The failure
  (`test_generate_example_dataset`, missing `configs/scenarios/example.yaml`) is pre-existing:
  the config was deleted in f77344f, already an ancestor of the pre-merge branch (verified via
  `git ls-tree 43b9cc9 configs/scenarios/`). — VERIFIED (failure documented, not introduced)
- **Runtime behavior of merged code** (workflows, GCS scripts, inference backends): not
  exercised beyond imports/unit tests. — UNVERIFIED
