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
