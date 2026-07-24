# Verification Log

## 2026-07-24 — Explorer: six entries published

- Six-entry export verified before publish: entry order/default correct
  (rollout default), target entry 45 samples seeds [0..4], MFA re-attached
  (assert), JSON roundtrip + node --check; payload trimmed to 15.2MB (big
  two entries lost '+user' fits, raw-centering fits, and assistant-scope
  t-SNE/UMAP — first/steps scopes and all task-centered PCA intact).
  Republished same URL. In-browser render not re-verified. — VERIFIED (build)

## 2026-07-24 — Target-semantics sweep (run 383cb5f89bda)

- **Run**: 45/45 done 0 failed (~97 min; log read). step_mode=target ("Time
  target" = future offset), all steps expanded (max 24 assistant turns; plans
  up to 12 steps fully collected), 2 tasks × task-appropriate horizons ×
  5 rollouts @T0.7. 4/45 conversations expanded all steps but never emitted
  "Plan Completed" and ran to the 48-turn cap (benign; steps intact).
- **Semantics adopted** (computed from stored raw text with the mode-aware
  parser; 53 tests pass): 93% of conversations have perfectly monotone
  non-decreasing target sequences; median last-target/total = 1.00 with 95%
  within ±20% — plans land on their horizon.
- **The turn-tracking question resolved by design**: pooled corr(turn index,
  log step target) = 0.52 (p=4e-23) vs 0.10 for duration mode — targets are
  cumulative, so step time now tracks conversation position, as intended.
- **Probes**: step-target grouped-CV R²=0.822/ρ=0.91 (n=316, p=4e-121, L25
  role); total R²=1.000. Raw PCA: step target on PC2 rho=0.70 (p=2e-48,
  n=482) — step-time now visible in raw PCA, unlike duration mode. — VERIFIED
  (script outputs read)

## 2026-07-24 — MFA region decomposition (arXiv:2602.02464 method on our data)

- **turn-vs-fraction check**: Spearman(turn index, step/total fraction) = 0.02
  (p=0.44, n=1752) — computed directly; the two variables are independent in
  this data (user's tracking assumption disconfirmed with numbers).
- **MFA fit** (authors' released implementation, cloned repo; K=24, rank 8,
  30 epochs, k-means init per paper recipe) on 10,965 assistant boundary
  activations at L37 of the 32B rollout run: NLL 4947→3766 (converging).
  Results read from script output; saved to figures/mfa_regions.json +
  mfa_assignment.npz. — VERIFIED (run + outputs read); NLL not fully
  converged (30 epochs, small-scale adaptation of a 100M-point method)
- **Findings**: regions are 100% pure in token kind (NMI 0.71) and organized
  by conversation phase (NMI 0.40); task NMI 0.005, total-horizon NMI 0.004.
  In 18/24 regions a local factor correlates with log step horizon
  (|rho| 0.30-0.59). Region=structure, local-direction=time — consistent
  with both the paper's claim and our probe results.

## 2026-07-24 — 32B rollout expansion (13 horizons × 4 tasks × 5 seeds @ T0.7)

- **Run 9ccf36267612**: 260/260 done, 0 failed, ~7.6h (log read; per-sample
  monitor + 15-min content spot-checks throughout: exact parses, boundary
  counts, npz = boundaries×3, all finite on every check). Seeds [0..4]
  confirmed; rollout diversity verified early (5/5 distinct plans per prompt);
  step coverage 1752/1815 (97%). — VERIFIED
- **Probes**: total horizon grouped-CV R²=1.000/ρ=1.00 (n=260, p≈5.5e-290),
  LOTO R²=0.904; step horizon R²=0.878/ρ=0.94 (n=1752, p≈0, L50 think_close).
  Raw PCA PC1 ρ=0.94 (p=1.2e-118) at d40 think_close first turns. Read from
  script outputs. — VERIFIED
- **Rollout-variance vs horizon**: across-seed std of mean log step horizon
  per prompt, computed directly from stored conversations: ~0.2 (3d–6mo),
  0.5–0.8 (30min–1d), 0.88 (20y), 1.44 (50y), 1.39 (200y), 1.99 (1000y).
  — VERIFIED (computed, printed)
- **Explorer**: 5 entries, rollout entry first with default:true (UI opens on
  it), seeds present, 1752 predY chips; payload trimmed 21.5→15.2MB (dropped
  rollout +user-scope fits and raw-centering t-SNE/UMAP, coords ×100, null
  stripping — coordinate spot-check post-rescale OK); JSON roundtrip + node
  --check passed; republished same URL. In-browser render NOT re-verified.
  — VERIFIED (build)

## 2026-07-23 — Debug-explorer data/coloring verification + t-SNE

- **Exported PCA coords vs independent recompute** (32B L37 think_close steps
  slice, n=493): max abs diff 5e-4 = quantization floor, after per-component
  sign alignment. — VERIFIED
- **Row metadata vs ground truth**: 200 boundary rows' step_horizon_years
  cross-checked against conversation.json turns — 0 mismatches. — VERIFIED
- **Coloring**: example slice has 52 distinct step-horizon values → switched
  point coloring + legend from 9 quantized bins to continuous interpolated
  log gradient (user-reported issue confirmed real). Click-through-on-drag
  bug fixed (drag distance guard). — VERIFIED (data); UI render not
  re-verified in-browser
- **t-SNE**: 252 fits exported (per kind × layer × scope × centering,
  perplexity min(30,(n-1)/3), seed 0, PCA-50 init); index alignment with PCA
  fits asserted. — VERIFIED

## 2026-07-23 — v2/v3 exact-step-horizon reruns (step-horizon-first)

- **Protocol change**: step horizons must be one exact duration (no ranges,
  windows, frequencies). 14B/Llama/Gemma v2 ran with the first wording;
  Qwen3-32B rebelled ("Day 2–Day 5" windows, verified by reading samples 1-3
  of run dbee9e1f6749) and was restarted as v3 (run 864237268525) with one
  strengthened sentence. This wording asymmetry is documented; content
  otherwise identical. Parser extended for windows + unit-repeated windows
  (tests added; 50→52 tests pass... final tally checked below).
- **Format adherence verified by reading samples directly**: 14B first 3
  samples exact; Llama sample 1 exact; Gemma mostly exact w/ "Ongoing"; 32B v3
  samples 1-3 exact incl. degenerate 1-day prompt. Coverage counted over ALL
  samples: 14B 142/147 (97%), Llama 149/149 (100%), Gemma 122/135 (90%),
  32B 482/493 (98%). — VERIFIED
- **Behavioral finding**: with exact durations forced, 32B now fits steps
  within the 1-day total horizon ("1 day" per step) instead of multi-month
  windows — instruction wording changed constraint-following, not just format.
- **Step-horizon probes (headline)**: 32B v3 step grouped-CV R²=0.845/ρ=0.94
  (p=1.3e-220, n=482, L50 think_close); 14B 0.821 (n=142); Gemma 0.796
  (n=122); Llama 0.745 (n=149). Total-horizon: 32B R²=0.988, others ≥0.96.
  Read from probe script outputs. — VERIFIED
- **All four runs**: 20/20, 20/20, 20/20, 72/72, 0 failures (logs read).
  — VERIFIED
- **Debug explorer artifact**: re-exported from the four new run ids
  (round-trip JSON parse checked, node --check on app JS, model/run/sample
  counts printed and confirmed), republished at same URL. In-browser
  rendering NOT re-verified this cycle. — VERIFIED (build); UI render
  UNVERIFIED

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

- **Gemma-2-9B run** (9dd33616af46): 20/20 done 0 failed (log read). Probes:
  target LOTO R²=0.848/ρ=0.93 (L16 role) — strongest cross-task transfer;
  step grouped-CV R²=0.763/ρ=0.90 n=104, step LOTO R²=0.670. Figure
  d40_L16_turn_start_first__target_horizon.png viewed WITH IMAGE TOKENS
  (clean diagonal gradient, all 4 tasks aligned). — VERIFIED (exemplar)
- **Qwen3-32B-4bit MLX run** (f42ab94c4a17): 72/72 done 0 failed (log read;
  manifest complete). Independent verifier agent: all 72 sample dirs complete,
  3 random samples fully validated incl. token_id↔abs_pos crosschecks; headline
  independently reproduced from raw activations (PC1 Spearman rho=-0.912,
  p=8.9e-29, n=72 at L50 role first turns). — VERIFIED
- **32B probes** (after schedule-window parser fix, 50 tests pass): target
  grouped-CV R²=0.988/ρ=0.99 p=1.5e-62; target LOTO R²=0.857; step grouped-CV
  R²=0.816/ρ=0.92 p=6e-151 n=365. Read from script output. — VERIFIED
- **Figures viewed WITH IMAGE TOKENS** (by me; two also by verifier agent):
  d60_L37_think_close_first__target_horizon.png (smooth two-band manifold,
  log-horizon gradient along PC1 in all 4 tasks),
  d60_L37_think_close_first__phrasing.png (bands = phrasings, clean PC2
  separation → phrasing-equivariant horizon manifold),
  d60_L37_turn_end_trajectories.png (all 72 conversations traverse one
  reproducible phase loop). Bulk figures not individually viewed. — VERIFIED
  (exemplars)
- **Cross-model probe table** (best per model, from probe_results.json):
  target-R²(cv)/target-R²(LOTO)/step-R²(cv): Qwen3-14B .965/.835/.751;
  Llama-3.1-8B .963/.191/.670; Gemma-2-9B .970/.949/.805; Qwen3-32B
  .988/.857/.816. — VERIFIED (recomputed from stored results)

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
