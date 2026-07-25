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

## 2026-07-24 — Explorer entry renames (DURATION label on all duration-mode entries)
- WHAT: republished artifact with all five entries carrying explicit step-semantics labels.
- HOW: grepped the built temporal-manifolds-results.html for `"name":...steps"` — exactly 5 entries, all with "· DURATION steps" or "· TARGET steps"; grep for tsne/umap returned 0 matches; `node --check app_check.js` passed.
- RESULT: VERIFIED (names/data in the published file); artifact page itself not re-opened in a browser this round — visual render UNVERIFIED, structure verified.

## 2026-07-24 — Restored '+user' (all-scope) PCA fits on both Qwen3-32B x5 entries
- WHAT: 60 fits (2 entries x 5 kinds x 3 layers x centered/raw) merged back from full export with b64 Int16 coords; artifact republished (11.6MB).
- HOW: merge script printed 'restored 60 fits'; re-opened debug_data_trim.json and confirmed all-scope keys present on both entries and decoded coord count == 3x idx count; node --check passed.
- RESULT: VERIFIED (data-level); in-browser render of the +user scope UNVERIFIED (user will see it on reload).

## 2026-07-24 — Split explorer into DURATION and TARGET artifacts
- WHAT: main artifact (bd9903c8...) now 4 DURATION entries only (9.9MB); new artifact (748fc4a4-6eea-4e89-8c59-01b3e9bef01e) holds the Qwen3-32B x5 TARGET entry (1.6MB).
- HOW: split script printed entry lists per file; node --check passed; both published successfully.
- RESULT: VERIFIED (build-level); browser render UNVERIFIED.

## 2026-07-24 — Launched TARGET-mode 260-runs (Llama-8B run 57ea1d7cacbb, Gemma-9B run 73a3c5ec77c2)
- WHAT: 52 prompts x 5 rollouts each, step_mode=target, max_assistant_turns=24, MLX 4-bit, running in parallel; Qwen-32B queued for after.
- HOW: both logs show correct run_id/model/prompt count; per-sample monitors + first-sample compliance checks armed.
- RESULT: IN PROGRESS (outputs UNVERIFIED until runs finish and are inspected).

## 2026-07-24 — Sensible-grid TARGET runs relaunched + control mode added
- WHAT: (a) killed full-grid target runs (absurd pairings); relaunched with per-task horizons (20 prompts x 5 rollouts = 100/model): Llama run bcd154c149fb, Gemma run 19665970da25. (b) Aborted run dirs 57ea1d7cacbb, 73a3c5ec77c2 remain on disk (1 partial sample) — ignore. (c) Added step_mode=target_control (silent steps, final "Time assignments:" turn) + store backfill with horizon_source tag.
- HOW: first sample of each relaunched run re-opened and read: both models emit parsable "Time target:" lines, non-decreasing (VERIFIED for those 2 samples; rest in progress). Control mode: 4 new unit tests pass (assignment parsing, no cross-contamination with step turns, prompt rendering); full suite 53+4 passed. Control runs NOT yet executed — store backfill on real data UNVERIFIED until first control run.
- RESULT: runs IN PROGRESS; control code VERIFIED at unit level.

## 2026-07-24 — Llama-8B TARGET run finished + CONTROL run validated
- WHAT: (a) run bcd154c149fb complete. (b) Llama CONTROL run 397e77f6a867 launched; first sample verified.
- HOW: (a) re-opened all 100 conversations + npz: manifest complete, 100/100 non-empty npz, 99/100 completed, 675/703 targets parsed, monotone 85/100, last/total median 1.00; independent verifier agent also spawned (report pending). (b) read first control conversation in full: 9 step turns with ZERO time words (regex-checked), assignments turn parses; ran store.query on the real sample — step rows return backfilled horizons with horizon_source='assigned' matching the assignment turn exactly.
- RESULT: target run VERIFIED (data-level, verifier report pending); control protocol + backfill VERIFIED on sample 1; remaining control samples IN PROGRESS.

## 2026-07-24 — Independent verifier report: Llama-8B target run bcd154c149fb
- WHAT: verifier agent re-opened manifest, 10 random conversations, all 100 boundaries.json, and every key of all 100 npz files.
- RESULT: VERIFIED on all aspects. Manifest complete (20 prompts x 5 rollouts, 1:1 with disk); every step turn in all 100 samples has a "Time target:" line; activations all finite float32 d=4096, layers {12,18,25}, key count = 3 x boundaries everywhere. Flags: (a) depth 0.6 -> layer 18 matches the project convention round(0.6*32)-1 (my brief mis-stated 19 — the run is correct); (b) sample 3970e34ec2703601 never emitted "Plan Completed" (turn-capped, completed=false) — keep, it is honest data; (c) 21 steps in 11 samples slightly overshoot the horizon — model behavior, faithfully recorded.

## 2026-07-24 — Independent verifier report: Gemma-9B target run 19665970da25 + Gemma control launch
- WHAT: verifier re-opened manifest, all 100 boundaries + npz programmatically, 10 conversations deep-read.
- RESULT: VERIFIED all aspects — 100 samples 1:1 with manifest, correct prompts per sample, gemma markup token ids match token_ids[abs_pos] with 0 mismatches, activations exactly {L16,L24,L33} x boundaries, all finite float32 d=3584. Caveats (honest model behavior): 24 samples with non-numeric target lines (mostly dinner_party "Tonight"/"Tomorrow"; note "Year N" forms DO parse in our target-mode parser), 15 samples overshoot horizon, 1 completed=False. Downstream filters already exclude unparsed/zero.
- ALSO: Gemma CONTROL run 87b32dddd585 launched; first sample re-opened and read: 9/9 steps time-silent, assignments parse 7/9 (2 "Ongoing" honestly unparsed).

## 2026-07-24 — Llama control run finished + probe recovery analysis
- WHAT: run 397e77f6a867 complete (100 samples, 746/757 steps with parsed assigned targets, 608 time-silent). probe_control.py written and run; results saved to output/runs/397e77f6a867/.../figures/probe_control_results.json.
- HOW: re-opened all conversations/npz (100/100 non-empty); ran analysis and read full printed table; baselines computed (step-index, total-horizon, within-conversation demeaned).
- RESULT: VERIFIED. Transfer probe (trained on stated targets) on silent steps: pooled R2=0.80/rho=0.90 (L12 turn_start), silent subset R2=0.81 > cadence subset — not driven by leaks. Caveat honestly noted: total-horizon-only baseline gets pooled R2=0.84, so pooled recovery mostly reflects the total-horizon code; the novel finding is within-conversation ordinal recovery: median Spearman 0.73, 67/79 conversations positive; magnitude calibration within-plan poor (negative demeaned R2).

## 2026-07-24 — Gemma control finished + probe recovery replicated; 32B target launched
- WHAT: Gemma control run 87b32dddd585 complete; probe_control run; Qwen3-32B target run 74c172105e25 started.
- HOW: re-opened all 100 conversations + npz (100/100 non-empty, 100/100 completed); ran probe_control and read full table; within-conversation analysis run at L16 turn_start.
- RESULT: VERIFIED. Gemma transfer probe on silent steps: R2=0.81 rho=0.92 (L16 turn_start), silent subset 0.82 >= cadence 0.69 — replicates Llama. Within-conversation ordinal recovery: median Spearman 0.64, 34/41 convs positive. LOTO negative for Gemma (task-entangled). Results JSON saved under the control run's figures dir.

## 2026-07-24 — Independent verifier: Qwen3-32B target run 74c172105e25; 32B control launched
- WHAT: verifier checked ALL 100 samples programmatically + read 10 conversations.
- RESULT: VERIFIED all aspects. Manifest 1:1 with disk; qwen3 markup with empty think blocks preserved (151667/151668 adjacent); depth_to_layer {25,37,50}, d=5120; every npz position set exactly equals boundary abs_pos set; parser cross-check 738/827 steps parse with ZERO value mismatches vs stored step_horizon_years. Caveats: 89 unparseable slot forms (mostly dinner_party 'Tonight'/'Year 0+'), 4 malformed headers, 5 samples completed=false (plan fully expanded, no terminal sentinel) — all honest model behavior, dropped by >0 filters downstream.
- ALSO: Qwen3-32B CONTROL run launched (last in queue).

## 2026-07-24 — 32B control stopped early per user (63 samples) + probe recovery: 3-model comparison complete
- WHAT: user stopped 32B control at 63/100 ("just enough to see if probe would work"). Analysis run on partial data (455 assigned step rows).
- HOW: verified 63/63 samples have non-empty npz; ran probe_control.py and within-conversation analysis; read full tables.
- RESULT: VERIFIED. 32B transfer: pooled rho=0.88 (R2 0.58-0.68 best combos), silent >= cadence. Within-conversation ordinal recovery STRONGEST of all models: mean Spearman 0.86, median 0.90, 50/50 convs positive (L25 turn_start). Comparison: Llama median 0.73 (67/79 pos), Gemma 0.64 (34/41 pos), Qwen3-32B 0.90 (50/50 pos) — internal temporal ordering of silent plans scales with model capability.

## 2026-07-24 — TARGET artifact updated: 3 target grids + 3 control entries
- WHAT: republished target artifact (748fc4a4...) with 6 entries: Qwen3-32B/Llama-8B/Gemma-9B TARGET grids + 3 CONTROL silent-steps entries (563 conversations total). Old 2-task sweep entry dropped (superseded; data preserved on disk). Size fixes: prompt dedupe (40 unique), idx arrays b64-packed, +user-scope fits kept on default entry only. 15.5MB.
- HOW: node functional test re-decoded ALL 426 fits (idx lengths, ranges vs rows, xyz byte counts) and all 563 prompts (string + contains Scenario); control prompt verified to contain "Time assignments:"; node --check passed; publish succeeded.
- RESULT: VERIFIED at data/build level; in-browser render UNVERIFIED until user loads it.

## 2026-07-25 — Target artifact rebuilt: joint TARGET+CONTROL PCA per model
- WHAT: republished target artifact with 3 joint entries (PCA fit on both conditions together); new color modes tstep/cstep/cond; control conv turns backfilled with assigned horizons ("X — assigned in final turn"); norm_q fixed from global-RMS to 97.5th-pct-radius scaling (outliers no longer compress the cloud, hard cap 3.2).
- HOW: node functional test decoded all 204 fits (idx/xyz byte-level), 563 prompts, and counted control conv steps with assigned horizons (Qwen 462/549, Llama 751/757, Gemma 361/405 — gaps are honestly-unparsed "Ongoing" forms); node --check passed; published successfully (15.8MB).
- RESULT: VERIFIED (build/data level); in-browser render UNVERIFIED until loaded.

## 2026-07-25 — Fixed 5x viewport overflow in target artifact
- WHAT: joint entries' packed coords were 5x the scale the template projection was tuned for (fresh export quantized x1000; the published duration data had been re-quantized to x100-equivalent during earlier size trims). Rescaled all 204 fits' coords /5 in place; export_joint_data.py quantizer fixed (q x200 with the 97.5pct-radius norm).
- HOW: measured mean-abs coordinate: duration reference 69.4, joint before 347.5, after rescale 69.5 — byte-level re-read of the packed data; republished.
- RESULT: VERIFIED (data-level; scale now matches the artifact whose rendering was known good). Render UNVERIFIED until user reloads.

## 2026-07-25 — Theme-controlled transfer (transfer-loto) added to probe_control and run for all 3 models
- WHAT: 4th test in probe_control.py: train on TARGET rows of other tasks, test on CONTROL rows of held-out task (per-task rho recorded in results JSON). Results regenerated for all 3 control runs.
- HOW: script rerun for all pairs; full tables read; key rows recorded below.
- RESULT: VERIFIED. Turn_start rho pooled, transfer -> transfer-loto: Qwen32B 0.88 -> 0.87 (theme contributes ~nothing); Llama 0.90 -> 0.52; Gemma 0.92 -> 0.66 (small models ride heavily on theme/task band). 32B keeps R2=0.58 even theme-controlled at L25 turn_start.

## 2026-07-25 — Time-scale classification probe + theme-split, all 3 models
- WHAT: probe_timescale.py (classes hours/days/weeks/months/years/decades+; multinomial logistic, inner-CV C; within-target CV, transfer to control, per-task same-theme vs held-out-theme). Results JSONs written to each control run figures dir.
- HOW: ran for all 3 model pairs; full printed tables read (task output b5i6swhpb).
- RESULT: VERIFIED. Within-target: exact 0.62-0.75 vs baseline 0.26-0.33, adjacent 0.95-0.99. Transfer to silent steps: exact ~0.5, adjacent 0.88-0.94. Theme-split: held-out-theme exact collapses for out-of-band themes (dinner/marathon ~0.0-0.09) but survives for climate/archive (0.17-0.56, adj up to 0.98); consistent with regression: ordinal time transfers, absolute band calibration is theme-anchored.
