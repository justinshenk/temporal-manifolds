# Verification Log

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
