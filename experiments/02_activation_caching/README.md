# 02 — Activation Caching

> **Category**: Activation Caching · **Members**: Shantanu, Justin · **Status**: Not started
> **Dates**: 2026-05-23 → 2026-05-25
> **Output**: Activation tensors pushed to Hugging Face Hub.
> **Comments**: Contrary to previous runs, this run caches at **5 key positions** identified previously. Results pushed to HF.

## TODO
- [ ] Confirm the 5 token positions (from the Qwen3-32B steering work) — document indices here.
- [ ] Pick model(s) for the workshop track (default: Qwen3-32B; mirror to a smaller model for fast iteration).
- [ ] Implement caching in `src/temporal_manifolds/activations/`.
- [ ] HF Hub upload helper (use `huggingface_hub` directly).

Reference impl: `/Users/justinshenk/projects/temporal/latents/latents/extract_steering_vectors.py`.
