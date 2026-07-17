"""Residual-stream activation caching + Hugging Face Hub upload.

Reference implementation to adapt:
``/Users/justinshenk/projects/temporal/latents/latents/extract_steering_vectors.py``.

Caches at 5 configurable token positions (the positions identified as
informative in the prior Qwen3-32B steering work; confirm and pin the exact
indices in ``experiments/02_activation_caching/README.md``).
"""
