"""Model engines: chat rendering, generation, targeted residual-stream capture."""

from ..core.auto_export import auto_export


def load_engine(model_id: str, backend: str = "auto"):
    """Load an engine. backend: 'hf' | 'mlx' | 'auto' (mlx only if requested
    explicitly — 'auto' currently always chooses HF for determinism)."""
    if backend == "mlx":
        from .mlx_engine import MLXEngine

        return MLXEngine(model_id)
    from .hf_engine import HFEngine

    return HFEngine(model_id)


__all__ = auto_export(__file__, __name__, globals()) + ["load_engine"]
