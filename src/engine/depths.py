"""Depth-fraction -> layer-index convention.

`layer_at_depth(0.4, n_layers=36)` -> 13: we capture resid_post of 0-indexed
block `round(depth * n_layers) - 1`, i.e. the residual stream after 40% of
the blocks have run. Documented here once; everything else imports it.
"""

from __future__ import annotations

DEFAULT_DEPTHS: tuple[float, ...] = (0.4, 0.6, 0.8)


def layer_at_depth(depth: float, n_layers: int) -> int:
    if not 0.0 < depth <= 1.0:
        raise ValueError(f"depth must be in (0, 1], got {depth}")
    return max(0, round(depth * n_layers) - 1)


def layers_for_depths(depths: tuple[float, ...], n_layers: int) -> dict[float, int]:
    """{depth: layer_index}; raises if two depths collapse to one layer."""
    result = {d: layer_at_depth(d, n_layers) for d in depths}
    if len(set(result.values())) != len(result):
        raise ValueError(
            f"Depths {depths} collapse to duplicate layers {result} for "
            f"n_layers={n_layers}; pick more separated depths."
        )
    return result
