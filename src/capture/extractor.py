"""Extract residual-stream activations at boundary positions.

One forward pass over the finished conversation per sample; hidden states are
captured only at the requested depths, then sliced to boundary positions.
"""

from __future__ import annotations

import numpy as np

from ..chat_markup.markup import ChatMarkup
from ..conversation.records import ConversationRecord
from ..engine.base import Engine
from ..engine.depths import layers_for_depths
from .boundaries import BoundaryMap, find_boundaries


def extract_boundary_activations(
    engine: Engine,
    markup: ChatMarkup,
    record: ConversationRecord,
    depths: tuple[float, ...],
) -> tuple[BoundaryMap, dict[float, int], dict[str, np.ndarray]]:
    """Returns (boundary_map, {depth: layer}, {f"L{layer}_p{pos}": vec}).

    Every returned vector is float32 [d_model]; a non-finite vector raises.
    """
    bmap = find_boundaries(record.token_ids, engine.tokenizer, markup, record)
    depth_to_layer = layers_for_depths(depths, engine.n_layers)
    layers = sorted(set(depth_to_layer.values()))

    per_layer = engine.capture_resid_post(record.token_ids, layers)

    seq_len = len(record.token_ids)
    acts: dict[str, np.ndarray] = {}
    for layer, mat in per_layer.items():
        if mat.shape[0] != seq_len:
            raise RuntimeError(
                f"Capture length mismatch at layer {layer}: "
                f"{mat.shape[0]} vs {seq_len} tokens"
            )
        for b in bmap.boundaries:
            vec = mat[b.abs_pos].astype(np.float32)
            if not np.isfinite(vec).all():
                raise RuntimeError(
                    f"Non-finite activation at layer {layer} pos {b.abs_pos} "
                    f"(sample {record.sample_uid})"
                )
            acts[f"L{layer}_p{b.abs_pos}"] = vec
    return bmap, depth_to_layer, acts
