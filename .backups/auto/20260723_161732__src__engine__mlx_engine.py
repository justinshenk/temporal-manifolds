"""MLX engine for Apple Silicon (large quantized models, e.g. Qwen3-32B 4-bit).

Generation uses mlx_lm.stream_generate. Capture monkey-patches the decoder
layer class's __call__ (MLX has no hook system) — same technique as the
legacy MLX backend — and slices per-layer hidden states after the forward.
Requires the optional `mlx-lm` dependency (uv sync --extra mlx).
"""

from __future__ import annotations

import numpy as np

from .base import Engine


class MLXEngine(Engine):
    def __init__(self, model_id: str):
        from mlx_lm import load

        self.model_id = model_id
        print(f"[MLXEngine] loading {model_id} ...")
        self._model, tok = load(model_id)
        # mlx_lm wraps the HF tokenizer; unwrap for full API compatibility
        self._tokenizer = getattr(tok, "_tokenizer", tok)
        self._layers = self._model.model.layers if hasattr(self._model, "model") else self._model.layers
        print(f"[MLXEngine] loaded: n_layers={self.n_layers} d_model={self.d_model}")

    @property
    def tokenizer(self):
        return self._tokenizer

    @property
    def n_layers(self) -> int:
        return len(self._layers)

    @property
    def d_model(self) -> int:
        args = self._model.args
        for attr in ("hidden_size", "dim", "d_model"):
            if hasattr(args, attr):
                return getattr(args, attr)
        raise AttributeError(f"Cannot find hidden size in {args}")

    def render(
        self,
        messages: list[dict],
        add_generation_prompt: bool = True,
        enable_thinking: bool | None = None,
    ) -> str:
        kwargs = {}
        if enable_thinking is not None:
            kwargs["enable_thinking"] = enable_thinking
        try:
            return self._tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=add_generation_prompt,
                **kwargs,
            )
        except TypeError:
            return self._tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=add_generation_prompt
            )

    def generate_ids(
        self,
        input_ids: list[int],
        max_new_tokens: int,
        temperature: float = 0.0,
        stop_token_ids: tuple[int, ...] = (),
    ) -> list[int]:
        from mlx_lm import stream_generate

        kwargs = {}
        if temperature > 0:
            from mlx_lm.sample_utils import make_sampler

            kwargs["sampler"] = make_sampler(temp=temperature)

        new_ids: list[int] = []
        stops = set(stop_token_ids)
        for response in stream_generate(
            self._model,
            self._tokenizer,
            prompt=input_ids,
            max_tokens=max_new_tokens,
            **kwargs,
        ):
            new_ids.append(response.token)
            if response.token in stops or response.finish_reason == "stop":
                break
        return new_ids

    def capture_resid_post(
        self, token_ids: list[int], layer_indices: list[int]
    ) -> dict[int, np.ndarray]:
        import mlx.core as mx

        wanted = set(layer_indices)
        for k in wanted:
            if not 0 <= k < self.n_layers:
                raise ValueError(f"layer {k} out of range 0..{self.n_layers - 1}")

        captured: dict[int, np.ndarray] = {}
        layer_to_idx = {id(layer): i for i, layer in enumerate(self._layers)}
        layer_class = type(self._layers[0])
        original_call = layer_class.__call__

        def hooked_call(self_layer, x, *args, **kwargs):
            result = original_call(self_layer, x, *args, **kwargs)
            idx = layer_to_idx.get(id(self_layer))
            if idx in wanted:
                hidden = result[0] if isinstance(result, tuple) else result
                h32 = hidden.astype(mx.float32)
                mx.eval(h32)
                captured[idx] = np.array(h32)[0]  # [seq, d_model]
            return result

        layer_class.__call__ = hooked_call
        try:
            logits = self._model(mx.array([token_ids]))
            mx.eval(logits)
        finally:
            layer_class.__call__ = original_call

        missing = [k for k in layer_indices if k not in captured]
        if missing:
            raise RuntimeError(f"MLX capture produced no output at layers {missing}")
        return {k: captured[k].astype(np.float32) for k in layer_indices}
