"""HuggingFace transformers engine (MPS / CUDA / CPU).

Generation uses HF `generate` with KV caching. Capture registers forward
hooks on exactly the requested decoder blocks and runs ONE full forward pass;
hook outputs are the block outputs = TransformerLens `resid_post`.
"""

from __future__ import annotations

import numpy as np
import torch

from ..core.device import clear_gpu_memory, get_device
from .base import Engine


def _decoder_layers(model):
    """Locate the decoder block list for Llama/Qwen/Gemma/GPT-style models."""
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        return model.model.layers  # llama / qwen / mistral / gemma(text)
    if hasattr(model, "transformer") and hasattr(model.transformer, "h"):
        return model.transformer.h  # gpt2
    raise ValueError(f"Cannot locate decoder layers on {type(model)}")


class HFEngine(Engine):
    def __init__(
        self,
        model_id: str,
        device: str | None = None,
        dtype: torch.dtype | None = None,
    ):
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.model_id = model_id
        self.device = device or get_device()
        if dtype is None:
            dtype = torch.bfloat16 if self.device in ("mps", "cuda") else torch.float32
        self.dtype = dtype

        print(f"[HFEngine] loading {model_id} on {self.device} ({dtype}) ...")
        self._tokenizer = AutoTokenizer.from_pretrained(model_id)
        self._model = AutoModelForCausalLM.from_pretrained(
            model_id, dtype=dtype
        ).to(self.device)
        self._model.eval()
        self._layers = _decoder_layers(self._model)
        print(
            f"[HFEngine] loaded: n_layers={self.n_layers} d_model={self.d_model}"
        )

    # -- info ---------------------------------------------------------------

    @property
    def tokenizer(self):
        return self._tokenizer

    @property
    def n_layers(self) -> int:
        return len(self._layers)

    @property
    def d_model(self) -> int:
        return self._model.config.hidden_size

    # -- chat template ------------------------------------------------------

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
            # Template does not accept enable_thinking — caller (ThinkingPolicy)
            # is responsible for having chosen a compatible no_think_mode.
            return self._tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=add_generation_prompt,
            )

    # -- generation ---------------------------------------------------------

    def generate_ids(
        self,
        input_ids: list[int],
        max_new_tokens: int,
        temperature: float = 0.0,
        stop_token_ids: tuple[int, ...] = (),
    ) -> list[int]:
        ids = torch.tensor([input_ids], device=self.device)
        eos_ids = list(stop_token_ids) or None
        gen_kwargs = dict(
            max_new_tokens=max_new_tokens,
            do_sample=temperature > 0,
            pad_token_id=(
                self._tokenizer.pad_token_id or self._tokenizer.eos_token_id
            ),
            eos_token_id=eos_ids,
            use_cache=True,
        )
        if temperature > 0:
            gen_kwargs["temperature"] = temperature
        with torch.no_grad():
            out = self._model.generate(ids, **gen_kwargs)
        return out[0, len(input_ids) :].tolist()

    # -- capture ------------------------------------------------------------

    def capture_resid_post(
        self, token_ids: list[int], layer_indices: list[int]
    ) -> dict[int, np.ndarray]:
        captured: dict[int, torch.Tensor] = {}
        hooks = []

        def make_hook(layer_idx):
            def hook(_mod, _inp, out):
                hidden = out[0] if isinstance(out, tuple) else out
                captured[layer_idx] = hidden.detach()[0].float().cpu()

            return hook

        for k in layer_indices:
            if not 0 <= k < self.n_layers:
                raise ValueError(f"layer {k} out of range 0..{self.n_layers - 1}")
            hooks.append(self._layers[k].register_forward_hook(make_hook(k)))
        try:
            ids = torch.tensor([token_ids], device=self.device)
            with torch.no_grad():
                self._model(ids)
        finally:
            for h in hooks:
                h.remove()

        missing = [k for k in layer_indices if k not in captured]
        if missing:
            raise RuntimeError(f"Capture hooks fired for no output at layers {missing}")
        return {k: v.numpy().astype(np.float32) for k, v in captured.items()}

    def unload(self) -> None:
        del self._model
        self._model = None
        clear_gpu_memory(aggressive=True)
