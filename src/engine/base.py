"""Engine interface: chat rendering, generation, and targeted activation capture.

An Engine wraps one loaded model. It exposes exactly what the conversation
driver and capture pipeline need — nothing else:

  - render(messages, ...)        chat-template a message list to text
  - encode / decode              tokenizer access (no special tokens added)
  - generate_ids(input_ids, ...) continue a token sequence, return NEW ids
  - capture_resid_post(...)      one forward pass; residual-stream vectors at
                                 the requested (layer, position) targets
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np


class Engine(ABC):
    model_id: str

    @property
    @abstractmethod
    def tokenizer(self):  # HF-compatible tokenizer
        ...

    @property
    @abstractmethod
    def n_layers(self) -> int: ...

    @property
    @abstractmethod
    def d_model(self) -> int: ...

    @abstractmethod
    def render(
        self,
        messages: list[dict],
        add_generation_prompt: bool = True,
        enable_thinking: bool | None = None,
    ) -> str:
        """Apply the chat template. `enable_thinking` is forwarded to the
        template only when not None AND the template supports it."""

    def encode(self, text: str) -> list[int]:
        return self.tokenizer.encode(text, add_special_tokens=False)

    def decode(self, ids: list[int]) -> str:
        return self.tokenizer.decode(ids, skip_special_tokens=False)

    @abstractmethod
    def generate_ids(
        self,
        input_ids: list[int],
        max_new_tokens: int,
        temperature: float = 0.0,
        stop_token_ids: tuple[int, ...] = (),
    ) -> list[int]:
        """Continue `input_ids`; return ONLY the newly generated ids
        (including the stop token if one was hit)."""

    @abstractmethod
    def capture_resid_post(
        self, token_ids: list[int], layer_indices: list[int]
    ) -> dict[int, np.ndarray]:
        """One forward pass over `token_ids`.

        Returns {layer_index: float32 array [seq_len, d_model]} where the
        vector at position p is the residual stream AFTER block layer_index
        at token p (TransformerLens `resid_post` convention).
        """

    def unload(self) -> None:  # optional
        pass
