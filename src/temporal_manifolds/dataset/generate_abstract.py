"""Generate the complete collection of abstract-dataset prompts.

Rendering is delegated to :mod:`temporal_manifolds.dataset.generate` so number
words, singular units, validation, and next-smaller-unit conversions continue
to use the shared rules in :mod:`temporal_manifolds.dataset.utils`.
"""

from __future__ import annotations

from temporal_manifolds.dataset.generate import PromptRecord, generate_task_dataset


def generate_abstract_prompt_records() -> list[PromptRecord]:
    """Return every abstract prompt together with its generation metadata.

    The result includes numeric and word-number renderings. For every time unit
    with a configured smaller unit, it also includes an equivalent prompt
    expressed in that unit. Records are not deduplicated because identical text
    can originate from different canonical time horizons.
    """
    return generate_task_dataset(dataset="abstract")


def generate_abstract_prompts() -> list[str]:
    """Return the ordered text of every generated abstract prompt.

    Duplicate strings are retained so this list remains a lossless text-only
    projection of :func:`generate_abstract_prompt_records`.
    """
    return [record["text"] for record in generate_abstract_prompt_records()]


__all__ = ["generate_abstract_prompt_records", "generate_abstract_prompts"]
