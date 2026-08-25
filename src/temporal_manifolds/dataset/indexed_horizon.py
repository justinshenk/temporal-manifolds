"""Planning prompts whose horizon is encoded as a difference between indices.

Each prompt defines the cadence of a monotonically increasing counter, gives its
current reading, and gives the reading at which planning must stop.  Recovering
the time horizon therefore requires subtracting the two readings; the duration
itself is never written in the prompt.

The task grid and canonical horizon grid are copied from :mod:`conversational`.
Five fixed-start and four fixed-end counter framings in numeric and word-number
forms provide eighteen renderings for every task/horizon cell, including seconds
and horizons measured in millennia.  The fixed-end variants deliberately reuse
the same closing index for every value, so that index alone carries no horizon
information.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

try:
    from . import conversational
    from .utils import NUMBER_FORMATS, NumberFormat, number_to_words, render_unit, singularize_unit
except ImportError:  # Support running dataset/generate.py directly.
    import conversational  # type: ignore

    from temporal_manifolds.dataset.utils import (  # type: ignore
        NUMBER_FORMATS,
        NumberFormat,
        number_to_words,
        render_unit,
        singularize_unit,
    )


REQUEST = "Write a plan to {task} that fits before the closing index is reached."


@dataclass(frozen=True)
class Variant:
    """One indexed timeline anchored at either its start or its end."""

    id: str
    fixed_index: int
    template: str
    fixed_endpoint: bool = False

    def render(self, value: int, unit: str, number_format: NumberFormat) -> str:
        if self.fixed_endpoint:
            start_index = self.fixed_index - value
            end_index = self.fixed_index
        else:
            start_index = self.fixed_index
            end_index = self.fixed_index + value
        start = _number(start_index, number_format)
        end = _number(end_index, number_format)
        return self.template.format(
            start=start,
            end=end,
            singular_unit=singularize_unit(unit),
        )


def _number(value: int, number_format: NumberFormat) -> str:
    return str(value) if number_format == "numeric" else number_to_words(value)


variants: tuple[Variant, ...] = (
    Variant(
        "schedule_marks",
        317,
        "The schedule advances by one numbered mark at the end of each {singular_unit}. "
        "Its current mark is {start}, and work must stop when it reaches mark {end}.",
    ),
    Variant(
        "access_slots",
        542,
        "The access ledger moves to its next numbered slot once per {singular_unit}. "
        "It is now in slot {start}; access closes as slot {end} begins.",
    ),
    Variant(
        "meter_readings",
        683,
        "A pacing meter increases by one at the end of every {singular_unit}. "
        "The meter reads {start} now, and the cut-off reading is {end}.",
    ),
    Variant(
        "checkpoint_indices",
        791,
        "A new numbered checkpoint is reached after each {singular_unit}. "
        "Checkpoint {start} has just been reached, and everything must finish when "
        "checkpoint {end} is reached.",
    ),
    Variant(
        "sequence_positions",
        863,
        "The project clock moves forward one sequence position per {singular_unit}. "
        "Its present position is {start}, and no work may continue at position {end}.",
    ),
    Variant(
        "fixed_gate_index",
        941,
        "A gate counter advances by one index after every {singular_unit}. "
        "It currently shows {start}, and the gate closes when it shows {end}.",
        fixed_endpoint=True,
    ),
    Variant(
        "fixed_terminal_marker",
        887,
        "The timeline reaches one new marker per {singular_unit}. Its current marker is "
        "{start}, and marker {end} is always the terminal marker.",
        fixed_endpoint=True,
    ),
    Variant(
        "fixed_ledger_boundary",
        773,
        "The ledger increments its index once each {singular_unit}. The current entry is "
        "{start}; entries close as soon as the ledger reaches its fixed boundary at {end}.",
        fixed_endpoint=True,
    ),
    Variant(
        "fixed_sequence_limit",
        659,
        "A sequence advances by one position per {singular_unit}. Position {start} is active "
        "now, and position {end} is the fixed point at which activity must cease.",
        fixed_endpoint=True,
    ),
)

VARIANTS_BY_ID = {variant.id: variant for variant in variants}

templates = [
    {
        "id": variant.id,
        "template": f"<indexed:{variant.id}>\n\n{REQUEST}",
        "prompt_framing": "indexed",
        "horizon_reference": "counter_difference",
    }
    for variant in variants
]


def _copy_task_config(config: dict[str, object]) -> dict[str, object]:
    copied = dict(config)
    copied["units"] = set(config["units"])  # type: ignore[arg-type]
    return copied


tasks = {task: _copy_task_config(config) for task, config in conversational.tasks.items()}
values = list(conversational.values)


def build_prompt_records() -> list[dict[str, Any]]:
    """Return every indexed-horizon prompt and its canonical metadata."""
    records: list[dict[str, Any]] = []
    for variant in variants:
        template_metadata = {
            "prompt_framing": "indexed",
            "horizon_reference": "counter_difference",
        }
        for task, task_config in sorted(tasks.items()):
            task_metadata = {
                key: str(config_value)
                for key, config_value in task_config.items()
                if key != "units"
            }
            for unit in sorted(task_config["units"]):  # type: ignore[arg-type]
                for value in values:
                    for number_format in NUMBER_FORMATS:
                        situation = variant.render(value, unit, number_format)
                        records.append(
                            {
                                "text": f"{situation}\n\n{REQUEST.format(task=task)}",
                                "template_id": variant.id,
                                "template_metadata": template_metadata,
                                "task": task,
                                "task_metadata": task_metadata,
                                "base_value": value,
                                "base_unit": unit,
                                "unit_variant": "original",
                                "number_format": number_format,
                                "value": value,
                                "value_text": _number(value, number_format),
                                "unit": render_unit(value, unit),
                            }
                        )
    return records


def build_prompts() -> list[str]:
    """Return the ordered text of every generated prompt."""
    return [record["text"] for record in build_prompt_records()]


__all__ = [
    "REQUEST",
    "VARIANTS_BY_ID",
    "Variant",
    "build_prompt_records",
    "build_prompts",
    "tasks",
    "templates",
    "values",
    "variants",
]
