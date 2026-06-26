"""Generate prompt records from selectable time-planning parameter grids."""

from __future__ import annotations

import argparse
import json
import random
from collections.abc import Iterable
from pathlib import Path
from typing import Literal, TypedDict

try:
    from ..utils.dataset_utils import (
        NUMBER_FORMATS,
        NumberFormat,
        UnitVariant,
        number_to_words,
        render_unit,
        smaller_unit_value,
        validate_task_units,
    )
    from . import abstract_dataset, conversational_dataset
except ImportError:
    import abstract_dataset  # type: ignore
    import conversational_dataset  # type: ignore

    from temporal_manifolds.utils.dataset_utils import (  # type: ignore
        NUMBER_FORMATS,
        NumberFormat,
        UnitVariant,
        number_to_words,
        render_unit,
        smaller_unit_value,
        validate_task_units,
    )


class TemplateConfig(TypedDict):
    id: str
    template: str


class PromptRecord(TypedDict):
    text: str
    template_id: str
    task: str
    quantity: int | None
    quantity_text: str | None
    base_value: int
    base_unit: str
    unit_variant: Literal["original", "smaller"]
    number_format: Literal["numeric", "words"]
    value: int
    value_text: str
    unit: str


DatasetName = Literal["conversational", "abstract"]

DATASETS = {
    "conversational": conversational_dataset,
    "abstract": abstract_dataset,
}


def normalize_dataset_name(dataset: str) -> DatasetName:
    """Return a supported dataset name."""
    if dataset not in DATASETS:
        supported = ", ".join(sorted(DATASETS))
        raise ValueError(f"Unsupported dataset {dataset!r}. Choose one of: {supported}")
    return dataset  # type: ignore[return-value]


def load_dataset_config(
    dataset: str,
) -> tuple[Iterable[TemplateConfig], dict[str, set[str]], Iterable[int], Iterable[int | None]]:
    """Return templates, task units, time values, and quantities for a named dataset."""
    dataset_name = normalize_dataset_name(dataset)
    dataset_module = DATASETS[dataset_name]
    return (
        dataset_module.templates,
        dataset_module.tasks,
        dataset_module.values,
        getattr(dataset_module, "quantities", (None,)),
    )


def make_prompt_record(
    template: str,
    template_id: str,
    task: str,
    quantity: int | None,
    base_value: int,
    base_unit: str,
    value: int,
    unit: str,
    unit_variant: UnitVariant,
    number_format: NumberFormat,
) -> PromptRecord:
    """Return one formatted prompt plus the parameters that generated it."""
    rendered_unit = render_unit(value, unit)
    value_text = str(value) if number_format == "numeric" else number_to_words(value)
    quantity_text = (
        None
        if quantity is None
        else str(quantity)
        if number_format == "numeric"
        else number_to_words(quantity)
    )

    return {
        "text": template.format(
            task=task,
            quantity=quantity_text,
            value=value_text,
            unit=rendered_unit,
        ),
        "template_id": template_id,
        "task": task,
        "quantity": quantity,
        "quantity_text": quantity_text,
        "base_value": base_value,
        "base_unit": base_unit,
        "unit_variant": unit_variant,
        "number_format": number_format,
        "value": value,
        "value_text": value_text,
        "unit": rendered_unit,
    }


def append_prompt_variants(
    records: list[PromptRecord],
    template: str,
    template_id: str,
    task: str,
    quantity: int | None,
    base_value: int,
    base_unit: str,
    value: int,
    unit: str,
    unit_variant: UnitVariant,
) -> None:
    """Append numeric and word-number records for one prompt."""
    for number_format in NUMBER_FORMATS:
        records.append(
            make_prompt_record(
                template=template,
                template_id=template_id,
                task=task,
                quantity=quantity,
                base_value=base_value,
                base_unit=base_unit,
                value=value,
                unit=unit,
                unit_variant=unit_variant,
                number_format=number_format,
            )
        )


def generate_task_dataset(
    template_list: Iterable[TemplateConfig] | None = None,
    task_units: dict[str, set[str]] | None = None,
    time_values: Iterable[int] | None = None,
    quantity_values: Iterable[int | None] | None = None,
    output_path: str | Path | None = None,
    randomize_template: bool = False,
    dataset: str = "conversational",
) -> list[PromptRecord]:
    """Return formatted prompt records from the configured templates and tasks.

    When ``randomize_template`` is false, return the full cartesian product of
    templates and task parameters. When true, generate each task-parameter sample
    once and choose one template from ``template_list`` at random for that sample.
    """
    dataset_templates, dataset_tasks, dataset_values, dataset_quantities = load_dataset_config(
        dataset
    )
    if template_list is None:
        template_list = dataset_templates
    if task_units is None:
        task_units = dataset_tasks
    if time_values is None:
        time_values = dataset_values
    if quantity_values is None:
        quantity_values = dataset_quantities

    records: list[PromptRecord] = []
    template_configs = tuple(template_list)
    quantity_configs = tuple(quantity_values)
    if not template_configs:
        raise ValueError("template_list must contain at least one template")
    if not quantity_configs:
        raise ValueError("quantity_values must contain at least one value")
    validate_task_units(task_units)

    if randomize_template:
        for task, units in sorted(task_units.items()):
            for quantity in quantity_configs:
                for unit in sorted(units):
                    for value in time_values:
                        template_config = random.choice(template_configs)
                        append_prompt_variants(
                            records=records,
                            template=template_config["template"],
                            template_id=template_config["id"],
                            task=task,
                            quantity=quantity,
                            base_value=value,
                            base_unit=unit,
                            value=value,
                            unit=unit,
                            unit_variant="original",
                        )

                        smaller = smaller_unit_value(value, unit)
                        if smaller is not None:
                            smaller_value, smaller_unit = smaller
                            template_config = random.choice(template_configs)
                            append_prompt_variants(
                                records=records,
                                template=template_config["template"],
                                template_id=template_config["id"],
                                task=task,
                                quantity=quantity,
                                base_value=value,
                                base_unit=unit,
                                value=smaller_value,
                                unit=smaller_unit,
                                unit_variant="smaller",
                            )
    else:
        for template_config in template_configs:
            template_id = template_config["id"]
            template = template_config["template"]
            for task, units in sorted(task_units.items()):
                for quantity in quantity_configs:
                    for unit in sorted(units):
                        for value in time_values:
                            append_prompt_variants(
                                records=records,
                                template=template,
                                template_id=template_id,
                                task=task,
                                quantity=quantity,
                                base_value=value,
                                base_unit=unit,
                                value=value,
                                unit=unit,
                                unit_variant="original",
                            )

                            smaller = smaller_unit_value(value, unit)
                            if smaller is not None:
                                smaller_value, smaller_unit = smaller
                                append_prompt_variants(
                                    records=records,
                                    template=template,
                                    template_id=template_id,
                                    task=task,
                                    quantity=quantity,
                                    base_value=value,
                                    base_unit=unit,
                                    value=smaller_value,
                                    unit=smaller_unit,
                                    unit_variant="smaller",
                                )

    if output_path is not None:
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        output_file.write_text(json.dumps(records, indent=2), encoding="utf-8")

    return records


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset",
        choices=sorted(DATASETS),
        default="conversational",
        help="Dataset configuration to generate.",
    )
    parser.add_argument(
        "--output-path",
        type=Path,
        default=None,
        help="Optional JSON path to write generated records.",
    )
    parser.add_argument(
        "--randomize-template",
        action="store_true",
        help="Choose one random template per task/time sample.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    dataset_records: list[PromptRecord] = generate_task_dataset(
        dataset=args.dataset,
        output_path=args.output_path,
        randomize_template=args.randomize_template,
    )
    print(json.dumps(dataset_records, indent=2))
