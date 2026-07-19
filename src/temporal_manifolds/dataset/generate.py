"""Generate prompt records from selectable time-planning parameter grids."""

from __future__ import annotations

import argparse
import json
import random
import re
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Literal, TypedDict

try:
    from . import abstract as abstract_dataset
    from . import conversational as conversational_dataset
    from .utils import (
        NUMBER_FORMATS,
        NumberFormat,
        UnitVariant,
        number_to_words,
        render_unit,
        smaller_unit_value,
        validate_task_units,
    )
except ImportError:
    import abstract as abstract_dataset  # type: ignore
    import conversational as conversational_dataset  # type: ignore

    from temporal_manifolds.dataset.utils import (  # type: ignore
        NUMBER_FORMATS,
        NumberFormat,
        UnitVariant,
        number_to_words,
        render_unit,
        smaller_unit_value,
        validate_task_units,
    )


TemplateConfig = dict[str, object]


TaskConfig = set[str] | dict[str, object]


class NormalizedTaskConfig(TypedDict):
    units: set[str]
    metadata: dict[str, str]


class PromptRecord(TypedDict):
    text: str
    template_id: str
    template_metadata: dict[str, str]
    task: str
    task_metadata: dict[str, str]
    quantity: int | None
    quantity_text: str | None
    base_value: int | None
    base_unit: str | None
    unit_variant: Literal["original", "smaller"] | None
    number_format: Literal["numeric", "words"] | None
    value: int | None
    value_text: str | None
    unit: str | None


DatasetName = Literal["conversational", "abstract"]

DATASETS = {
    "conversational": conversational_dataset,
    "abstract": abstract_dataset,
}

TIME_CONSTRAINT_LINE = re.compile(
    r"^\s*(?:available time|time budget|deadline|time constraint|time window)\s*:",
    re.IGNORECASE,
)


def remove_time_constraint_lines(template: str) -> str:
    """Remove explicit time-constraint fields from a prompt template."""
    lines = [line for line in template.splitlines() if not TIME_CONSTRAINT_LINE.match(line)]
    return "\n".join(lines)


def normalize_dataset_name(dataset: str) -> DatasetName:
    """Return a supported dataset name."""
    if dataset not in DATASETS:
        supported = ", ".join(sorted(DATASETS))
        raise ValueError(f"Unsupported dataset {dataset!r}. Choose one of: {supported}")
    return dataset  # type: ignore[return-value]


def load_dataset_config(
    dataset: str,
) -> tuple[Iterable[TemplateConfig], Mapping[str, TaskConfig], Iterable[int], Iterable[int | None]]:
    """Return templates, task units, time values, and quantities for a named dataset."""
    dataset_name = normalize_dataset_name(dataset)
    dataset_module = DATASETS[dataset_name]
    return (
        dataset_module.templates,
        dataset_module.tasks,
        dataset_module.values,
        getattr(dataset_module, "quantities", (None,)),
    )


def normalize_task_configs(task_configs: Mapping[str, TaskConfig]) -> dict[str, NormalizedTaskConfig]:
    """Return task units plus optional analysis metadata for each task."""
    normalized: dict[str, NormalizedTaskConfig] = {}
    units_by_task: dict[str, set[str]] = {}

    for task, config in task_configs.items():
        if isinstance(config, set):
            units = set(config)
            metadata: dict[str, str] = {}
        else:
            raw_units = config.get("units")
            if not isinstance(raw_units, set):
                raise ValueError(f"Task {task!r} must define units as a set of time units")
            units = set(raw_units)
            metadata = {key: str(value) for key, value in config.items() if key != "units"}

        units_by_task[task] = units
        normalized[task] = {"units": units, "metadata": metadata}

    validate_task_units(units_by_task)
    return normalized


def make_prompt_record(
    template: str,
    template_id: str,
    template_metadata: dict[str, str],
    task: str,
    task_metadata: dict[str, str],
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
        "template_metadata": template_metadata,
        "task": task,
        "task_metadata": task_metadata,
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
    template_metadata: dict[str, str],
    task: str,
    task_metadata: dict[str, str],
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
                template_metadata=template_metadata,
                task=task,
                task_metadata=task_metadata,
                quantity=quantity,
                base_value=base_value,
                base_unit=base_unit,
                value=value,
                unit=unit,
                unit_variant=unit_variant,
                number_format=number_format,
            )
        )


def append_unconstrained_prompt_variants(
    records: list[PromptRecord],
    template: str,
    template_id: str,
    template_metadata: dict[str, str],
    task: str,
    task_metadata: dict[str, str],
    quantity: int | None,
) -> None:
    """Append only variants that still change an unconstrained prompt's text."""
    number_formats: tuple[NumberFormat | None, ...] = (
        (None,) if quantity is None else NUMBER_FORMATS
    )
    for number_format in number_formats:
        quantity_text = (
            None
            if quantity is None
            else str(quantity)
            if number_format == "numeric"
            else number_to_words(quantity)
        )
        records.append(
            {
                "text": template.format(
                    task=task,
                    quantity=quantity_text,
                    value="",
                    unit="",
                ),
                "template_id": template_id,
                "template_metadata": template_metadata,
                "task": task,
                "task_metadata": task_metadata,
                "quantity": quantity,
                "quantity_text": quantity_text,
                "base_value": None,
                "base_unit": None,
                "unit_variant": None,
                "number_format": number_format,
                "value": None,
                "value_text": None,
                "unit": None,
            }
        )


def write_prompt_records(records: list[PromptRecord], output_path: str | Path | None) -> None:
    """Write prompt records when an output path was requested."""
    if output_path is None:
        return
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    output_file.write_text(json.dumps(records, indent=2), encoding="utf-8")


def generate_task_dataset(
    template_list: Iterable[TemplateConfig] | None = None,
    task_units: Mapping[str, TaskConfig] | None = None,
    time_values: Iterable[int] | None = None,
    quantity_values: Iterable[int | None] | None = None,
    output_path: str | Path | None = None,
    randomize_template: bool = False,
    dataset: str = "conversational",
    remove_time_constraints: bool = False,
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
    task_configs = normalize_task_configs(task_units)

    if remove_time_constraints:
        if randomize_template:
            for task, task_config in sorted(task_configs.items()):
                for quantity in quantity_configs:
                    template_config = random.choice(template_configs)
                    template_id = str(template_config["id"])
                    template = remove_time_constraint_lines(str(template_config["template"]))
                    template_metadata = {
                        key: str(value)
                        for key, value in template_config.items()
                        if key not in {"id", "template"}
                    }
                    append_unconstrained_prompt_variants(
                        records,
                        template,
                        template_id,
                        template_metadata,
                        task,
                        task_config["metadata"],
                        quantity,
                    )
        else:
            for template_config in template_configs:
                template_id = str(template_config["id"])
                template = remove_time_constraint_lines(str(template_config["template"]))
                template_metadata = {
                    key: str(value)
                    for key, value in template_config.items()
                    if key not in {"id", "template"}
                }
                for task, task_config in sorted(task_configs.items()):
                    for quantity in quantity_configs:
                        append_unconstrained_prompt_variants(
                            records,
                            template,
                            template_id,
                            template_metadata,
                            task,
                            task_config["metadata"],
                            quantity,
                        )

        write_prompt_records(records, output_path)
        return records

    if randomize_template:
        for task, task_config in sorted(task_configs.items()):
            units = task_config["units"]
            task_metadata = task_config["metadata"]
            for quantity in quantity_configs:
                for unit in sorted(units):
                    for value in time_values:
                        template_config = random.choice(template_configs)
                        template_id = str(template_config["id"])
                        template = str(template_config["template"])
                        if remove_time_constraints:
                            template = remove_time_constraint_lines(template)
                        template_metadata = {
                            key: str(value)
                            for key, value in template_config.items()
                            if key not in {"id", "template"}
                        }
                        append_prompt_variants(
                            records=records,
                            template=template,
                            template_id=template_id,
                            template_metadata=template_metadata,
                            task=task,
                            task_metadata=task_metadata,
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
                            template_id = str(template_config["id"])
                            template = str(template_config["template"])
                            if remove_time_constraints:
                                template = remove_time_constraint_lines(template)
                            template_metadata = {
                                key: str(value)
                                for key, value in template_config.items()
                                if key not in {"id", "template"}
                            }
                            append_prompt_variants(
                                records=records,
                                template=template,
                                template_id=template_id,
                                template_metadata=template_metadata,
                                task=task,
                                task_metadata=task_metadata,
                                quantity=quantity,
                                base_value=value,
                                base_unit=unit,
                                value=smaller_value,
                                unit=smaller_unit,
                                unit_variant="smaller",
                            )
    else:
        for template_config in template_configs:
            template_id = str(template_config["id"])
            template = str(template_config["template"])
            if remove_time_constraints:
                template = remove_time_constraint_lines(template)
            template_metadata = {
                key: str(value)
                for key, value in template_config.items()
                if key not in {"id", "template"}
            }
            for task, task_config in sorted(task_configs.items()):
                units = task_config["units"]
                task_metadata = task_config["metadata"]
                for quantity in quantity_configs:
                    for unit in sorted(units):
                        for value in time_values:
                            append_prompt_variants(
                                records=records,
                                template=template,
                                template_id=template_id,
                                template_metadata=template_metadata,
                                task=task,
                                task_metadata=task_metadata,
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
                                    template_metadata=template_metadata,
                                    task=task,
                                    task_metadata=task_metadata,
                                    quantity=quantity,
                                    base_value=value,
                                    base_unit=unit,
                                    value=smaller_value,
                                    unit=smaller_unit,
                                    unit_variant="smaller",
                                )

    write_prompt_records(records, output_path)

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
    parser.add_argument(
        "--remove-time-constraints",
        action="store_true",
        help="Remove explicit Available time, Deadline, and similar prompt fields.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    dataset_records: list[PromptRecord] = generate_task_dataset(
        dataset=args.dataset,
        output_path=args.output_path,
        randomize_template=args.randomize_template,
        remove_time_constraints=args.remove_time_constraints,
    )
    print(json.dumps(dataset_records, indent=2))
