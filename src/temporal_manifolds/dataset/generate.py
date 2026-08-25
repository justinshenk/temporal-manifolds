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
    from . import conversational_no_time as conversational_no_time_dataset
    from . import event_anchored as event_anchored_dataset
    from . import indirect_horizon as indirect_horizon_dataset
    from . import indexed_horizon as indexed_horizon_dataset
    from . import plain_english as plain_english_dataset
    from . import plain_long as plain_long_dataset
    from . import task_only as task_only_dataset
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
    import conversational_no_time as conversational_no_time_dataset  # type: ignore
    import event_anchored as event_anchored_dataset  # type: ignore
    import indirect_horizon as indirect_horizon_dataset  # type: ignore
    import indexed_horizon as indexed_horizon_dataset  # type: ignore
    import plain_english as plain_english_dataset  # type: ignore
    import plain_long as plain_long_dataset  # type: ignore
    import task_only as task_only_dataset  # type: ignore

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
    base_value: int | None
    base_unit: str | None
    unit_variant: Literal["original", "smaller"] | None
    number_format: Literal["numeric", "words"] | None
    value: int | None
    value_text: str | None
    unit: str | None


DatasetName = Literal[
    "conversational",
    "conversational_no_time",
    "event_anchored",
    "abstract",
    "plain_english",
    "plain_long",
    "task_only",
    "indirect_horizon",
    "indexed_horizon",
]

DATASETS = {
    "conversational": conversational_dataset,
    "conversational_no_time": conversational_no_time_dataset,
    "event_anchored": event_anchored_dataset,
    "abstract": abstract_dataset,
    "plain_english": plain_english_dataset,
    "plain_long": plain_long_dataset,
    "task_only": task_only_dataset,
    "indirect_horizon": indirect_horizon_dataset,
    "indexed_horizon": indexed_horizon_dataset,
}

TIME_CONSTRAINT_LINE = re.compile(
    r"^\s*(?:available time|time budget|deadline|time constraint|time window)\s*:",
    re.IGNORECASE,
)


def remove_time_constraint_lines(template: str) -> str:
    """Remove explicit time-constraint fields from a prompt template."""
    lines = [line for line in template.splitlines() if not TIME_CONSTRAINT_LINE.match(line)]
    return "\n".join(lines)


def unconstrained_template(template_config: TemplateConfig) -> str:
    """Return a template's natural time-free form or remove labelled time fields."""
    configured = template_config.get("unconstrained_template")
    if configured is not None:
        return str(configured)
    return remove_time_constraint_lines(str(template_config["template"]))


def template_metadata(template_config: TemplateConfig) -> dict[str, str]:
    """Return analysis metadata, excluding strings used only for rendering."""
    return {
        key: str(value)
        for key, value in template_config.items()
        if key not in {"id", "template", "unconstrained_template"}
    }


def normalize_dataset_name(dataset: str) -> DatasetName:
    """Return a supported dataset name."""
    if dataset not in DATASETS:
        supported = ", ".join(sorted(DATASETS))
        raise ValueError(f"Unsupported dataset {dataset!r}. Choose one of: {supported}")
    return dataset  # type: ignore[return-value]


def load_dataset_config(
    dataset: str,
) -> tuple[Iterable[TemplateConfig], Mapping[str, TaskConfig], Iterable[int]]:
    """Return templates, task units, and time values for a named dataset."""
    dataset_name = normalize_dataset_name(dataset)
    dataset_module = DATASETS[dataset_name]
    return (
        dataset_module.templates,
        dataset_module.tasks,
        dataset_module.values,
    )


def is_time_free_dataset(dataset: str) -> bool:
    """Return whether a dataset's prompts never express a time horizon."""
    dataset_module = DATASETS[normalize_dataset_name(dataset)]
    return bool(getattr(dataset_module, "time_free", False))


def dataset_record_builder(dataset: str):
    """Return a dataset's own record builder, or ``None`` for template-driven datasets.

    Most datasets render a prompt by formatting one template string. A dataset whose
    horizon is implied rather than written -- clock times, calendar dates, fractions
    of a larger allowance -- has to compute its text from the horizon instead, so it
    builds its own records and this returns that builder.
    """
    dataset_module = DATASETS[normalize_dataset_name(dataset)]
    return getattr(dataset_module, "build_prompt_records", None)


def normalize_task_configs(
    task_configs: Mapping[str, TaskConfig],
) -> dict[str, NormalizedTaskConfig]:
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

    return {
        "text": template.format(
            task=task,
            value=value_text,
            unit=rendered_unit,
        ),
        "template_id": template_id,
        "template_metadata": template_metadata,
        "task": task,
        "task_metadata": task_metadata,
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
                base_value=base_value,
                base_unit=base_unit,
                value=value,
                unit=unit,
                unit_variant=unit_variant,
                number_format=number_format,
            )
        )


def append_unconstrained_prompt(
    records: list[PromptRecord],
    template: str,
    template_id: str,
    template_metadata: dict[str, str],
    task: str,
    task_metadata: dict[str, str],
) -> None:
    """Append one prompt with no horizon metadata."""
    records.append(
        {
            "text": template.format(task=task, value="", unit=""),
            "template_id": template_id,
            "template_metadata": template_metadata,
            "task": task,
            "task_metadata": task_metadata,
            "base_value": None,
            "base_unit": None,
            "unit_variant": None,
            "number_format": None,
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
    output_path: str | Path | None = None,
    randomize_template: bool = False,
    dataset: str = "conversational",
    remove_time_constraints: bool = False,
) -> list[PromptRecord]:
    """Return formatted prompt records from the configured templates and tasks.

    When ``randomize_template`` is false, return the full cartesian product of
    templates and task parameters. When true, generate each task-parameter sample
    once and choose one template from ``template_list`` at random for that sample.
    Datasets whose prompts never express a horizon declare ``time_free = True``
    and are always generated without time parameters, because iterating the
    horizon grid would emit identical duplicate prompts rather than new
    conditions.
    """
    record_builder = dataset_record_builder(dataset)
    if record_builder is not None:
        unsupported = [
            name
            for name, argument in (
                ("template_list", template_list),
                ("task_units", task_units),
                ("time_values", time_values),
            )
            if argument is not None
        ]
        unsupported += [
            name
            for name, flag in (
                ("randomize_template", randomize_template),
                ("remove_time_constraints", remove_time_constraints),
            )
            if flag
        ]
        if unsupported:
            # These options all assume a prompt is one template string with the
            # horizon substituted into it, which is precisely what this dataset is
            # not. Silently ignoring them would emit prompts that contradict the
            # caller's request; task_only and conversational_no_time provide
            # explicit time-free controls instead.
            raise ValueError(
                f"Dataset {dataset!r} builds its own prompts and does not support: "
                f"{', '.join(sorted(unsupported))}."
            )
        built_records: list[PromptRecord] = record_builder()
        write_prompt_records(built_records, output_path)
        return built_records

    dataset_templates, dataset_tasks, dataset_values = load_dataset_config(dataset)
    if is_time_free_dataset(dataset):
        remove_time_constraints = True
    if template_list is None:
        template_list = dataset_templates
    if task_units is None:
        task_units = dataset_tasks
    if time_values is None:
        time_values = dataset_values

    records: list[PromptRecord] = []
    template_configs = tuple(template_list)
    time_configs = tuple(time_values)
    if not template_configs:
        raise ValueError("template_list must contain at least one template")
    task_configs = normalize_task_configs(task_units)

    if remove_time_constraints:
        if randomize_template:
            for task, task_config in sorted(task_configs.items()):
                template_config = random.choice(template_configs)
                template_id = str(template_config["id"])
                template = unconstrained_template(template_config)
                metadata = template_metadata(template_config)
                append_unconstrained_prompt(
                    records,
                    template,
                    template_id,
                    metadata,
                    task,
                    task_config["metadata"],
                )
        else:
            for template_config in template_configs:
                template_id = str(template_config["id"])
                template = unconstrained_template(template_config)
                metadata = template_metadata(template_config)
                for task, task_config in sorted(task_configs.items()):
                    append_unconstrained_prompt(
                        records,
                        template,
                        template_id,
                        metadata,
                        task,
                        task_config["metadata"],
                    )

        write_prompt_records(records, output_path)
        return records

    if randomize_template:
        for task, task_config in sorted(task_configs.items()):
            units = task_config["units"]
            task_metadata = task_config["metadata"]
            for unit in sorted(units):
                for value in time_configs:
                    template_config = random.choice(template_configs)
                    template_id = str(template_config["id"])
                    template = str(template_config["template"])
                    if remove_time_constraints:
                        template = remove_time_constraint_lines(template)
                    metadata = template_metadata(template_config)
                    append_prompt_variants(
                        records=records,
                        template=template,
                        template_id=template_id,
                        template_metadata=metadata,
                        task=task,
                        task_metadata=task_metadata,
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
                        metadata = template_metadata(template_config)
                        append_prompt_variants(
                            records=records,
                            template=template,
                            template_id=template_id,
                            template_metadata=metadata,
                            task=task,
                            task_metadata=task_metadata,
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
            metadata = template_metadata(template_config)
            for task, task_config in sorted(task_configs.items()):
                units = task_config["units"]
                task_metadata = task_config["metadata"]
                for unit in sorted(units):
                    for value in time_configs:
                        append_prompt_variants(
                            records=records,
                            template=template,
                            template_id=template_id,
                            template_metadata=metadata,
                            task=task,
                            task_metadata=task_metadata,
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
                                template_metadata=metadata,
                                task=task,
                                task_metadata=task_metadata,
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
