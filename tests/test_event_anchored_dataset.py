from __future__ import annotations

import re
import sys

import pytest

from temporal_manifolds.dataset import conversational, event_anchored
from temporal_manifolds.dataset.generate import (
    DATASETS,
    dataset_record_builder,
    generate_task_dataset,
)


def test_event_anchored_dataset_is_registered_with_its_custom_builder() -> None:
    assert DATASETS["event_anchored"] is event_anchored
    assert dataset_record_builder("event_anchored") is event_anchored.build_prompt_records


def test_event_anchored_tasks_are_independent_copies_of_conversational_tasks() -> None:
    assert event_anchored.tasks == conversational.tasks
    assert event_anchored.tasks is not conversational.tasks
    assert all(
        event_anchored.tasks[task]["units"] is not config["units"]
        for task, config in conversational.tasks.items()
    )


def test_event_anchored_templates_are_explicit_and_cover_each_base_unit_twice() -> None:
    expected_units = {
        unit for config in conversational.tasks.values() for unit in config["units"]
    }
    templates_by_unit = {
        unit: [
            template
            for template in event_anchored.templates
            if template["base_unit"] == unit
        ]
        for unit in expected_units
    }

    assert len(event_anchored.templates) == 2 * len(expected_units) == 20
    assert len({template["id"] for template in event_anchored.templates}) == 20
    assert all(len(templates) == 2 for templates in templates_by_unit.values())
    assert all("{task}" in template["template"] for template in event_anchored.templates)
    assert all("{value}" not in template["template"] for template in event_anchored.templates)
    assert all("{unit}" not in template["template"] for template in event_anchored.templates)


def test_event_anchored_prompts_have_no_explicit_duration_or_clock_value() -> None:
    duration_language = re.compile(
        r"\b(?:seconds?|minutes?|hours?|days?|weeks?|months?|years?|decades?"
        r"|centur(?:y|ies)|millenni(?:um|a)|deadline|time budget|available time)\b",
        re.IGNORECASE,
    )

    for template in event_anchored.templates:
        prompt_without_task = template["template"].format(task="")
        assert duration_language.search(prompt_without_task) is None
        assert re.search(r"\d", prompt_without_task) is None


def test_event_anchored_generation_only_uses_plausible_task_unit_pairs() -> None:
    records = generate_task_dataset(dataset="event_anchored")
    task_unit_pairs = {
        (task, unit)
        for task, config in conversational.tasks.items()
        for unit in config["units"]
    }

    assert len(records) == 2 * len(task_unit_pairs) == 254
    assert len({record["text"] for record in records}) == len(records)
    assert {(record["task"], record["base_unit"]) for record in records} == task_unit_pairs
    assert all(record["base_unit"] in conversational.tasks[record["task"]]["units"] for record in records)


def test_event_anchored_records_retain_event_and_task_metadata() -> None:
    records = generate_task_dataset(dataset="event_anchored")

    assert {record["template_metadata"]["prompt_framing"] for record in records} == {
        "before_event",
        "between_events",
        "through_event",
        "until_event",
    }
    assert all(record["task_metadata"] for record in records)
    assert all(record["task"] in record["text"] for record in records)
    assert all(record["base_value"] is None for record in records)
    assert all(record["value"] is None and record["unit"] is None for record in records)
    assert all(record["number_format"] is None and record["unit_variant"] is None for record in records)


@pytest.mark.parametrize(
    "unsupported",
    [
        {"time_values": [1]},
        {"remove_time_constraints": True},
        {"randomize_template": True},
    ],
)
def test_event_anchored_dataset_rejects_inapplicable_generation_options(
    unsupported: dict[str, object],
) -> None:
    with pytest.raises(ValueError, match="builds its own prompts"):
        generate_task_dataset(dataset="event_anchored", **unsupported)  # type: ignore[arg-type]


def test_event_anchored_dataset_is_available_from_the_cli(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from temporal_manifolds.dataset.generate import parse_args

    monkeypatch.setattr(sys, "argv", ["generate", "--dataset", "event_anchored"])

    assert parse_args().dataset == "event_anchored"
