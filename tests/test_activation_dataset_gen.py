"""Tests for activation prompt dataset generation."""

from __future__ import annotations

import sys

import pytest


def test_conversational_records_include_task_metadata() -> None:
    from temporal_manifolds.dataset.generate import generate_task_dataset

    records = generate_task_dataset(
        dataset="conversational",
        template_list=[
            {
                "id": "test_template",
                "template": "Task: {task}\nTime: {value} {unit}",
            }
        ],
        task_units={
            "answer a yes-or-no question": {
                "units": {"seconds"},
                "domain": "communication",
                "complexity": "low",
                "planning_type": "reactive",
                "stakes": "low",
                "agency": "individual",
            }
        },
        time_values=[1],
    )

    assert records[0]["task_metadata"] == {
        "domain": "communication",
        "complexity": "low",
        "planning_type": "reactive",
        "stakes": "low",
        "agency": "individual",
    }
    assert records[0]["template_metadata"] == {}


def test_plain_task_unit_sets_remain_supported() -> None:
    from temporal_manifolds.dataset.generate import generate_task_dataset

    records = generate_task_dataset(
        template_list=[
            {
                "id": "test_template",
                "template": "Task: {task}\nTime: {value} {unit}",
            }
        ],
        task_units={"test task": {"minutes"}},
        time_values=[1],
    )

    assert records[0]["task_metadata"] == {}
    assert records[0]["unit"] == "minute"


def test_one_shot_time_value_iterables_cover_the_full_generation_grid() -> None:
    from temporal_manifolds.dataset.generate import generate_task_dataset

    kwargs = {
        "template_list": [
            {"id": "first", "template": "{task}: {value} {unit}"},
            {"id": "second", "template": "{task}: {value} {unit}"},
        ],
        "task_units": {"test task": {"minutes", "hours"}},
    }
    from_list = generate_task_dataset(time_values=[1, 2], **kwargs)
    from_iterator = generate_task_dataset(time_values=iter([1, 2]), **kwargs)

    assert from_iterator == from_list


def test_months_include_an_approximate_four_week_variant() -> None:
    from temporal_manifolds.dataset.utils import smaller_unit_value

    assert smaller_unit_value(1, "months") == (4, "weeks")


def test_time_constraints_can_be_removed_from_generated_prompts() -> None:
    from temporal_manifolds.dataset.generate import generate_task_dataset

    records = generate_task_dataset(
        template_list=[
            {
                "id": "deadline",
                "template": "Task: {task}\nDeadline: {value} {unit} from now\n\nMake a plan.",
            }
        ],
        task_units={"test task": {"minutes"}},
        time_values=[1, 2, 3],
        remove_time_constraints=True,
    )

    assert len(records) == 1
    assert records[0]["text"] == "Task: test task\n\nMake a plan."
    assert "Deadline:" not in records[0]["text"]
    assert records[0]["base_value"] is None
    assert records[0]["base_unit"] is None
    assert records[0]["unit_variant"] is None
    assert records[0]["number_format"] is None


def test_unconstrained_conversational_dataset_has_no_redundant_variants() -> None:
    from temporal_manifolds.dataset import conversational
    from temporal_manifolds.dataset.generate import generate_task_dataset

    records = generate_task_dataset(
        dataset="conversational",
        remove_time_constraints=True,
    )

    assert len(records) == len(conversational.tasks) * len(conversational.templates)
    assert len({record["text"] for record in records}) == len(records)


def test_conversational_dataset_has_crossed_difficulty_controls() -> None:
    from temporal_manifolds.dataset import conversational

    controlled_families = {"knowledge_archive"}
    variants_by_family = {family: [] for family in controlled_families}

    for config in conversational.tasks.values():
        family = config["task_family"]
        if family in variants_by_family:
            variants_by_family[family].append(config)

    for family, variants in variants_by_family.items():
        difficulties = {variant["difficulty"] for variant in variants}
        assert difficulties == {"low", "medium", "high"}, family
        assert all(len(variant["units"]) >= 3 for variant in variants), family


def test_every_difficulty_and_stakes_combination_has_two_plausibly_ranged_tasks() -> None:
    from collections import defaultdict

    from temporal_manifolds.dataset import conversational
    from temporal_manifolds.dataset.utils import SUPPORTED_UNITS

    tasks_by_combination: dict[tuple[str, str], set[str]] = defaultdict(set)
    for task, config in conversational.tasks.items():
        tasks_by_combination[(config["difficulty"], config["stakes"])].add(task)
        assert config["units"] != SUPPORTED_UNITS
        assert 1 <= len(config["units"]) <= 4

    difficulties = {config["difficulty"] for config in conversational.tasks.values()}
    stakes = {config["stakes"] for config in conversational.tasks.values()}
    expected_combinations = {
        (difficulty, stake) for difficulty in difficulties for stake in stakes
    }

    assert set(tasks_by_combination) == expected_combinations
    assert min(len(tasks) for tasks in tasks_by_combination.values()) >= 2


def test_conversational_templates_cover_all_prompt_framings() -> None:
    from temporal_manifolds.dataset import conversational

    prompt_framings = {template["prompt_framing"] for template in conversational.templates}
    assert prompt_framings == {
        "task_available_time",
        "task_time_budget",
        "task_deadline",
        "goal_available_time",
        "goal_time_budget",
        "goal_deadline",
        "objective_available_time",
        "objective_time_budget",
        "objective_deadline",
    }

    assert len(conversational.templates) == len(prompt_framings) == 9


def test_conversational_templates_follow_prompt_framings() -> None:
    from temporal_manifolds.dataset import conversational

    assert conversational.templates == [
        {
            "id": framing["id"],
            "template": framing["body"],
            "prompt_framing": framing["id"],
        }
        for framing in conversational.prompt_framings
    ]
    assert len(conversational.templates) == 9


def test_plain_english_dataset_reuses_conversational_tasks_and_time_values() -> None:
    from temporal_manifolds.dataset import conversational, plain_english

    assert plain_english.tasks == conversational.tasks
    assert plain_english.tasks is not conversational.tasks
    assert all(
        plain_english.tasks[task]["units"] is not config["units"]
        for task, config in conversational.tasks.items()
    )
    assert plain_english.values == conversational.values


def test_plain_english_prompts_are_natural_and_format_neutral() -> None:
    from temporal_manifolds.dataset import plain_english
    from temporal_manifolds.dataset.generate import generate_task_dataset

    records = generate_task_dataset(
        dataset="plain_english",
        task_units={"make a cup of tea": {"minutes"}},
        time_values=[1],
    )

    assert len(records) == len(plain_english.templates) * 4
    assert {record["template_id"] for record in records} == {
        template["id"] for template in plain_english.templates
    }
    assert all("make a cup of tea" in record["text"] for record in records)
    assert all(record["value_text"] in record["text"] for record in records)
    assert all(record["unit"] in record["text"] for record in records)
    assert all("\n" not in record["text"] for record in records)
    assert all(
        label not in record["text"]
        for record in records
        for label in ("Task:", "Goal:", "Objective:")
    )


def test_plain_english_dataset_is_available_from_the_cli(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from temporal_manifolds.dataset.generate import parse_args

    monkeypatch.setattr(sys, "argv", ["generate", "--dataset", "plain_english"])

    assert parse_args().dataset == "plain_english"


def test_plain_english_time_constraints_can_be_removed_naturally() -> None:
    from temporal_manifolds.dataset import plain_english
    from temporal_manifolds.dataset.generate import generate_task_dataset

    records = generate_task_dataset(
        dataset="plain_english",
        task_units={"make a cup of tea": {"minutes"}},
        time_values=[1, 2],
        remove_time_constraints=True,
    )

    assert len(records) == len(plain_english.templates)
    assert all("make a cup of tea" in record["text"] for record in records)
    assert all("{value}" not in record["text"] and "{unit}" not in record["text"] for record in records)
    assert all(record["value"] is None and record["unit"] is None for record in records)


def test_task_only_dataset_reuses_conversational_tasks_without_sharing_state() -> None:
    from temporal_manifolds.dataset import conversational, task_only

    assert task_only.tasks == conversational.tasks
    assert task_only.tasks is not conversational.tasks
    assert all(
        task_only.tasks[task]["units"] is not config["units"]
        for task, config in conversational.tasks.items()
    )


def test_task_only_prompts_state_the_task_and_never_a_horizon() -> None:
    import re

    from temporal_manifolds.dataset import task_only
    from temporal_manifolds.dataset.generate import generate_task_dataset

    records = generate_task_dataset(dataset="task_only")
    time_words = re.compile(
        r"\b(seconds?|minutes?|hours?|weeks?|months?|years?|decades?|centur\w*"
        r"|millenni\w*|deadline|available time|time budget)\b",
        re.IGNORECASE,
    )

    assert records
    assert all(record["task"] in record["text"] for record in records)
    assert all("{" not in record["text"] for record in records)
    # A task such as "pack a bag with essentials for the day" legitimately
    # contains a time word, so only the template wording around it is checked.
    assert not [
        record
        for record in records
        if time_words.search(record["text"].replace(record["task"], ""))
    ]
    assert {record["template_id"] for record in records} == {
        template["id"] for template in task_only.templates
    }


def test_task_only_records_carry_no_time_parameters() -> None:
    from temporal_manifolds.dataset.generate import generate_task_dataset

    records = generate_task_dataset(dataset="task_only")

    assert all(record["value"] is None and record["unit"] is None for record in records)
    assert all(
        record["base_value"] is None and record["base_unit"] is None for record in records
    )
    assert all(record["unit_variant"] is None for record in records)
    assert all(record["value_text"] is None for record in records)


def test_task_only_generation_skips_the_horizon_grid_instead_of_duplicating() -> None:
    from temporal_manifolds.dataset import task_only
    from temporal_manifolds.dataset.generate import generate_task_dataset

    records = generate_task_dataset(dataset="task_only")

    expected = len(task_only.templates) * len(task_only.tasks)
    assert len(records) == expected
    assert len({record["text"] for record in records}) == expected
    # A wider horizon grid must not multiply prompts that never render one.
    assert len(generate_task_dataset(dataset="task_only", time_values=[1, 2, 3])) == expected


def test_task_only_dataset_ignores_the_remove_time_constraints_flag() -> None:
    from temporal_manifolds.dataset.generate import generate_task_dataset

    without_flag = generate_task_dataset(dataset="task_only")
    with_flag = generate_task_dataset(dataset="task_only", remove_time_constraints=True)

    assert [record["text"] for record in without_flag] == [
        record["text"] for record in with_flag
    ]


def test_time_free_datasets_are_declared_explicitly() -> None:
    from temporal_manifolds.dataset.generate import is_time_free_dataset

    assert is_time_free_dataset("task_only") is True
    assert is_time_free_dataset("conversational_no_time") is True
    assert is_time_free_dataset("conversational") is False
    assert is_time_free_dataset("plain_english") is False


def test_task_only_dataset_is_available_from_the_cli(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from temporal_manifolds.dataset.generate import parse_args

    monkeypatch.setattr(sys, "argv", ["generate", "--dataset", "task_only"])

    assert parse_args().dataset == "task_only"


def test_conversational_no_time_templates_are_explicit_and_complete() -> None:
    from temporal_manifolds.dataset import conversational_no_time

    assert len(conversational_no_time.templates) == 3
    assert len({template["id"] for template in conversational_no_time.templates}) == 3
    assert {template["prompt_framing"] for template in conversational_no_time.templates} == {
        "task",
        "goal",
        "objective",
    }
    assert all("{value}" not in template["template"] for template in conversational_no_time.templates)
    assert all("{unit}" not in template["template"] for template in conversational_no_time.templates)


def test_conversational_no_time_generates_one_record_per_template_and_task() -> None:
    import re

    from temporal_manifolds.dataset import conversational_no_time
    from temporal_manifolds.dataset.generate import generate_task_dataset

    records = generate_task_dataset(dataset="conversational_no_time")
    temporal_language = re.compile(
        r"\b(seconds?|minutes?|hours?|weeks?|months?|years?|decades?|centur\w*"
        r"|millenni\w*|deadline|available time|time budget|time window)\b",
        re.IGNORECASE,
    )

    assert len(records) == len(conversational_no_time.templates) * len(
        conversational_no_time.tasks
    )
    assert all(record["base_value"] is None and record["base_unit"] is None for record in records)
    assert all(record["value"] is None and record["unit"] is None for record in records)
    assert not [
        record
        for record in records
        if temporal_language.search(record["text"].replace(record["task"], ""))
    ]


def test_conversational_no_time_dataset_is_available_from_the_cli(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from temporal_manifolds.dataset.generate import parse_args

    monkeypatch.setattr(sys, "argv", ["generate", "--dataset", "conversational_no_time"])

    assert parse_args().dataset == "conversational_no_time"


def test_abstract_tasks_cover_the_complete_supported_time_range() -> None:
    from temporal_manifolds.dataset import abstract
    from temporal_manifolds.dataset.utils import SUPPORTED_UNITS

    task_unit_sets = []
    for config in abstract.tasks.values():
        assert isinstance(config, dict)
        assert config["units"] == SUPPORTED_UNITS
        assert config["temporal_coverage"] == "full_range"
        task_unit_sets.append(config["units"])

    # A future mutation of one task's coverage must not silently affect another.
    assert len({id(units) for units in task_unit_sets}) == len(task_unit_sets)


def test_abstract_dataset_covers_distinct_task_families() -> None:
    from temporal_manifolds.dataset import abstract

    task_families = {config["task_family"] for config in abstract.tasks.values()}

    assert len(task_families) == len(abstract.tasks)
    assert {
        "adaptation",
        "coordination",
        "knowledge_transfer",
        "resource_allocation",
        "state_stabilization",
        "system_understanding",
    } <= task_families


def test_abstract_templates_use_the_conversational_prompt_framing_grid() -> None:
    from temporal_manifolds.dataset import abstract, conversational

    expected_framing_ids = {framing["id"] for framing in conversational.prompt_framings}
    actual_framing_ids = {framing["id"] for framing in abstract.prompt_framings}

    assert actual_framing_ids == expected_framing_ids
    assert len(abstract.templates) == len(expected_framing_ids)
    assert {template["prompt_framing"] for template in abstract.templates} == expected_framing_ids
    assert {template["subject_framing"] for template in abstract.templates} == {
        "task",
        "goal",
        "objective",
    }
    assert {template["time_framing"] for template in abstract.templates} == {
        "available_time",
        "time_budget",
        "deadline",
    }


def test_abstract_templates_share_a_completion_request() -> None:
    from temporal_manifolds.dataset import abstract

    assert all(
        template["template"].endswith(abstract.FORMAT_NEUTRAL_COMPLETION_REQUEST)
        for template in abstract.templates
    )


def test_generated_abstract_tasks_each_span_every_base_time_unit() -> None:
    from temporal_manifolds.dataset import abstract
    from temporal_manifolds.dataset.generate import generate_task_dataset

    records = generate_task_dataset(
        dataset="abstract",
        template_list=[abstract.templates[0]],
        time_values=[1],
    )
    base_units_by_task = {task: set() for task in abstract.tasks}

    for record in records:
        if record["unit_variant"] == "original" and record["number_format"] == "numeric":
            base_units_by_task[record["task"]].add(record["base_unit"])

    assert all(units == abstract.time_units for units in base_units_by_task.values())


def test_abstract_time_values_add_at_most_twenty_log_spaced_points() -> None:
    import math

    from temporal_manifolds.dataset import abstract

    original_values = {1, 2, 3, 4, 5, 10, 20, 30, 50, 70, 100}
    added_values = set(abstract.values) - original_values

    assert abstract.values == sorted(set(abstract.values))
    assert original_values <= set(abstract.values)
    assert 0 < len(added_values) <= 20

    # Below 10 there are no integer points between adjacent small values.  In
    # the decade where finer integer spacing is possible, avoid large log gaps.
    values_from_ten = [value for value in abstract.values if value >= 10]
    log_gaps = [
        math.log10(right) - math.log10(left)
        for left, right in zip(values_from_ten, values_from_ten[1:])
    ]
    assert max(log_gaps) < 0.075
