"""Tests for activation prompt dataset generation."""

from __future__ import annotations


def test_conversational_records_include_task_metadata() -> None:
    from temporal_manifolds.dataset.generate import generate_task_dataset

    records = generate_task_dataset(
        dataset="conversational",
        template_list=[
            {
                "id": "test_template",
                "template": "Task: {task}\nTime: {value} {unit}",
                "output_format": "strategy_steps",
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
    assert records[0]["template_metadata"] == {"output_format": "strategy_steps"}


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


def test_conversational_dataset_has_crossed_difficulty_controls() -> None:
    from temporal_manifolds.dataset import conversational

    controlled_families = {
        "communication_plan",
        "organization_project",
        "software_project",
        "knowledge_archive",
    }
    variants_by_family = {family: [] for family in controlled_families}

    for config in conversational.tasks.values():
        family = config["task_family"]
        if family in variants_by_family:
            variants_by_family[family].append(config)

    for family, variants in variants_by_family.items():
        difficulties = {variant["difficulty"] for variant in variants}
        assert difficulties == {"low", "medium", "high"}, family
        assert all(len(variant["units"]) >= 3 for variant in variants), family


def test_conversational_dataset_has_output_format_variations() -> None:
    from temporal_manifolds.dataset import conversational

    output_formats = {template["output_format"] for template in conversational.templates}
    assert output_formats == {"strategy_steps", "summary_checklist", "approach_actions"}

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

    assert len(conversational.templates) == len(output_formats) * len(prompt_framings)
