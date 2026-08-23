"""Planning prompts whose horizons are expressed as event boundaries.

The prompts in this dataset contain no duration, date, clock time, or named time
unit.  Instead, the planning window ends at a milestone such as a handoff,
review, transition, lifecycle boundary, or civilizational epoch.  Each event
anchor is paired only with tasks whose conversational configuration already
allows the anchor's latent base unit, keeping the implied horizons plausible.

Two explicit event anchors are provided for every supported base unit.  The
base unit is retained as metadata so these records can be compared with direct
and indirect horizon datasets without exposing that unit in the prompt text.
"""

from __future__ import annotations

from typing import Any

try:
    from . import conversational
except ImportError:  # Support running dataset/generate.py directly.
    import conversational  # type: ignore


def _copy_task_config(config: dict[str, object]) -> dict[str, object]:
    """Copy a task config without sharing its mutable unit set."""
    copied = dict(config)
    copied["units"] = set(config["units"])  # type: ignore[arg-type]
    return copied


tasks = {task: _copy_task_config(config) for task, config in conversational.tasks.items()}


# Keep every prompt explicit: there is no combinatorial sentence-fragment
# assembly whose output has to be inferred by readers or tests.
templates = [
    {
        "id": "seconds_response_window",
        "base_unit": "seconds",
        "prompt_framing": "before_event",
        "template": (
            "Task: {task}\n"
            "Event constraint: Complete it before the immediate response window closes.\n\n"
            "Write a plan suited to that event boundary."
        ),
    },
    {
        "id": "seconds_signal_window",
        "base_unit": "seconds",
        "prompt_framing": "between_events",
        "template": (
            "Task: {task}\n"
            "Event constraint: Act between the go-ahead and the stop signal.\n\n"
            "Write a plan suited to that event boundary."
        ),
    },
    {
        "id": "minutes_activity_boundary",
        "base_unit": "minutes",
        "prompt_framing": "before_event",
        "template": (
            "Task: {task}\n"
            "Event constraint: Complete it before the current activity concludes.\n\n"
            "Write a plan suited to that event boundary."
        ),
    },
    {
        "id": "minutes_local_handoff",
        "base_unit": "minutes",
        "prompt_framing": "between_events",
        "template": (
            "Task: {task}\n"
            "Event constraint: Carry it out between the opening cue and the scheduled handoff.\n\n"
            "Write a plan suited to that event boundary."
        ),
    },
    {
        "id": "hours_operating_window",
        "base_unit": "hours",
        "prompt_framing": "before_event",
        "template": (
            "Task: {task}\n"
            "Event constraint: Complete it before the current operating window closes.\n\n"
            "Write a plan suited to that event boundary."
        ),
    },
    {
        "id": "hours_scheduled_checkpoint",
        "base_unit": "hours",
        "prompt_framing": "until_event",
        "template": (
            "Task: {task}\n"
            "Event constraint: Work toward completion until the next scheduled checkpoint.\n\n"
            "Write a plan suited to that event boundary."
        ),
    },
    {
        "id": "days_operational_handoff",
        "base_unit": "days",
        "prompt_framing": "before_event",
        "template": (
            "Task: {task}\n"
            "Event constraint: Reach the intended outcome before the next operational handoff.\n\n"
            "Write a plan suited to that event boundary."
        ),
    },
    {
        "id": "days_operational_cycle",
        "base_unit": "days",
        "prompt_framing": "through_event",
        "template": (
            "Task: {task}\n"
            "Event constraint: Carry the work through the current operational cycle.\n\n"
            "Write a plan suited to that event boundary."
        ),
    },
    {
        "id": "weeks_project_milestone",
        "base_unit": "weeks",
        "prompt_framing": "before_event",
        "template": (
            "Task: {task}\n"
            "Event constraint: Reach the intended outcome before the next major project milestone.\n\n"
            "Write a plan suited to that event boundary."
        ),
    },
    {
        "id": "weeks_project_phase",
        "base_unit": "weeks",
        "prompt_framing": "through_event",
        "template": (
            "Task: {task}\n"
            "Event constraint: Carry the work through the current project phase.\n\n"
            "Write a plan suited to that event boundary."
        ),
    },
    {
        "id": "months_formal_review",
        "base_unit": "months",
        "prompt_framing": "before_event",
        "template": (
            "Task: {task}\n"
            "Event constraint: Reach the intended outcome before the next formal review.\n\n"
            "Write a plan suited to that event boundary."
        ),
    },
    {
        "id": "months_planning_cycle",
        "base_unit": "months",
        "prompt_framing": "through_event",
        "template": (
            "Task: {task}\n"
            "Event constraint: Carry the work through the current planning cycle.\n\n"
            "Write a plan suited to that event boundary."
        ),
    },
    {
        "id": "years_strategic_transition",
        "base_unit": "years",
        "prompt_framing": "before_event",
        "template": (
            "Task: {task}\n"
            "Event constraint: Reach the intended outcome before the next strategic transition.\n\n"
            "Write a plan suited to that event boundary."
        ),
    },
    {
        "id": "years_programme_lifecycle",
        "base_unit": "years",
        "prompt_framing": "through_event",
        "template": (
            "Task: {task}\n"
            "Event constraint: Sustain the work through the full programme lifecycle.\n\n"
            "Write a plan suited to that event boundary."
        ),
    },
    {
        "id": "decades_generational_transition",
        "base_unit": "decades",
        "prompt_framing": "before_event",
        "template": (
            "Task: {task}\n"
            "Event constraint: Reach durable success before the next generational transition.\n\n"
            "Write a plan suited to that event boundary."
        ),
    },
    {
        "id": "decades_infrastructure_lifecycle",
        "base_unit": "decades",
        "prompt_framing": "through_event",
        "template": (
            "Task: {task}\n"
            "Event constraint: Sustain the outcome through the current infrastructure lifecycle.\n\n"
            "Write a plan suited to that event boundary."
        ),
    },
    {
        "id": "centuries_institutional_succession",
        "base_unit": "centuries",
        "prompt_framing": "before_event",
        "template": (
            "Task: {task}\n"
            "Event constraint: Establish a durable outcome before the present institutional era gives way to its successor.\n\n"
            "Write a plan suited to that event boundary."
        ),
    },
    {
        "id": "centuries_institutional_eras",
        "base_unit": "centuries",
        "prompt_framing": "through_event",
        "template": (
            "Task: {task}\n"
            "Event constraint: Sustain the outcome through successive institutional eras.\n\n"
            "Write a plan suited to that event boundary."
        ),
    },
    {
        "id": "millennia_civilizational_transition",
        "base_unit": "millennia",
        "prompt_framing": "before_event",
        "template": (
            "Task: {task}\n"
            "Event constraint: Establish a durable outcome before the next civilizational epoch begins.\n\n"
            "Write a plan suited to that event boundary."
        ),
    },
    {
        "id": "millennia_civilizational_epochs",
        "base_unit": "millennia",
        "prompt_framing": "through_event",
        "template": (
            "Task: {task}\n"
            "Event constraint: Sustain the outcome through successive civilizational epochs.\n\n"
            "Write a plan suited to that event boundary."
        ),
    },
]


# Present only to satisfy the common dataset module interface.  The custom
# builder does not iterate numeric values because no duration is rendered.
values = [1]


def build_prompt_records() -> list[dict[str, Any]]:
    """Return every explicit event anchor crossed only with plausible tasks."""
    records: list[dict[str, Any]] = []

    for template_config in templates:
        base_unit = str(template_config["base_unit"])
        for task, task_config in sorted(tasks.items()):
            task_units = task_config["units"]
            if base_unit not in task_units:  # type: ignore[operator]
                continue

            records.append(
                {
                    "text": str(template_config["template"]).format(task=task),
                    "template_id": str(template_config["id"]),
                    "template_metadata": {
                        "prompt_framing": str(template_config["prompt_framing"]),
                    },
                    "task": task,
                    "task_metadata": {
                        key: str(value)
                        for key, value in task_config.items()
                        if key != "units"
                    },
                    "base_value": None,
                    "base_unit": base_unit,
                    "unit_variant": None,
                    "number_format": None,
                    "value": None,
                    "value_text": None,
                    "unit": None,
                }
            )

    return records


__all__ = ["build_prompt_records", "tasks", "templates", "values"]
