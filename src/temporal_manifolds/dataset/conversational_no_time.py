"""Explicit conversational planning prompts with no time horizon."""

from __future__ import annotations

try:
    from . import conversational
except ImportError:  # Support running dataset/generate.py directly.
    import conversational  # type: ignore


templates = [
    {
        "id": "task",
        "template": "Task: {task}\n\nWrite a plan.",
        "prompt_framing": "task",
    },
    {
        "id": "goal",
        "template": "Goal: {task}\n\nWrite a plan.",
        "prompt_framing": "goal",
    },
    {
        "id": "objective",
        "template": "Objective: {task}\n\nWrite a plan.",
        "prompt_framing": "objective",
    },
]


def _copy_task_config(config: dict[str, object]) -> dict[str, object]:
    """Copy a task config without sharing its mutable unit set."""
    copied = dict(config)
    copied["units"] = set(config["units"])  # type: ignore[arg-type]
    return copied


tasks = {task: _copy_task_config(config) for task, config in conversational.tasks.items()}

time_free = True
values = [1]


__all__ = ["tasks", "templates", "time_free", "values"]
