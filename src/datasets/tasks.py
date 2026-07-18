"""Curated planning tasks with naturally different time scales.

Adapted from the legacy conversational dataset's task vocabulary (kept in
src/temporal_manifolds/dataset/conversational.py): task families are crossed
with difficulty and natural horizon span so analyses can compare within-task
across horizons and within-horizon across tasks.

The main-experiment tasks (`CORE_TASKS`) deliberately span very different
natural timescales — the same target-horizon sweep lands "short" for one task
and "long" for another, which is exactly the contrast we want colored in the
PCA plots.
"""

from __future__ import annotations

from .schema import PlanningTask

CORE_TASKS: tuple[PlanningTask, ...] = (
    PlanningTask(
        task_id="climate_city",
        description="reduce your city's carbon emissions to net zero",
        natural_units=("years", "decades"),
        task_family="civic_transformation",
        difficulty="very_high",
        domain="civilization",
        planning_type="strategic",
        stakes="high",
        agency="organization",
        scenario=(
            "You are the sustainability director of a mid-sized coastal city "
            "(population 400,000) with aging infrastructure and a limited "
            "budget."
        ),
    ),
    PlanningTask(
        task_id="marathon",
        description="prepare for and complete a marathon",
        natural_units=("weeks", "months"),
        task_family="physical_training",
        difficulty="medium",
        domain="personal_lifestyle",
        planning_type="training",
        stakes="medium",
        agency="individual",
        scenario=(
            "You are an office worker who currently runs about 10 km per week "
            "and has never raced beyond a 10K."
        ),
    ),
    PlanningTask(
        task_id="dinner_party",
        description="host a dinner party for eight guests",
        natural_units=("hours", "days"),
        task_family="event_planning",
        difficulty="low",
        domain="personal_lifestyle",
        planning_type="logistical",
        stakes="low",
        agency="individual",
        scenario=(
            "You live in a small apartment with a standard kitchen, and two "
            "of the guests have dietary restrictions you only just learned "
            "about."
        ),
    ),
    PlanningTask(
        task_id="knowledge_archive",
        description="build an archive that preserves your organization's knowledge",
        natural_units=("years", "decades", "centuries"),
        task_family="knowledge_archive",
        difficulty="high",
        domain="knowledge_preservation",
        planning_type="systems",
        stakes="high",
        agency="organization",
        scenario=(
            "You lead documentation at a research institute whose founding "
            "staff are beginning to retire."
        ),
    ),
)


def get_task(task_id: str) -> PlanningTask:
    for t in CORE_TASKS:
        if t.task_id == task_id:
            return t
    raise KeyError(f"Unknown task_id: {task_id}")
