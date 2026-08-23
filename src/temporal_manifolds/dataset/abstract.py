"""Domain-neutral tasks crossed with the complete temporal range.

Every task is deliberately available at every time unit so that a task traces a
continuous path through time instead of appearing in only one short part of the
manifold. The task families exercise different kinds of temporal reasoning.
Tasks describe terminal outcomes so the same
completion-focused request remains meaningful across all prompt framings.
"""


def _task(*, task_family: str, planning_type: str) -> dict[str, object]:
    """Return metadata for a task that covers the full configured time range."""
    return {
        # Give every task its own set so changing one task cannot narrow another.
        "units": set(time_units),
        "task_family": task_family,
        "planning_type": planning_type,
        "domain": "abstract",
        "temporal_coverage": "full_range",
    }


# Use the conversational dataset's 3 x 3 framing grid, but keep the request
# itself common so framing syntax can be controlled independently.
subject_framings = [
    {
        "id": "task",
        "label": "Task",
    },
    {
        "id": "goal",
        "label": "Goal",
    },
    {
        "id": "objective",
        "label": "Objective",
    },
]

time_framings = [
    {
        "id": "available_time",
        "label": "Available time",
        "suffix": "",
    },
    {
        "id": "time_budget",
        "label": "Time budget",
        "suffix": "",
    },
    {
        "id": "deadline",
        "label": "Deadline",
        "suffix": " from now",
    },
]

prompt_framings = [
    {
        "id": f"{subject['id']}_{time['id']}",
        "subject_framing": subject["id"],
        "subject_label": subject["label"],
        "time_framing": time["id"],
        "time_label": time["label"],
        "time_suffix": time["suffix"],
    }
    for subject in subject_framings
    for time in time_framings
]

FORMAT_NEUTRAL_COMPLETION_REQUEST = (
    "Complete it within the stated time. Explain what you would do to accomplish it."
)

templates = [
    {
        "id": framing["id"],
        "template": (
            f"{framing['subject_label']}: {{task}}\n"
            f"{framing['time_label']}: {{value}} {{unit}}{framing['time_suffix']}\n\n"
            f"{FORMAT_NEUTRAL_COMPLETION_REQUEST}"
        ),
        "prompt_framing": framing["id"],
        "subject_framing": framing["subject_framing"],
        "time_framing": framing["time_framing"],
    }
    for framing in prompt_framings
]


# Keep this as a set: dataset.generate accepts mutable sets as task-unit configs.
time_units = {
    "seconds",
    "minutes",
    "hours",
    "days",
    "weeks",
    "months",
    "years",
    "decades",
    "centuries",
    "millennia",
}


tasks = {
    "bring a system into a useful stable state": _task(
        task_family="state_stabilization",
        planning_type="stabilization",
    ),
    "produce a reliably functioning process": _task(
        task_family="process_improvement",
        planning_type="iterative",
    ),
    "determine how an unfamiliar system behaves": _task(
        task_family="system_understanding",
        planning_type="investigative",
    ),
    "revise a strategy for changed conditions": _task(
        task_family="adaptation",
        planning_type="adaptive",
    ),
    "coordinate contributors to produce an agreed result": _task(
        task_family="coordination",
        planning_type="coordination",
    ),
    "reduce a known failure risk to an acceptable level": _task(
        task_family="risk_reduction",
        planning_type="risk_management",
    ),
    "recover a disrupted system to a stable state": _task(
        task_family="system_recovery",
        planning_type="recovery",
    ),
    "transfer a body of knowledge to a new recipient": _task(
        task_family="knowledge_transfer",
        planning_type="knowledge_preservation",
    ),
    "establish an effective shared standard": _task(
        task_family="standard_governance",
        planning_type="governance",
    ),
    "allocate limited resources among competing needs": _task(
        task_family="resource_allocation",
        planning_type="allocation",
    ),
}


# Rounded samples from an approximately uniform log grid over 1--100.  The
# original anchors are retained; below 10, integer values limit finer spacing.
values = [
    1,
    2,
    3,
    4,
    5,
    6,
    7,
    8,
    9,
    10,
    11,
    13,
    14,
    16,
    18,
    20,
    22,
    25,
    28,
    30,
    32,
    35,
    40,
    45,
    50,
    56,
    63,
    70,
    79,
    89,
    100,
]
