"""Materialize PlanningPromptDatasets: tasks × target horizons × phrasings.

The rendered first-user-turn prompt follows a strict, parsable format that
also reads naturally to humans (see PROMPT_TEMPLATE). The conversation
protocol (src/conversation/protocol.py) relies on the response format wording
here, so the two are kept in one place and cross-imported.
"""

from __future__ import annotations

from ..core.time_value import TimeValue
from .phrasings import DEFAULT_PHRASINGS, HorizonPhrasing
from .schema import PlanningPrompt, PlanningPromptDataset, PlanningTask
from .tasks import CORE_TASKS

# The response-format contract. Two step-time semantics are supported:
#   "duration" — Time horizon: how long the step takes (original protocol)
#   "target"   — Time target: how far into the future the step's outcome lies,
#                measured from the start of the plan (cumulative offset; should
#                be non-decreasing across steps)
RESPONSE_FORMAT_INSTRUCTIONS_TARGET = """Response format (follow it strictly):
1. In this first reply, give only the high-level plan: a one-line goal \
restatement, then the numbered list of step titles, each with the rough \
point in the future it aims at, in parentheses. Do not detail any step yet.
2. Each time I reply "Continue.", expand exactly one step, in order, using \
this exact header format:

Step: <number>
Time target: <how far into the future this step's outcome lies>

<detailed plan for this step>

The Time target line states WHEN, counted from today, this step's outcome \
is reached — not how long the step takes. It must be one exact offset with \
a single number and unit, such as "3 weeks" or "5 years". Never a range, a \
calendar date, a duration of work, or a vague word ("soon", "ongoing"). \
Because steps move the plan forward, each step's time target should be at \
or beyond the previous step's.

3. After the final step has been expanded, reply to my next "Continue." with \
exactly: Plan Completed

Important: reply one message at a time and then stop. Never write "Continue." \
yourself, never expand more than one step per message, and never write \
"Plan Completed" in the same message as a step."""

RESPONSE_FORMAT_INSTRUCTIONS = """Response format (follow it strictly):
1. In this first reply, give only the high-level plan: a one-line goal \
restatement, then the numbered list of step titles with the rough time \
horizon of each step in parentheses. Do not detail any step yet.
2. Each time I reply "Continue.", expand exactly one step, in order, using \
this exact header format:

Step: <number>
Time horizon: <duration of this step>

<detailed plan for this step>

The Time horizon line must state how long that single step takes, as one \
exact duration — a single number and unit such as "3 days" or "6 weeks". \
Do not write calendar positions or spans ("Day 2-Day 5", "Weeks 1-2", \
"Months 3-12"), ranges ("2-6 months"), frequencies ("Yearly", "Ongoing"), \
or vague words ("Immediate"). If a step is recurring or open-ended, write \
the total time it occupies within this plan as a single duration.

3. After the final step has been expanded, reply to my next "Continue." with \
exactly: Plan Completed

Important: reply one message at a time and then stop. Never write "Continue." \
yourself, never expand more than one step per message, and never write \
"Plan Completed" in the same message as a step."""


PROMPT_TEMPLATE = """Scenario:
{scenario}

Task:
Create a plan to {task_description}.

Objective:
{horizon_sentence}

{response_format}"""


def render_prompt_text(
    task: PlanningTask,
    phrasing: HorizonPhrasing,
    horizon: TimeValue,
    step_mode: str = "duration",
) -> str:
    fmt = (
        RESPONSE_FORMAT_INSTRUCTIONS_TARGET
        if step_mode == "target"
        else RESPONSE_FORMAT_INSTRUCTIONS
    )
    return PROMPT_TEMPLATE.format(
        scenario=task.scenario,
        task_description=task.description,
        horizon_sentence=phrasing.render(horizon),
        response_format=fmt,
    )


def build_dataset(
    name: str,
    tasks: tuple[PlanningTask, ...] = CORE_TASKS,
    horizons: tuple[TimeValue, ...] = (),
    phrasings: tuple[HorizonPhrasing, ...] = DEFAULT_PHRASINGS[:1],
    seed: int = 0,
    step_mode: str = "duration",
    task_horizons: dict[str, tuple[TimeValue, ...]] | None = None,
) -> PlanningPromptDataset:
    """Cross tasks × horizons × phrasings into a PlanningPromptDataset.

    `task_horizons` overrides `horizons` per task_id (task-appropriate sweeps).
    prompt_id = "{task_id}__{value}{unit}__{phrasing_id}" — human-readable and
    unique within a dataset.
    """
    if not horizons and not task_horizons:
        raise ValueError("Provide at least one target horizon")
    prompts: list[PlanningPrompt] = []
    for task in tasks:
        for horizon in (task_horizons or {}).get(task.task_id, horizons):
            for phrasing in phrasings:
                value_str = (
                    str(int(horizon.value))
                    if horizon.value == int(horizon.value)
                    else str(horizon.value)
                )
                prompt_id = f"{task.task_id}__{value_str}{horizon.unit}__{phrasing.phrasing_id}"
                prompts.append(
                    PlanningPrompt(
                        prompt_id=prompt_id,
                        task_id=task.task_id,
                        phrasing_id=phrasing.phrasing_id,
                        target_horizon=horizon,
                        text=render_prompt_text(task, phrasing, horizon, step_mode),
                    )
                )
    return PlanningPromptDataset(
        name=name,
        prompts=prompts,
        tasks=list(tasks),
        phrasings=list(phrasings),
        seed=seed,
    )


def horizons_from_specs(specs: list) -> tuple[TimeValue, ...]:
    """Parse ["3 days", [2, "weeks"], {"value": 1, "unit": "years"}] etc."""
    return tuple(TimeValue.parse(s) for s in specs)
