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

# The response-format contract. Kept simple and rigid so that:
#  - step headers are trivially parsable (Step: N / Time horizon: ...),
#  - the model closes with the literal sentinel "Plan Completed",
#  - it reads clearly to humans.
RESPONSE_FORMAT_INSTRUCTIONS = """Response format (follow it strictly):
1. In this first reply, give only the high-level plan: a one-line goal \
restatement, then the numbered list of step titles with the rough time \
horizon of each step in parentheses. Do not detail any step yet.
2. Each time I reply "Continue.", expand exactly one step, in order, using \
this exact header format:

Step: <number>
Time horizon: <duration of this step>

<detailed plan for this step>

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
    task: PlanningTask, phrasing: HorizonPhrasing, horizon: TimeValue
) -> str:
    return PROMPT_TEMPLATE.format(
        scenario=task.scenario,
        task_description=task.description,
        horizon_sentence=phrasing.render(horizon),
        response_format=RESPONSE_FORMAT_INSTRUCTIONS,
    )


def build_dataset(
    name: str,
    tasks: tuple[PlanningTask, ...] = CORE_TASKS,
    horizons: tuple[TimeValue, ...] = (),
    phrasings: tuple[HorizonPhrasing, ...] = DEFAULT_PHRASINGS[:1],
    seed: int = 0,
) -> PlanningPromptDataset:
    """Cross tasks × horizons × phrasings into a PlanningPromptDataset.

    prompt_id = "{task_id}__{value}{unit}__{phrasing_id}" — human-readable and
    unique within a dataset.
    """
    if not horizons:
        raise ValueError("Provide at least one target horizon")
    prompts: list[PlanningPrompt] = []
    for task in tasks:
        for horizon in horizons:
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
                        text=render_prompt_text(task, phrasing, horizon),
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
