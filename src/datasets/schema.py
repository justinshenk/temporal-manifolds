"""Schemas for planning-conversation prompt datasets.

A PlanningPrompt is a task description + response-format instructions with a
`{target_time_horizon}` slot. The horizon is substituted at materialization
time using an ecologically valid phrasing, producing a PromptSample (what is
actually sent to the model as the first user turn).
"""

from __future__ import annotations

from dataclasses import dataclass, field

from ..core.schema import BaseSchema
from ..core.time_value import TimeValue

HORIZON_SLOT = "{target_time_horizon}"


@dataclass
class PlanningTask(BaseSchema):
    """A task that admits plans at several time horizons.

    Metadata mirrors the controls used in the legacy conversational dataset
    (task_family / difficulty / domain / planning_type / stakes / agency) so
    downstream analyses can separate horizon effects from task effects.
    """

    task_id: str
    description: str  # imperative task text, e.g. "address climate change in your city"
    natural_units: tuple[str, ...]  # units the task naturally spans, e.g. ("years","decades")
    task_family: str = ""
    difficulty: str = "medium"
    domain: str = ""
    planning_type: str = ""
    stakes: str = "medium"
    agency: str = "individual"
    scenario: str = ""  # optional scene-setting paragraph shown before the task


@dataclass
class HorizonPhrasing(BaseSchema):
    """An ecologically valid way to state the target time horizon.

    `template` must contain '{horizon}' where the time value goes, e.g.
    "You have {horizon} to accomplish this." Phrasings replace the stilted
    "The target time horizon is {horizon}." wording.
    """

    phrasing_id: str
    template: str

    def render(self, horizon: TimeValue) -> str:
        return self.template.format(horizon=str(horizon))


@dataclass
class PlanningPrompt(BaseSchema):
    """One materialized first-user-turn prompt (task × horizon × phrasing)."""

    prompt_id: str
    task_id: str
    phrasing_id: str
    target_horizon: TimeValue = field(default_factory=lambda: TimeValue(1, "years"))
    text: str = ""  # full first user message (scenario+task+horizon+format)

    @property
    def target_horizon_years(self) -> float:
        return self.target_horizon.to_years()


@dataclass
class PlanningPromptDataset(BaseSchema):
    """A set of PlanningPrompts plus the tasks/phrasings that produced them."""

    name: str
    prompts: list[PlanningPrompt] = field(default_factory=list)
    tasks: list[PlanningTask] = field(default_factory=list)
    phrasings: list[HorizonPhrasing] = field(default_factory=list)
    seed: int = 0

    def get_prompt(self, prompt_id: str) -> PlanningPrompt:
        for p in self.prompts:
            if p.prompt_id == prompt_id:
                return p
        raise KeyError(f"prompt_id {prompt_id!r} not in dataset {self.name!r}")

    def get_task(self, task_id: str) -> PlanningTask:
        for t in self.tasks:
            if t.task_id == task_id:
                return t
        raise KeyError(f"task_id {task_id!r} not in dataset {self.name!r}")
