"""Scenario templates.

A scenario is a string with a single ``{horizon}`` placeholder. At dataset
generation time, ``{horizon}`` is replaced with each phrasing in turn.

Templates here are starter examples — real scenarios live in
``configs/scenarios/*.yaml`` so they can be edited without touching code.
"""

from __future__ import annotations

EXAMPLE_TEMPLATES: dict[str, str] = {
    "savings_plan": "I want to save up for a new laptop. I plan to save over the next {horizon}.",
    "fitness_goal": "I'm starting a new workout routine and aim to see results within {horizon}.",
    "career_milestone": "My next career milestone is something I want to reach in {horizon}.",
    "vacation_planning": "I'm planning a vacation that's coming up in {horizon}.",
}
