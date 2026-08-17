"""Planning prompts that state the task and nothing about time.

Unlike :mod:`temporal_manifolds.dataset.conversational` and
:mod:`temporal_manifolds.dataset.plain_english`, these prompts never mention a
horizon, a deadline, or an amount of available time.  They are the
time-free control condition: activations cached from this dataset show what
the residual stream encodes for a planning task when no horizon is supplied,
so horizon-dependent structure found in the other datasets can be compared
against a baseline that has no horizon to encode.

The task definitions are copied from the conversational dataset so records can
be matched to their time-constrained counterparts by task, task family,
difficulty, domain, planning style, stakes, and agency.

Because no template interpolates ``{value}`` or ``{unit}``, the time grid is a
single placeholder value.  Iterating a wider grid would emit byte-identical
duplicate prompts rather than new conditions.
"""

from __future__ import annotations

try:
    from . import conversational
except ImportError:  # Support running dataset/generate.py directly.
    import conversational  # type: ignore


# The templates mirror the plain-English voices so that wording, perspective,
# and request style stay comparable across the two datasets; only the time
# clause is absent here.
templates = [
    {
        "id": "direct_help",
        "template": "Could you help me plan how to {task}? Please keep the advice realistic.",
        "perspective": "first_person",
        "request_style": "direct",
    },
    {
        "id": "practical_walkthrough",
        "template": "I need to {task}. Walk me through a practical way to pull it off.",
        "perspective": "first_person",
        "request_style": "walkthrough",
    },
    {
        "id": "hypothetical_advice",
        "template": "Imagine you had to {task}. How would you approach it?",
        "perspective": "second_person",
        "request_style": "hypothetical",
    },
    {
        "id": "starting_now",
        "template": (
            "If I started now and wanted to {task}, what would a sensible plan look like?"
        ),
        "perspective": "first_person",
        "request_style": "reflective",
    },
    {
        "id": "best_way",
        "template": "What's the best way for me to {task}?",
        "perspective": "first_person",
        "request_style": "optimization",
    },
    {
        "id": "what_to_focus_on",
        "template": "I'd like to {task}. What should I focus on?",
        "perspective": "first_person",
        "request_style": "prioritization",
    },
    {
        "id": "think_it_through",
        "template": (
            "Help me think through how to {task}. Where should I begin and what "
            "should happen after that?"
        ),
        "perspective": "first_person",
        "request_style": "collaborative",
    },
    {
        "id": "external_request",
        "template": "Someone has asked me to {task}. Can you suggest a realistic approach?",
        "perspective": "first_person",
        "request_style": "advice",
    },
    {
        "id": "break_it_down",
        "template": "My goal is to {task}. How would you break that down?",
        "perspective": "first_person",
        "request_style": "decomposition",
    },
    {
        "id": "recommendation",
        "template": "I'm trying to {task}. What would you recommend?",
        "perspective": "first_person",
        "request_style": "advice",
    },
    {
        "id": "how_to_go_about_it",
        "template": "Say I wanted to {task}. How could I go about it?",
        "perspective": "first_person",
        "request_style": "conversational",
    },
    {
        "id": "advice_request",
        "template": (
            "I could use some advice: I want to {task}. What's a sensible way forward?"
        ),
        "perspective": "first_person",
        "request_style": "advice",
    },
]

# Every template is already time-free, so removing time constraints must leave
# the prompt untouched rather than fall back to line-stripping.
for template in templates:
    template["unconstrained_template"] = template["template"]


def _copy_task_config(config: dict[str, object]) -> dict[str, object]:
    """Copy a task config without sharing its mutable unit set."""
    copied = dict(config)
    copied["units"] = set(config["units"])  # type: ignore[arg-type]
    return copied


tasks = {task: _copy_task_config(config) for task, config in conversational.tasks.items()}

# Marks the dataset as having no horizon to vary, so generation skips the time
# grid instead of emitting one identical duplicate prompt per time value.
time_free = True

# A single placeholder horizon that no template renders, kept so the module
# satisfies the same interface as the time-constrained datasets.
values = [1]
