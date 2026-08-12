"""Natural-language planning prompts based on the conversational task set.

Unlike :mod:`temporal_manifolds.dataset.conversational`, these prompts read like
ordinary requests rather than forms with labelled fields or prescribed output
formats.  The task definitions and time grid are copied from the conversational
dataset so that results from the two datasets can be compared directly.
"""

from __future__ import annotations

try:
    from . import conversational
except ImportError:  # Support running dataset/generate.py directly.
    import conversational  # type: ignore


# The different voices make wording less of a confound while keeping every
# prompt explicit about both the task and the amount of time available.
templates = [
    {
        "id": "direct_help",
        "template": (
            "Could you help me plan how to {task}? I have {value} {unit} available, "
            "so please keep the advice realistic for that amount of time."
        ),
        "perspective": "first_person",
        "time_framing": "available_time",
        "request_style": "direct",
    },
    {
        "id": "practical_walkthrough",
        "template": (
            "I need to {task} within {value} {unit}. Walk me through a practical "
            "way to pull it off."
        ),
        "perspective": "first_person",
        "time_framing": "time_limit",
        "request_style": "walkthrough",
    },
    {
        "id": "hypothetical_advice",
        "template": (
            "Imagine you had {value} {unit} to {task}. How would you approach it?"
        ),
        "perspective": "second_person",
        "time_framing": "time_budget",
        "request_style": "hypothetical",
    },
    {
        "id": "starting_now",
        "template": (
            "If I started now and wanted to {task} by {value} {unit} from now, what "
            "would a sensible plan look like?"
        ),
        "perspective": "first_person",
        "time_framing": "deadline",
        "request_style": "reflective",
    },
    {
        "id": "use_the_time",
        "template": (
            "What's the best way for me to use the next {value} {unit} if my goal is "
            "to {task}?"
        ),
        "perspective": "first_person",
        "time_framing": "time_window",
        "request_style": "optimization",
    },
    {
        "id": "limited_availability",
        "template": (
            "I'd like to {task}, but I can only devote {value} {unit} to it. What "
            "should I focus on?"
        ),
        "perspective": "first_person",
        "time_framing": "limited_availability",
        "request_style": "prioritization",
    },
    {
        "id": "think_it_through",
        "template": (
            "Help me think through how to {task}. Given a window of {value} {unit}, "
            "where should I begin and what should happen after that?"
        ),
        "perspective": "first_person",
        "time_framing": "time_window",
        "request_style": "collaborative",
    },
    {
        "id": "external_deadline",
        "template": (
            "Someone has asked me to {task}, and it needs to be handled in {value} "
            "{unit}. Can you suggest a realistic approach?"
        ),
        "perspective": "first_person",
        "time_framing": "deadline",
        "request_style": "advice",
    },
    {
        "id": "break_it_down",
        "template": (
            "My goal is to {task}, with {value} {unit} to make it happen. How would "
            "you break that down?"
        ),
        "perspective": "first_person",
        "time_framing": "time_budget",
        "request_style": "decomposition",
    },
    {
        "id": "recommendation",
        "template": (
            "I'm trying to {task} over the next {value} {unit}. What would you "
            "recommend?"
        ),
        "perspective": "first_person",
        "time_framing": "time_window",
        "request_style": "advice",
    },
    {
        "id": "set_aside_time",
        "template": (
            "Say I set aside {value} {unit} to {task}. How could I make the most of "
            "that time?"
        ),
        "perspective": "first_person",
        "time_framing": "time_budget",
        "request_style": "conversational",
    },
    {
        "id": "advice_before_deadline",
        "template": (
            "I could use some advice: I want to {task} and have {value} {unit} before "
            "the deadline. What's a sensible way forward?"
        ),
        "perspective": "first_person",
        "time_framing": "deadline",
        "request_style": "advice",
    },
]

_UNCONSTRAINED_TEMPLATES = {
    "direct_help": "Could you help me plan how to {task}? Please keep the advice realistic.",
    "practical_walkthrough": "I need to {task}. Walk me through a practical way to pull it off.",
    "hypothetical_advice": "Imagine you had to {task}. How would you approach it?",
    "starting_now": "If I started now and wanted to {task}, what would a sensible plan look like?",
    "use_the_time": "What's the best way for me to {task}?",
    "limited_availability": "I'd like to {task}. What should I focus on?",
    "think_it_through": (
        "Help me think through how to {task}. Where should I begin and what should happen after that?"
    ),
    "external_deadline": (
        "Someone has asked me to {task}. Can you suggest a realistic approach?"
    ),
    "break_it_down": "My goal is to {task}. How would you break that down?",
    "recommendation": "I'm trying to {task}. What would you recommend?",
    "set_aside_time": "Say I wanted to {task}. How could I go about it?",
    "advice_before_deadline": (
        "I could use some advice: I want to {task}. What's a sensible way forward?"
    ),
}

for template in templates:
    template["unconstrained_template"] = _UNCONSTRAINED_TEMPLATES[template["id"]]


def _copy_task_config(config: dict[str, object]) -> dict[str, object]:
    """Copy a task config without sharing its mutable unit set."""
    copied = dict(config)
    copied["units"] = set(config["units"])  # type: ignore[arg-type]
    return copied


tasks = {
    task: _copy_task_config(config)
    for task, config in conversational.tasks.items()
}

values = list(conversational.values)
