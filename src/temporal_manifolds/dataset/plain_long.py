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
            "so please keep the advice realistic, practical, and grounded for that "
            "amount of time. I want a clear, actionable plan I can actually follow, "
            "not just vague or generic suggestions, so feel free to be specific about "
            "what to do first, next, and last."
        ),
        "perspective": "first_person",
        "time_framing": "available_time",
        "request_style": "direct",
    },
    {
        "id": "practical_walkthrough",
        "template": (
            "I need to {task} within {value} {unit}. Walk me through a practical, "
            "step-by-step way to pull it off, including any preparation, key "
            "milestones, and potential pitfalls I should watch out for along the way. "
            "Assume I'm starting from scratch and want a concrete, no-nonsense plan "
            "rather than abstract advice."
        ),
        "perspective": "first_person",
        "time_framing": "time_limit",
        "request_style": "walkthrough",
    },
    {
        "id": "hypothetical_advice",
        "template": (
            "Imagine you had {value} {unit} to {task}. How would you approach it? "
            "What order would you tackle things in, what would you prioritize, and "
            "what would you deliberately skip or postpone in order to make the most "
            "of the time you've got?"
        ),
        "perspective": "second_person",
        "time_framing": "time_budget",
        "request_style": "hypothetical",
    },
    {
        "id": "starting_now",
        "template": (
            "If I started now and wanted to {task} by {value} {unit} from now, what "
            "would a sensible plan look like? I'd like a rough timeline with clear "
            "checkpoints, so I can tell early on whether I'm on track or falling "
            "behind and need to adjust my approach."
        ),
        "perspective": "first_person",
        "time_framing": "deadline",
        "request_style": "reflective",
    },
    {
        "id": "use_the_time",
        "template": (
            "What's the best way for me to use the next {value} {unit} if my goal is "
            "to {task}? I want to make the most efficient, effective use of that "
            "window, avoiding wasted effort, so please suggest how to sequence tasks "
            "and where to focus my energy."
        ),
        "perspective": "first_person",
        "time_framing": "time_window",
        "request_style": "optimization",
    },
    {
        "id": "limited_availability",
        "template": (
            "I'd like to {task}, but I can only devote {value} {unit} to it. What "
            "should I focus on given that constraint? I'd rather do a few important "
            "things well than spread myself thin, so help me identify the highest-"
            "impact priorities and what can safely be left out or deferred."
        ),
        "perspective": "first_person",
        "time_framing": "limited_availability",
        "request_style": "prioritization",
    },
    {
        "id": "think_it_through",
        "template": (
            "Help me think through how to {task}. Given a window of {value} {unit}, "
            "where should I begin, what should happen after that, and how should the "
            "later stages build on the earlier ones? I'd appreciate a collaborative "
            "back-and-forth style of thinking rather than just a final answer."
        ),
        "perspective": "first_person",
        "time_framing": "time_window",
        "request_style": "collaborative",
    },
    {
        "id": "external_deadline",
        "template": (
            "Someone has asked me to {task}, and it needs to be handled in {value} "
            "{unit}. Can you suggest a realistic approach, including how to pace "
            "myself, what to prioritize under this externally imposed deadline, and "
            "how to avoid last-minute scrambling as the time runs out?"
        ),
        "perspective": "first_person",
        "time_framing": "deadline",
        "request_style": "advice",
    },
    {
        "id": "break_it_down",
        "template": (
            "My goal is to {task}, with {value} {unit} to make it happen. How would "
            "you break that down into smaller, manageable steps or phases? I want to "
            "understand how the pieces fit together and roughly how much time each "
            "part should reasonably take."
        ),
        "perspective": "first_person",
        "time_framing": "time_budget",
        "request_style": "decomposition",
    },
    {
        "id": "recommendation",
        "template": (
            "I'm trying to {task} over the next {value} {unit}. What would you "
            "recommend as the smartest way to approach this, given the time I have "
            "to work with? Feel free to suggest a general strategy as well as any "
            "specific tips that would make a real difference."
        ),
        "perspective": "first_person",
        "time_framing": "time_window",
        "request_style": "advice",
    },
    {
        "id": "set_aside_time",
        "template": (
            "Say I set aside {value} {unit} to {task}. How could I make the most of "
            "that time, structure it well, and avoid common traps like "
            "procrastination, distraction, or spending too long on the wrong things? "
            "I'm looking for practical, down-to-earth suggestions."
        ),
        "perspective": "first_person",
        "time_framing": "time_budget",
        "request_style": "conversational",
    },
    {
        "id": "advice_before_deadline",
        "template": (
            "I could use some advice: I want to {task} and have {value} {unit} before "
            "the deadline. What's a sensible way forward that balances thoroughness "
            "with the limited time available, and how should I decide what to cut if "
            "I end up running short on time?"
        ),
        "perspective": "first_person",
        "time_framing": "deadline",
        "request_style": "advice",
    },
]


def _copy_task_config(config: dict[str, object]) -> dict[str, object]:
    """Copy a task config without sharing its mutable unit set."""
    copied = dict(config)
    copied["units"] = set(config["units"])  # type: ignore[arg-type]
    return copied


tasks = {task: _copy_task_config(config) for task, config in conversational.tasks.items()}

values = list(conversational.values)
