"""Control condition (step_mode="target_control"): steps carry no time
information; a final "Time assignments:" reply gives every step's target
offset retrospectively."""

from src.conversation.parsing import (
    parse_step_horizon,
    parse_step_index,
    parse_time_assignments,
)
from src.core.time_value import TimeValue
from src.datasets.generator import render_prompt_text
from src.datasets.phrasings import get_phrasing
from src.datasets.tasks import get_task

ASSIGNMENT_REPLY = """Time assignments:
Step 1: 6 months
Step 2: 1 year
**Step 3**: 18 months
Step 4: 2 years"""


def test_parse_time_assignments():
    a = parse_time_assignments(ASSIGNMENT_REPLY)
    assert a == {
        1: ("6 months", 0.5),
        2: ("1 year", 1.0),
        3: ("18 months", 1.5),
        4: ("2 years", 2.0),
    }


def test_assignment_reply_is_not_a_step_turn():
    assert parse_step_index(ASSIGNMENT_REPLY) is None
    assert parse_step_horizon(ASSIGNMENT_REPLY) == (None, None)


def test_step_turn_is_not_assignments():
    text = "Step: 2\n\nDetails referring back to Step 1: the audit."
    assert parse_time_assignments(text) == {}


def test_control_prompt_renders():
    txt = render_prompt_text(
        get_task("marathon"),
        get_phrasing("available_time"),
        TimeValue.parse("6 months"),
        step_mode="target_control",
    )
    assert "Time assignments:" in txt
    assert "never mention times" in txt
    assert "Time target:" not in txt
    assert "Time horizon:" not in txt
