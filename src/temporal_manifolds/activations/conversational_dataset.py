"""Conversational planning prompts for probing LLM planning-horizon representations.

The task metadata is intended for downstream controls: analyses can separate
time-horizon effects from task family, difficulty, domain, planning style,
stakes, and whether the plan is mostly individual or requires coordination.

Several task families are deliberately crossed with difficulty and horizon:
records can be compared within the same task family across horizons, or within
the same horizon across difficulty levels.
"""


def _task(
    units: set[str],
    *,
    task_family: str,
    difficulty: str,
    domain: str,
    planning_type: str,
    stakes: str,
    agency: str,
) -> dict[str, object]:
    return {
        "units": units,
        "task_family": task_family,
        "difficulty": difficulty,
        "domain": domain,
        "complexity": difficulty,
        "planning_type": planning_type,
        "stakes": stakes,
        "agency": agency,
    }


prompt_framings = [
    {
        "id": "task_available_time",
        "body": """Task: {task}
Available time: {value} {unit}

Write a plan optimized for the available time.""",
    },
    {
        "id": "task_time_budget",
        "body": """Task: {task}
Time budget: {value} {unit}

Provide a plan that fits this time budget.""",
    },
    {
        "id": "task_deadline",
        "body": """Task: {task}
Deadline: {value} {unit} from now

Give a plan appropriate for this deadline.""",
    },
    {
        "id": "goal_available_time",
        "body": """Goal: {task}
Available time: {value} {unit}

Write a plan optimized for the available time.""",
    },
    {
        "id": "goal_time_budget",
        "body": """Goal: {task}
Time budget: {value} {unit}

Provide a plan that fits this time budget.""",
    },
    {
        "id": "goal_deadline",
        "body": """Goal: {task}
Deadline: {value} {unit} from now

Give a plan appropriate for this deadline.""",
    },
    {
        "id": "object_available_time",
        "body": """Object: {task}
Available time: {value} {unit}

Write a plan optimized for the available time.""",
    },
    {
        "id": "object_time_budget",
        "body": """Object: {task}
Time budget: {value} {unit}

Provide a plan that fits this time budget.""",
    },
    {
        "id": "object_deadline",
        "body": """Object: {task}
Deadline: {value} {unit} from now

Give a plan appropriate for this deadline.""",
    },
]


output_formats = [
    {
        "id": "strategy_steps",
        "instructions": """Output format:
Strategy: <one sentence>
Steps:
1. ...
2. ...
3. ...""",
    },
    {
        "id": "summary_checklist",
        "instructions": """Output format:
Summary: <one sentence>
Checklist:
- ...
- ...
- ...""",
    },
    {
        "id": "approach_actions",
        "instructions": """Output format:
Approach: <one sentence>
Actions:
1. ...
2. ...
3. ...""",
    },
]


templates = [
    {
        "id": f"{framing['id']}__{output_format['id']}",
        "template": f"{framing['body']}\n\n{output_format['instructions']}",
        "prompt_framing": framing["id"],
        "output_format": output_format["id"],
    }
    for framing in prompt_framings
    for output_format in output_formats
]

tasks = {
    # Fast baseline tasks
    "answer a yes-or-no question": _task(
        {"seconds"},
        task_family="quick_decision",
        difficulty="low",
        domain="communication",
        planning_type="reactive",
        stakes="low",
        agency="individual",
    ),
    "press a button when a light turns green": _task(
        {"seconds"},
        task_family="quick_action",
        difficulty="low",
        domain="personal_lifestyle",
        planning_type="reactive",
        stakes="low",
        agency="individual",
    ),
    "copy a short code from one screen to another": _task(
        {"seconds", "minutes"},
        task_family="copy_information",
        difficulty="low",
        domain="administrative",
        planning_type="procedural",
        stakes="low",
        agency="individual",
    ),
    "choose between two lunch options": _task(
        {"seconds", "minutes"},
        task_family="personal_choice",
        difficulty="low",
        domain="personal_lifestyle",
        planning_type="decision",
        stakes="low",
        agency="individual",
    ),
    "make a cup of tea": _task(
        {"minutes"},
        task_family="prepare_drink",
        difficulty="low",
        domain="personal_lifestyle",
        planning_type="procedural",
        stakes="low",
        agency="individual",
    ),
    "pack a small bag for the day": _task(
        {"minutes"},
        task_family="personal_logistics",
        difficulty="low",
        domain="personal_lifestyle",
        planning_type="procedural",
        stakes="low",
        agency="individual",
    ),
    "tidy a desk before a meeting": _task(
        {"minutes"},
        task_family="workspace_organization",
        difficulty="low",
        domain="administrative",
        planning_type="procedural",
        stakes="low",
        agency="individual",
    ),
    "prepare for a marathon": _task(
        {"weeks", "months", "years"},
        task_family="physical_training",
        difficulty="medium",
        domain="personal_lifestyle",
        planning_type="training",
        stakes="medium",
        agency="individual",
    ),
    "plan an international trip": _task(
        {"hours", "days"},
        task_family="travel_planning",
        difficulty="medium",
        domain="personal_lifestyle",
        planning_type="logistical",
        stakes="medium",
        agency="individual",
    ),
    "organize my friend's wedding": _task(
        {"days", "weeks", "months"},
        task_family="event_planning",
        difficulty="high",
        domain="personal_lifestyle",
        planning_type="coordination",
        stakes="medium",
        agency="multi_agent",
    ),
    "move to a new city": _task(
        {"hours", "days", "weeks"},
        task_family="relocation",
        difficulty="medium",
        domain="personal_lifestyle",
        planning_type="logistical",
        stakes="medium",
        agency="individual",
    ),
    "learn a new language": _task(
        {"days", "weeks", "months"},
        task_family="skill_learning",
        difficulty="medium",
        domain="personal_lifestyle",
        planning_type="learning",
        stakes="medium",
        agency="individual",
    ),
    "write a short story": _task(
        {"hours", "days"},
        task_family="creative_writing",
        difficulty="medium",
        domain="creative",
        planning_type="creative",
        stakes="low",
        agency="individual",
    ),
    "write a novel": _task(
        {"weeks", "months", "years"},
        task_family="creative_writing",
        difficulty="high",
        domain="creative",
        planning_type="creative",
        stakes="medium",
        agency="individual",
    ),
    "renovate an apartment": _task(
        {"hours", "days"},
        task_family="home_renovation",
        difficulty="medium",
        domain="personal_lifestyle",
        planning_type="logistical",
        stakes="medium",
        agency="multi_agent",
    ),
    "help me make an investment strategy for long-term wealth creation": _task(
        {"months", "years", "decades"},
        task_family="investment_strategy",
        difficulty="high",
        domain="business",
        planning_type="strategic",
        stakes="high",
        agency="individual",
    ),
    "expand a local business to new markets": _task(
        {"months", "years", "decades"},
        task_family="business_expansion",
        difficulty="high",
        domain="business",
        planning_type="strategic",
        stakes="high",
        agency="organization",
    ),
    "coordinate disaster relief efforts": _task(
        {"days", "weeks", "months"},
        task_family="emergency_response",
        difficulty="high",
        domain="safety",
        planning_type="coordination",
        stakes="high",
        agency="multi_agent",
    ),
    "establish a self-sustaining civilization on a new planet": _task(
        {"decades", "centuries", "millennia"},
        task_family="civilization_survival",
        difficulty="very_high",
        domain="civilization",
        planning_type="strategic",
        stakes="existential",
        agency="civilization",
    ),
    "restore a degraded ecosystem to long-term stability": _task(
        {"years", "decades", "centuries"},
        task_family="ecosystem_restoration",
        difficulty="very_high",
        domain="safety",
        planning_type="systems",
        stakes="high",
        agency="multi_agent",
    ),
    "build a city designed to survive natural disasters": _task(
        {"decades", "centuries"},
        task_family="resilient_infrastructure",
        difficulty="very_high",
        domain="infrastructure",
        planning_type="systems",
        stakes="high",
        agency="organization",
    ),
    "ensure long-term safe containment of hazardous materials": _task(
        {"months", "years", "decades"},
        task_family="hazard_containment",
        difficulty="high",
        domain="safety",
        planning_type="risk_management",
        stakes="high",
        agency="organization",
    ),
    "create an institution that remains stable and effective": _task(
        {"months", "years", "decades"},
        task_family="institution_design",
        difficulty="high",
        domain="civilization",
        planning_type="strategic",
        stakes="high",
        agency="organization",
    ),
    "plan the long-term survival strategy of humanity": _task(
        {"centuries", "millennia"},
        task_family="civilization_survival",
        difficulty="very_high",
        domain="civilization",
        planning_type="strategic",
        stakes="existential",
        agency="civilization",
    ),
    "design infrastructure resilient to climate change": _task(
        {"decades", "centuries"},
        task_family="resilient_infrastructure",
        difficulty="very_high",
        domain="infrastructure",
        planning_type="systems",
        stakes="high",
        agency="organization",
    ),
    # Crossed task-family controls: compare within a family across difficulty
    # and compare within a difficulty level across available time units.
    "write a one-sentence email reply": _task(
        {"minutes", "hours", "days"},
        task_family="communication_plan",
        difficulty="low",
        domain="communication",
        planning_type="procedural",
        stakes="low",
        agency="individual",
    ),
    "write a sensitive email to resolve a team conflict": _task(
        {"minutes", "hours", "days"},
        task_family="communication_plan",
        difficulty="medium",
        domain="communication",
        planning_type="coordination",
        stakes="medium",
        agency="individual",
    ),
    "design a communication plan for a company restructuring": _task(
        {"hours", "days", "weeks"},
        task_family="communication_plan",
        difficulty="high",
        domain="communication",
        planning_type="strategic",
        stakes="high",
        agency="organization",
    ),
    "organize files on a desk": _task(
        {"minutes", "hours", "days"},
        task_family="organization_project",
        difficulty="low",
        domain="administrative",
        planning_type="procedural",
        stakes="low",
        agency="individual",
    ),
    "organize a shared drive for a small team": _task(
        {"hours", "days", "weeks"},
        task_family="organization_project",
        difficulty="medium",
        domain="administrative",
        planning_type="logistical",
        stakes="medium",
        agency="multi_agent",
    ),
    "organize a company-wide knowledge management system": _task(
        {"days", "weeks", "months"},
        task_family="organization_project",
        difficulty="high",
        domain="knowledge_preservation",
        planning_type="systems",
        stakes="high",
        agency="organization",
    ),
    "create a single-page personal website": _task(
        {"hours", "days", "weeks"},
        task_family="software_project",
        difficulty="low",
        domain="software",
        planning_type="project",
        stakes="low",
        agency="individual",
    ),
    "create an e-commerce website": _task(
        {"hours", "days", "weeks"},
        task_family="software_project",
        difficulty="medium",
        domain="software",
        planning_type="project",
        stakes="medium",
        agency="individual",
    ),
    "create a scalable marketplace platform": _task(
        {"days", "weeks", "months"},
        task_family="software_project",
        difficulty="high",
        domain="software",
        planning_type="project",
        stakes="high",
        agency="organization",
    ),
    "preserve a small folder of important documents": _task(
        {"days", "weeks", "months"},
        task_family="knowledge_archive",
        difficulty="low",
        domain="knowledge_preservation",
        planning_type="procedural",
        stakes="medium",
        agency="individual",
    ),
    "preserve an organization's records for future teams": _task(
        {"months", "years", "decades"},
        task_family="knowledge_archive",
        difficulty="medium",
        domain="knowledge_preservation",
        planning_type="systems",
        stakes="high",
        agency="organization",
    ),
    "design and preserve a knowledge archive for future civilizations": _task(
        {"years", "decades", "centuries"},
        task_family="knowledge_archive",
        difficulty="high",
        domain="knowledge_preservation",
        planning_type="strategic",
        stakes="high",
        agency="civilization",
    ),
}


values = [1, 2, 3, 4, 5, 10, 50, 100]
