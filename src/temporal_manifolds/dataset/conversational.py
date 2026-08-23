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
        "id": "objective_available_time",
        "body": """Objective: {task}
Available time: {value} {unit}

Write a plan optimized for the available time.""",
    },
    {
        "id": "objective_time_budget",
        "body": """Objective: {task}
Time budget: {value} {unit}

Provide a plan that fits this time budget.""",
    },
    {
        "id": "objective_deadline",
        "body": """Objective: {task}
Deadline: {value} {unit} from now

Give a plan appropriate for this deadline.""",
    },
]


templates = [
    {
        "id": framing["id"],
        "template": framing["body"],
        "prompt_framing": framing["id"],
    }
    for framing in prompt_framings
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
    "pack a bag with essentials for the day": _task(
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
    "train for a marathon": _task(
        {"days", "weeks", "months"},
        task_family="physical_training",
        difficulty="medium",
        domain="personal_lifestyle",
        planning_type="training",
        stakes="medium",
        agency="individual",
    ),
    "plan an international trip": _task(
        {"hours", "days", "weeks", "months"},
        task_family="travel_planning",
        difficulty="medium",
        domain="personal_lifestyle",
        planning_type="logistical",
        stakes="medium",
        agency="individual",
    ),
    "organize a wedding for a friend": _task(
        {"weeks", "months", "years"},
        task_family="event_planning",
        difficulty="high",
        domain="personal_lifestyle",
        planning_type="coordination",
        stakes="medium",
        agency="multi_agent",
    ),
    "plan a move to a new city": _task(
        {"days", "weeks", "months"},
        task_family="relocation",
        difficulty="medium",
        domain="personal_lifestyle",
        planning_type="logistical",
        stakes="medium",
        agency="individual",
    ),
    "build proficiency in a new language": _task(
        {"weeks", "months", "years"},
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
        stakes="low",
        agency="individual",
    ),
    "renovate an apartment": _task(
        {"days", "weeks", "months"},
        task_family="home_renovation",
        difficulty="medium",
        domain="personal_lifestyle",
        planning_type="logistical",
        stakes="medium",
        agency="multi_agent",
    ),
    "develop a long-term personal investment strategy": _task(
        {"months", "years", "decades"},
        task_family="investment_strategy",
        difficulty="high",
        domain="business",
        planning_type="strategic",
        stakes="medium",
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
    "coordinate a disaster relief effort": _task(
        {"days", "weeks", "months"},
        task_family="emergency_response",
        difficulty="high",
        domain="safety",
        planning_type="coordination",
        stakes="high",
        agency="multi_agent",
    ),
    "establish a self-sustaining human settlement on another planet": _task(
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
    "design and build a city resilient to natural disasters": _task(
        {"decades", "centuries"},
        task_family="resilient_infrastructure",
        difficulty="very_high",
        domain="infrastructure",
        planning_type="systems",
        stakes="high",
        agency="organization",
    ),
    "develop a long-term plan for safely containing hazardous materials": _task(
        {"months", "years", "decades"},
        task_family="hazard_containment",
        difficulty="high",
        domain="safety",
        planning_type="risk_management",
        stakes="high",
        agency="organization",
    ),
    "design an institution for long-term stability and effectiveness": _task(
        {"months", "years", "decades"},
        task_family="institution_design",
        difficulty="high",
        domain="civilization",
        planning_type="strategic",
        stakes="high",
        agency="organization",
    ),
    "develop a long-term strategy for humanity's survival": _task(
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
    # Additional domain controls. The knowledge-archive family below is retained
    # as an explicit within-family difficulty ladder.
    "design a communication plan for a company restructuring": _task(
        {"hours", "days", "weeks"},
        task_family="communication_plan",
        difficulty="high",
        domain="communication",
        planning_type="strategic",
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
    "preserve a small folder of important documents": _task(
        {"days", "weeks", "months"},
        task_family="knowledge_archive",
        difficulty="low",
        domain="knowledge_preservation",
        planning_type="procedural",
        stakes="medium",
        agency="individual",
    ),
    "develop a process to preserve an organization's records for future teams": _task(
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
    # Fully cover the difficulty x stakes grid without crossing tasks with
    # implausible horizons. These controls fill combinations that the natural
    # task collection above represents fewer than twice.
    "replace a lost identification card using a standard process": _task(
        {"hours", "days", "weeks"},
        task_family="administrative_recovery",
        difficulty="low",
        domain="administrative",
        planning_type="procedural",
        stakes="medium",
        agency="individual",
    ),
    "activate a building's emergency alarm after confirming a fire": _task(
        {"seconds", "minutes"},
        task_family="predefined_emergency_action",
        difficulty="low",
        domain="safety",
        planning_type="reactive",
        stakes="high",
        agency="individual",
    ),
    "follow a shutdown checklist for overheating laboratory equipment": _task(
        {"minutes", "hours"},
        task_family="predefined_emergency_action",
        difficulty="low",
        domain="safety",
        planning_type="procedural",
        stakes="high",
        agency="individual",
    ),
    "trigger a preauthorized shutdown of a runaway autonomous weapons system": _task(
        {"seconds", "minutes"},
        task_family="existential_safeguard_activation",
        difficulty="low",
        domain="civilization",
        planning_type="reactive",
        stakes="existential",
        agency="individual",
    ),
    "send a prewritten global warning for a confirmed extinction-level impact": _task(
        {"seconds", "minutes"},
        task_family="existential_safeguard_activation",
        difficulty="low",
        domain="civilization",
        planning_type="procedural",
        stakes="existential",
        agency="organization",
    ),
    "organize a community board-game tournament": _task(
        {"days", "weeks", "months"},
        task_family="recreational_event",
        difficulty="medium",
        domain="personal_lifestyle",
        planning_type="logistical",
        stakes="low",
        agency="multi_agent",
    ),
    "restore drinking-water distribution after a regional outage": _task(
        {"hours", "days", "weeks"},
        task_family="utility_recovery",
        difficulty="medium",
        domain="infrastructure",
        planning_type="recovery",
        stakes="high",
        agency="organization",
    ),
    "execute a prepared evacuation plan for a confirmed planet-wide impact": _task(
        {"hours", "days", "weeks"},
        task_family="prepared_existential_response",
        difficulty="very_high",
        domain="civilization",
        planning_type="coordination",
        stakes="existential",
        agency="multi_agent",
    ),
    "restore a failed component in a global biosecurity containment system": _task(
        {"hours", "days", "weeks"},
        task_family="prepared_existential_response",
        difficulty="high",
        domain="safety",
        planning_type="recovery",
        stakes="existential",
        agency="organization",
    ),
    "design a detailed ruleset for a fictional competitive league": _task(
        {"weeks", "months"},
        task_family="complex_recreational_design",
        difficulty="high",
        domain="creative",
        planning_type="systems",
        stakes="low",
        agency="individual",
    ),
    "build a high-fidelity simulation of an imaginary city's transit network": _task(
        {"days", "weeks", "months"},
        task_family="complex_recreational_design",
        difficulty="high",
        domain="software",
        planning_type="project",
        stakes="low",
        agency="individual",
    ),
    "coordinate international containment of a self-propagating engineered pathogen": _task(
        {"hours", "days", "weeks", "months"},
        task_family="existential_threat_response",
        difficulty="high",
        domain="safety",
        planning_type="coordination",
        stakes="existential",
        agency="multi_agent",
    ),
    "develop a global response strategy for a newly detected extinction-level asteroid": _task(
        {"days", "weeks", "months", "years"},
        task_family="existential_threat_response",
        difficulty="very_high",
        domain="civilization",
        planning_type="strategic",
        stakes="existential",
        agency="civilization",
    ),
    "apply a validated patch that prevents an imminent uncontrolled nuclear escalation": _task(
        {"hours", "days"},
        task_family="bounded_existential_intervention",
        difficulty="medium",
        domain="safety",
        planning_type="procedural",
        stakes="existential",
        agency="organization",
    ),
    "restart a planetary-defense tracking network using its tested recovery procedure": _task(
        {"hours", "days"},
        task_family="bounded_existential_intervention",
        difficulty="medium",
        domain="infrastructure",
        planning_type="recovery",
        stakes="existential",
        agency="organization",
    ),
    "coordinate deployment of a proven atmospheric intervention after a supervolcanic eruption": _task(
        {"days", "weeks", "months"},
        task_family="existential_threat_response",
        difficulty="high",
        domain="safety",
        planning_type="coordination",
        stakes="existential",
        agency="multi_agent",
    ),
    "simulate the complete history of a richly detailed fictional civilization": _task(
        {"years", "decades"},
        task_family="extreme_recreational_research",
        difficulty="very_high",
        domain="creative",
        planning_type="investigative",
        stakes="low",
        agency="organization",
    ),
    "design an exhaustive taxonomy for an open-ended generative art universe": _task(
        {"years", "decades"},
        task_family="extreme_recreational_research",
        difficulty="very_high",
        domain="creative",
        planning_type="systems",
        stakes="low",
        agency="organization",
    ),
    "coordinate worldwide preservation of public-domain cultural works": _task(
        {"years", "decades", "centuries"},
        task_family="global_knowledge_coordination",
        difficulty="very_high",
        domain="knowledge_preservation",
        planning_type="strategic",
        stakes="medium",
        agency="civilization",
    ),
    "design a globally interoperable scientific metadata system": _task(
        {"years", "decades"},
        task_family="global_knowledge_coordination",
        difficulty="very_high",
        domain="research_infrastructure",
        planning_type="systems",
        stakes="medium",
        agency="organization",
    ),
}


values = [1, 2, 3, 4, 5, 10, 20, 30, 50, 70, 100]
