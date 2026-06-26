"""Abstract quantity-distribution prompts over time constraints."""

templates = [
    {  # replace with something informative
        "id": "quantity_time_constraint",
        "template": """Quantity type: {task}
Quantity amount: {quantity} 
Time constraint: {value} {unit}

Distribute the quantity across the available time.

Output format:
Allocation rule: <one sentence>
Schedule:
1. ...
2. ...
3. ...""",
    },
    {
        "id": "resource_budget_window",
        "template": """Resource to distribute: {task}
Amount to distribute: {quantity}
Time window: {value} {unit}

Allocate this resource over the full time window.

Output format:
Allocation rule: <one sentence>
Schedule:
1. ...
2. ...
3. ...""",
    },
    {
        "id": "abstract_quantity_deadline",
        "template": """Abstract quantity: {task}
Amount: {quantity}
Deadline: {value} {unit} from now

Create a distribution plan that fits the deadline.

Output format:
Allocation rule: <one sentence>
Schedule:
1. ...
2. ...
3. ...""",
    },
]

time_units = {
    "seconds",
    "minutes",
    "hours",
    "days",
    "weeks",
    "months",
    "years",
    "decades",
    "centuries",
    "millennia",
}

tasks = {
    "attention": time_units,
    "explanation tokens": time_units,
    "compute": time_units,
    "storage": time_units,
    "risk": time_units,
    "budget": time_units,
    "effort": time_units,
    "maintenance": time_units,
}

quantities = [1, 2, 3, 4, 5, 10, 50, 100, 500, 1000]
values = [1, 2, 3, 4, 5, 10, 50, 100]
