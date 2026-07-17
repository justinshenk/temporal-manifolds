"""Materialise a parametric prompt dataset from a scenario config.

Config schema (YAML)::

    name: example
    seed: 0
    test_phrasing_groups: [1.0, 12.0]      # horizons (months) held out for test
    scenarios:                              # name -> template with {horizon}
      savings_plan: "I plan to save over the next {horizon}."
      fitness_goal: "I aim to see results within {horizon}."
    phrasing_groups: null                   # null => use DEFAULT_PHRASING_GROUPS

Output: a list of dicts (or a pandas.DataFrame via ``to_dataframe``) with
columns ``scenario, horizon_months, phrasing_group, phrasing, prompt, split``.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from src.datasets.other.phrasings import (
    DEFAULT_PHRASING_GROUPS,
    PhrasingGroup,
)


@dataclass
class DatasetConfig:
    name: str
    scenarios: dict[str, str]
    seed: int = 0
    test_phrasing_groups: list[float] = field(default_factory=list)
    phrasing_groups: tuple[PhrasingGroup, ...] = DEFAULT_PHRASING_GROUPS

    @classmethod
    def from_yaml(cls, path: str | Path) -> "DatasetConfig":
        with open(path) as f:
            raw = yaml.safe_load(f)
        groups = raw.get("phrasing_groups")
        if groups is None:
            phrasing_groups = DEFAULT_PHRASING_GROUPS
        else:
            phrasing_groups = tuple(
                PhrasingGroup(float(g["horizon_months"]), tuple(g["phrasings"]))
                for g in groups
            )
        return cls(
            name=raw["name"],
            scenarios=raw["scenarios"],
            seed=raw.get("seed", 0),
            test_phrasing_groups=[float(h) for h in raw.get("test_phrasing_groups", [])],
            phrasing_groups=phrasing_groups,
        )


def generate_dataset(config: DatasetConfig) -> list[dict[str, Any]]:
    """Materialise the dataset described by ``config`` as a list of row dicts."""
    rng = random.Random(config.seed)
    test_horizons = set(config.test_phrasing_groups)
    rows: list[dict[str, Any]] = []
    for scenario_name, template in config.scenarios.items():
        if "{horizon}" not in template:
            raise ValueError(
                f"Scenario {scenario_name!r} template missing '{{horizon}}' placeholder"
            )
        for group in config.phrasing_groups:
            split = "test" if group.horizon_months in test_horizons else "train"
            for phrasing in group.phrasings:
                rows.append(
                    {
                        "scenario": scenario_name,
                        "horizon_months": group.horizon_months,
                        "phrasing_group": group.horizon_months,
                        "phrasing": phrasing,
                        "prompt": template.format(horizon=phrasing),
                        "split": split,
                    }
                )
    rng.shuffle(rows)
    return rows
