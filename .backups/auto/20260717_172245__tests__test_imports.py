"""Smoke test: every package and the dataset generator are importable + runnable."""

from __future__ import annotations

from pathlib import Path

import yaml


def test_package_imports() -> None:
    import src  # noqa: F401
    from src import binary_choice, common, datasets, geometry, inference  # noqa: F401
    from src.datasets import other, preference, prompt  # noqa: F401


def test_generate_example_dataset(tmp_path: Path) -> None:
    from src.datasets import generate_dataset
    from src.datasets.other.generate import DatasetConfig

    config_path = Path(__file__).parent.parent / "configs" / "scenarios" / "example.yaml"
    config = DatasetConfig.from_yaml(config_path)
    rows = generate_dataset(config)

    assert len(rows) > 0
    expected_cols = {"scenario", "horizon_months", "phrasing_group", "phrasing", "prompt", "split"}
    assert expected_cols <= set(rows[0].keys())

    splits = {r["split"] for r in rows}
    assert splits == {"train", "test"}, splits

    raw = yaml.safe_load(config_path.read_text())
    test_horizons = set(raw["test_phrasing_groups"])
    for row in rows:
        in_test = row["horizon_months"] in test_horizons
        assert (row["split"] == "test") == in_test
