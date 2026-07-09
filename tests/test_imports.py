"""Smoke test: every package and the dataset generator are importable + runnable."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import yaml


def test_package_imports() -> None:
    import temporal_manifolds  # noqa: F401
    from temporal_manifolds import dataset, activations, geometry, models, evaluation, viz  # noqa: F401


def test_eap_ig_plots_imports_with_colab_inline_backend() -> None:
    env = os.environ.copy()
    env["MPLBACKEND"] = "module://matplotlib_inline.backend_inline"

    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "import os; import temporal_manifolds.viz.eap_ig_plots; print(os.environ['MPLBACKEND'])",
        ],
        check=True,
        capture_output=True,
        env=env,
        text=True,
    )

    assert result.stdout.strip() == "Agg"


def test_generate_example_dataset(tmp_path: Path) -> None:
    from temporal_manifolds.dataset import generate_dataset
    from temporal_manifolds.dataset.generate import DatasetConfig

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
