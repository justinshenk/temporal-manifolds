"""Tests for residual-stream regression helpers."""

from __future__ import annotations

import importlib.util
import json
import math
from pathlib import Path

import numpy as np
import pytest
import torch

pytest.importorskip("sklearn")

SCRIPT_PATH = Path(__file__).parents[1] / "scripts" / "fit_residual_stream_regressions.py"
SCRIPT_SPEC = importlib.util.spec_from_file_location("fit_residual_stream_regressions", SCRIPT_PATH)
assert SCRIPT_SPEC is not None and SCRIPT_SPEC.loader is not None
SCRIPT_MODULE = importlib.util.module_from_spec(SCRIPT_SPEC)
SCRIPT_SPEC.loader.exec_module(SCRIPT_MODULE)


def test_horizon_months_uses_canonical_base_value_and_unit() -> None:
    record = {
        "prompt_metadata": {
            "base_value": 2,
            "base_unit": "hours",
            "value": 120,
            "unit": "minutes",
        }
    }
    assert SCRIPT_MODULE.horizon_months(record) == pytest.approx(2.0 / (30.4375 * 24.0))


@pytest.mark.parametrize(
    ("unit", "years"),
    [("decades", 10), ("centuries", 100), ("millennia", 1000)],
)
def test_horizon_months_supports_long_units(unit: str, years: int) -> None:
    record = {"prompt_metadata": {"base_value": 2, "base_unit": unit}}
    assert SCRIPT_MODULE.horizon_months(record) == 2 * years * 12


def test_load_log_targets_uses_base_10_log_months(tmp_path: Path) -> None:
    path = tmp_path / "completions.jsonl"
    records = [
        {"prompt_metadata": {"base_value": 1, "base_unit": "months"}},
        {"prompt_metadata": {"base_value": 100, "base_unit": "months"}},
    ]
    path.write_text("".join(json.dumps(record) + "\n" for record in records), encoding="utf-8")

    targets = SCRIPT_MODULE.load_log_targets(path)

    assert targets.tolist() == pytest.approx([0.0, 2.0])


def test_load_position_features_orders_samples_and_concatenates_layers(tmp_path: Path) -> None:
    payload = {
        "cached_position_indices": [0, 1, 2],
        "sample_indices": [4, 2],
        "residual_stream_activations": {
            "layer_out/1": torch.tensor(
                [[[10.0], [11.0], [12.0]], [[20.0], [21.0], [22.0]]]
            ),
            "layer_out/0": torch.tensor(
                [[[1.0], [2.0], [3.0]], [[4.0], [5.0], [6.0]]]
            ),
        },
    }
    path = tmp_path / "chunk.pt"
    torch.save(payload, path)

    features, indices = SCRIPT_MODULE.load_position_features([path], 1)

    assert indices.tolist() == [2, 4]
    assert features.tolist() == [[5.0, 21.0], [2.0, 11.0]]


def test_metrics_distinguish_squared_correlation_from_R2() -> None:
    metrics = SCRIPT_MODULE.regression_metrics(
        np.asarray([0.0, 1.0, 2.0]), np.asarray([1.0, 3.0, 5.0])
    )
    assert metrics["r2"] == pytest.approx(1.0)
    assert metrics["R2"] < 0.0
    assert metrics["rmse"] == pytest.approx(math.sqrt(14.0 / 3.0))
