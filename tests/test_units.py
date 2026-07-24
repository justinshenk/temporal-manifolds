"""Unit tests: parsing, dataset generation, uids, depths, PCA."""

from __future__ import annotations

import numpy as np

from src.analysis.pca import fit_pca
from src.conversation import parsing
from src.conversation.records import make_sample_uid
from src.core.time_value import TimeValue
from src.datasets.generator import build_dataset, horizons_from_specs
from src.datasets.tasks import CORE_TASKS
from src.engine.depths import layer_at_depth, layers_for_depths


def test_parse_step_header():
    text = "Step: 3\nTime horizon: 2 months\n\nDo the thing."
    assert parsing.parse_step_index(text) == 3
    raw, years = parsing.parse_step_horizon(text)
    assert raw == "2 months"
    assert abs(years - 2 / 12) < 1e-9


def test_parse_horizon_ranges_and_noise():
    assert parsing.parse_step_horizon("Time horizon: ~3 weeks")[1] is not None
    _, years = parsing.parse_step_horizon("Time horizon: 1-2 years")
    assert abs(years - 1.5) < 1e-9
    assert parsing.parse_step_horizon("no header here") == (None, None)


def test_plan_completed_and_overview_count():
    assert parsing.is_plan_completed("Plan Completed")
    assert parsing.is_plan_completed("  plan completed.")
    overview = "Goal: X\n1. First (2 days)\n2. Second (1 week)\n3. Third (3 weeks)"
    assert parsing.count_overview_steps(overview) == 3


def test_strip_think_block():
    text = "<think>\n\n</think>\n\nHello"
    assert parsing.strip_think_block(text, "<think>", "</think>") == "Hello"
    text2 = "<think>reasoning</think>\nanswer"
    assert parsing.strip_think_block(text2, "<think>", "</think>") == "answer"
    assert parsing.strip_think_block("plain", "<think>", "</think>") == "plain"


def test_build_dataset_cross_product():
    horizons = horizons_from_specs(["3 days", [2, "weeks"]])
    ds = build_dataset("t", tasks=CORE_TASKS[:2], horizons=horizons)
    assert len(ds.prompts) == 2 * 2 * 1
    p = ds.prompts[0]
    assert "{target_time_horizon}" not in p.text
    assert "Continue." in p.text and "Plan Completed" in p.text
    assert str(p.target_horizon) in p.text
    ids = [q.prompt_id for q in ds.prompts]
    assert len(set(ids)) == len(ids)


def test_sample_uid_deterministic():
    a = make_sample_uid("p1", "m1", 1.0, {"x": 1}, 0)
    b = make_sample_uid("p1", "m1", 1.0, {"x": 1}, 0)
    c = make_sample_uid("p1", "m1", 2.0, {"x": 1}, 0)
    assert a == b != c


def test_depths():
    assert layer_at_depth(0.4, 36) == 13
    assert layer_at_depth(0.6, 36) == 21
    assert layer_at_depth(0.8, 36) == 28
    assert layer_at_depth(1.0, 36) == 35
    mapping = layers_for_depths((0.4, 0.6, 0.8), 28)
    assert len(set(mapping.values())) == 3


def test_time_value_parse():
    assert TimeValue.parse("3 days").to_years() > 0
    assert TimeValue.parse([2, "weeks"]).unit == "weeks"


def test_pca_shapes_and_variance():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(40, 16)).astype(np.float32)
    X[:, 0] *= 10  # dominant direction
    res = fit_pca(X, n_components=3)
    assert res.projected.shape == (40, 3)
    assert res.components.shape == (3, 16)
    assert res.explained_variance_ratio[0] > 0.5
    assert abs(res.components[0, 0]) > 0.9


def test_parse_schedule_window_horizons():
    assert abs(parsing._parse_duration_years("Months 1–2") - 2 / 12) < 1e-9
    assert abs(parsing._parse_duration_years("Years 3-5") - 3.0) < 1e-9
    assert abs(parsing._parse_duration_years("Month 3") - 1 / 12) < 1e-9
    assert abs(parsing._parse_duration_years("Week 2") - 7 / 365.25) < 1e-9
    # plain durations still work
    assert abs(parsing._parse_duration_years("2 months") - 2 / 12) < 1e-9


def test_parse_unit_repeated_windows():
    assert abs(parsing._parse_duration_years("Day 2–Day 5") - 4 / 365.25) < 1e-9
    assert abs(parsing._parse_duration_years("Day 1") - 1 / 365.25) < 1e-9
    assert abs(parsing._parse_duration_years("Week 3-Week 4") - 14 / 365.25) < 1e-9


def test_target_mode_and_task_horizons():
    from src.datasets.generator import build_dataset
    from src.datasets.tasks import get_task
    ds = build_dataset(
        "t2",
        tasks=(get_task("climate_city"), get_task("marathon")),
        step_mode="target",
        task_horizons={
            "climate_city": horizons_from_specs(["2 years", "5 years"]),
            "marathon": horizons_from_specs(["6 weeks"]),
        },
    )
    assert len(ds.prompts) == 3
    assert "Time target:" in ds.prompts[0].text
    assert "not how long the step takes" in ds.prompts[0].text
    # parser accepts the new header
    raw, years = parsing.parse_step_horizon("Time target: 5 years")
    assert raw == "5 years" and abs(years - 5) < 1e-9


def test_mode_aware_parsing():
    # target mode: slot form = future offset; compound values sum
    assert abs(parsing.parse_step_horizon("Time target: Year 2")[1] - 2.0) < 1e-9
    assert abs(parsing.parse_step_horizon("Time target: 1 year, 6 months")[1] - 1.5) < 1e-9
    # duration mode: slot form stays a 1-unit schedule slot (old data unchanged)
    assert abs(parsing.parse_step_horizon("Time horizon: Year 2")[1] - 1.0) < 1e-9
    assert abs(parsing.parse_step_horizon("Time horizon: 3 weeks")[1] - 21 / 365.25) < 1e-9
