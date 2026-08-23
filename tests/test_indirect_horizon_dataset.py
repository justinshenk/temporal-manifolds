"""Contract tests for the indirect-horizon dataset.

The dataset's value rests on two claims: every prompt is self-contained, and every
(task, horizon) cell is written at least ten different ways. The shared dataset
conversion table supplies smaller-unit variants, including four weeks per month.
"""

from __future__ import annotations

import re
from collections import defaultdict
from datetime import datetime, timedelta

import pytest

from temporal_manifolds.activations.extraction_policy import (
    NOT_APPLICABLE,
    canonical_prompt_metadata,
)
from temporal_manifolds.dataset import conversational, indirect_horizon
from temporal_manifolds.dataset.generate import (
    DATASETS,
    dataset_record_builder,
    generate_task_dataset,
)
from temporal_manifolds.horizon.cache import UNIT_TO_MONTHS

MINIMUM_VARIANTS_PER_CELL = 10


@pytest.fixture(scope="module")
def records() -> list[dict]:
    return generate_task_dataset(dataset="indirect_horizon")


def test_dataset_is_registered_and_builds_its_own_records() -> None:
    assert DATASETS["indirect_horizon"] is indirect_horizon
    assert dataset_record_builder("indirect_horizon") is indirect_horizon.build_prompt_records
    assert dataset_record_builder("conversational") is None


def test_tasks_and_horizons_match_the_conversational_grid() -> None:
    assert indirect_horizon.tasks == conversational.tasks
    assert indirect_horizon.tasks is not conversational.tasks
    assert indirect_horizon.values == conversational.values
    for task, config in indirect_horizon.tasks.items():
        assert config["units"] is not conversational.tasks[task]["units"]


def test_every_record_carries_one_resolvable_horizon(records: list[dict]) -> None:
    for record in records:
        unit = record["base_unit"]
        assert record["base_value"] in conversational.values
        assert unit in conversational.tasks[record["task"]]["units"]
        # A horizon that cannot be converted to months would be dropped by the
        # explorer's preparation step rather than plotted.
        assert unit in UNIT_TO_MONTHS


def test_prompts_are_rendered_not_templated(records: list[dict]) -> None:
    for record in records:
        text = record["text"]
        situation, request = text.split("\n\n")
        assert "{" not in text and "}" not in text
        assert request == indirect_horizon.REQUEST.format(task=record["task"])
        assert situation.endswith(".")
        assert ".." not in situation
        assert "  " not in situation


def test_no_prompt_states_its_horizon_as_a_plain_duration(records: list[dict]) -> None:
    """The point of the dataset is that ``base_value base_unit`` never appears verbatim."""
    for record in records:
        situation = record["text"].split("\n\n")[0]
        value, unit = record["base_value"], record["base_unit"]
        singular = unit[:-1] if unit.endswith("s") else unit
        stated = re.search(
            rf"\b{value}\s+({re.escape(unit)}|{re.escape(singular)})\b",
            situation,
        )
        # A stated duration is only acceptable as one term of an arithmetic
        # relation, never as the horizon itself.
        assert stated is None or record["template_id"] in {
            "remainder_spent",
            "elapsed_window",
            "delayed_start",
        }, situation


def test_every_task_and_horizon_is_rendered_at_least_ten_ways(records: list[dict]) -> None:
    cells: dict[tuple[str, int, str], set[str]] = defaultdict(set)
    for record in records:
        cells[(record["task"], record["base_value"], record["base_unit"])].add(
            record["template_id"]
        )

    assert cells, "the dataset generated no prompts"
    thinnest = min(cells.items(), key=lambda item: len(item[1]))
    assert len(thinnest[1]) >= MINIMUM_VARIANTS_PER_CELL, thinnest

    expected_cells = sum(
        len(config["units"]) * len(indirect_horizon.values)
        for config in indirect_horizon.tasks.values()
    )
    assert len(cells) == expected_cells


def test_both_number_formats_are_exercised(records: list[dict]) -> None:
    formats = {record["number_format"] for record in records}
    assert formats == {"numeric", "words"}
    spelled = [
        record
        for record in records
        if record["number_format"] == "words" and record["template_id"] == "proportion_half"
    ]
    assert spelled and all(char.isdigit() is False for char in spelled[0]["text"])


@pytest.mark.parametrize(
    ("variant_id", "expected"),
    [
        ("clock_12h", "It is 9:00 a.m., and this has to be finished by 9:30 a.m."),
        ("clock_24h", "The current time is 09:00, and the cut-off is 09:30."),
        (
            "clock_spoken",
            "It is nine o'clock in the morning, and the cut-off is thirty minutes past "
            "nine in the morning.",
        ),
        (
            "clock_three_point",
            "Work opened at 8:30 a.m., it is 9:00 a.m. now, and everything must be "
            "finished by 9:30 a.m.",
        ),
    ],
)
def test_clock_variants_render_the_expected_wall_clock_times(
    variant_id: str, expected: str
) -> None:
    variant = indirect_horizon.VARIANTS_BY_ID[variant_id]
    rendering = variant.render(30, "minutes", variant.number_formats[0])
    assert rendering is not None
    assert rendering.text == expected


def test_clock_variants_stay_inside_one_day() -> None:
    for variant in indirect_horizon.variants:
        if variant.family != "clock":
            continue
        for unit in ("seconds", "minutes", "hours"):
            for value in indirect_horizon.values:
                rendering = variant.render(value, unit, variant.number_formats[0])
                seconds = indirect_horizon.horizon_seconds(value, unit)
                assert seconds is not None
                if seconds > indirect_horizon.MAX_CLOCK_SECONDS:
                    assert rendering is None
                else:
                    assert rendering is not None


def test_calendar_variants_name_the_deadline_the_horizon_implies() -> None:
    """Recompute each deadline from the horizon and look for it in the prompt."""
    anchor = indirect_horizon.ANCHOR
    checks = [
        ("date_iso", 3, "weeks", "2025-03-25"),
        ("date_long", 3, "weeks", "25 March 2025"),
        ("date_us", 3, "weeks", "March 25, 2025"),
        ("date_weekday", 3, "weeks", "Tuesday, 25 March 2025"),
        ("datetime_span", 100, "hours", "13:00 on 8 March 2025"),
        ("year_span", 2, "decades", "the start of 2045"),
    ]
    for variant_id, value, unit, expected in checks:
        variant = indirect_horizon.VARIANTS_BY_ID[variant_id]
        rendering = variant.render(value, unit, variant.number_formats[0])
        assert rendering is not None, (variant_id, value, unit)
        assert expected in rendering.text, rendering.text

    # The weeks case above, recomputed without the module's calendar helpers.
    assert anchor + timedelta(weeks=3) == datetime(2025, 3, 25, 9, 0, 0)


def test_month_and_year_arithmetic_is_exact() -> None:
    assert indirect_horizon.add_months(datetime(2025, 3, 4), 100) == datetime(2033, 7, 4)
    assert indirect_horizon.add_months(datetime(2025, 12, 4), 1) == datetime(2026, 1, 4)
    assert indirect_horizon.add_years(datetime(2024, 2, 29), 1) == datetime(2025, 2, 28)
    with pytest.raises(ValueError, match="cannot be written as a date"):
        indirect_horizon.add_years(datetime(2025, 3, 4), 100_000)


def test_horizons_beyond_the_calendar_still_get_a_year_span() -> None:
    variant = indirect_horizon.VARIANTS_BY_ID["year_span"]
    rendering = variant.render(100, "millennia", "numeric")
    assert rendering is not None
    assert rendering.text.endswith("to the start of 102025.")
    # Dates are refused rather than truncated for horizons the calendar cannot hold.
    assert indirect_horizon.VARIANTS_BY_ID["date_iso"].render(100, "millennia", "numeric") is None


def test_a_month_uses_the_shared_four_week_conversion() -> None:
    for variant_id in ("two_blocks", "multiple_reference"):
        variant = indirect_horizon.VARIANTS_BY_ID[variant_id]
        month_rendering = variant.render(1, "months", "numeric")
        assert month_rendering is not None
        assert month_rendering.value == 4
        assert month_rendering.unit == "weeks"
        assert month_rendering.unit_variant == "smaller"

        rendering = variant.render(1, "hours", "numeric")
        assert rendering is not None
        assert rendering.unit == "minutes"
        assert rendering.unit_variant == "smaller"


def test_records_project_onto_the_cached_metadata_schema(records: list[dict]) -> None:
    metadata = canonical_prompt_metadata(records[0])
    assert metadata["template_id"] in indirect_horizon.VARIANTS_BY_ID
    assert metadata["template_metadata"]["prompt_framing"] in {
        "arithmetic",
        "clock",
        "calendar",
    }
    assert metadata["base_unit"] != NOT_APPLICABLE
    assert metadata["base_value"] != NOT_APPLICABLE
    assert metadata["task_metadata"]["task_family"] != NOT_APPLICABLE


def test_template_driven_options_are_refused() -> None:
    for kwargs in (
        {"randomize_template": True},
        {"remove_time_constraints": True},
        {"time_values": [1, 2]},
    ):
        with pytest.raises(ValueError, match="builds its own prompts"):
            generate_task_dataset(dataset="indirect_horizon", **kwargs)


def test_the_expanded_caching_run_has_an_isolated_scenario() -> None:
    """The multi-layer sweep must cache this dataset into its own namespace."""
    import importlib.util
    import sys
    from pathlib import Path

    script_path = Path.cwd() / "scripts" / "cache_expanded_selected_acts.py"
    spec = importlib.util.spec_from_file_location("cache_expanded_selected_acts", script_path)
    assert spec is not None and spec.loader is not None
    script = importlib.util.module_from_spec(spec)
    # The module defines a dataclass, which resolves annotations through sys.modules.
    sys.modules[spec.name] = script
    try:
        spec.loader.exec_module(script)
        scenario = script.SCENARIOS_BY_NAME["indirect_horizon"]
        assert scenario.dataset == "indirect_horizon"
        prefixes = {other.gcs_prefix for other in script.SCENARIOS if other is not scenario}
        directories = {other.output_dir_name for other in script.SCENARIOS if other is not scenario}
        assert scenario.gcs_prefix not in prefixes
        assert scenario.output_dir_name not in directories
    finally:
        sys.modules.pop(spec.name, None)


def test_records_are_written_when_an_output_path_is_given(tmp_path) -> None:
    output_path = tmp_path / "indirect.json"
    generated = generate_task_dataset(dataset="indirect_horizon", output_path=output_path)
    assert output_path.exists()
    assert len(generated) > 10_000
