from __future__ import annotations

import importlib.util
import re
import sys
from pathlib import Path

import pytest

from temporal_manifolds.dataset import conversational, indexed_horizon
from temporal_manifolds.dataset.generate import (
    DATASETS,
    dataset_record_builder,
    generate_task_dataset,
)


@pytest.fixture(scope="module")
def records() -> list[dict]:
    return generate_task_dataset(dataset="indexed_horizon")


def test_dataset_is_registered_and_uses_the_conversational_grid() -> None:
    assert DATASETS["indexed_horizon"] is indexed_horizon
    assert dataset_record_builder("indexed_horizon") is indexed_horizon.build_prompt_records
    assert indexed_horizon.tasks == conversational.tasks
    assert indexed_horizon.tasks is not conversational.tasks
    assert indexed_horizon.values == conversational.values
    for task, config in indexed_horizon.tasks.items():
        assert config is not conversational.tasks[task]
        assert config["units"] is not conversational.tasks[task]["units"]


def test_every_cell_has_eighteen_distinct_renderings(records: list[dict]) -> None:
    expected_cells = sum(
        len(config["units"]) * len(indexed_horizon.values)
        for config in indexed_horizon.tasks.values()
    )
    grouped: dict[tuple[str, int, str], set[str]] = {}
    for record in records:
        key = (record["task"], record["base_value"], record["base_unit"])
        grouped.setdefault(key, set()).add(record["text"])

    assert len(grouped) == expected_cells
    assert all(len(prompts) == 18 for prompts in grouped.values())
    assert len(records) == expected_cells * 18


def test_indices_encode_exact_horizons_without_printing_a_duration(records: list[dict]) -> None:
    for record in records:
        situation, request = record["text"].split("\n\n", 1)
        numbers = [int(value) for value in re.findall(r"\d+", situation)]
        if record["number_format"] == "numeric":
            assert len(numbers) == 2
            assert numbers[1] - numbers[0] == record["base_value"]
        else:
            assert not numbers
        explicit_duration = f'{record["value_text"]} {record["unit"]}'
        assert explicit_duration not in situation
        assert request == indexed_horizon.REQUEST.format(task=record["task"])


def test_all_units_and_number_formats_are_supported(records: list[dict]) -> None:
    assert {record["base_unit"] for record in records} == {
        unit for config in conversational.tasks.values() for unit in config["units"]
    }
    assert {record["number_format"] for record in records} == {"numeric", "words"}
    assert all(record["unit_variant"] == "original" for record in records)


def test_fixed_endpoint_variants_do_not_encode_horizon_in_closing_index() -> None:
    fixed_endpoint_variants = [
        variant for variant in indexed_horizon.variants if variant.fixed_endpoint
    ]
    assert len(fixed_endpoint_variants) == 4

    for variant in fixed_endpoint_variants:
        endpoints = set()
        starts = set()
        for value in indexed_horizon.values:
            situation = variant.render(value, "days", "numeric")
            start, end = [int(number) for number in re.findall(r"\d+", situation)]
            starts.add(start)
            endpoints.add(end)
            assert end - start == value

        assert endpoints == {variant.fixed_index}
        assert len(starts) == len(indexed_horizon.values)


def test_template_driven_options_are_refused() -> None:
    with pytest.raises(ValueError, match="builds its own prompts"):
        generate_task_dataset(dataset="indexed_horizon", randomize_template=True)


def test_all_selected_acts_workflow_includes_the_dataset() -> None:
    script_path = Path.cwd() / "scripts" / "cache_all_selected_acts.py"
    spec = importlib.util.spec_from_file_location("cache_all_selected_acts", script_path)
    assert spec is not None and spec.loader is not None
    script = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = script
    try:
        spec.loader.exec_module(script)
        target = script.DATASET_TARGETS_BY_NAME["indexed_horizon"]
        assert target.output_dir_name == "indexed_horizon_selected_acts"
        assert target.gcs_prefix == "indexed_horizon_selected_acts"
    finally:
        sys.modules.pop(spec.name, None)
