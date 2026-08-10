from __future__ import annotations

import importlib.util
from collections.abc import Mapping
from pathlib import Path
from types import ModuleType
from typing import Any

import torch


def load_script(name: str, filename: str) -> ModuleType:
    path = Path.cwd() / "scripts" / filename
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


CONVERSATIONAL = load_script(
    "cache_conversational_selected_acts_schema_test",
    "cache_conversational_selected_acts.py",
)
ABSTRACT = load_script(
    "cache_abstract_selected_acts_schema_test",
    "cache_abstract_selected_acts.py",
)


def recursive_keys(value: Any) -> Any:
    if isinstance(value, Mapping):
        return tuple((key, recursive_keys(nested)) for key, nested in value.items())
    return None


def test_all_caching_modes_emit_the_conversational_metadata_schema() -> None:
    common = {
        "text": "Prompt",
        "template_id": "template",
        "task": "Task",
        "quantity": None,
        "quantity_text": None,
        "base_value": 1,
        "base_unit": "days",
        "unit_variant": "original",
        "number_format": "numeric",
        "value": 1,
        "value_text": "1",
        "unit": "day",
    }
    conversational_record = {
        **common,
        "template_metadata": {
            "prompt_framing": "task_available_time",
            "output_format": "strategy_steps",
        },
        "task_metadata": {
            "task_family": "planning",
            "difficulty": "medium",
            "domain": "personal",
            "complexity": "medium",
            "planning_type": "project",
            "stakes": "medium",
            "agency": "individual",
        },
    }
    no_output_format_record = {
        **common,
        "template_metadata": {"prompt_framing": "task_available_time"},
        "task_metadata": conversational_record["task_metadata"],
    }
    abstract_record = {
        **common,
        "template_metadata": {
            "prompt_framing": "task_available_time",
            "subject_framing": "task",
            "time_framing": "available_time",
        },
        "task_metadata": {
            "task_family": "state_stabilization",
            "domain": "abstract",
            "planning_type": "stabilization",
            "temporal_coverage": "full_range",
        },
    }

    activation = torch.ones(1, 1, 2)
    conversational_payload = CONVERSATIONAL.build_payload(
        activation=activation,
        records=[conversational_record],
        sample_indices=[0],
        batch_index=0,
        model_name="test-model",
    )
    no_output_format_payload = CONVERSATIONAL.build_payload(
        activation=activation,
        records=[no_output_format_record],
        sample_indices=[0],
        batch_index=0,
        model_name="test-model",
    )
    abstract_payload = ABSTRACT.build_payload(
        activation=activation,
        records=[abstract_record],
        sample_indices=[0],
        batch_index=0,
        model_name="test-model",
    )

    metadata_rows = [
        conversational_payload["prompt_metadata"][0],
        no_output_format_payload["prompt_metadata"][0],
        abstract_payload["prompt_metadata"][0],
    ]
    assert recursive_keys(metadata_rows[0]) == recursive_keys(metadata_rows[1])
    assert recursive_keys(metadata_rows[0]) == recursive_keys(metadata_rows[2])
    assert no_output_format_payload["prompt_metadata"][0]["template_metadata"][
        "output_format"
    ] == "N/A"
    assert abstract_payload["prompt_metadata"][0]["template_metadata"]["output_format"] == (
        "N/A"
    )
    assert abstract_payload["prompt_metadata"][0]["task_metadata"]["difficulty"] == "N/A"
    assert all(metadata["quantity"] == "N/A" for metadata in metadata_rows)
    assert "subject_framing" not in abstract_payload["prompt_metadata"][0][
        "template_metadata"
    ]
    assert "temporal_coverage" not in abstract_payload["prompt_metadata"][0]["task_metadata"]
