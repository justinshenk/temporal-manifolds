from __future__ import annotations

import json
from pathlib import Path


NOTEBOOK_PATH = Path.cwd() / "notebooks" / "download_filtered_abstract_selected_activations.ipynb"


def load_notebook_source() -> tuple[dict[str, object], str]:
    notebook = json.loads(NOTEBOOK_PATH.read_text(encoding="utf-8"))
    source = "\n".join("".join(cell.get("source", [])) for cell in notebook["cells"])
    return notebook, source


def test_abstract_notebook_uses_matching_dataset_paths_and_payload_validation() -> None:
    notebook, source = load_notebook_source()

    assert notebook["nbformat"] == 4
    assert "conversational" not in source.lower()
    assert "scripts/run_activation_caching_abstract_selected_acts.sh" in source
    assert "gs://temporal-research-bucket/abstract_selected_acts" in source
    assert "filtered_abstract_selected_activations" in source
    assert "EXPECTED_DATASET = 'abstract'" in source
    assert "payload_dataset != EXPECTED_DATASET" in source


def test_abstract_notebook_preserves_per_task_temporal_trajectories() -> None:
    _notebook, source = load_notebook_source()

    assert "AGG_BY: list[str] | None = ['task', 'time_horizon_months']" in source
    assert "'template_metadata.prompt_framing'" in source
    assert "'template_metadata.subject_framing'" in source
    assert "'template_metadata.time_framing'" in source
    assert "'number_format'" in source
    assert "'unit_variant'" in source
    assert "template_metadata.output_format" not in source
