from __future__ import annotations

import json
from pathlib import Path


NOTEBOOK_PATH = (
    Path.cwd()
    / "notebooks"
    / "download_filtered_plain_english_selected_activations.ipynb"
)
CONVERSATIONAL_NOTEBOOK_PATH = (
    Path.cwd()
    / "notebooks"
    / "download_filtered_conversational_selected_activations.ipynb"
)


def load_notebook(path: Path) -> tuple[dict[str, object], str]:
    notebook = json.loads(path.read_text(encoding="utf-8"))
    source = "\n".join("".join(cell.get("source", [])) for cell in notebook["cells"])
    return notebook, source


def test_plain_english_notebook_uses_matching_dataset_paths_and_payload_validation() -> None:
    notebook, source = load_notebook(NOTEBOOK_PATH)

    assert notebook["nbformat"] == 4
    assert "conversational" not in source.lower()
    assert "scripts/run_activation_caching_plain_english_selected_acts.sh" in source
    assert "gs://temporal-research-bucket/plain_english_selected_acts" in source
    assert "filtered_plain_english_selected_activations" in source
    assert "EXPECTED_DATASET = 'plain_english'" in source
    assert "payload_dataset != EXPECTED_DATASET" in source


def test_plain_english_notebook_preserves_the_conversational_workflow() -> None:
    notebook, source = load_notebook(NOTEBOOK_PATH)
    conversational, _conversational_source = load_notebook(CONVERSATIONAL_NOTEBOOK_PATH)

    assert [cell["id"] for cell in notebook["cells"]] == [
        cell["id"] for cell in conversational["cells"]
    ]
    assert [cell["cell_type"] for cell in notebook["cells"]] == [
        cell["cell_type"] for cell in conversational["cells"]
    ]
    dataset_specific_cells = {"title", "configuration", "filter-load"}
    conversational_by_id = {cell["id"]: cell for cell in conversational["cells"]}
    for cell in notebook["cells"]:
        if cell["id"] not in dataset_specific_cells:
            assert cell["source"] == conversational_by_id[cell["id"]]["source"]
    assert "# 'template_id': 'direct_help'" in source
    assert "PHRASING_FIELDS = ['template_id']" in source
    assert "AGG_BY: list[str] | None = ['time_horizon_months']" in source
