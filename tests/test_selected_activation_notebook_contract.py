from __future__ import annotations

import json
from pathlib import Path

import pytest


NOTEBOOKS = (
    "download_filtered_abstract_selected_activations.ipynb",
    "download_filtered_conversational_selected_activations.ipynb",
    "download_filtered_conversational_selected_activations_no_output_format.ipynb",
)


@pytest.mark.parametrize("notebook_name", NOTEBOOKS)
def test_notebook_rejects_any_activation_outside_fixed_contract(
    notebook_name: str,
) -> None:
    path = Path.cwd() / "notebooks" / notebook_name
    notebook = json.loads(path.read_text(encoding="utf-8"))
    source = "\n".join("".join(cell.get("source", [])) for cell in notebook["cells"])

    assert "LAYER_COMPONENT = 'layer_out/21'" in source
    assert "POSITION_INDEX = 0  # The only cached position; payload position value is -1." in source
    assert "payload.get('layer_component') != LAYER_COMPONENT" in source
    assert "set(payload.get('activations', {})) != {LAYER_COMPONENT}" in source
    assert "positions != [-1]" in source
