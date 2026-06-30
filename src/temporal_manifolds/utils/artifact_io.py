"""Generic helpers for writing artifact files."""

from __future__ import annotations

import json
import pickle
from pathlib import Path
from typing import Any


def write_pickle(path: Path, payload: Any) -> None:
    """Write a pickle artifact, creating parent directories as needed."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as f:
        pickle.dump(payload, f)


def write_json(path: Path, payload: Any) -> None:
    """Write a JSON artifact, creating parent directories as needed."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
