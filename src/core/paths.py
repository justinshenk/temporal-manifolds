"""Project path helpers.

All experiment artifacts live under output/ (gitignored), organized by run:

    output/
      runs/<run_fingerprint>/       # one prompt-dataset + protocol + gen config
        run_config.json
        prompt_dataset.json
        <model_name>/               # sanitized model id
          manifest.json
          samples/<sample_uid>/...
          figures/
"""

from __future__ import annotations

import re
from pathlib import Path


def get_project_root() -> Path:
    """Repo root (this file lives at src/core/)."""
    return Path(__file__).parents[2]


def get_output_dir() -> Path:
    return get_project_root() / "output"


def get_runs_dir() -> Path:
    return get_output_dir() / "runs"


def get_configs_dir() -> Path:
    return get_project_root() / "experiments" / "configs"


def sanitize_model_name(model_id: str) -> str:
    """Model id -> safe directory name. 'Qwen/Qwen3-14B' -> 'Qwen__Qwen3-14B'."""
    name = model_id.replace("/", "__")
    return re.sub(r"[^A-Za-z0-9._-]+", "-", name)
