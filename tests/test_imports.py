"""Smoke tests for the remaining importable packages."""

from __future__ import annotations

import os
import subprocess
import sys


def test_package_imports() -> None:
    import temporal_manifolds  # noqa: F401
    from temporal_manifolds import activations, dataset, viz  # noqa: F401


def test_eap_ig_plots_imports_with_colab_inline_backend() -> None:
    env = os.environ.copy()
    env["MPLBACKEND"] = "module://matplotlib_inline.backend_inline"

    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "import os; import temporal_manifolds.viz.eap_ig_plots; print(os.environ['MPLBACKEND'])",
        ],
        check=True,
        capture_output=True,
        env=env,
        text=True,
    )

    assert result.stdout.strip() == "Agg"
