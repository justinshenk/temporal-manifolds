"""Make both package roots importable in tests.

The repo has two coexisting layouts after merging dev:
- `src` as a package (import src.common, src.datasets, ...)
- the legacy `temporal_manifolds` package living under src/
  (import temporal_manifolds.workflows, ...)

Putting src/ on sys.path keeps the legacy absolute imports working
without reintroducing the old package-dir mapping in pyproject.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))
