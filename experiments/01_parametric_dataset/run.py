"""Generate a parametric prompt dataset from a scenario config.

Usage:
    uv run python -m experiments.01_parametric_dataset.run \\
        --config configs/scenarios/example.yaml \\
        --out data/example/
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

from temporal_manifolds.dataset import generate_dataset
from temporal_manifolds.dataset.generate import DatasetConfig


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()

    config = DatasetConfig.from_yaml(args.config)
    rows = generate_dataset(config)

    args.out.mkdir(parents=True, exist_ok=True)
    out_file = args.out / "dataset.jsonl"
    with out_file.open("w") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")
    shutil.copy(args.config, args.out / "config.snapshot.yaml")

    n_train = sum(1 for r in rows if r["split"] == "train")
    n_test = sum(1 for r in rows if r["split"] == "test")
    print(
        f"Wrote {len(rows)} rows ({n_train} train, {n_test} test) "
        f"across {len(config.scenarios)} scenarios → {out_file}"
    )


if __name__ == "__main__":
    main()
