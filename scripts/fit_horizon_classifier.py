"""Classify prompts into bands of log10 time horizon and print the accuracy report.

    python scripts/fit_horizon_classifier.py --bands 4 --cache .tmp/horizon_cache

Regression on this representation bottoms out near 0.5 decades RMSE, so the useful question is
how coarse a banding has to be before it can be read reliably. ``--sweep`` answers that directly
by reporting accuracy at every band count.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from temporal_manifolds.horizon.cache import build_cache, load_cache
from temporal_manifolds.horizon.experiment import DEFAULT_CONFIG, run_band_classification
from temporal_manifolds.horizon.pipeline import ModelConfig, ViewConfig

WINDOWS = {
    "full": (-np.inf, np.inf),
    "10min-1yr": (float(np.log10(10 / (30.4375 * 1440))), float(np.log10(12.0))),
    "1min-100yr": (float(np.log10(1 / (30.4375 * 1440))), float(np.log10(1200.0))),
    "1hr-10yr": (float(np.log10(1 / (30.4375 * 24))), float(np.log10(120.0))),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--acts", default=".acts")
    parser.add_argument("--cache", default=".tmp/horizon_cache")
    parser.add_argument("--rebuild", action="store_true")
    parser.add_argument("--bands", type=int, default=4)
    parser.add_argument("--sweep", default="", help="Comma-separated band counts to compare.")
    parser.add_argument("--unit-bands", action="store_true", help="Use a-priori unit midpoints.")
    parser.add_argument("--window", default="full", choices=sorted(WINDOWS))
    parser.add_argument("--supervised", type=int, default=DEFAULT_CONFIG.views[0].n_supervised)
    parser.add_argument("--residual", type=int, default=DEFAULT_CONFIG.views[0].n_residual)
    parser.add_argument("--poly-head", type=int, default=DEFAULT_CONFIG.poly_head)
    parser.add_argument("--alpha", type=float, default=DEFAULT_CONFIG.alpha)
    parser.add_argument("--out", default="")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cache_dir = Path(args.cache)
    if args.rebuild or not (cache_dir / "cache_info.json").exists():
        cache = build_cache(args.acts, cache_dir, exclude_folders=("conv",))
    else:
        cache = load_cache(cache_dir)

    config = ModelConfig(
        views=(ViewConfig("raw", args.supervised, args.residual),),
        poly_degree=4,
        poly_head=args.poly_head,
        alpha=args.alpha,
    )
    window = WINDOWS[args.window]
    counts = [int(b) for b in args.sweep.split(",")] if args.sweep else [args.bands]

    for n_bands in counts:
        report, details = run_band_classification(
            cache, n_bands=n_bands, config=config, unit_bands=args.unit_bands, window=window
        )
        print(json.dumps(details, indent=2))
        print(report.summary(top_tasks=10))
        print("\nCONFUSION (rows = actual band)")
        print(report.confusion.to_string())
        print()
        if args.out:
            out = Path(args.out) / f"bands_{details['n_bands']}_{args.window}"
            out.mkdir(parents=True, exist_ok=True)
            report.per_band.to_csv(out / "accuracy_by_band.csv", index=False)
            report.by_task.to_csv(out / "accuracy_by_task.csv", index=False)
            report.by_folder.to_csv(out / "accuracy_by_folder.csv", index=False)
            report.confusion.to_csv(out / "confusion.csv")
            (out / "bands.json").write_text(json.dumps(details, indent=2))
            print(f"written to {out}\n")


if __name__ == "__main__":
    main()
