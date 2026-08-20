"""Fit log10 time horizon from cached activations and print the per-task / per-folder report.

    python scripts/fit_horizon_model.py --acts .acts --cache .tmp/horizon_cache

The first run streams every ``.pt`` batch into a float16 memmap plus a metadata table; later
runs reuse it. ``conv`` is excluded as an outlier source, and ``--drop-outlier-tasks`` re-runs
the fit without the tasks whose RMSE is a robust outlier on the first pass.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from temporal_manifolds.horizon.cache import build_cache, load_cache
from temporal_manifolds.horizon.experiment import DEFAULT_CONFIG, run_experiment
from temporal_manifolds.horizon.pipeline import ModelConfig, ViewConfig
from temporal_manifolds.horizon.report import flag_outlier_tasks, task_folder_matrix


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--acts", default=".acts", help="Root of the cached activation folders.")
    parser.add_argument("--cache", default=".tmp/horizon_cache", help="Where to keep the memmap.")
    parser.add_argument("--rebuild", action="store_true", help="Rebuild the memmap cache.")
    parser.add_argument("--exclude-folders", default="conv", help="Comma-separated outlier folders.")
    parser.add_argument("--supervised", type=int, default=DEFAULT_CONFIG.views[0].n_supervised)
    parser.add_argument("--residual", type=int, default=DEFAULT_CONFIG.views[0].n_residual)
    parser.add_argument("--degree", type=int, default=DEFAULT_CONFIG.poly_degree)
    parser.add_argument("--poly-head", type=int, default=DEFAULT_CONFIG.poly_head)
    parser.add_argument("--alpha", type=float, default=DEFAULT_CONFIG.alpha)
    parser.add_argument("--splits", type=int, default=DEFAULT_CONFIG.n_splits)
    parser.add_argument("--subsample", type=int, default=0, help="Fit on a random row subset.")
    parser.add_argument(
        "--drop-outlier-tasks",
        action="store_true",
        help="Re-fit without the tasks flagged as robust RMSE outliers.",
    )
    parser.add_argument("--out", default="", help="Directory for the scorecard CSVs.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.degree > 4:
        raise SystemExit("Polynomial degree is capped at 4 for this experiment.")

    cache_dir = Path(args.cache)
    if args.rebuild or not (cache_dir / "cache_info.json").exists():
        print(f"building cache from {args.acts} -> {cache_dir}", flush=True)
        cache = build_cache(
            args.acts,
            cache_dir,
            exclude_folders=tuple(f for f in args.exclude_folders.split(",") if f),
        )
    else:
        cache = load_cache(cache_dir)
    print(f"cache: {cache.n_rows} rows x {cache.n_features} dims @ {cache.layer_component}")

    config = ModelConfig(
        views=(ViewConfig("raw", args.supervised, args.residual),),
        poly_degree=args.degree,
        poly_head=args.poly_head,
        alpha=args.alpha,
        n_splits=args.splits,
    )

    result = run_experiment(cache, config=config, subsample=args.subsample)
    print(json.dumps(result.details, indent=2, default=str))
    print(result.scorecards.summary(top_tasks=15))

    flagged = flag_outlier_tasks(result.scorecards)
    print(f"\noutlier tasks flagged: {flagged or 'none'}")
    if flagged and args.drop_outlier_tasks:
        print("\nre-fitting without the flagged tasks")
        result = run_experiment(
            cache, config=config, drop_tasks=tuple(flagged), subsample=args.subsample
        )
        print(json.dumps(result.details, indent=2, default=str))
        print(result.scorecards.summary(top_tasks=15))

    if args.out:
        out = Path(args.out)
        out.mkdir(parents=True, exist_ok=True)
        result.scorecards.by_task.to_csv(out / "metrics_by_task.csv", index=False)
        result.scorecards.by_folder.to_csv(out / "metrics_by_folder.csv", index=False)
        result.scorecards.by_fold.to_csv(out / "metrics_by_fold.csv", index=False)
        result.scorecards.by_unit.to_csv(out / "metrics_by_unit.csv", index=False)
        task_folder_matrix(result.scorecards).to_csv(out / "rmse_task_by_folder.csv")
        print(f"\nscorecards written to {out}")


if __name__ == "__main__":
    main()
