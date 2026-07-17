#!/usr/bin/env python3
"""Generate (or load) samples and activations via the content-addressed DataManager.

Output goes to data/<prompt_hash>/<model_name>/ and generation is skipped
entirely when a complete cache already exists for the same config + model.
See src/datasets/data_manager.py for the layout and caching rules.

Usage:
    # Full experiment config (dataset + model + targets in one JSON)
    uv run python utils/generate_samples_with_activation.py --config configs/experiments/run.json

    # Bare dataset config (name from configs/prompt_datasets/, or a path);
    # model and targets come from flags / defaults
    uv run python utils/generate_samples_with_activation.py --config saving --model Qwen/Qwen3-4B

    # Default dataset (GEOMETRY_CFG), limited samples
    uv run python utils/generate_samples_with_activation.py --max-samples 20

    # Force regeneration despite an existing cache
    uv run python utils/generate_samples_with_activation.py --config saving --force
"""

import argparse
import json
import logging
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.common.file_io import parse_file_path  # noqa: E402
from src.common.project_paths import get_prompt_dataset_configs_dir  # noqa: E402
from src.datasets.data_manager import DataConfig, DataManager  # noqa: E402
from src.datasets.default_configs import DEFAULT_MODEL, FULL_EXPERIMENT_CONFIG  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate samples and extract activations (content-addressed cache)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Experiment config JSON (with a 'dataset' key), or a bare dataset "
        "config: a path, or a name from configs/prompt_datasets/. "
        "Default: built-in GEOMETRY_CFG.",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help=f"Model identifier (default: config value or {DEFAULT_MODEL})",
    )
    parser.add_argument("--seed", type=int, default=None, help="Random seed (default: 42)")
    parser.add_argument(
        "--max-samples", type=int, default=None, help="Max samples to extract (default: all)"
    )
    parser.add_argument(
        "--layers", type=int, nargs="+", default=None, help="Layers to extract (default: all)"
    )
    parser.add_argument(
        "--components", type=str, nargs="+", default=None,
        help="Components to extract (default: all)",
    )
    parser.add_argument(
        "--positions", type=str, nargs="+", default=None,
        help="Semantic positions to extract (default: all)",
    )
    parser.add_argument(
        "--data-root", type=str, default=None,
        help="Root cache directory (default: <repo>/data)",
    )
    parser.add_argument(
        "--force", action="store_true", help="Regenerate even if a complete cache exists"
    )
    return parser.parse_args()


def build_config(args: argparse.Namespace) -> DataConfig:
    """Resolve --config into a DataConfig, applying CLI overrides."""
    overrides = dict(
        model=args.model,
        seed=args.seed,
        max_samples=args.max_samples,
        layers=args.layers,
        components=args.components,
        positions=args.positions,
    )

    if args.config is None:
        dataset_cfg = FULL_EXPERIMENT_CONFIG["dataset_config"]
        logger.info("Using built-in default dataset config (GEOMETRY_CFG)")
        return DataConfig.from_dict(
            {"dataset": dataset_cfg, "model": args.model or DEFAULT_MODEL}, **overrides
        )

    filepath = parse_file_path(
        args.config,
        default_dir_path=str(get_prompt_dataset_configs_dir()),
        default_ext=".json",
    )
    if not filepath.exists():
        raise FileNotFoundError(f"Config not found: {filepath}")

    with open(filepath) as f:
        raw = json.load(f)

    if "dataset" in raw:
        logger.info(f"Loaded experiment config: {filepath}")
        return DataConfig.from_dict(raw, **overrides)

    # Bare dataset config: model/targets come from flags or defaults
    logger.info(f"Loaded dataset config: {raw.get('name', filepath.stem)} from {filepath}")
    return DataConfig.from_dict(
        {"dataset": raw, "model": args.model or DEFAULT_MODEL}, **overrides
    )


def main() -> int:
    args = get_args()
    config = build_config(args)
    manager = DataManager(data_root=args.data_root)

    logger.info("=" * 60)
    logger.info("GENERATE SAMPLES WITH ACTIVATIONS")
    logger.info("=" * 60)
    logger.info(f"Dataset: {config.dataset_cfg.get('name', 'unnamed')}")
    logger.info(f"Model: {config.model}")
    logger.info(f"Fingerprint: {config.prompt_fingerprint}")
    logger.info(f"Output: {manager.model_dir(config)}")
    logger.info(
        f"Targets: {len(config.targets)} "
        f"({len(config.layers)} layers x {len(config.components)} components "
        f"x {len(config.positions)} positions)"
    )
    if manager.is_complete(config) and not args.force:
        logger.info("Complete cache found - loading without generation.")

    data = manager.get_or_generate(config, force=args.force)

    logger.info("=" * 60)
    logger.info("DONE")
    logger.info("=" * 60)
    logger.info(f"Samples: {len(data.samples)}")
    logger.info(f"Targets with data: {len(data.get_target_keys())}")
    logger.info(f"Data directory: {manager.model_dir(config) / 'data'}")
    logger.info(f"Manifest: {manager.manifest_path(config)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
