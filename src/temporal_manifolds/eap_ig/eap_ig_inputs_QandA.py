"""
Run Q&A EAP-style attribution on clean vs corrupted prompts and save scores to NPZ.
"""

import argparse
from pathlib import Path

import torch

try:
    from .eap_ig_qanda_pipeline import run_eap, run_eap_ig
except ImportError:
    from eap_ig_qanda_pipeline import run_eap, run_eap_ig

torch.set_grad_enabled(False)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run EAP-style attribution on clean vs corrupted prompts"
    )
    parser.add_argument(
        "--method",
        choices=("eap-ig", "eap"),
        default="eap-ig",
        help=(
            "Attribution method to run. eap-ig uses path-averaged gradients; "
            "eap uses the selected prompt-side gradient."
        ),
    )
    parser.add_argument(
        "--compute-gradient-at",
        choices=("clean", "corrupted"),
        default="clean",
        help="Prompt side where vanilla EAP computes gradients.",
    )
    parser.add_argument(
        "--config",
        type=Path,
        required=True,
        help="Path to config YAML file.",
    )
    parser.add_argument(
        "--save-to-gcp",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Upload generated NPZ files to the configured GCS bucket.",
    )
    parser.add_argument(
        "--results-root",
        type=Path,
        default=None,
        help="Optional root directory overriding the results/ prefix in config save_loc paths.",
    )
    args = parser.parse_args()
    runner = run_eap_ig if args.method == "eap-ig" else run_eap
    if args.method == "eap":
        runner(
            args.config,
            save_to_gcp=args.save_to_gcp,
            results_root=args.results_root,
            compute_gradient_at=args.compute_gradient_at,
        )
    else:
        runner(args.config, save_to_gcp=args.save_to_gcp, results_root=args.results_root)


if __name__ == "__main__":
    main()
