"""Plotting helpers for Q&A EAP-IG artifacts."""

from __future__ import annotations

import math
import os
from pathlib import Path

if os.environ.get("MPLBACKEND") in {None, "module://matplotlib_inline.backend_inline"}:
    os.environ["MPLBACKEND"] = "Agg"

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import pearsonr

from temporal_manifolds.utils.eap_ig_artifacts import (
    LOGIT_ARRAY_NAMES,
    CaseResults,
    MergedArrayGroup,
)

COMPLETENESS_PLOT_ROWS = 3
LOGIT_INDEX_BY_LABEL = {"A": 0, "B": 1}


def safe_pearsonr(x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    """Compute Pearson correlation without crashing on constant arrays."""
    try:
        result = pearsonr(x, y)
        return float(result.correlation), float(result.pvalue)  # type: ignore
    except ValueError:
        return float("nan"), float("nan")


def compute_completeness_arrays(
    group: MergedArrayGroup,
    logit_index: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Return the notebook-style predicted and target completeness arrays."""
    target = group["clean_logits"][:, logit_index] - group["corrupted_logits"][:, logit_index]
    component_predictions = [
        values.sum(axis=-1) for key, values in group.items() if key not in LOGIT_ARRAY_NAMES
    ]
    predicted = np.stack(component_predictions, axis=-1).sum(axis=-1)
    return predicted, target


def plot_case_completeness(
    case_name: str,
    case_results: CaseResults,
    output_path: Path,
    *,
    logit_label: str = "B",
) -> Path:
    """Save the completeness scatter plots produced in the notebook workflow."""
    variants = list(case_results.keys())
    columns = max(1, math.ceil(len(variants) / COMPLETENESS_PLOT_ROWS))
    fig, axes = plt.subplots(
        COMPLETENESS_PLOT_ROWS,
        columns,
        figsize=(5 * columns, 4.5 * COMPLETENESS_PLOT_ROWS),
    )
    axes = np.atleast_1d(axes).ravel()  # type: ignore

    for axis, variant_name in zip(axes, variants, strict=False):
        group = case_results[variant_name][f"short_first_logit_{logit_label}"]
        predicted, target = compute_completeness_arrays(
            group,
            LOGIT_INDEX_BY_LABEL[logit_label],
        )
        correlation, p_value = safe_pearsonr(predicted, target)
        axis.scatter(predicted, target, s=5)
        axis.set_title(
            f"{variant_name}_r:{correlation:.2f}_p:{p_value:.4f}",
            fontsize=10,
        )
        axis.set_xlabel("Predicted sum")
        axis.set_ylabel("Target logit delta")

    for axis in axes[len(variants) :]:
        axis.axis("off")

    fig.suptitle(f"{case_name}_{logit_label}")
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return output_path
