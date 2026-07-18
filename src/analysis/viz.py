"""2D / 3D PCA scatter figures for boundary activations.

Colorings supported:
  target_horizon  log-scale continuous colormap over target_horizon_years
  step_horizon    log-scale continuous colormap over step_horizon_years
                  (rows without a parsed step horizon are drawn gray)
  prompt_id / task_id / turn_index / kind / step_index   categorical

Figures are matplotlib (Agg), saved as PNG. Each call returns the paths it
wrote so callers can verify the artifacts.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import colormaps
from matplotlib.colors import LogNorm

from ..capture.store import QueryRow
from .pca import PCAResult

_CATEGORICAL_CMAP = "tab10"
_CONTINUOUS_CMAP = "viridis"


def _horizon_values(rows: list[QueryRow], field: str) -> np.ndarray:
    vals = []
    for r in rows:
        v = getattr(r, field)
        vals.append(np.nan if v is None else float(v))
    return np.array(vals)


def _scatter(ax, pts, rows, color_by: str, three_d: bool):
    if color_by in ("target_horizon", "step_horizon"):
        field = (
            "target_horizon_years"
            if color_by == "target_horizon"
            else "step_horizon_years"
        )
        vals = _horizon_values(rows, field)
        finite = np.isfinite(vals) & (vals > 0)
        # gray for missing values
        if (~finite).any():
            coords = pts[~finite]
            args = (coords[:, 0], coords[:, 1], coords[:, 2]) if three_d else (
                coords[:, 0],
                coords[:, 1],
            )
            ax.scatter(*args, c="lightgray", s=14, alpha=0.5, label="no value")
        if finite.any():
            coords = pts[finite]
            vmin, vmax = vals[finite].min(), vals[finite].max()
            norm = LogNorm(vmin=vmin, vmax=max(vmax, vmin * 1.0001))
            args = (coords[:, 0], coords[:, 1], coords[:, 2]) if three_d else (
                coords[:, 0],
                coords[:, 1],
            )
            sc = ax.scatter(
                *args, c=vals[finite], cmap=_CONTINUOUS_CMAP, norm=norm, s=18
            )
            return sc
        return None
    # categorical
    cats = [str(getattr(r, color_by)) for r in rows]
    uniq = sorted(set(cats))
    cmap = colormaps[_CATEGORICAL_CMAP]
    for i, cat in enumerate(uniq):
        mask = np.array([c == cat for c in cats])
        coords = pts[mask]
        args = (coords[:, 0], coords[:, 1], coords[:, 2]) if three_d else (
            coords[:, 0],
            coords[:, 1],
        )
        ax.scatter(*args, color=cmap(i % 10), s=18, label=cat, alpha=0.85)
    return None


def plot_pca(
    pca: PCAResult,
    rows: list[QueryRow],
    color_by: str,
    title: str,
    out_path: Path,
    three_d: bool = False,
) -> Path:
    pts = pca.projected
    if three_d and pts.shape[1] < 3:
        raise ValueError("Need >=3 PCA components for a 3D plot")

    fig = plt.figure(figsize=(8, 6.5))
    ax = fig.add_subplot(111, projection="3d" if three_d else None)
    sc = _scatter(ax, pts, rows, color_by, three_d)

    evr = pca.explained_variance_ratio
    ax.set_xlabel(f"PC1 ({evr[0]:.0%})")
    ax.set_ylabel(f"PC2 ({evr[1]:.0%})")
    if three_d:
        ax.set_zlabel(f"PC3 ({evr[2]:.0%})" if len(evr) > 2 else "PC3")
    ax.set_title(title, fontsize=10)
    if sc is not None:
        cbar = fig.colorbar(sc, ax=ax, shrink=0.75, pad=0.1 if three_d else 0.02)
        cbar.set_label(f"{color_by} (years, log)")
    else:
        handles, labels = ax.get_legend_handles_labels()
        if handles and len(labels) <= 14:
            ax.legend(fontsize=7, loc="best", framealpha=0.6)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)
    return out_path
