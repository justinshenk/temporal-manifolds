"""Focused horizon-geometry analysis: is the target time horizon geometrically
organized at change-of-turn tokens, and how does it evolve across turns?

Controls that the raw PCA (analyze_geometry.py) lacks:
  - slices condition on kind x role(assistant) x turn position
  - per-task mean-centering removes task identity before pooling
  - per-conversation TRAJECTORIES: the same boundary kind tracked across
    assistant turns, projected into a shared PCA basis

Usage:
    uv run python experiments/horizon_geometry.py --run <run_id> --model <model_id>

Figures -> output/runs/<run>/<model>/figures/horizon/
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parents[1]))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import colormaps
from matplotlib.colors import LogNorm

from src.analysis.pca import fit_pca
from src.capture.store import ResponseStore

KINDS = ("turn_end", "role", "think_close", "turn_start")


def task_centered(X: np.ndarray, rows) -> np.ndarray:
    """Subtract each task's mean vector (removes task identity)."""
    Xc = X.copy()
    by_task = defaultdict(list)
    for i, r in enumerate(rows):
        by_task[r.task_id].append(i)
    for idxs in by_task.values():
        Xc[idxs] -= X[idxs].mean(axis=0)
    return Xc


def horizon_colors(rows, field="target_horizon_years"):
    vals = np.array(
        [np.nan if getattr(r, field) is None else float(getattr(r, field)) for r in rows]
    )
    return vals


def scatter_horizon(ax, pts, vals, rows, marker_by_task=True):
    finite = np.isfinite(vals) & (vals > 0)
    norm = None
    sc = None
    if finite.any():
        norm = LogNorm(vals[finite].min(), max(vals[finite].max(), vals[finite].min() * 1.0001))
    markers = "o^sDv*P"
    tasks = sorted({r.task_id for r in rows})
    for t_i, task in enumerate(tasks):
        mask = np.array([r.task_id == task for r in rows]) & finite
        if not mask.any():
            continue
        sc = ax.scatter(
            pts[mask, 0],
            pts[mask, 1],
            c=vals[mask],
            cmap="viridis",
            norm=norm,
            s=42,
            marker=markers[t_i % len(markers)] if marker_by_task else "o",
            edgecolors="k",
            linewidths=0.3,
            label=task,
        )
    if (~finite).any():
        ax.scatter(
            pts[~finite, 0], pts[~finite, 1], c="lightgray", s=20, alpha=0.5
        )
    return sc


def fig_slice(store, model, layer, depth, kind, turn_sel, fig_dir, tag, results):
    """One controlled slice: assistant boundaries of `kind`, task-centered."""
    X, rows = store.query(model, layer=layer, kind=kind, role="assistant")
    if turn_sel == "first":
        keep = [i for i, r in enumerate(rows) if r.turn_index == 1]
    elif turn_sel == "steps":
        keep = [i for i, r in enumerate(rows) if r.step_index is not None]
    else:
        keep = list(range(len(rows)))
    if len(keep) < 8:
        return
    X, rows = X[keep], [rows[i] for i in keep]
    Xc = task_centered(X, rows)
    pca = fit_pca(Xc, n_components=3)

    for field, fname in (
        ("target_horizon_years", "target_horizon"),
        ("step_horizon_years", "step_horizon"),
    ):
        vals = horizon_colors(rows, field)
        if not (np.isfinite(vals) & (vals > 0)).any():
            continue
        fig, ax = plt.subplots(figsize=(8, 6.5))
        sc = scatter_horizon(ax, pca.projected, vals, rows)
        evr = pca.explained_variance_ratio
        ax.set_xlabel(f"PC1 ({evr[0]:.0%})")
        ax.set_ylabel(f"PC2 ({evr[1]:.0%})")
        ax.set_title(
            f"{tag} {kind} ({turn_sel} turns, task-centered) | color={fname} | "
            f"n={len(rows)}",
            fontsize=10,
        )
        if sc is not None:
            fig.colorbar(sc, ax=ax, shrink=0.8).set_label(f"{fname} (years, log)")
        ax.legend(fontsize=7, title="task (marker)", framealpha=0.6)
        out = fig_dir / f"{tag}_{kind}_{turn_sel}__{fname}.png"
        fig.tight_layout()
        fig.savefig(out, dpi=140)
        plt.close(fig)

        # quantify: rank-correlation of log-horizon with each of PC1..3
        finite = np.isfinite(vals) & (vals > 0)
        corr = {}
        if finite.sum() >= 5:
            from scipy.stats import spearmanr

            for pc in range(min(3, pca.projected.shape[1])):
                rho, p = spearmanr(
                    np.log(vals[finite]), pca.projected[finite, pc]
                )
                corr[f"PC{pc + 1}"] = {"rho": float(rho), "p": float(p)}
        results.append(
            {
                "slice": f"{tag}_{kind}_{turn_sel}",
                "coloring": fname,
                "n": len(rows),
                "evr": [float(v) for v in pca.explained_variance_ratio],
                "spearman_log_horizon": corr,
                "figure": str(out),
            }
        )


def fig_trajectories(store, model, layer, depth, kind, fig_dir, tag, results):
    """Per-conversation trajectory of `kind` across assistant turns, in a
    shared task-centered PCA basis; colored by target horizon."""
    X, rows = store.query(model, layer=layer, kind=kind, role="assistant")
    if len(rows) < 8:
        return
    Xc = task_centered(X, rows)
    pca = fit_pca(Xc, n_components=2)
    pts = pca.projected

    by_conv = defaultdict(list)
    for i, r in enumerate(rows):
        by_conv[r.sample_uid].append(i)

    vals_all = horizon_colors(rows)
    finite_vals = vals_all[np.isfinite(vals_all) & (vals_all > 0)]
    if finite_vals.size == 0:
        return
    norm = LogNorm(finite_vals.min(), max(finite_vals.max(), finite_vals.min() * 1.0001))
    cmap = colormaps["viridis"]

    fig, ax = plt.subplots(figsize=(8.5, 7))
    for uid, idxs in by_conv.items():
        idxs = sorted(idxs, key=lambda i: rows[i].turn_index)
        p = pts[idxs]
        h = rows[idxs[0]].target_horizon_years
        color = cmap(norm(h)) if h and h > 0 else (0.7, 0.7, 0.7, 1)
        ax.plot(p[:, 0], p[:, 1], "-", color=color, alpha=0.55, lw=1.2)
        ax.scatter(p[0, 0], p[0, 1], color=color, marker="o", s=45, ec="k", lw=0.5)
        ax.scatter(p[-1, 0], p[-1, 1], color=color, marker="X", s=55, ec="k", lw=0.5)
    evr = pca.explained_variance_ratio
    ax.set_xlabel(f"PC1 ({evr[0]:.0%})")
    ax.set_ylabel(f"PC2 ({evr[1]:.0%})")
    ax.set_title(
        f"{tag} {kind}: conversation trajectories (o=first turn, X=last), "
        "task-centered, color=target horizon",
        fontsize=10,
    )
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    fig.colorbar(sm, ax=ax, shrink=0.8).set_label("target horizon (years, log)")
    out = fig_dir / f"{tag}_{kind}_trajectories.png"
    fig.tight_layout()
    fig.savefig(out, dpi=140)
    plt.close(fig)
    results.append(
        {"slice": f"{tag}_{kind}_trajectories", "n_convs": len(by_conv), "figure": str(out)}
    )


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run", required=True)
    ap.add_argument("--model", required=True)
    args = ap.parse_args()

    store = ResponseStore(args.run)
    uids = store.sample_uids(args.model)
    if not uids:
        print("no samples")
        return 1
    fig_dir = store.model_dir(args.model) / "figures" / "horizon"
    fig_dir.mkdir(parents=True, exist_ok=True)

    binfo = store.load_boundaries(args.model, uids[0])
    depth_to_layer = {float(k): v for k, v in binfo["depth_to_layer"].items()}

    results: list[dict] = []
    for depth, layer in sorted(depth_to_layer.items()):
        tag = f"d{int(depth * 100)}_L{layer}"
        for kind in KINDS:
            for turn_sel in ("first", "steps", "all"):
                fig_slice(
                    store, args.model, layer, depth, kind, turn_sel, fig_dir, tag, results
                )
            fig_trajectories(store, args.model, layer, depth, kind, fig_dir, tag, results)

    with open(fig_dir / "horizon_results.json", "w") as f:
        json.dump(results, f, indent=2)
    n_figs = len([r for r in results if "figure" in r])
    print(f"[horizon] {len(uids)} samples -> {n_figs} figures in {fig_dir}")
    # print the strongest correlations
    scored = [
        (
            max(
                (abs(c["rho"]) for c in r.get("spearman_log_horizon", {}).values()),
                default=0.0,
            ),
            r,
        )
        for r in results
        if r.get("spearman_log_horizon")
    ]
    scored.sort(reverse=True, key=lambda x: x[0])
    for rho, r in scored[:10]:
        best = max(
            r["spearman_log_horizon"].items(), key=lambda kv: abs(kv[1]["rho"])
        )
        print(
            f"[horizon] |rho|={rho:.2f} {r['slice']} ({r['coloring']}) "
            f"{best[0]} p={best[1]['p']:.1e} n={r['n']}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
