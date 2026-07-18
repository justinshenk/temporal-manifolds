"""PCA + 2D/3D visualization of boundary activations from a stored run.

Usage:
    uv run python experiments/analyze_geometry.py --run <run_id> --model <model_id>
    # optional: --kinds turn_end,role  --colorings target_horizon,prompt_id

For every (layer/depth × boundary kind) slice it fits a PCA over all matching
vectors and renders 2D and 3D scatters for each coloring. Also renders
"assistant-boundary" combined plots (all boundary kinds at assistant turn
starts pooled) and per-turn-position plots. Figures ->
output/runs/<run_id>/<model>/figures/.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1]))

from src.analysis.pca import fit_pca
from src.analysis.viz import plot_pca
from src.capture.store import ResponseStore
from src.core.paths import sanitize_model_name

DEFAULT_COLORINGS = (
    "target_horizon",
    "step_horizon",
    "prompt_id",
    "task_id",
    "turn_index",
)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run", required=True, help="run_id under output/runs/")
    ap.add_argument("--model", required=True)
    ap.add_argument("--kinds", default="turn_end,turn_start,role,think_close")
    ap.add_argument("--colorings", default=",".join(DEFAULT_COLORINGS))
    ap.add_argument("--min-points", type=int, default=8)
    args = ap.parse_args()

    store = ResponseStore(args.run)
    fig_dir = store.model_dir(args.model) / "figures"
    kinds = [k.strip() for k in args.kinds.split(",") if k.strip()]
    colorings = [c.strip() for c in args.colorings.split(",") if c.strip()]

    uids = store.sample_uids(args.model)
    if not uids:
        print(f"No samples for model {args.model} in run {args.run}")
        return 1
    print(f"[analyze] {len(uids)} samples; figures -> {fig_dir}")

    # discover layers from one sample
    binfo = store.load_boundaries(args.model, uids[0])
    depth_to_layer = {float(k): v for k, v in binfo["depth_to_layer"].items()}

    written: list[str] = []
    manifest: list[dict] = []

    def render(X, rows, slice_name: str):
        if X.shape[0] < args.min_points:
            print(f"[analyze] skip {slice_name}: only {X.shape[0]} points")
            return
        pca = fit_pca(X, n_components=3)
        for coloring in colorings:
            for three_d in (False, True):
                suffix = "3d" if three_d else "2d"
                path = fig_dir / f"{slice_name}__{coloring}__{suffix}.png"
                title = (
                    f"{slice_name} | color={coloring} | n={X.shape[0]} | "
                    f"model={sanitize_model_name(args.model)}"
                )
                plot_pca(pca, rows, coloring, title, path, three_d=three_d)
                written.append(str(path))
                manifest.append(
                    {
                        "figure": str(path),
                        "slice": slice_name,
                        "coloring": coloring,
                        "n_points": int(X.shape[0]),
                        "evr": [float(v) for v in pca.explained_variance_ratio],
                    }
                )
        print(
            f"[analyze] {slice_name}: n={X.shape[0]} "
            f"evr={[round(float(v), 3) for v in pca.explained_variance_ratio]}"
        )

    for depth, layer in sorted(depth_to_layer.items()):
        # per boundary-kind slices (assistant turns are where plans live)
        for kind in kinds:
            X, rows = store.query(args.model, layer=layer, kind=kind)
            render(X, rows, f"d{int(depth * 100)}_L{layer}_{kind}")
        # pooled assistant-turn boundary tokens
        X, rows = store.query(args.model, layer=layer, role="assistant")
        render(X, rows, f"d{int(depth * 100)}_L{layer}_assistant-all")

    fig_dir.mkdir(parents=True, exist_ok=True)
    with open(fig_dir / "figures_manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"[analyze] wrote {len(written)} figures + figures_manifest.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
