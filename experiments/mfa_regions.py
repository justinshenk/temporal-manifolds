"""Region decomposition of boundary activations via Mixture of Factor Analyzers.

Applies the method of Shafran et al., "From Directions to Regions: Decomposing
Activations in Language Models via Local Geometry" (arXiv:2602.02464, ICML
2026) to this project's boundary-token activations, using the authors'
released MFA implementation (github.com/ordavid-s/decomposing-activations-
local-geometry, cloned locally and passed via --mfa-repo).

What it does:
  1. loads all assistant boundary-token activations of one run at one layer
     (all kinds pooled — the corpus analog of their residual-stream fit),
  2. k-means-initializes and gradient-trains an MFA (their model + trainer),
  3. assigns each activation to a region (argmax posterior responsibility),
  4. asks the paper's interpretability question against OUR variables:
     do regions align with token kind / task / conversation phase / horizon?
     (NMI + per-region composition), and
  5. asks the local-geometry question: inside each region, do the local
     factor coordinates E[z|x] encode log step horizon? (per-region best
     factor correlation — the "local direction" analog.)

Usage:
    uv run python experiments/mfa_regions.py --run 9ccf36267612 \
        --model mlx-community/Qwen3-32B-4bit --layer-depth 0.6 \
        --mfa-repo <path-to-cloned-repo> [--k 24] [--rank 8] [--epochs 30]
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parents[1]))

from src.capture.store import ResponseStore

KINDS = ["turn_end", "turn_start", "role", "think_open", "think_close"]


def horizon_band(years: float | None) -> str:
    if years is None or years <= 0:
        return "none"
    days = years * 365.25
    if days < 1:
        return "sub-day"
    if days < 30:
        return "days-weeks"
    if years < 1:
        return "months"
    if years < 10:
        return "years"
    return "decades+"


def phase_band(turn: int) -> str:
    if turn <= 1:
        return "opening"
    if turn <= 7:
        return "early"
    if turn <= 13:
        return "mid"
    return "late"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--layer-depth", type=float, default=0.6)
    ap.add_argument("--mfa-repo", required=True)
    ap.add_argument("--k", type=int, default=24)
    ap.add_argument("--rank", type=int, default=8)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    sys.path.insert(0, str(Path(args.mfa_repo).resolve()))
    from modeling.mfa import MFA  # noqa: E402 (paper repo)

    from scipy.stats import spearmanr
    from sklearn.cluster import KMeans
    from sklearn.metrics import normalized_mutual_info_score as nmi

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    store = ResponseStore(args.run)
    uids = store.sample_uids(args.model)
    binfo = store.load_boundaries(args.model, uids[0])
    d2l = {float(k): v for k, v in binfo["depth_to_layer"].items()}
    layer = d2l[args.layer_depth]

    # ---- gather activations + labels -------------------------------------
    X_parts, labels = [], defaultdict(list)
    for kind in KINDS:
        X, rows = store.query(args.model, layer=layer, kind=kind, role="assistant")
        if not rows:
            continue
        X_parts.append(X)
        for r in rows:
            labels["kind"].append(kind)
            labels["task"].append(r.task_id)
            labels["phase"].append(phase_band(r.turn_index))
            labels["total_band"].append(horizon_band(r.target_horizon_years))
            labels["step_band"].append(horizon_band(r.step_horizon_years))
            labels["log_step"].append(
                np.log(r.step_horizon_years)
                if r.step_horizon_years and r.step_horizon_years > 0
                else np.nan
            )
    X = np.concatenate(X_parts).astype(np.float32)
    log_step = np.array(labels.pop("log_step"))
    n, D = X.shape
    print(f"[mfa] {n} assistant boundary activations, D={D}, layer {layer}")

    # ---- fit (paper's recipe at our scale) --------------------------------
    km = KMeans(n_clusters=args.k, n_init=4, random_state=args.seed).fit(X)
    centroids = torch.tensor(km.cluster_centers_, dtype=torch.float32)
    model = MFA(centroids, rank=args.rank, psi_init=1.0)

    Xt = torch.tensor(X)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    bs = 512
    for ep in range(args.epochs):
        perm = torch.randperm(n)
        tot = 0.0
        for i in range(0, n, bs):
            xb = Xt[perm[i : i + bs]]
            opt.zero_grad(set_to_none=True)
            nll = model.nll(xb)
            nll.backward()
            opt.step()
            tot += nll.item() * len(xb)
        if ep % 5 == 0 or ep == args.epochs - 1:
            print(f"[mfa] epoch {ep:3d} nll/point = {tot / n:.1f}")

    # ---- region assignment + alignment ------------------------------------
    model.eval()
    with torch.no_grad():
        resp_parts, ez_parts = [], []
        for i in range(0, n, 2048):
            xb = Xt[i : i + 2048]
            resp_parts.append(model.responsibilities(xb))
            ez_parts.append(model.component_posterior(xb)[0])
        resp = torch.cat(resp_parts)          # (n, K)
        Ez = torch.cat(ez_parts)              # (n, K, q)
    region = resp.argmax(1).numpy()
    conf = resp.max(1).values.numpy()

    print(f"\n[mfa] region occupancy (K={args.k}):",
          dict(sorted(Counter(region.tolist()).items())))
    print(f"[mfa] mean assignment confidence: {conf.mean():.3f}")

    results = {"run": args.run, "model": args.model, "layer": layer,
               "K": args.k, "rank": args.rank, "n": int(n),
               "mean_confidence": float(conf.mean()), "nmi": {}, "regions": []}

    print("\n[mfa] region <-> variable alignment (NMI):")
    for var, vals in labels.items():
        score = nmi(region, np.array(vals))
        results["nmi"][var] = float(score)
        print(f"  {var:12s} NMI = {score:.3f}")

    # ---- per-region composition + local horizon directions ------------------
    print("\n[mfa] per-region: majority labels + best local-factor ~ log step horizon")
    for k in sorted(set(region.tolist())):
        mask = region == k
        row = {"region": int(k), "n": int(mask.sum())}
        for var, vals in labels.items():
            top, cnt = Counter(np.array(vals)[mask].tolist()).most_common(1)[0]
            row[var] = f"{top} ({cnt / mask.sum():.0%})"
        ok = mask & np.isfinite(log_step)
        if ok.sum() >= 40:
            best = (0.0, -1)
            for j in range(args.rank):
                z = Ez[ok, k, j].numpy()
                rho, p = spearmanr(z, log_step[ok])
                if abs(rho) > abs(best[0]):
                    best = (float(rho), j)
            row["local_horizon_factor"] = {"factor": best[1], "rho": best[0],
                                           "n_steps": int(ok.sum())}
        results["regions"].append(row)
        lf = row.get("local_horizon_factor")
        lf_s = (f" | local z{lf['factor']} ~ log step horizon rho={lf['rho']:+.2f}"
                f" (n={lf['n_steps']})" if lf else "")
        print(f"  R{k:02d} n={row['n']:5d} kind={row['kind']:22s} "
              f"phase={row['phase']:14s} task={row['task']:22s}{lf_s}")

    out_dir = store.model_dir(args.model) / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "mfa_regions.json", "w") as f:
        json.dump(results, f, indent=2)
    np.savez(out_dir / "mfa_assignment.npz", region=region, confidence=conf)
    print(f"\n[mfa] wrote {out_dir / 'mfa_regions.json'} and mfa_assignment.npz")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
