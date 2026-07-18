"""Linear probes: read the time horizon out of boundary activations.

For each (layer × boundary kind × target):
    target-horizon probe   rows = first-assistant-turn boundaries,
                           y = log(target_horizon_years)
    step-horizon probe     rows = step-expansion turns with a parsed
                           "Time horizon:", y = log(step_horizon_years)

Ridge regression (dual form, alpha swept on train), two evaluation schemes:
    grouped-cv    5-fold CV grouped by conversation (no conversation appears
                  in both train and test)
    loto          leave-one-task-out (cross-task generalization, raw features
                  — the hard test)

Metrics: held-out R^2 and Spearman rho, aggregated over folds.

Usage:
    uv run python experiments/probe_horizon.py --run <run_id> --model <model_id>
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

sys.path.insert(0, str(Path(__file__).parents[1]))

from src.capture.store import ResponseStore

KINDS = ("turn_end", "role", "turn_start", "think_close")
ALPHAS = (1e0, 1e1, 1e2, 1e3, 1e4, 1e5)


def ridge_fit_predict(Xtr, ytr, Xte, alpha):
    """Dual ridge: predictions for Xte. Standardizes on train."""
    mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-6
    Xtr = (Xtr - mu) / sd
    Xte = (Xte - mu) / sd
    ym = ytr.mean()
    yc = ytr - ym
    n = Xtr.shape[0]
    K = Xtr @ Xtr.T
    beta = np.linalg.solve(K + alpha * np.eye(n), yc)
    return Xte @ (Xtr.T @ beta) + ym


def pick_alpha(Xtr, ytr, groups_tr):
    """Inner 3-fold grouped CV on train to select alpha."""
    uniq = sorted(set(groups_tr))
    if len(uniq) < 3:
        return ALPHAS[len(ALPHAS) // 2]
    folds = [uniq[i::3] for i in range(3)]
    best, best_err = ALPHAS[0], np.inf
    for alpha in ALPHAS:
        errs = []
        for fold in folds:
            te = np.array([g in fold for g in groups_tr])
            if te.all() or (~te).sum() < 3 or te.sum() == 0:
                continue
            pred = ridge_fit_predict(Xtr[~te], ytr[~te], Xtr[te], alpha)
            errs.append(np.mean((pred - ytr[te]) ** 2))
        if errs and np.mean(errs) < best_err:
            best, best_err = alpha, float(np.mean(errs))
    return best

def evaluate(X, y, conv_groups, task_groups, scheme):
    """Returns pooled held-out predictions and fold assignment."""
    preds = np.full_like(y, np.nan)
    if scheme == "grouped-cv":
        uniq = sorted(set(conv_groups))
        folds = [set(uniq[i::5]) for i in range(5)]
        group_arr = conv_groups
    elif scheme == "loto":
        uniq = sorted(set(task_groups))
        folds = [{t} for t in uniq]
        group_arr = task_groups
    else:
        raise ValueError(scheme)
    for fold in folds:
        te = np.array([g in fold for g in group_arr])
        if te.sum() == 0 or (~te).sum() < 5:
            continue
        alpha = pick_alpha(X[~te], y[~te], [g for g, m in zip(conv_groups, te) if not m])
        preds[te] = ridge_fit_predict(X[~te], y[~te], X[te], alpha)
    return preds


def score(y, preds):
    ok = np.isfinite(preds)
    if ok.sum() < 5:
        return None
    ss_res = np.sum((y[ok] - preds[ok]) ** 2)
    ss_tot = np.sum((y[ok] - y[ok].mean()) ** 2)
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else np.nan
    rho, p = spearmanr(y[ok], preds[ok])
    return {"n": int(ok.sum()), "r2": float(r2), "rho": float(rho), "p": float(p)}


def gather(store, model, layer, kind, target):
    X, rows = store.query(model, layer=layer, kind=kind, role="assistant")
    if target == "target":
        keep = [i for i, r in enumerate(rows) if r.turn_index == 1]
        y = [rows[i].target_horizon_years for i in keep]
    else:
        keep = [
            i
            for i, r in enumerate(rows)
            if r.step_index is not None
            and r.step_horizon_years is not None
            and r.step_horizon_years > 0
        ]
        y = [rows[i].step_horizon_years for i in keep]
    if len(keep) < 12:
        return None
    X = X[keep]
    y = np.log(np.array(y, dtype=np.float64))
    convs = [rows[i].sample_uid for i in keep]
    tasks = [rows[i].task_id for i in keep]
    return X.astype(np.float64), y, convs, tasks


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
    binfo = store.load_boundaries(args.model, uids[0])
    depth_to_layer = {float(k): v for k, v in binfo["depth_to_layer"].items()}

    results = []
    for depth, layer in sorted(depth_to_layer.items()):
        for kind in KINDS:
            for target in ("target", "step"):
                data = gather(store, args.model, layer, kind, target)
                if data is None:
                    continue
                X, y, convs, tasks = data
                for scheme in ("grouped-cv", "loto"):
                    preds = evaluate(X, y, convs, tasks, scheme)
                    s = score(y, preds)
                    if s is None:
                        continue
                    results.append(
                        {
                            "depth": depth,
                            "layer": layer,
                            "kind": kind,
                            "target": target,
                            "scheme": scheme,
                            **s,
                        }
                    )

    out_dir = store.model_dir(args.model) / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "probe_results.json", "w") as f:
        json.dump(results, f, indent=2)

    results.sort(key=lambda r: -r["r2"])
    print(f"[probe] {args.model} run={args.run}: {len(results)} probe evals")
    print(f"{'depth':>5} {'layer':>5} {'kind':>12} {'tgt':>6} {'scheme':>10} "
          f"{'n':>4} {'R2':>7} {'rho':>6} {'p':>9}")
    for r in results[:14]:
        print(
            f"{r['depth']:>5} {r['layer']:>5} {r['kind']:>12} {r['target']:>6} "
            f"{r['scheme']:>10} {r['n']:>4} {r['r2']:>7.3f} {r['rho']:>6.2f} "
            f"{r['p']:>9.1e}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
