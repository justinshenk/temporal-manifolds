"""Time-SCALE classification probe: can activations classify a step's
horizon into human-natural log-spaced buckets?

Classes (boundaries in years):
    hours    < 1 day        (< 1/365)
    days     < 1 week       (< 7/365)
    weeks    < 1 month      (< 1/12)
    months   < 1 year
    years    1 - 10 y
    decades+ >= 10 y

Multinomial logistic probe (standardized activations, C swept by inner
grouped CV). Evaluations per (layer x kind):
    within-target   grouped-CV by conversation on the TARGET run
    transfer        train on all TARGET rows, test on CONTROL rows
    theme split     per task: accuracy on that task's CONTROL rows when the
                    task's TARGET data was IN training (transfer) vs OUT
                    (train on the other tasks only)

Metrics: accuracy, balanced accuracy, adjacent accuracy (|pred-true| <= 1
class on the log scale), majority-class baseline, confusion matrix.

Usage:
    uv run python experiments/probe_timescale.py \
        --target-run <run_id> --control-run <run_id> --model <model_id>
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression

sys.path.insert(0, str(Path(__file__).parents[1]))
sys.path.insert(0, str(Path(__file__).parent))

from probe_control import gather_steps

from src.capture.store import ResponseStore

CLASSES = ["hours", "days", "weeks", "months", "years", "decades+"]
BOUNDS = [1 / 365, 7 / 365, 1 / 12, 1.0, 10.0]  # upper bounds, in years
C_GRID = (1e-3, 1e-2, 1e-1, 1.0)
KINDS = ("turn_start", "role", "turn_end")


def to_class(years: float) -> int:
    for i, b in enumerate(BOUNDS):
        if years < b:
            return i
    return len(BOUNDS)


def fit_logreg(Xtr, ctr, C):
    mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-6
    clf = LogisticRegression(C=C, max_iter=3000, solver="lbfgs")
    clf.fit((Xtr - mu) / sd, ctr)
    return clf, mu, sd


def pick_C(Xtr, ctr, groups):
    uniq = sorted(set(groups))
    if len(uniq) < 3 or len(set(ctr)) < 2:
        return C_GRID[1]
    folds = [set(uniq[i::3]) for i in range(3)]
    best, best_acc = C_GRID[0], -1.0
    for C in C_GRID:
        accs = []
        for fold in folds:
            te = np.array([g in fold for g in groups])
            if te.all() or te.sum() == 0 or len(set(ctr[~te])) < 2:
                continue
            clf, mu, sd = fit_logreg(Xtr[~te], ctr[~te], C)
            accs.append(float((clf.predict((Xtr[te] - mu) / sd) == ctr[te]).mean()))
        if accs and np.mean(accs) > best_acc:
            best, best_acc = C, float(np.mean(accs))
    return best


def metrics(true, pred):
    true, pred = np.asarray(true), np.asarray(pred)
    n = len(true)
    if n == 0:
        return None
    acc = float((pred == true).mean())
    adj = float((np.abs(pred - true) <= 1).mean())
    recalls = [float((pred[true == c] == c).mean())
               for c in sorted(set(true.tolist())) if (true == c).sum() > 0]
    bal = float(np.mean(recalls)) if recalls else float("nan")
    base = float(max(np.bincount(true, minlength=len(CLASSES))) / n)
    K = len(CLASSES)
    conf = [[int(((true == i) & (pred == j)).sum()) for j in range(K)]
            for i in range(K)]
    return {"n": n, "acc": round(acc, 3), "balanced_acc": round(bal, 3),
            "adjacent_acc": round(adj, 3), "majority_baseline": round(base, 3),
            "confusion": conf}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--target-run", required=True)
    ap.add_argument("--control-run", required=True)
    ap.add_argument("--model", required=True)
    args = ap.parse_args()

    tstore = ResponseStore(args.target_run)
    cstore = ResponseStore(args.control_run)
    uids = cstore.sample_uids(args.model)
    binfo = cstore.load_boundaries(args.model, uids[0])
    depth_to_layer = {float(k): v for k, v in binfo["depth_to_layer"].items()}

    results = []
    for depth, layer in sorted(depth_to_layer.items()):
        for kind in KINDS:
            tgt = gather_steps(tstore, args.model, layer, kind)
            ctl = gather_steps(cstore, args.model, layer, kind,
                               require_source="assigned")
            if tgt is None or ctl is None:
                continue
            Xt, yt, metat = tgt
            Xc, yc, metac = ctl
            ct = np.array([to_class(float(np.exp(v))) for v in yt])
            cc = np.array([to_class(float(np.exp(v))) for v in yc])
            t_groups = [m.sample_uid for m in metat]
            t_tasks = np.array([m.task_id for m in metat])
            c_tasks = np.array([m.task_id for m in metac])

            # within-target grouped-CV
            uniq = sorted(set(t_groups))
            folds = [set(uniq[i::5]) for i in range(5)]
            preds = np.full(len(ct), -1)
            for fold in folds:
                te = np.array([g in fold for g in t_groups])
                if te.sum() == 0 or len(set(ct[~te])) < 2:
                    continue
                C = pick_C(Xt[~te], ct[~te],
                           [g for g, m_ in zip(t_groups, te) if not m_])
                clf, mu, sd = fit_logreg(Xt[~te], ct[~te], C)
                preds[te] = clf.predict((Xt[te] - mu) / sd)
            ok = preds >= 0
            m = metrics(ct[ok], preds[ok])
            if m:
                results.append(dict(depth=depth, layer=layer, kind=kind,
                                    test="within-target/grouped-cv", **m))

            # transfer: train all target, test control
            C = pick_C(Xt, ct, t_groups)
            clf, mu, sd = fit_logreg(Xt, ct, C)
            cpred = clf.predict((Xc - mu) / sd)
            m = metrics(cc, cpred)
            if m:
                results.append(dict(depth=depth, layer=layer, kind=kind,
                                    test="transfer", **m))

            # theme split: same-theme (in-training) vs held-out-theme
            theme = {}
            for task in sorted(set(c_tasks)):
                te = c_tasks == task
                if te.sum() < 8:
                    continue
                in_acc = float((cpred[te] == cc[te]).mean())
                tr = t_tasks != task
                if tr.sum() < 12 or len(set(ct[tr])) < 2:
                    continue
                C2 = pick_C(Xt[tr], ct[tr],
                            [g for g, keep in zip(t_groups, tr) if keep])
                clf2, mu2, sd2 = fit_logreg(Xt[tr], ct[tr], C2)
                out_pred = clf2.predict((Xc[te] - mu2) / sd2)
                theme[task] = {
                    "n": int(te.sum()),
                    "same_theme_acc": round(in_acc, 3),
                    "held_out_theme_acc": round(float((out_pred == cc[te]).mean()), 3),
                    "held_out_adjacent": round(float((np.abs(out_pred - cc[te]) <= 1).mean()), 3),
                }
            if theme:
                results.append(dict(depth=depth, layer=layer, kind=kind,
                                    test="theme-split", per_task=theme))

    out_dir = cstore.model_dir(args.model) / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "probe_timescale_results.json", "w") as f:
        json.dump({"classes": CLASSES, "bounds_years": BOUNDS,
                   "results": results}, f, indent=2)

    print(f"[timescale] {args.model} {args.target_run}->{args.control_run}")
    print(f"{'depth':>5} {'kind':>10} {'test':>24} {'n':>5} {'acc':>6} "
          f"{'bal':>6} {'adj':>6} {'base':>6}")
    for r in results:
        if "acc" in r:
            print(f"{r['depth']:>5} {r['kind']:>10} {r['test']:>24} "
                  f"{r['n']:>5} {r['acc']:>6.2f} {r['balanced_acc']:>6.2f} "
                  f"{r['adjacent_acc']:>6.2f} {r['majority_baseline']:>6.2f}")
    for r in results:
        if r.get("test") == "theme-split":
            print(f"  theme-split d{r['depth']} {r['kind']}: " + " | ".join(
                f"{t}: same {v['same_theme_acc']} vs held-out {v['held_out_theme_acc']} (adj {v['held_out_adjacent']})"
                for t, v in r["per_task"].items()))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
