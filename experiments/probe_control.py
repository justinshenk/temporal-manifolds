"""Does the probe recover step time-targets the model never verbalized?

Control condition (step_mode="target_control"): steps are expanded with no
time information; a final "Time assignments:" turn assigns each step's target
offset retrospectively. If step-boundary activations encode the step's
temporal position, a probe should read it out even though nothing temporal
was written in the step turn.

Four tests per (layer x kind):
    within-control  ridge on control activations vs log(assigned target),
                    grouped-CV by conversation + leave-one-task-out
    transfer        probe TRAINED on the verbalized-target run's step rows
                    (y = stated target), TESTED on control rows
                    (y = assigned target) — the headline recovery test
    transfer-loto   theme-controlled transfer: train on TARGET rows of the
                    OTHER tasks only, test on CONTROL rows of the held-out
                    task. Removes the shared task-theme confound (target and
                    control conversations of the same task share content
                    that correlates with time); the gap between transfer and
                    transfer-loto measures the theme contribution.
    leak split      transfer metrics on strictly time-silent control steps
                    vs steps containing incidental time words (cadence)

Usage:
    uv run python experiments/probe_control.py \
        --target-run <run_id> --control-run <run_id> --model <model_id>
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parents[1]))
sys.path.insert(0, str(Path(__file__).parent))

from probe_horizon import evaluate, pick_alpha, ridge_fit_predict, score

from src.capture.store import ResponseStore

KINDS = ("turn_end", "role", "turn_start")
TIME_WORD_RE = re.compile(r"\b(hour|day|week|month|year|minute)s?\b", re.I)


def leak_map(store: ResponseStore, model: str) -> dict[tuple[str, int], bool]:
    """(sample_uid, turn_index) -> does the step turn contain time words."""
    leaks: dict[tuple[str, int], bool] = {}
    for uid in store.sample_uids(model):
        record = store.load_record(model, uid)
        for t in record.turns:
            if t.step_index is not None:
                leaks[(uid, t.turn_index)] = bool(TIME_WORD_RE.search(t.text))
    return leaks


def gather_steps(store, model, layer, kind, require_source=None):
    X, rows = store.query(model, layer=layer, kind=kind, role="assistant")
    keep = [
        i
        for i, r in enumerate(rows)
        if r.step_index is not None
        and r.step_horizon_years is not None
        and r.step_horizon_years > 0
        and (require_source is None or r.horizon_source == require_source)
    ]
    if len(keep) < 12:
        return None
    y = np.log(np.array([rows[i].step_horizon_years for i in keep]))
    meta = [rows[i] for i in keep]
    return X[keep].astype(np.float64), y, meta


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
    leaks = leak_map(cstore, args.model)

    results = []
    for depth, layer in sorted(depth_to_layer.items()):
        for kind in KINDS:
            ctl = gather_steps(
                cstore, args.model, layer, kind, require_source="assigned"
            )
            if ctl is None:
                continue
            Xc, yc, metac = ctl

            # -- within-control probes ----------------------------------
            convs = [m.sample_uid for m in metac]
            tasks = [m.task_id for m in metac]
            for scheme in ("grouped-cv", "loto"):
                s = score(yc, evaluate(Xc, yc, convs, tasks, scheme))
                if s:
                    results.append(
                        dict(depth=depth, layer=layer, kind=kind,
                             test=f"within/{scheme}", **s)
                    )

            # -- transfer: train on stated, test on assigned ------------
            tgt = gather_steps(tstore, args.model, layer, kind)
            if tgt is None:
                continue
            Xt, yt, metat = tgt
            alpha = pick_alpha(Xt, yt, [m.sample_uid for m in metat])
            preds = ridge_fit_predict(Xt, yt, Xc, alpha)
            s = score(yc, preds)
            if s:
                per_task = {}
                ctasks_arr = np.array([m.task_id for m in metac])
                for task in sorted(set(ctasks_arr)):
                    te = ctasks_arr == task
                    if te.sum() >= 8:
                        st = score(yc[te], preds[te])
                        if st:
                            per_task[task] = round(st["rho"], 2)
                results.append(
                    dict(depth=depth, layer=layer, kind=kind,
                         test="transfer", per_task_rho=per_task, **s)
                )
            # theme-controlled transfer: hold each task out of training
            # entirely (both conditions share a task's theme, so plain
            # transfer can ride on theme-time correlations)
            t_tasks = np.array([m.task_id for m in metat])
            c_tasks = np.array([m.task_id for m in metac])
            lpreds = np.full(len(yc), np.nan)
            for task in sorted(set(c_tasks)):
                tr = t_tasks != task
                te = c_tasks == task
                if tr.sum() < 12 or te.sum() == 0:
                    continue
                a2 = pick_alpha(
                    Xt[tr], yt[tr],
                    [m.sample_uid for m, keep in zip(metat, tr) if keep],
                )
                lpreds[te] = ridge_fit_predict(Xt[tr], yt[tr], Xc[te], a2)
            s = score(yc, lpreds)
            if s:
                per_task = {}
                for task in sorted(set(c_tasks)):
                    te = c_tasks == task
                    ok = te & np.isfinite(lpreds)
                    if ok.sum() >= 8:
                        st = score(yc[ok], lpreds[ok])
                        if st:
                            per_task[task] = round(st["rho"], 2)
                results.append(
                    dict(depth=depth, layer=layer, kind=kind,
                         test="transfer-loto", per_task_rho=per_task, **s)
                )

            # leak split on the same transfer predictions
            is_leaky = np.array(
                [leaks.get((m.sample_uid, m.turn_index), False) for m in metac]
            )
            for label, mask in (("silent", ~is_leaky), ("cadence", is_leaky)):
                if mask.sum() >= 12:
                    s = score(yc[mask], preds[mask])
                    if s:
                        results.append(
                            dict(depth=depth, layer=layer, kind=kind,
                                 test=f"transfer/{label}", **s)
                        )

    out_dir = cstore.model_dir(args.model) / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "probe_control_results.json", "w") as f:
        json.dump(results, f, indent=2)

    print(f"[control-probe] {args.model}")
    print(f"  target run {args.target_run} -> control run {args.control_run}")
    print(f"{'depth':>5} {'layer':>5} {'kind':>10} {'test':>17} "
          f"{'n':>5} {'R2':>7} {'rho':>6} {'p':>9}")
    for r in sorted(results, key=lambda r: (r["depth"], r["kind"], r["test"])):
        print(
            f"{r['depth']:>5} {r['layer']:>5} {r['kind']:>10} {r['test']:>17} "
            f"{r['n']:>5} {r['r2']:>7.3f} {r['rho']:>6.2f} {r['p']:>9.1e}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
