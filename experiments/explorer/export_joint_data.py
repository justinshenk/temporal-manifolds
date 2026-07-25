"""Export debug-explorer data: conversations, boundaries, PCA + t-SNE fits.

Run from the repo root:
    uv run python <scratchpad>/export_debug_data.py
"""

import json
import sys
import warnings
from collections import defaultdict
from pathlib import Path

import numpy as np

warnings.filterwarnings("ignore")
sys.path.insert(0, ".")
import umap
from sklearn.manifold import TSNE

from src.analysis.pca import fit_pca
from src.capture.store import ResponseStore
from src.conversation.parsing import _parse_duration_years, parse_time_assignments

# (name, engine label, run id, model id, default-in-UI)
MODELS = [
    ("Qwen3-32B ×5 · TARGET grid", "4-bit · MLX · T0.7 · 5 rollouts", "74c172105e25",
     "mlx-community/Qwen3-32B-4bit", True),
    ("Qwen3-32B ×5 · TARGET sweep (2 tasks)", "4-bit · MLX · T0.7 · target mode", "383cb5f89bda",
     "mlx-community/Qwen3-32B-4bit", False),
    ("Llama-3.1-8B ×5 · TARGET grid", "4-bit · MLX · T0.7 · 5 rollouts", "bcd154c149fb",
     "mlx-community/Meta-Llama-3.1-8B-Instruct-4bit", False),
    ("Gemma-2-9B ×5 · TARGET grid", "4-bit · MLX · T0.7 · 5 rollouts", "19665970da25",
     "mlx-community/gemma-2-9b-it-4bit", False),
    ("Qwen3-32B · CONTROL silent steps", "4-bit · MLX · T0.7 · 63 samples", "9d44c500dc1b",
     "mlx-community/Qwen3-32B-4bit", False),
    ("Llama-3.1-8B · CONTROL silent steps", "4-bit · MLX · T0.7", "397e77f6a867",
     "mlx-community/Meta-Llama-3.1-8B-Instruct-4bit", False),
    ("Gemma-2-9B · CONTROL silent steps", "4-bit · MLX · T0.7", "87b32dddd585",
     "mlx-community/gemma-2-9b-it-4bit", False),
]
KINDS = ["turn_end", "turn_start", "role", "think_close", "think_open"]
TSNE_SCOPES = ()  # t-SNE/UMAP removed for now per user request
SEED = 0

# probe (predicted horizon per step) ----------------------------------------
import importlib.util as _ilu

_spec = _ilu.spec_from_file_location(
    "probe_horizon", Path("experiments/probe_horizon.py").resolve()
)
_ph = _ilu.module_from_spec(_spec)
_spec.loader.exec_module(_ph)


def step_predictions(store, model_id, kinds_available, depth_to_layer):
    """Grouped-CV ridge predictions of log step horizon; {(si_uid, turn): years}."""
    kind = "think_close" if "think_close" in kinds_available else "role"
    layers = sorted(depth_to_layer.values())
    layer = layers[len(layers) // 2]  # 60% depth
    X, rows = store.query(model_id, layer=layer, kind=kind, role="assistant")
    keep = [i for i, r in enumerate(rows)
            if r.step_index is not None and r.step_horizon_years
            and r.step_horizon_years > 0]
    if len(keep) < 20:
        return {}, kind, layer
    Xk = X[keep].astype(np.float64)
    y = np.log([rows[i].step_horizon_years for i in keep])
    convs = [rows[i].sample_uid for i in keep]
    tasks = [rows[i].task_id for i in keep]
    preds = _ph.evaluate(Xk, y, convs, tasks, "grouped-cv")
    out = {}
    for j, i in enumerate(keep):
        if np.isfinite(preds[j]):
            out[(rows[i].sample_uid, rows[i].turn_index)] = float(np.exp(preds[j]))
    return out, kind, layer


def q(x):
    return int(round(x * 1000))


def norm_q(P):
    # robust scale: 97.5th-percentile radius maps to 1.6, hard cap at 3.2 —
    # a few extreme outlier activations must not compress the whole cloud
    # (global RMS did exactly that).
    r = np.sqrt((np.asarray(P) ** 2).sum(axis=1))
    s = float(np.quantile(r, 0.975)) / 1.6 or 1.0
    Pn = np.clip(np.asarray(P) / s, -3.2, 3.2)
    return [[q(v) for v in row] for row in Pn.tolist()]


def reduce50(X):
    if X.shape[1] > 50:
        return fit_pca(X, n_components=min(50, X.shape[0] - 1)).projected
    return X


def tsne_fit(X):
    n = X.shape[0]
    if n < 12:
        return None
    perp = float(min(30, max(5, (n - 1) // 3)))
    emb = TSNE(
        n_components=2, perplexity=perp, random_state=SEED, init="pca",
        max_iter=750,
    ).fit_transform(np.asarray(reduce50(X), dtype=np.float64))
    return emb, perp


def umap_fit(X):
    n = X.shape[0]
    if n < 12:
        return None
    nn = int(min(15, max(4, n - 2)))
    emb = umap.UMAP(
        n_components=2, n_neighbors=nn, min_dist=0.1, random_state=SEED,
    ).fit_transform(np.asarray(reduce50(X), dtype=np.float64))
    return emb, nn



out = {"models": []}
JOINT = [
    ("Qwen3-32B · TARGET + CONTROL", "4-bit · MLX · T0.7 · 5 rollouts",
     [("74c172105e25", "target"), ("9d44c500dc1b", "control")],
     "mlx-community/Qwen3-32B-4bit", True),
    ("Llama-3.1-8B · TARGET + CONTROL", "4-bit · MLX · T0.7 · 5 rollouts",
     [("bcd154c149fb", "target"), ("397e77f6a867", "control")],
     "mlx-community/Meta-Llama-3.1-8B-Instruct-4bit", False),
    ("Gemma-2-9B · TARGET + CONTROL", "4-bit · MLX · T0.7 · 5 rollouts",
     [("19665970da25", "target"), ("87b32dddd585", "control")],
     "mlx-community/gemma-2-9b-it-4bit", False),
]

for name, eng, run_conds, model_id, is_default in JOINT:
    stores = [(ResponseStore(run), cond) for run, cond in run_conds]
    store0 = stores[0][0]
    uids0 = store0.sample_uids(model_id)
    binfo0 = store0.load_boundaries(model_id, uids0[0])
    depth_to_layer = {float(k): v for k, v in binfo0["depth_to_layer"].items()}

    kinds_present = set()
    for store, cond in stores:
        b0 = store.load_boundaries(model_id, store.sample_uids(model_id)[0])
        kinds_present |= {b["kind"] for b in b0["boundary_map"]["boundaries"]}

    samples, key_to_i, preds_all = [], {}, {}
    for store, cond in stores:
        preds, pk, pl = step_predictions(store, model_id, kinds_present, depth_to_layer)
        for k, v in preds.items():
            preds_all[(cond,) + k] = v
        print(f"  {name} [{cond}]: {len(preds)} step preds ({pk} L{pl})", flush=True)
        dataset = store.load_dataset()
        for uid in store.sample_uids(model_id):
            rec = store.load_record(model_id, uid)
            binfo = store.load_boundaries(model_id, uid)
            p = dataset.get_prompt(rec.prompt_id)
            assigns = {}
            if cond == "control":
                for t in reversed(rec.turns):
                    if t.role == "assistant":
                        a = parse_time_assignments(t.text)
                        if a:
                            assigns = a
                            break
            turns = []
            for t in rec.turns:
                shY, shT = t.step_horizon_years, t.step_horizon_text
                if shY is None and shT:
                    shY = _parse_duration_years(shT, target_mode=True)
                if (shY is None and t.step_index is not None
                        and t.step_index in assigns):
                    raw, val = assigns[t.step_index]
                    if val is not None:
                        shY, shT = val, f"{raw} — assigned in final turn"
                turns.append({
                    "i": t.turn_index, "role": t.role, "text": t.text,
                    "step": t.step_index, "shText": shT,
                    "shYears": shY, "trunc": t.truncated,
                    "forced": t.think_forced_closed,
                    "predY": preds_all.get((cond, uid, t.turn_index)),
                })
            bounds = [{
                "p": b["abs_pos"], "k": b["kind"], "r": b["role"],
                "t": b["turn_index"], "s": b["step_index"], "tok": b["token_str"],
            } for b in binfo["boundary_map"]["boundaries"]]
            key_to_i[(cond, uid)] = len(samples)
            v = p.target_horizon.value
            hl = f"{int(v) if v == int(v) else v} {p.target_horizon.unit}"
            samples.append({
                "seed": (rec.protocol or {}).get("seed", 0), "cond": cond,
                "uid": uid, "promptId": rec.prompt_id, "task": p.task_id,
                "phrasing": p.phrasing_id, "hYears": rec.target_horizon_years,
                "hLabel": hl + (" · CONTROL" if cond == "control" else ""),
                "completed": rec.completed, "planned": rec.n_steps_planned,
                "expanded": rec.n_steps_expanded, "nTok": len(rec.token_ids),
                "turns": turns, "bounds": bounds,
            })

    kinds_data = {}
    for kind in KINDS:
        per_layer = {}
        for depth, layer in sorted(depth_to_layer.items()):
            Xs_all, rows_all = [], []
            for store, cond in stores:
                X, rows = store.query(model_id, layer=layer, kind=kind)
                if len(rows) == 0:
                    continue
                Xs_all.append(X)
                rows_all += [(r, cond) for r in rows]
            if rows_all:
                per_layer[layer] = (np.vstack(Xs_all), rows_all)
        if not per_layer:
            continue
        first_layer = sorted(per_layer)[0]
        row_meta = [{
            "si": key_to_i[(cond, r.sample_uid)], "p": r.abs_pos, "role": r.role,
            "turn": r.turn_index, "step": r.step_index,
            "shY": r.step_horizon_years, "cond": cond,
        } for r, cond in per_layer[first_layer][1]]
        fits = {}
        for layer, (X, rows_c) in sorted(per_layer.items()):
            rows = [r for r, _ in rows_c]
            conds = [c for _, c in rows_c]
            scopes = {
                "first": [i for i, r in enumerate(rows)
                          if r.turn_index == 1 and r.role == "assistant"],
                "steps": [i for i, r in enumerate(rows)
                          if r.step_index is not None and r.role == "assistant"],
                "assistant": [i for i, r in enumerate(rows)
                              if r.role == "assistant"],
                "all": list(range(len(rows))),
            }
            for scope, idxs in scopes.items():
                if len(idxs) < 8:
                    continue
                Xs = X[idxs]
                for centered in (True, False):
                    Xc = Xs.copy()
                    if centered:
                        by_task = defaultdict(list)
                        for j, i in enumerate(idxs):
                            by_task[rows[i].task_id].append(j)
                        for jj in by_task.values():
                            Xc[jj] -= Xs[jj].mean(0)
                    tag = "c" if centered else "r"
                    pca = fit_pca(Xc, n_components=3)
                    fits[f"{layer}|{scope}|{tag}"] = {
                        "idx": idxs,
                        "xyz": norm_q(pca.projected),
                        "evr": [round(float(v), 4)
                                for v in pca.explained_variance_ratio],
                    }
        kinds_data[kind] = {"rows": row_meta, "fits": fits}
        print(f"  {name} {kind}: {len(row_meta)} rows, {len(fits)} fits", flush=True)

    tstore = stores[0][0]
    fig_dir = tstore.model_dir(model_id) / "figures"
    probes = json.loads((fig_dir / "probe_results.json").read_text())
    horizon = json.loads((fig_dir / "horizon" / "horizon_results.json").read_text())
    hslim = [
        {k: r.get(k) for k in ("slice", "coloring", "n", "evr", "spearman_log_horizon")}
        for r in horizon if r.get("spearman_log_horizon")
    ]
    out["models"].append({
        "name": name, "engine": eng, "run": "+".join(r for r, _ in run_conds),
        "modelId": model_id, "default": is_default, "joint": True,
        "depthToLayer": {str(k): v for k, v in sorted(depth_to_layer.items())},
        "samples": samples, "kinds": kinds_data,
        "probes": probes, "horizon": hslim,
    })
    print(name, len(samples), "samples", flush=True)

dest = Path(__file__).parent / "debug_data_joint.json"
dest.write_text(json.dumps(out, separators=(",", ":")))
print("wrote", dest, dest.stat().st_size // 1024, "KB")
