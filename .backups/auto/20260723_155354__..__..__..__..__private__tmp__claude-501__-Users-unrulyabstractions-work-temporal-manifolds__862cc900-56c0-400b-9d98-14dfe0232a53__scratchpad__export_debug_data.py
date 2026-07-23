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
from src.conversation.parsing import _parse_duration_years

MODELS = [
    ("Qwen3-14B", "bf16 · HF/MPS", "d196f33fb864", "Qwen/Qwen3-14B"),
    ("Llama-3.1-8B", "bf16 · HF/MPS", "4940b5401c34", "meta-llama/Llama-3.1-8B-Instruct"),
    ("Gemma-2-9B", "bf16 · HF/MPS", "32b7d3ceba2b", "google/gemma-2-9b-it"),
    ("Qwen3-32B", "4-bit · MLX", "864237268525", "mlx-community/Qwen3-32B-4bit"),
]
KINDS = ["turn_end", "turn_start", "role", "think_close"]
TSNE_SCOPES = ("first", "steps", "assistant")  # skip 'all' (too many, slow)
SEED = 0


def q(x):
    return int(round(x * 1000))


def norm_q(P):
    rms = float(np.sqrt((P**2).mean())) or 1.0
    return [[q(v) for v in row] for row in (P / rms).tolist()]


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
for name, eng, run, model_id in MODELS:
    store = ResponseStore(run)
    uids = store.sample_uids(model_id)
    binfo0 = store.load_boundaries(model_id, uids[0])
    depth_to_layer = {float(k): v for k, v in binfo0["depth_to_layer"].items()}
    dataset = store.load_dataset()

    samples, uid_to_i = [], {}
    for uid in uids:
        rec = store.load_record(model_id, uid)
        binfo = store.load_boundaries(model_id, uid)
        p = dataset.get_prompt(rec.prompt_id)
        turns = []
        for t in rec.turns:
            shY = t.step_horizon_years
            if shY is None and t.step_horizon_text:
                shY = _parse_duration_years(t.step_horizon_text)
            turns.append({
                "i": t.turn_index, "role": t.role, "text": t.text,
                "step": t.step_index, "shText": t.step_horizon_text,
                "shYears": shY, "trunc": t.truncated,
                "forced": t.think_forced_closed,
            })
        bounds = [{
            "p": b["abs_pos"], "k": b["kind"], "r": b["role"],
            "t": b["turn_index"], "s": b["step_index"], "tok": b["token_str"],
        } for b in binfo["boundary_map"]["boundaries"]]
        uid_to_i[uid] = len(samples)
        v = p.target_horizon.value
        samples.append({
            "uid": uid, "promptId": rec.prompt_id, "task": p.task_id,
            "phrasing": p.phrasing_id, "hYears": rec.target_horizon_years,
            "hLabel": f"{int(v) if v == int(v) else v} {p.target_horizon.unit}",
            "completed": rec.completed, "planned": rec.n_steps_planned,
            "expanded": rec.n_steps_expanded, "nTok": len(rec.token_ids),
            "turns": turns, "bounds": bounds,
        })

    kinds_data = {}
    for kind in KINDS:
        first_layer = sorted(depth_to_layer.values())[0]
        X0, rows0 = store.query(model_id, layer=first_layer, kind=kind)
        if not rows0:
            continue
        row_meta = [{
            "si": uid_to_i[r.sample_uid], "p": r.abs_pos, "role": r.role,
            "turn": r.turn_index, "step": r.step_index, "shY": r.step_horizon_years,
        } for r in rows0]
        fits = {}
        for depth, layer in sorted(depth_to_layer.items()):
            X, rows = store.query(model_id, layer=layer, kind=kind)
            scopes = {
                "first": [i for i, r in enumerate(rows)
                          if r.turn_index == 1 and r.role == "assistant"],
                "steps": [i for i, r in enumerate(rows)
                          if r.step_index is not None and r.role == "assistant"],
                "assistant": [i for i, r in enumerate(rows) if r.role == "assistant"],
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
                    if scope in TSNE_SCOPES:
                        res = tsne_fit(Xc)
                        if res is not None:
                            emb, perp = res
                            emb3 = np.column_stack([emb, np.zeros(len(emb))])
                            fits[f"{layer}|{scope}|{tag}|tsne"] = {
                                "idx": idxs,
                                "xyz": norm_q(emb3),
                                "perp": perp,
                            }
        kinds_data[kind] = {"rows": row_meta, "fits": fits}
        print(f"  {name} {kind}: {len(row_meta)} rows, {len(fits)} fits", flush=True)

    fig_dir = store.model_dir(model_id) / "figures"
    probes = json.loads((fig_dir / "probe_results.json").read_text())
    horizon = json.loads((fig_dir / "horizon" / "horizon_results.json").read_text())
    hslim = [
        {k: r.get(k) for k in ("slice", "coloring", "n", "evr", "spearman_log_horizon")}
        for r in horizon if r.get("spearman_log_horizon")
    ]
    out["models"].append({
        "name": name, "engine": eng, "run": run, "modelId": model_id,
        "depthToLayer": {str(k): v for k, v in sorted(depth_to_layer.items())},
        "samples": samples, "kinds": kinds_data,
        "probes": probes, "horizon": hslim,
    })
    print(name, len(samples), "samples", flush=True)

dest = Path(__file__).parent / "debug_data.json"
dest.write_text(json.dumps(out, separators=(",", ":")))
print("wrote", dest, dest.stat().st_size // 1024, "KB")
