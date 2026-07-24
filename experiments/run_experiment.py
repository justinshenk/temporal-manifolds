"""generate_llm_responses: run planning conversations + capture activations.

Usage:
    uv run python experiments/run_experiment.py --config experiments/configs/smoke_small.json
    uv run python experiments/run_experiment.py --config ... --model Qwen/Qwen3-14B
    uv run python experiments/run_experiment.py --config ... --max-samples 2

Config JSON schema (see experiments/configs/):
    {
      "name": "main_nothink",
      "model": "Qwen/Qwen3-14B",
      "backend": "hf",                  // hf | mlx
      "tasks": ["climate_city", ...],   // optional, default all CORE_TASKS
      "horizons": ["3 days", [2,"weeks"], "1 years", "10 years"],
      "phrasings": ["available_time"],  // optional
      "depths": [0.4, 0.6, 0.8],
      "protocol": { ... ProtocolConfig fields ... }
    }

Outputs land in output/runs/<run_id>/<model>/samples/<uid>/ (resumable:
completed samples are skipped on re-run).
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1]))

from src.capture.extractor import extract_boundary_activations
from src.capture.store import ResponseStore, run_fingerprint
from src.chat_markup.registry import detect_markup, verify_markup
from src.conversation.driver import ConversationDriver, ProtocolConfig
from src.datasets.generator import build_dataset, horizons_from_specs
from src.datasets.phrasings import DEFAULT_PHRASINGS, get_phrasing
from src.datasets.tasks import CORE_TASKS, get_task
from src.engine import load_engine


def load_config(path: str) -> dict:
    with open(path) as f:
        return json.load(f)


def build_prompt_dataset(cfg: dict):
    tasks = (
        tuple(get_task(t) for t in cfg["tasks"])
        if cfg.get("tasks")
        else CORE_TASKS
    )
    horizons = horizons_from_specs(cfg.get("horizons", []))
    task_horizons = None
    if cfg.get("task_horizons"):
        task_horizons = {
            tid: horizons_from_specs(specs)
            for tid, specs in cfg["task_horizons"].items()
        }
    phrasings = (
        tuple(get_phrasing(p) for p in cfg["phrasings"])
        if cfg.get("phrasings")
        else DEFAULT_PHRASINGS[:1]
    )
    return build_dataset(
        name=cfg["name"],
        tasks=tasks,
        horizons=horizons,
        phrasings=phrasings,
        seed=cfg.get("protocol", {}).get("seed", 0),
        step_mode=cfg.get("step_mode", "duration"),
        task_horizons=task_horizons,
    )


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", required=True)
    ap.add_argument("--model", default=None, help="override config model")
    ap.add_argument("--backend", default=None, help="hf | mlx (override)")
    ap.add_argument("--max-samples", type=int, default=None)
    ap.add_argument("--force", action="store_true", help="re-run existing samples")
    args = ap.parse_args()

    cfg = load_config(args.config)
    model_id = args.model or cfg["model"]
    backend = args.backend or cfg.get("backend", "hf")
    depths = tuple(cfg.get("depths", [0.4, 0.6, 0.8]))
    protocol = ProtocolConfig.from_dict(cfg.get("protocol", {}))

    rollouts = int(cfg.get("rollouts", 1))
    dataset = build_prompt_dataset(cfg)
    run_id = run_fingerprint(
        {
            "name": cfg["name"],
            "rollouts": rollouts,
            "step_mode": cfg.get("step_mode", "duration"),
            # hash the ACTUAL prompt texts so any template/phrasing change
            # produces a fresh run instead of silently resuming stale samples
            "prompts": {p.prompt_id: p.text for p in dataset.prompts},
        },
        protocol.to_dict(),
        depths,
    )
    store = ResponseStore(run_id)
    store.init_run(
        {"config": cfg, "run_id": run_id, "depths": list(depths)}, dataset
    )
    print(f"[run] run_id={run_id} model={model_id} prompts={len(dataset.prompts)}")
    print(f"[run] output: {store.run_dir}")

    prompts = dataset.prompts
    if args.max_samples is not None:
        prompts = prompts[: args.max_samples]

    engine = load_engine(model_id, backend=backend)
    markup = detect_markup(model_id)
    marker_ids = verify_markup(markup, engine.tokenizer)
    print(f"[run] markup family={markup.family} verified: {marker_ids}")

    from src.conversation.records import make_sample_uid

    jobs = [
        (prompt, r)
        for prompt in prompts
        for r in range(rollouts)
    ]
    n_done = n_skip = n_fail = 0
    t0 = time.time()
    for i, (prompt, rollout) in enumerate(jobs):
        proto_r = ProtocolConfig.from_dict({**protocol.to_dict(), "seed": rollout})
        uid = make_sample_uid(
            prompt.prompt_id,
            model_id,
            prompt.target_horizon_years,
            proto_r.to_dict(),
            proto_r.seed,
        )
        if not args.force and store.has_sample(model_id, uid):
            n_skip += 1
            continue
        print(
            f"[run] ({i + 1}/{len(jobs)}) {prompt.prompt_id} r{rollout} "
            f"(horizon={prompt.target_horizon.value} {prompt.target_horizon.unit})"
        )
        try:
            engine.set_seed(rollout)
            driver = ConversationDriver(engine, markup, proto_r)
            record = driver.run(prompt)
            bmap, depth_to_layer, acts = extract_boundary_activations(
                engine, markup, record, depths
            )
            store.save_sample(
                model_id,
                record,
                bmap,
                depth_to_layer,
                acts,
                extra_meta={
                    "markup_family": markup.family,
                    "marker_ids": marker_ids,
                    "d_model": engine.d_model,
                    "n_layers": engine.n_layers,
                },
            )
            n_done += 1
            print(
                f"[run]   turns={len(record.turns)} steps="
                f"{record.n_steps_expanded}/{record.n_steps_planned} "
                f"completed={record.completed} tokens={len(record.token_ids)} "
                f"boundaries={len(bmap.boundaries)} "
                f"({time.time() - t0:.0f}s elapsed)"
            )
        except Exception:
            n_fail += 1
            print(f"[run]   FAILED on {prompt.prompt_id}:")
            traceback.print_exc()

    if n_fail == 0 and (n_done + n_skip) == len(jobs):
        store.mark_complete(model_id)
    print(
        f"[run] finished: done={n_done} skipped={n_skip} failed={n_fail} "
        f"run_id={run_id}"
    )
    return 1 if n_fail else 0


if __name__ == "__main__":
    raise SystemExit(main())
