"""ResponseStore: content-addressed on-disk store for conversations +
boundary activations, with a retrieval API.

Layout (mirrors the proven DataManager scheme):

    output/runs/<run_id>/               # fingerprint(dataset+protocol+depths+seed)
      run_config.json
      prompt_dataset.json
      <model_dir>/                      # sanitize_model_name(model_id)
        manifest.json                   # {status, samples: {uid: prompt_id}, ...}
        samples/<sample_uid>/
          conversation.json             # ConversationRecord
          boundaries.json               # BoundaryMap + depth->layer + d_model
          activations.npz               # keys "L{layer}_p{abs_pos}" -> [d_model]

Retrieval: `store.query(model_id, layer=..., kind=..., role=...)` returns
(matrix [n, d_model], rows metadata) gathered lazily across samples.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np

from ..capture.boundaries import BoundaryMap
from ..conversation.records import ConversationRecord
from ..core.file_io import load_json, save_json
from ..core.paths import get_runs_dir, sanitize_model_name
from ..datasets.schema import PlanningPromptDataset


def canonical_json(obj) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), default=str)


def run_fingerprint(dataset_cfg: dict, protocol_cfg: dict, depths: tuple) -> str:
    payload = canonical_json(
        {"dataset": dataset_cfg, "protocol": protocol_cfg, "depths": list(depths)}
    )
    return hashlib.sha256(payload.encode()).hexdigest()[:12]


@dataclass
class QueryRow:
    """Metadata for one activation vector returned by ResponseStore.query."""

    sample_uid: str
    prompt_id: str
    model_id: str
    target_horizon_years: float
    layer: int
    depth: float
    abs_pos: int
    kind: str
    role: str
    turn_index: int
    step_index: int | None
    step_horizon_years: float | None
    task_id: str = ""
    phrasing_id: str = ""
    # "stated"   — horizon verbalized in the step turn itself
    # "assigned" — control mode: backfilled from the final "Time assignments:"
    horizon_source: str = "stated"


class ResponseStore:
    def __init__(self, run_id: str, runs_root: Path | None = None):
        self.run_id = run_id
        self.run_dir = (runs_root or get_runs_dir()) / run_id

    # -- paths ---------------------------------------------------------------

    def model_dir(self, model_id: str) -> Path:
        return self.run_dir / sanitize_model_name(model_id)

    def sample_dir(self, model_id: str, sample_uid: str) -> Path:
        return self.model_dir(model_id) / "samples" / sample_uid

    def manifest_path(self, model_id: str) -> Path:
        return self.model_dir(model_id) / "manifest.json"

    # -- write ---------------------------------------------------------------

    def init_run(self, run_config: dict, dataset: PlanningPromptDataset) -> None:
        self.run_dir.mkdir(parents=True, exist_ok=True)
        cfg_path = self.run_dir / "run_config.json"
        if not cfg_path.exists():
            save_json(run_config, cfg_path, readable_text=False)
        ds_path = self.run_dir / "prompt_dataset.json"
        if not ds_path.exists():
            save_json(dataset.to_dict(), ds_path)

    def read_manifest(self, model_id: str) -> dict:
        path = self.manifest_path(model_id)
        if not path.exists():
            return {"status": "in_progress", "samples": {}}
        return load_json(path)

    def _write_manifest(self, model_id: str, manifest: dict) -> None:
        manifest["updated_at"] = datetime.now().isoformat(timespec="seconds")
        self.model_dir(model_id).mkdir(parents=True, exist_ok=True)
        save_json(manifest, self.manifest_path(model_id), readable_text=False)

    def has_sample(self, model_id: str, sample_uid: str) -> bool:
        d = self.sample_dir(model_id, sample_uid)
        return (
            (d / "conversation.json").exists()
            and (d / "boundaries.json").exists()
            and (d / "activations.npz").exists()
        )

    def save_sample(
        self,
        model_id: str,
        record: ConversationRecord,
        bmap: BoundaryMap,
        depth_to_layer: dict[float, int],
        acts: dict[str, np.ndarray],
        extra_meta: dict | None = None,
    ) -> None:
        d = self.sample_dir(model_id, record.sample_uid)
        d.mkdir(parents=True, exist_ok=True)
        save_json(record.to_dict(), d / "conversation.json")
        save_json(
            {
                "boundary_map": bmap.to_dict(),
                "depth_to_layer": {str(k): v for k, v in depth_to_layer.items()},
                **(extra_meta or {}),
            },
            d / "boundaries.json",
            readable_text=False,
        )
        np.savez(d / "activations.npz", **acts)

        manifest = self.read_manifest(model_id)
        manifest.setdefault("samples", {})[record.sample_uid] = record.prompt_id
        manifest["model_id"] = model_id
        manifest["run_id"] = self.run_id
        self._write_manifest(model_id, manifest)

    def mark_complete(self, model_id: str) -> None:
        manifest = self.read_manifest(model_id)
        manifest["status"] = "complete"
        self._write_manifest(model_id, manifest)

    # -- read ----------------------------------------------------------------

    def load_dataset(self) -> PlanningPromptDataset:
        return PlanningPromptDataset.from_dict(
            load_json(self.run_dir / "prompt_dataset.json")
        )

    def sample_uids(self, model_id: str) -> list[str]:
        samples_dir = self.model_dir(model_id) / "samples"
        if not samples_dir.exists():
            return []
        return sorted(p.name for p in samples_dir.iterdir() if p.is_dir())

    def load_record(self, model_id: str, sample_uid: str) -> ConversationRecord:
        return ConversationRecord.from_dict(
            load_json(self.sample_dir(model_id, sample_uid) / "conversation.json")
        )

    def load_boundaries(self, model_id: str, sample_uid: str) -> dict:
        return load_json(self.sample_dir(model_id, sample_uid) / "boundaries.json")

    def query(
        self,
        model_id: str,
        layer: int | None = None,
        depth: float | None = None,
        kind: str | None = None,
        role: str | None = None,
        prompt_id: str | None = None,
        turn_index: int | None = None,
        step_index: int | None = None,
        include_system: bool = False,
    ) -> tuple[np.ndarray, list[QueryRow]]:
        """Gather boundary activation vectors matching the filters.

        Returns (X [n, d_model] float32, rows) in a stable order
        (sample_uid, layer, abs_pos).
        """
        dataset = None
        try:
            dataset = self.load_dataset()
        except FileNotFoundError:
            pass

        vectors: list[np.ndarray] = []
        rows: list[QueryRow] = []
        for uid in self.sample_uids(model_id):
            record = self.load_record(model_id, uid)
            if prompt_id is not None and record.prompt_id != prompt_id:
                continue
            binfo = self.load_boundaries(model_id, uid)
            depth_to_layer = {
                float(k): v for k, v in binfo["depth_to_layer"].items()
            }
            layer_to_depth = {v: k for k, v in depth_to_layer.items()}
            bmap = BoundaryMap.from_dict(binfo["boundary_map"])

            task_id, phrasing_id = "", ""
            if dataset is not None:
                try:
                    p = dataset.get_prompt(record.prompt_id)
                    task_id, phrasing_id = p.task_id, p.phrasing_id
                except KeyError:
                    pass

            # control mode: the last assistant turn with a "Time assignments:"
            # block carries every step's target offset retrospectively
            assignments: dict[int, tuple[str, float | None]] = {}
            for t in reversed(record.turns):
                if t.role == "assistant":
                    from ..conversation.parsing import parse_time_assignments

                    assignments = parse_time_assignments(t.text)
                    if assignments:
                        break

            wanted_layers = sorted(layer_to_depth)
            if layer is not None:
                wanted_layers = [x for x in wanted_layers if x == layer]
            if depth is not None:
                wanted_layers = [
                    x for x in wanted_layers if depth_to_layer.get(depth) == x
                ]

            with np.load(self.sample_dir(model_id, uid) / "activations.npz") as npz:
                for b in bmap.boundaries:
                    if not include_system and b.turn_index < 0:
                        continue
                    if kind is not None and b.kind != kind:
                        continue
                    if role is not None and b.role != role:
                        continue
                    if turn_index is not None and b.turn_index != turn_index:
                        continue
                    if step_index is not None and b.step_index != step_index:
                        continue
                    step_horizon = None
                    horizon_source = "stated"
                    if 0 <= b.turn_index < len(record.turns):
                        turn = record.turns[b.turn_index]
                        step_horizon = turn.step_horizon_years
                        if (
                            step_horizon is None
                            and not turn.step_horizon_text
                            and b.step_index is not None
                            and b.step_index in assignments
                        ):
                            step_horizon = assignments[b.step_index][1]
                            horizon_source = "assigned"
                        if step_horizon is None and turn.step_horizon_text:
                            # stored raw text; parser may have improved since.
                            # Mode is recoverable from the prompt instructions.
                            from ..conversation.parsing import _parse_duration_years

                            tmode = "Time target:" in (record.turns[0].text or "")
                            step_horizon = _parse_duration_years(
                                turn.step_horizon_text, target_mode=tmode
                            )
                    for lay in wanted_layers:
                        key = f"L{lay}_p{b.abs_pos}"
                        if key not in npz:
                            raise KeyError(
                                f"Missing activation {key} in sample {uid}"
                            )
                        vectors.append(npz[key])
                        rows.append(
                            QueryRow(
                                sample_uid=uid,
                                prompt_id=record.prompt_id,
                                model_id=model_id,
                                target_horizon_years=record.target_horizon_years,
                                layer=lay,
                                depth=layer_to_depth[lay],
                                abs_pos=b.abs_pos,
                                kind=b.kind,
                                role=b.role,
                                turn_index=b.turn_index,
                                step_index=b.step_index,
                                step_horizon_years=step_horizon,
                                task_id=task_id,
                                phrasing_id=phrasing_id,
                                horizon_source=horizon_source,
                            )
                        )
        if not vectors:
            return np.zeros((0, 0), dtype=np.float32), []
        return np.stack(vectors).astype(np.float32), rows
