"""Build a disk-backed cache of last-position activations plus prompt metadata.

The ``.acts`` tree holds ~2k torch batches whose activations do not fit in this machine's
free RAM, so every downstream fit streams from a float16 memmap written once here. float16
is lossless for the cached bfloat16 activations in this value range (|x| < 34) because it
carries strictly more mantissa bits over the same magnitudes.
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

UNIT_TO_MONTHS = {
    "second": 1 / (30.4375 * 86400),
    "minute": 1 / (30.4375 * 1440),
    "hour": 1 / (30.4375 * 24),
    "day": 1 / 30.4375,
    "week": 7 / 30.4375,
    "month": 1.0,
    "year": 12.0,
    "decade": 120.0,
    "century": 1200.0,
    "millennium": 12000.0,
}
UNIT_TO_MONTHS.update({f"{unit}s": value for unit, value in list(UNIT_TO_MONTHS.items())})
UNIT_TO_MONTHS["centuries"] = 1200.0
UNIT_TO_MONTHS["millennia"] = 12000.0

NOT_APPLICABLE = {"n/a", "na", "none", "nan", ""}

TASK_METADATA_FIELDS = (
    "task_family",
    "difficulty",
    "domain",
    "complexity",
    "planning_type",
    "stakes",
    "agency",
)
TEMPLATE_METADATA_FIELDS = ("prompt_framing",)
SCALAR_METADATA_FIELDS = (
    "template_id",
    "task",
    "base_value",
    "base_unit",
    "unit_variant",
    "number_format",
    "value",
    "value_text",
    "unit",
)


def _is_not_applicable(value: Any) -> bool:
    return value is None or str(value).strip().lower() in NOT_APPLICABLE


def time_horizon_months(value: Any, unit: Any) -> float:
    """Convert a stated horizon to months, or NaN when the prompt states no horizon."""
    if _is_not_applicable(value) or _is_not_applicable(unit):
        return float("nan")
    key = str(unit).strip().lower()
    if key not in UNIT_TO_MONTHS:
        raise ValueError(f"Cannot convert time-horizon unit to months: {unit!r}")
    return round(float(value) * UNIT_TO_MONTHS[key], 12)


def flatten_prompt_metadata(metadata: dict[str, Any]) -> dict[str, Any]:
    """Flatten one prompt's nested metadata into a single row of scalars."""
    row: dict[str, Any] = {field: metadata.get(field) for field in SCALAR_METADATA_FIELDS}
    task_metadata = metadata.get("task_metadata") or {}
    template_metadata = metadata.get("template_metadata") or {}
    row.update({field: task_metadata.get(field) for field in TASK_METADATA_FIELDS})
    row.update({field: template_metadata.get(field) for field in TEMPLATE_METADATA_FIELDS})
    return row


@dataclass(frozen=True)
class ActivationCache:
    """A memmapped activation matrix aligned row-for-row with a metadata frame."""

    activations: np.memmap
    metadata: pd.DataFrame
    layer_component: str

    @property
    def n_rows(self) -> int:
        return int(self.activations.shape[0])

    @property
    def n_features(self) -> int:
        return int(self.activations.shape[1])

    def rows(self, positions: np.ndarray, *, chunk_size: int = 8192) -> np.ndarray:
        """Read the requested rows as float32, in chunks so peak memory stays bounded."""
        positions = np.asarray(positions, dtype=np.int64)
        out = np.empty((positions.size, self.n_features), dtype=np.float32)
        for start in range(0, positions.size, chunk_size):
            block = positions[start : start + chunk_size]
            out[start : start + block.size] = np.asarray(self.activations[block], dtype=np.float32)
        return out

    def iter_chunks(
        self, positions: np.ndarray, *, chunk_size: int = 8192
    ) -> Iterable[tuple[np.ndarray, np.ndarray]]:
        """Yield ``(row_positions, float32 block)`` pairs for streaming ``partial_fit`` passes."""
        positions = np.asarray(positions, dtype=np.int64)
        for start in range(0, positions.size, chunk_size):
            block = positions[start : start + chunk_size]
            yield block, np.asarray(self.activations[block], dtype=np.float32)


def _batch_files(folder: Path) -> list[Path]:
    return sorted(folder.glob("activations_batch_*.pt"))


def build_cache(
    acts_root: str | Path,
    cache_dir: str | Path,
    *,
    folders: Sequence[str] | None = None,
    exclude_folders: Sequence[str] = ("conv",),
    layer_component: str | None = None,
    position_index: int = -1,
) -> ActivationCache:
    """Write ``activations.f16`` and ``metadata.parquet`` for the selected folders.

    ``conv`` is excluded by default: its activations sit far off the manifold the other
    folders share, so it is treated as an outlier source rather than training data.
    """
    acts_root = Path(acts_root)
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    if folders is None:
        folders = sorted(p.name for p in acts_root.iterdir() if p.is_dir())
    selected = [name for name in folders if name not in set(exclude_folders)]
    if not selected:
        raise ValueError("No activation folders remain after exclusions.")

    files = [(name, path) for name in selected for path in _batch_files(acts_root / name)]
    if not files:
        raise ValueError(f"No activation batches found under {acts_root}.")

    # First pass reads only shapes and metadata so the memmap can be sized exactly.
    row_counts: list[int] = []
    metadata_rows: list[dict[str, Any]] = []
    resolved_component = layer_component
    n_features: int | None = None
    for folder_name, path in files:
        payload = torch.load(path, map_location="cpu", weights_only=False)
        component = resolved_component or payload["layer_component"]
        if resolved_component is None:
            resolved_component = component
        if component not in payload["activations"]:
            raise ValueError(f"{path} has no activations for component {component!r}.")
        tensor = payload["activations"][component]
        if n_features is None:
            n_features = int(tensor.shape[-1])
        elif int(tensor.shape[-1]) != n_features:
            raise ValueError(f"{path} has {tensor.shape[-1]} features, expected {n_features}.")
        row_counts.append(int(tensor.shape[0]))
        for sample_index, metadata in zip(
            payload["sample_indices"], payload["prompt_metadata"], strict=True
        ):
            row = flatten_prompt_metadata(metadata)
            row["source_folder"] = folder_name
            row["dataset"] = payload["dataset"]
            row["batch_index"] = int(payload["batch_index"])
            row["sample_index"] = int(sample_index)
            metadata_rows.append(row)

    total_rows = int(sum(row_counts))
    assert n_features is not None and resolved_component is not None
    matrix_path = cache_dir / "activations.f16"
    matrix = np.memmap(matrix_path, dtype=np.float16, mode="w+", shape=(total_rows, n_features))

    offset = 0
    for (_, path), count in zip(files, row_counts, strict=True):
        payload = torch.load(path, map_location="cpu", weights_only=False)
        tensor = payload["activations"][resolved_component]
        block = tensor[:, position_index, :] if tensor.ndim == 3 else tensor
        matrix[offset : offset + count] = block.to(torch.float32).numpy().astype(np.float16)
        offset += count
    matrix.flush()

    metadata = pd.DataFrame(metadata_rows)
    # Prompt metadata mixes numbers with the "N/A" sentinel in the same field, so the raw
    # columns are stored as strings and every numeric view is derived explicitly.
    for column in metadata.columns:
        if metadata[column].dtype == object:
            metadata[column] = metadata[column].astype(str)
    metadata["time_horizon_months"] = [
        time_horizon_months(value, unit)
        for value, unit in zip(metadata["base_value"], metadata["base_unit"], strict=True)
    ]
    metadata["log10_time_horizon_months"] = np.log10(metadata["time_horizon_months"])
    metadata["row"] = np.arange(len(metadata), dtype=np.int64)
    metadata.to_parquet(cache_dir / "metadata.parquet", index=False)
    (cache_dir / "cache_info.json").write_text(
        json.dumps(
            {
                "layer_component": resolved_component,
                "position_index": position_index,
                "n_rows": total_rows,
                "n_features": n_features,
                "folders": selected,
                "excluded_folders": list(exclude_folders),
            },
            indent=2,
        )
    )
    return load_cache(cache_dir)


def load_cache(cache_dir: str | Path) -> ActivationCache:
    """Open a cache written by :func:`build_cache` without copying it into RAM."""
    cache_dir = Path(cache_dir)
    info = json.loads((cache_dir / "cache_info.json").read_text())
    matrix = np.memmap(
        cache_dir / "activations.f16",
        dtype=np.float16,
        mode="r",
        shape=(int(info["n_rows"]), int(info["n_features"])),
    )
    metadata = pd.read_parquet(cache_dir / "metadata.parquet")
    return ActivationCache(
        activations=matrix, metadata=metadata, layer_component=str(info["layer_component"])
    )
