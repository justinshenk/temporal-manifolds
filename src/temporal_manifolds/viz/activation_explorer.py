"""Core data preparation for the local activation PCA explorer."""

from __future__ import annotations

import io
import os
from collections.abc import Callable, Iterable, Mapping, Sequence
from hashlib import sha256
from pathlib import Path, PurePosixPath
from typing import Any, BinaryIO

import numpy as np
import pandas as pd
import torch
from sklearn.decomposition import PCA, IncrementalPCA

MISSING = object()

UNIT_TO_MONTHS = {
    "second": 1 / (30.4375 * 86400), "minute": 1 / (30.4375 * 1440),
    "hour": 1 / (30.4375 * 24), "day": 1 / 30.4375, "week": 7 / 30.4375,
    "month": 1, "year": 12, "decade": 120, "century": 1200,
    "millennium": 12000,
}
UNIT_TO_MONTHS.update({f"{unit}s": value for unit, value in list(UNIT_TO_MONTHS.items())})
UNIT_TO_MONTHS["centuries"] = 1200
UNIT_TO_MONTHS["millennia"] = 12000


def activation_batch_basename(name: str) -> str | None:
    """Return a valid activation batch basename from an upload-relative path."""
    basename = PurePosixPath(name.replace("\\", "/")).name
    if basename.startswith("activations_batch_") and basename.endswith(".pt"):
        return basename
    return None


def flatten_scalar_metadata(metadata: Mapping[str, Any], prefix: str = "") -> dict[str, Any]:
    """Flatten nested scalar metadata using the notebook's dotted paths."""
    flattened: dict[str, Any] = {}
    for key, value in metadata.items():
        path = f"{prefix}.{key}" if prefix else key
        if isinstance(value, Mapping):
            flattened.update(flatten_scalar_metadata(value, path))
        elif not isinstance(value, (list, tuple, set)):
            flattened[path] = value
    return flattened


def get_metadata_field(metadata: Mapping[str, Any], dotted_path: str) -> Any:
    value: Any = metadata
    for key in dotted_path.split("."):
        if not isinstance(value, Mapping) or key not in value:
            return MISSING
        value = value[key]
    return value


def metadata_value_matches(actual_value: Any, expected_value: Any) -> bool:
    if actual_value is MISSING:
        return False
    allowed_values = expected_value if isinstance(expected_value, list) else [expected_value]
    return any(actual_value == allowed_value for allowed_value in allowed_values)


def _load(source: str | Path | bytes | BinaryIO) -> dict[str, Any]:
    if isinstance(source, bytes):
        source = io.BytesIO(source)
    elif hasattr(source, "seek"):
        source.seek(0)
    kwargs: dict[str, Any] = {"map_location": "cpu", "weights_only": True}
    if isinstance(source, (str, Path)):
        kwargs["mmap"] = True
    return torch.load(source, **kwargs)


def inspect_sources(sources: Sequence[str | Path | bytes | BinaryIO]) -> dict[str, Any]:
    """Return available components, positions, and flattened metadata fields."""
    if not sources:
        raise ValueError("Select at least one activation batch.")
    components: set[str] | None = None
    positions: list[Any] | None = None
    metadata_fields: set[str] = set()
    metadata_records: list[dict[str, Any]] = []
    source_row_counts: list[int] = []
    rows = 0
    for source_index, source in enumerate(sources):
        payload = _load(source)
        current_components = set(payload["activations"])
        components = current_components if components is None else components & current_components
        current_positions = list(payload["positions"])
        if positions is None:
            positions = current_positions
        elif current_positions != positions:
            raise ValueError("Selected batches have inconsistent cached positions.")
        metadata_rows = payload["prompt_metadata"]
        sample_indices = payload["sample_indices"]
        if len(sample_indices) != len(metadata_rows):
            raise ValueError("A selected batch has misaligned sample indices and metadata rows.")
        source_row_counts.append(len(metadata_rows))
        rows += len(metadata_rows)
        for row_offset, (sample_index, metadata) in enumerate(
            zip(sample_indices, metadata_rows, strict=True)
        ):
            flattened = flatten_scalar_metadata(metadata)
            metadata_fields.update(flattened)
            metadata_records.append({
                "sample_index": int(sample_index),
                "_source_index": source_index,
                "_row_offset": row_offset,
                **flattened,
            })
        del payload
    if not components:
        raise ValueError("Selected batches do not share an activation component.")
    return {
        "components": sorted(components), "positions": positions or [],
        "metadata_fields": sorted(metadata_fields), "batch_count": len(sources),
        "row_count": rows, "metadata_index": pd.DataFrame(metadata_records),
        "source_row_counts": source_row_counts,
    }


def extract_activation_slice(
    sources: Sequence[str | Path | bytes | BinaryIO], *, layer_component: str,
    position_index: int, source_row_counts: Sequence[int], cache_dir: str | Path | None = None,
    progress: Callable[[int, int], None] | None = None,
) -> tuple[np.ndarray, Path | None]:
    """Stream one component/position into a reusable RAM or disk-backed matrix."""
    if len(sources) != len(source_row_counts):
        raise ValueError("Source row counts are not aligned with activation batches.")
    total_rows = sum(source_row_counts)
    if total_rows == 0:
        raise ValueError("The selected batches do not contain any rows.")

    local_sources = all(isinstance(source, (str, Path)) for source in sources)
    cache_path: Path | None = None
    if cache_dir is not None and local_sources:
        signature = sha256()
        for source in sources:
            path = Path(source).resolve()
            stat = path.stat()
            signature.update(f"{path}|{stat.st_size}|{stat.st_mtime_ns}\n".encode())
        signature.update(f"{layer_component}|{position_index}".encode())
        cache_path = Path(cache_dir) / f"slice-{signature.hexdigest()[:20]}.npy"
        if cache_path.exists():
            cached = np.load(cache_path, mmap_mode="r")
            if cached.ndim == 2 and cached.shape[0] == total_rows:
                return cached, cache_path

    matrix: np.ndarray | None = None
    write_path = cache_path.with_suffix(".tmp") if cache_path is not None else None
    row_start = 0
    try:
        for source_index, (source, expected_rows) in enumerate(
            zip(sources, source_row_counts, strict=True)
        ):
            payload = _load(source)
            try:
                tensor = payload["activations"][layer_component]
            except KeyError as exc:
                raise ValueError(
                    f"Component {layer_component!r} is missing from batch {source_index + 1}."
                ) from exc
            if tensor.shape[0] != expected_rows:
                raise ValueError(f"Batch {source_index + 1} changed since metadata was indexed.")
            if not 0 <= position_index < tensor.shape[1]:
                raise ValueError(f"Position {position_index} is unavailable in batch {source_index + 1}.")
            if matrix is None:
                shape = (total_rows, int(tensor.shape[-1]))
                if write_path is None:
                    matrix = np.empty(shape, dtype=np.float32)
                else:
                    write_path.parent.mkdir(parents=True, exist_ok=True)
                    matrix = np.lib.format.open_memmap(
                        write_path, mode="w+", dtype=np.float32, shape=shape
                    )
            row_end = row_start + expected_rows
            matrix[row_start:row_end] = tensor[:, position_index, :].to(torch.float32).numpy()
            row_start = row_end
            del tensor, payload
            if progress is not None:
                progress(source_index + 1, len(sources))
        if matrix is None:
            raise ValueError("No activation data was extracted.")
        if isinstance(matrix, np.memmap):
            matrix.flush()
        if cache_path is not None and write_path is not None:
            del matrix
            os.replace(write_path, cache_path)
            return np.load(cache_path, mmap_mode="r"), cache_path
        return matrix, None
    except Exception:
        if write_path is not None and write_path.exists():
            write_path.unlink()
        raise


def prepare_analysis_data(
    activation_matrix: np.ndarray, metadata_index: pd.DataFrame, *,
    cached_position: Any, metadata_filters: Mapping[str, Any] | None,
    aggregation_fields: Sequence[str] | None, max_samples: int | None = None,
    chunk_size: int = 2048,
    phrasing_fields: Sequence[str] = (
        "template_metadata.prompt_framing", "template_metadata.output_format",
    ),
) -> tuple[np.ndarray | None, np.ndarray, pd.DataFrame, dict[str, Any]]:
    """Filter metadata and aggregate a memory map without copying all selected rows."""
    if len(activation_matrix) != len(metadata_index):
        raise ValueError("Activation rows and indexed metadata are misaligned.")
    selected = np.ones(len(metadata_index), dtype=bool)
    for field, expected in (metadata_filters or {}).items():
        if field not in metadata_index:
            selected[:] = False
            break
        allowed = expected if isinstance(expected, list) else [expected]
        selected &= metadata_index[field].isin(allowed).to_numpy()
    row_offsets = np.flatnonzero(selected)
    if max_samples is not None:
        row_offsets = row_offsets[:max_samples]
    if not len(row_offsets):
        raise ValueError("No activation rows match the selected filters.")

    metadata_df = metadata_index.iloc[row_offsets].drop(
        columns=["_source_index", "_row_offset"], errors="ignore"
    ).reset_index(drop=True)
    metadata_df.insert(1, "absolute_token_position", cached_position)
    _add_time_horizon_months(metadata_df)

    aggregation_fields = list(aggregation_fields or [])
    missing_fields = set(aggregation_fields) - set(metadata_df.columns)
    if missing_fields:
        raise ValueError(f"Missing aggregation fields: {sorted(missing_fields)}")
    value_field = "base_value" if "base_value" in metadata_df else "value"
    unit_field = "base_unit" if "base_unit" in metadata_df else "unit"
    available_phrasing = [field for field in phrasing_fields if field in metadata_df]
    if aggregation_fields:
        records: list[pd.Series] = []
        groups = metadata_df.groupby(aggregation_fields, dropna=False, sort=False).indices
        prepared_matrix = np.empty(
            (len(groups), activation_matrix.shape[1]), dtype=np.float32
        )
        for group_index, offsets in enumerate(groups.values()):
            group_offsets = np.asarray(offsets, dtype=np.int64)
            source_offsets = row_offsets[group_offsets]
            vector_sum = np.zeros(activation_matrix.shape[1], dtype=np.float32)
            for start in range(0, len(source_offsets), chunk_size):
                chunk_offsets = source_offsets[start:start + chunk_size]
                chunk = np.asarray(activation_matrix[chunk_offsets], dtype=np.float32)
                vector_sum += chunk.sum(axis=0, dtype=np.float32)
                del chunk
            prepared_matrix[group_index] = vector_sum / len(source_offsets)
            row = metadata_df.iloc[group_offsets[0]].copy()
            row["source_sample_count"] = len(group_offsets)
            for field in [value_field, unit_field, *available_phrasing]:
                if metadata_df.iloc[group_offsets][field].nunique(dropna=True) > 1:
                    row[field] = "<averaged>"
            records.append(row)
        analysis_metadata = pd.DataFrame(records).reset_index(drop=True)
    else:
        prepared_matrix = None
        analysis_metadata = metadata_df.copy()
        analysis_metadata["source_sample_count"] = 1

    details = {
        "loaded_samples": len(row_offsets),
        "analysis_rows": len(analysis_metadata),
        "feature_count": int(activation_matrix.shape[1]),
        "cached_position": cached_position,
        "aggregation_applied": bool(aggregation_fields),
    }
    return prepared_matrix, row_offsets, analysis_metadata, details


def _bounded_batches(length: int, batch_size: int, minimum_size: int) -> Iterable[slice]:
    """Yield batches while ensuring the final IncrementalPCA batch is large enough."""
    start = 0
    while start < length:
        remaining = length - start
        size = min(batch_size, remaining)
        if 0 < remaining - size < minimum_size:
            size = remaining
        yield slice(start, start + size)
        start += size


def fit_pca_projection(
    activation_matrix: np.ndarray, prepared_matrix: np.ndarray | None,
    row_offsets: np.ndarray, analysis_metadata: pd.DataFrame, *, n_components: int,
    details: Mapping[str, Any], batch_size: int = 2048,
) -> tuple[pd.DataFrame, PCA | IncrementalPCA, dict[str, Any]]:
    """Fit randomized PCA on aggregates or bounded-memory PCA on raw selected rows."""
    analysis_rows = len(analysis_metadata)
    max_components = min(analysis_rows, int(activation_matrix.shape[1]))
    if not 1 <= n_components <= max_components:
        raise ValueError(
            f"PCA components must be between 1 and {max_components} for the prepared matrix."
        )

    if prepared_matrix is not None:
        pca: PCA | IncrementalPCA = PCA(
            n_components=n_components, svd_solver="randomized", random_state=0
        )
        projections = pca.fit_transform(prepared_matrix)
        solver = "randomized"
    else:
        effective_batch_size = max(batch_size, n_components)
        pca = IncrementalPCA(n_components=n_components, batch_size=effective_batch_size)
        batches = list(_bounded_batches(len(row_offsets), effective_batch_size, n_components))
        for batch in batches:
            values = np.asarray(activation_matrix[row_offsets[batch]], dtype=np.float32)
            pca.partial_fit(values)
            del values
        projections = np.empty((len(row_offsets), n_components), dtype=np.float32)
        for batch in batches:
            values = np.asarray(activation_matrix[row_offsets[batch]], dtype=np.float32)
            projections[batch] = pca.transform(values).astype(np.float32, copy=False)
            del values
        solver = "incremental"

    result = analysis_metadata.copy().reset_index(drop=True)
    for index in range(n_components):
        result[f"PC{index + 1}"] = projections[:, index]
    result["log10_time_horizon_months"] = np.log10(result["time_horizon_months"])
    projection_details = {
        **details,
        "explained_variance": pca.explained_variance_ratio_,
        "pca_solver": solver,
    }
    return result, pca, projection_details


def prepare_projection_from_matrix(
    activation_matrix: np.ndarray, metadata_index: pd.DataFrame, *,
    cached_position: Any, metadata_filters: Mapping[str, Any] | None,
    aggregation_fields: Sequence[str] | None, n_components: int,
    max_samples: int | None = None,
    phrasing_fields: Sequence[str] = (
        "template_metadata.prompt_framing", "template_metadata.output_format",
    ),
) -> tuple[pd.DataFrame, PCA | IncrementalPCA, dict[str, Any]]:
    """Prepare filtered data and fit PCA using bounded memory."""
    prepared = prepare_analysis_data(
        activation_matrix, metadata_index, cached_position=cached_position,
        metadata_filters=metadata_filters, aggregation_fields=aggregation_fields,
        max_samples=max_samples, phrasing_fields=phrasing_fields,
    )
    return fit_pca_projection(activation_matrix, *prepared[:3], n_components=n_components,
                              details=prepared[3])


def prepare_projection(
    sources: Sequence[str | Path | bytes | BinaryIO], *, layer_component: str,
    position_index: int, metadata_filters: Mapping[str, Any] | None,
    aggregation_fields: Sequence[str] | None, n_components: int,
    max_samples: int | None = None,
    phrasing_fields: Sequence[str] = (
        "template_metadata.prompt_framing", "template_metadata.output_format",
    ),
) -> tuple[pd.DataFrame, PCA, dict[str, Any]]:
    """Filter, aggregate, and refit PCA in the notebook's operation order."""
    feature_parts: list[torch.Tensor] = []
    metadata_records: list[dict[str, Any]] = []
    loaded_indices: list[int] = []
    absolute_positions: list[Any] = []
    cached_position: Any = None

    for source in sources:
        payload = _load(source)
        sample_indices, prompts = payload["sample_indices"], payload["prompts"]
        metadata_rows = payload["prompt_metadata"]
        try:
            tensor = payload["activations"][layer_component]
        except KeyError as exc:
            raise ValueError(f"Component {layer_component!r} is missing from a selected batch.") from exc
        if not (len(sample_indices) == len(prompts) == len(metadata_rows) == tensor.shape[0]):
            raise ValueError("A selected batch has misaligned activation and metadata rows.")
        positions = list(payload["positions"])
        if not 0 <= position_index < len(positions):
            raise ValueError(f"Position index {position_index} is unavailable in a selected batch.")
        position_value = positions[position_index]
        if cached_position is None:
            cached_position = position_value
        elif position_value != cached_position:
            raise ValueError("Selected batches have inconsistent cached positions.")
        row_offsets = [
            offset for offset, metadata in enumerate(metadata_rows)
            if all(metadata_value_matches(get_metadata_field(metadata, field), expected)
                   for field, expected in (metadata_filters or {}).items())
        ]
        if max_samples is not None:
            row_offsets = row_offsets[: max(max_samples - len(loaded_indices), 0)]
        if row_offsets:
            rows = torch.as_tensor(row_offsets, dtype=torch.long)
            feature_parts.append(tensor[:, position_index, :].index_select(0, rows).to(torch.float32))
            for offset in row_offsets:
                loaded_indices.append(int(sample_indices[offset]))
                metadata_records.append(flatten_scalar_metadata(metadata_rows[offset]))
                absolute_positions.append(position_value)
        if max_samples is not None and len(loaded_indices) >= max_samples:
            break

    if not feature_parts:
        raise ValueError("No activation rows match the selected filters.")
    activation_matrix = torch.cat(feature_parts, dim=0)
    metadata_df = pd.DataFrame(metadata_records)
    metadata_df.insert(0, "sample_index", loaded_indices)
    metadata_df.insert(1, "absolute_token_position", absolute_positions)
    _add_time_horizon_months(metadata_df)

    aggregation_fields = list(aggregation_fields or [])
    missing_fields = set(aggregation_fields) - set(metadata_df.columns)
    if missing_fields:
        raise ValueError(f"Missing aggregation fields: {sorted(missing_fields)}")
    if len(set(aggregation_fields)) != len(aggregation_fields):
        raise ValueError("Aggregation fields must not contain duplicates.")
    value_field = "base_value" if "base_value" in metadata_df else "value"
    unit_field = "base_unit" if "base_unit" in metadata_df else "unit"
    available_phrasing = [field for field in phrasing_fields if field in metadata_df]
    if aggregation_fields:
        vectors, records = [], []
        groups = metadata_df.groupby(aggregation_fields, dropna=False, sort=False).indices
        for offsets in groups.values():
            row_offsets = list(offsets)
            vectors.append(activation_matrix.index_select(
                0, torch.as_tensor(row_offsets, dtype=torch.long)).mean(dim=0))
            row = metadata_df.iloc[row_offsets[0]].copy()
            row["source_sample_count"] = len(row_offsets)
            for field in [value_field, unit_field, *available_phrasing]:
                if field in metadata_df and metadata_df.iloc[row_offsets][field].nunique(dropna=True) > 1:
                    row[field] = "<averaged>"
            records.append(row)
        pca_matrix = torch.stack(vectors)
        analysis_metadata = pd.DataFrame(records).reset_index(drop=True)
    else:
        pca_matrix = activation_matrix
        analysis_metadata = metadata_df.copy()
        analysis_metadata["source_sample_count"] = 1

    pca_input = pca_matrix.cpu().numpy()
    max_components = min(pca_input.shape)
    if not 1 <= n_components <= max_components:
        raise ValueError(f"PCA components must be between 1 and {max_components} for the prepared matrix.")
    pca = PCA(n_components=n_components, svd_solver="randomized", random_state=0)
    projections = pca.fit_transform(pca_input)
    result = analysis_metadata.copy().reset_index(drop=True)
    for index in range(n_components):
        result[f"PC{index + 1}"] = projections[:, index]
    result["log10_time_horizon_months"] = np.log10(result["time_horizon_months"])
    details = {
        "loaded_samples": len(loaded_indices), "analysis_rows": len(result),
        "feature_count": int(pca_input.shape[1]), "cached_position": cached_position,
        "explained_variance": pca.explained_variance_ratio_,
    }
    return result, pca, details


def _add_time_horizon_months(metadata_df: pd.DataFrame) -> None:
    value_field = "base_value" if "base_value" in metadata_df else "value"
    unit_field = "base_unit" if "base_unit" in metadata_df else "unit"
    missing = [field for field in (value_field, unit_field) if field not in metadata_df]
    if missing:
        raise ValueError(f"Time-horizon metadata is missing fields: {missing}")
    units = metadata_df[unit_field].astype(str).str.lower()
    unknown_units = sorted(set(units) - set(UNIT_TO_MONTHS))
    if unknown_units:
        raise ValueError(f"Cannot convert time-horizon units to months: {unknown_units}")
    metadata_df["time_horizon_months"] = [
        round(float(value) * UNIT_TO_MONTHS[unit], 12)
        for value, unit in zip(metadata_df[value_field], units, strict=True)
    ]


def unique_filter_values(
    sources: Iterable[str | Path | bytes | BinaryIO], fields: Sequence[str]
) -> dict[str, list[Any]]:
    """Collect scalar metadata values for filter controls without reading activations."""
    values, seen = {field: [] for field in fields}, {field: set() for field in fields}
    for source in sources:
        payload = _load(source)
        for metadata in payload["prompt_metadata"]:
            for field in fields:
                value = get_metadata_field(metadata, field)
                if value is not MISSING and value not in seen[field]:
                    seen[field].add(value)
                    values[field].append(value)
    return {field: sorted(field_values, key=str) for field, field_values in values.items()}
