"""Core data preparation for the local activation PCA explorer."""

from __future__ import annotations

import io
import json
import os
from collections import Counter
from collections.abc import Callable, Iterable, Mapping, Sequence
from hashlib import sha256
from pathlib import Path, PurePosixPath
from typing import Any, BinaryIO

import joblib
import numpy as np
import pandas as pd
import torch
from sklearn import __version__ as sklearn_version
from sklearn.cross_decomposition import PLSRegression
from sklearn.decomposition import PCA, IncrementalPCA
from pandas.api.types import is_scalar

from temporal_manifolds.activations.extraction_policy import (
    CACHED_POSITION_INDEX,
    PROMPT_TOKEN_POSITION,
    TARGET_LAYER_COMPONENT,
    validate_activation_payload,
    validate_cached_position,
    validate_extraction_request,
)

MISSING = object()
SOURCE_FOLDER_FIELD = "source_folder"
PCA_MODEL_ARTIFACT_KIND = "temporal-manifolds.activation-pca"
PCA_MODEL_ARTIFACT_VERSION = 1
PLS_MODEL_ARTIFACT_KIND = "temporal-manifolds.activation-pls"
PLS_MODEL_ARTIFACT_VERSION = 1
PCA_PROJECTION_FINGERPRINT_VERSION = 2
MetadataValueToken = tuple[str, str, str]

UNIT_TO_MONTHS = {
    "second": 1 / (30.4375 * 86400),
    "minute": 1 / (30.4375 * 1440),
    "hour": 1 / (30.4375 * 24),
    "day": 1 / 30.4375,
    "week": 7 / 30.4375,
    "month": 1,
    "year": 12,
    "decade": 120,
    "century": 1200,
    "millennium": 12000,
}
UNIT_TO_MONTHS.update({f"{unit}s": value for unit, value in list(UNIT_TO_MONTHS.items())})
UNIT_TO_MONTHS["centuries"] = 1200
UNIT_TO_MONTHS["millennia"] = 12000


def _update_array_fingerprint(signature, value: Any) -> None:
    array = np.ascontiguousarray(value)
    signature.update(str(array.shape).encode("ascii"))
    signature.update(array.dtype.str.encode("ascii"))
    signature.update(array.tobytes())


def pca_projection_fingerprint(
    pca: Any, *, version: int = PCA_PROJECTION_FINGERPRINT_VERSION
) -> str:
    """Fingerprint every fitted PCA field that changes transformed coordinates.

    Version 1 reproduces the original components-and-mean digest so surface artifacts saved
    before GUI loading was added remain usable. Version 2 also identifies whitening and its
    fitted variance scale.
    """

    if version not in {1, PCA_PROJECTION_FINGERPRINT_VERSION}:
        raise ValueError(f"Unsupported PCA projection fingerprint version: {version!r}.")
    if not hasattr(pca, "components_") or not hasattr(pca, "mean_"):
        raise ValueError("The PCA model must be fitted before it can be fingerprinted.")

    signature = sha256()
    if version == PCA_PROJECTION_FINGERPRINT_VERSION:
        signature.update(b"temporal-manifolds.pca-coordinate-system.v2\0")
    _update_array_fingerprint(signature, pca.components_)
    _update_array_fingerprint(signature, pca.mean_)
    if version == PCA_PROJECTION_FINGERPRINT_VERSION:
        whiten = bool(getattr(pca, "whiten", False))
        signature.update(b"whiten=1" if whiten else b"whiten=0")
        if whiten:
            if not hasattr(pca, "explained_variance_"):
                raise ValueError("The whitened PCA model has no fitted variance scale.")
            _update_array_fingerprint(signature, pca.explained_variance_)
    return signature.hexdigest()


def activation_batch_basename(name: str) -> str | None:
    """Return a valid activation batch basename from an upload-relative path."""
    basename = PurePosixPath(name.replace("\\", "/")).name
    if basename.startswith("activations_batch_") and basename.endswith(".pt"):
        return basename
    return None


def _activation_source_parent(source: Any) -> tuple[tuple[str, str], tuple[str, ...]]:
    """Return a canonical parent identity and displayable path parts."""
    if isinstance(source, (str, Path)):
        parent = Path(source).expanduser().resolve().parent
        identity = ("local", os.path.normcase(str(parent)))
        parts = tuple(part.rstrip("\\/") or part for part in parent.parts)
        return identity, parts

    name = str(getattr(source, "name", ""))
    if not name:
        return ("unknown", "<unknown>"), ("<unknown>",)
    parent = PurePosixPath(name.replace("\\", "/")).parent
    if parent == PurePosixPath("."):
        return ("upload", "<root>"), ("<root>",)
    return ("upload", parent.as_posix()), parent.parts


def activation_source_folder_names(sources: Sequence[Any]) -> list[str]:
    """Return concise folder labels, expanding duplicate leaf names just enough to differ."""
    parents = [_activation_source_parent(source) for source in sources]
    unique_parents = dict(parents)
    labels: dict[tuple[str, str], str] = {}
    for identity, parts in unique_parents.items():
        for depth in range(1, len(parts) + 1):
            candidate = "/".join(parts[-depth:])
            if all(
                other_identity == identity
                or "/".join(other_parts[-depth:]) != candidate
                for other_identity, other_parts in unique_parents.items()
            ):
                labels[identity] = candidate
                break
        else:
            labels[identity] = "/".join(parts)

    duplicate_labels = Counter(labels.values())
    for identity, label in tuple(labels.items()):
        if duplicate_labels[label] > 1:
            labels[identity] = f"{identity[0]}:{label}"
    return [labels[identity] for identity, _ in parents]


def activation_source_folder_name(source: Any) -> str:
    """Return the folder label for one local path or directory upload."""
    return activation_source_folder_names([source])[0]


def select_activation_batch_uploads(files: Iterable[Any]) -> list[Any]:
    """Keep activation-batch uploads in order without merging duplicate filenames."""
    return [
        file
        for file in files
        if activation_batch_basename(str(getattr(file, "name", ""))) is not None
    ]


def discover_activation_batch_paths(
    folders: Iterable[str | Path],
) -> tuple[list[str], list[str]]:
    """Discover batches across roots, deduplicating only identical resolved paths."""
    roots: list[str] = []
    sources: list[str] = []
    seen_roots: set[str] = set()
    seen_sources: set[str] = set()
    for folder in folders:
        path = Path(folder).expanduser()
        if not path.is_dir():
            raise ValueError(f"Folder does not exist or is not accessible: {path}")
        resolved_root = str(path.resolve())
        if resolved_root in seen_roots:
            continue
        seen_roots.add(resolved_root)
        roots.append(resolved_root)
        for item in sorted(path.rglob("activations_batch_*.pt")):
            resolved_source = str(item.resolve())
            if resolved_source not in seen_sources:
                seen_sources.add(resolved_source)
                sources.append(resolved_source)
    if not roots:
        raise ValueError("Enter at least one folder path.")
    return sources, roots


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


def metadata_value_token(value: Any) -> MetadataValueToken:
    """Return a stable, typed token suitable for metadata selection widgets."""
    if is_scalar(value):
        try:
            if bool(pd.isna(value)):
                return ("missing", "", "")
        except (TypeError, ValueError):
            pass
    normalized = value.tolist() if hasattr(value, "tolist") else value
    try:
        representation = json.dumps(
            normalized,
            sort_keys=True,
            ensure_ascii=False,
            default=repr,
        )
    except (TypeError, ValueError):
        representation = repr(normalized)
    value_type = type(value)
    return (
        "value",
        f"{value_type.__module__}.{value_type.__qualname__}",
        representation,
    )


def metadata_filter_choices(values: Iterable[Any]) -> list[tuple[MetadataValueToken, str]]:
    """Return unique typed metadata tokens and unambiguous display labels."""
    representatives: dict[MetadataValueToken, Any] = {}
    for value in values:
        representatives.setdefault(metadata_value_token(value), value)

    base_labels = {
        token: (
            "<missing>"
            if token[0] == "missing"
            else "<empty string>"
            if isinstance(value, str) and value == ""
            else str(value)
        )
        for token, value in representatives.items()
    }
    label_counts = Counter(base_labels.values())
    labels: dict[MetadataValueToken, str] = {}
    used_labels: set[str] = set()
    for token, base_label in base_labels.items():
        type_label = "missing" if token[0] == "missing" else token[1].rsplit(".", 1)[-1]
        label = f"{base_label} · {type_label}" if label_counts[base_label] > 1 else base_label
        if label in used_labels:
            suffix = 2
            while f"{label} · {suffix}" in used_labels:
                suffix += 1
            label = f"{label} · {suffix}"
        used_labels.add(label)
        labels[token] = label
    return sorted(labels.items(), key=lambda item: (item[1].casefold(), item[0]))


def metadata_filter_mask(
    dataframe: pd.DataFrame,
    filters: Mapping[str, Sequence[MetadataValueToken]],
) -> np.ndarray:
    """Select rows with OR-within-field and AND-across-field semantics."""
    selected = np.ones(len(dataframe), dtype=bool)
    for field, allowed_tokens in filters.items():
        if field not in dataframe:
            raise ValueError(f"Metadata filter field is unavailable: {field}")
        allowed = set(allowed_tokens)
        if not allowed:
            continue
        selected &= np.fromiter(
            (metadata_value_token(value) in allowed for value in dataframe[field]),
            dtype=bool,
            count=len(dataframe),
        )
    return selected


def validate_pca_model(pca: Any) -> tuple[int, int]:
    """Validate a fitted PCA estimator and return component and feature counts."""
    if not isinstance(pca, (PCA, IncrementalPCA)):
        raise ValueError("The selected file does not contain a PCA or IncrementalPCA model.")
    components = getattr(pca, "components_", None)
    explained_variance = getattr(pca, "explained_variance_ratio_", None)
    if components is None or explained_variance is None:
        raise ValueError("The selected PCA model has not been fitted.")
    components = np.asarray(components)
    explained_variance = np.asarray(explained_variance)
    if components.ndim != 2 or not all(components.shape):
        raise ValueError("The selected PCA model has invalid fitted components.")
    component_count, feature_count = map(int, components.shape)
    if explained_variance.shape != (component_count,):
        raise ValueError("The selected PCA model has invalid explained-variance metadata.")
    fitted_feature_count = int(getattr(pca, "n_features_in_", feature_count))
    if fitted_feature_count != feature_count:
        raise ValueError("The selected PCA model has inconsistent feature metadata.")
    return component_count, feature_count


def serialize_pca_model(
    pca: PCA | IncrementalPCA,
    *,
    metadata: Mapping[str, Any] | None = None,
) -> bytes:
    """Serialize a fitted PCA estimator and provenance as a versioned joblib artifact."""
    component_count, feature_count = validate_pca_model(pca)
    artifact = {
        "kind": PCA_MODEL_ARTIFACT_KIND,
        "version": PCA_MODEL_ARTIFACT_VERSION,
        "model": pca,
        "sklearn_version": sklearn_version,
        "component_count": component_count,
        "feature_count": feature_count,
        "metadata": dict(metadata or {}),
    }
    buffer = io.BytesIO()
    joblib.dump(artifact, buffer, compress=3)
    return buffer.getvalue()


def load_pca_model(
    source: str | Path | bytes | BinaryIO,
) -> tuple[PCA | IncrementalPCA, dict[str, Any]]:
    """Load a trusted PCA artifact or raw estimator and return its provenance.

    Joblib and pickle files can execute arbitrary code while loading. Callers must
    only pass files from trusted sources.
    """
    if isinstance(source, bytes):
        source = io.BytesIO(source)
    elif hasattr(source, "seek"):
        source.seek(0)
    try:
        payload = joblib.load(source)
    except Exception as exc:  # noqa: BLE001 - normalize artifact errors for UI callers
        raise ValueError(f"The PCA model file could not be loaded: {exc}") from exc

    if isinstance(payload, (PCA, IncrementalPCA)):
        validate_pca_model(payload)
        return payload, {}
    if not isinstance(payload, Mapping) or payload.get("kind") != PCA_MODEL_ARTIFACT_KIND:
        raise ValueError("The selected file is not a supported PCA model artifact.")
    if payload.get("version") != PCA_MODEL_ARTIFACT_VERSION:
        raise ValueError(f"Unsupported PCA artifact version: {payload.get('version')!r}.")
    pca = payload.get("model")
    component_count, feature_count = validate_pca_model(pca)
    if payload.get("component_count") != component_count:
        raise ValueError("The PCA artifact's component count does not match its model.")
    if payload.get("feature_count") != feature_count:
        raise ValueError("The PCA artifact's feature count does not match its model.")
    metadata = payload.get("metadata", {})
    if not isinstance(metadata, Mapping):
        raise ValueError("The PCA artifact contains invalid provenance metadata.")
    provenance = {
        **dict(metadata),
        "artifact_kind": PCA_MODEL_ARTIFACT_KIND,
        "artifact_version": PCA_MODEL_ARTIFACT_VERSION,
        "sklearn_version": payload.get("sklearn_version"),
        "component_count": component_count,
        "feature_count": feature_count,
    }
    return pca, provenance


def validate_pls_model(pls: Any) -> tuple[int, int]:
    """Validate a fitted Activation Atlas PLS estimator."""
    if not isinstance(pls, PLSRegression):
        raise ValueError("The selected file does not contain a PLSRegression model.")

    components = getattr(pls, "components_", None)
    explained_variance = getattr(pls, "explained_variance_ratio_", None)
    rotations = getattr(pls, "x_rotations_", None)
    mean = getattr(pls, "mean_", None)
    x_scale = getattr(pls, "_x_std", None)
    if any(value is None for value in (components, explained_variance, rotations, mean, x_scale)):
        raise ValueError(
            "The selected PLS model is not a fitted Activation Atlas PLS artifact."
        )

    components = np.asarray(components)
    explained_variance = np.asarray(explained_variance)
    rotations = np.asarray(rotations)
    mean = np.asarray(mean)
    x_scale = np.asarray(x_scale)
    if components.ndim != 2 or not all(components.shape):
        raise ValueError("The selected PLS model has invalid fitted components.")
    component_count, feature_count = map(int, components.shape)
    if rotations.shape != (feature_count, component_count):
        raise ValueError("The selected PLS model has inconsistent rotations.")
    if explained_variance.shape != (component_count,):
        raise ValueError("The selected PLS model has invalid represented-variance metadata.")
    if mean.shape != (feature_count,) or x_scale.shape != (feature_count,):
        raise ValueError("The selected PLS model has inconsistent centering metadata.")
    if not np.isfinite(components).all() or not np.isfinite(mean).all():
        raise ValueError("The selected PLS model contains non-finite projection parameters.")
    if not np.isfinite(x_scale).all() or np.any(x_scale <= 0):
        raise ValueError("The selected PLS model has invalid activation scales.")
    fitted_feature_count = int(getattr(pls, "n_features_in_", feature_count))
    if fitted_feature_count != feature_count:
        raise ValueError("The selected PLS model has inconsistent feature metadata.")
    return component_count, feature_count


def serialize_pls_model(
    pls: PLSRegression,
    *,
    metadata: Mapping[str, Any] | None = None,
) -> bytes:
    """Serialize a fitted PLS estimator and provenance as a versioned joblib artifact."""
    component_count, feature_count = validate_pls_model(pls)
    artifact_metadata = dict(metadata or {})
    artifact_metadata.update(
        {
            "pls_scale": bool(pls.scale),
            "pls_max_iter": int(pls.max_iter),
            "pls_tolerance": float(pls.tol),
        }
    )
    artifact = {
        "kind": PLS_MODEL_ARTIFACT_KIND,
        "version": PLS_MODEL_ARTIFACT_VERSION,
        "model": pls,
        "sklearn_version": sklearn_version,
        "component_count": component_count,
        "feature_count": feature_count,
        "metadata": artifact_metadata,
    }
    buffer = io.BytesIO()
    joblib.dump(artifact, buffer, compress=3)
    return buffer.getvalue()


def load_pls_model(
    source: str | Path | bytes | BinaryIO,
) -> tuple[PLSRegression, dict[str, Any]]:
    """Load a trusted Activation Atlas PLS artifact and return its provenance.

    Joblib and pickle files can execute arbitrary code while loading. Callers must
    only pass files from trusted sources.
    """
    if isinstance(source, bytes):
        source = io.BytesIO(source)
    elif hasattr(source, "seek"):
        source.seek(0)
    try:
        payload = joblib.load(source)
    except Exception as exc:  # noqa: BLE001 - normalize artifact errors for UI callers
        raise ValueError(f"The PLS model file could not be loaded: {exc}") from exc

    if not isinstance(payload, Mapping) or payload.get("kind") != PLS_MODEL_ARTIFACT_KIND:
        raise ValueError("The selected file is not a supported PLS model artifact.")
    if payload.get("version") != PLS_MODEL_ARTIFACT_VERSION:
        raise ValueError(f"Unsupported PLS artifact version: {payload.get('version')!r}.")
    pls = payload.get("model")
    component_count, feature_count = validate_pls_model(pls)
    if payload.get("component_count") != component_count:
        raise ValueError("The PLS artifact's component count does not match its model.")
    if payload.get("feature_count") != feature_count:
        raise ValueError("The PLS artifact's feature count does not match its model.")
    metadata = payload.get("metadata", {})
    if not isinstance(metadata, Mapping):
        raise ValueError("The PLS artifact contains invalid provenance metadata.")
    provenance = {
        **dict(metadata),
        "pls_scale": bool(pls.scale),
        "pls_max_iter": int(pls.max_iter),
        "pls_tolerance": float(pls.tol),
        "artifact_kind": PLS_MODEL_ARTIFACT_KIND,
        "artifact_version": PLS_MODEL_ARTIFACT_VERSION,
        "sklearn_version": payload.get("sklearn_version"),
        "component_count": component_count,
        "feature_count": feature_count,
    }
    return pls, provenance


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
    source_folders = activation_source_folder_names(sources)
    for source_index, (source, source_folder) in enumerate(
        zip(sources, source_folders, strict=True)
    ):
        payload = _load(source)
        validate_activation_payload(
            payload,
            source_name=f"Activation batch {source_index + 1}",
        )
        current_components = {TARGET_LAYER_COMPONENT}
        components = current_components if components is None else components & current_components
        current_positions = [PROMPT_TOKEN_POSITION]
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
            metadata_fields.add(SOURCE_FOLDER_FIELD)
            metadata_records.append(
                {
                    "sample_index": int(sample_index),
                    "_source_index": source_index,
                    "_row_offset": row_offset,
                    **flattened,
                    SOURCE_FOLDER_FIELD: source_folder,
                }
            )
        del payload
    if not components:
        raise ValueError("Selected batches do not share an activation component.")
    return {
        "components": sorted(components),
        "positions": positions or [],
        "metadata_fields": sorted(metadata_fields),
        "batch_count": len(sources),
        "row_count": rows,
        "metadata_index": pd.DataFrame(metadata_records),
        "source_row_counts": source_row_counts,
    }


def extract_activation_slice(
    sources: Sequence[str | Path | bytes | BinaryIO],
    *,
    layer_component: str,
    position_index: int,
    source_row_counts: Sequence[int],
    cache_dir: str | Path | None = None,
    progress: Callable[[int, int], None] | None = None,
) -> tuple[np.ndarray, Path | None]:
    """Stream one component/position into a reusable RAM or disk-backed matrix."""
    validate_extraction_request(
        layer_component=layer_component,
        position_index=position_index,
    )
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
            tensor = validate_activation_payload(
                payload,
                source_name=f"Activation batch {source_index + 1}",
            )
            if tensor.shape[0] != expected_rows:
                raise ValueError(f"Batch {source_index + 1} changed since metadata was indexed.")
            if not 0 <= position_index < tensor.shape[1]:
                raise ValueError(
                    f"Position {position_index} is unavailable in batch {source_index + 1}."
                )
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
            matrix[row_start:row_end] = (
                tensor[:, CACHED_POSITION_INDEX, :].to(torch.float32).numpy()
            )
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
    activation_matrix: np.ndarray,
    metadata_index: pd.DataFrame,
    *,
    cached_position: Any,
    metadata_filters: Mapping[str, Any] | None,
    aggregation_fields: Sequence[str] | None,
    max_samples: int | None = None,
    chunk_size: int = 2048,
    phrasing_fields: Sequence[str] = (
        "template_metadata.prompt_framing",
        "template_metadata.output_format",
    ),
) -> tuple[np.ndarray | None, np.ndarray, pd.DataFrame, dict[str, Any]]:
    """Filter metadata and aggregate a memory map without copying all selected rows."""
    validate_cached_position(cached_position)
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

    metadata_df = (
        metadata_index.iloc[row_offsets]
        .drop(columns=["_source_index", "_row_offset"], errors="ignore")
        .reset_index(drop=True)
    )
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
        prepared_matrix = np.empty((len(groups), activation_matrix.shape[1]), dtype=np.float32)
        for group_index, offsets in enumerate(groups.values()):
            group_offsets = np.asarray(offsets, dtype=np.int64)
            source_offsets = row_offsets[group_offsets]
            vector_sum = np.zeros(activation_matrix.shape[1], dtype=np.float32)
            for start in range(0, len(source_offsets), chunk_size):
                chunk_offsets = source_offsets[start : start + chunk_size]
                chunk = np.asarray(activation_matrix[chunk_offsets], dtype=np.float32)
                vector_sum += chunk.sum(axis=0, dtype=np.float32)
                del chunk
            prepared_matrix[group_index] = vector_sum / len(source_offsets)
            row = metadata_df.iloc[group_offsets[0]].copy()
            row["source_sample_count"] = len(group_offsets)
            for field in [value_field, unit_field, *available_phrasing, SOURCE_FOLDER_FIELD]:
                if field not in metadata_df:
                    continue
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


def _build_projection_result(
    analysis_metadata: pd.DataFrame,
    projections: np.ndarray,
    pca: PCA | IncrementalPCA,
    details: Mapping[str, Any],
    *,
    pca_source: str,
    pca_solver: str,
) -> tuple[pd.DataFrame, PCA | IncrementalPCA, dict[str, Any]]:
    component_count, _ = validate_pca_model(pca)
    projections = np.asarray(projections)
    expected_shape = (len(analysis_metadata), component_count)
    if projections.shape != expected_shape:
        raise ValueError(
            "PCA projections and prepared metadata are misaligned: "
            f"expected {expected_shape}, got {projections.shape}."
        )
    result = analysis_metadata.copy().reset_index(drop=True)
    for index in range(component_count):
        result[f"PC{index + 1}"] = projections[:, index]
    result["log10_time_horizon_months"] = np.log10(result["time_horizon_months"])
    projection_details = {
        **details,
        "explained_variance": np.asarray(pca.explained_variance_ratio_),
        "pca_solver": pca_solver,
        "pca_source": pca_source,
    }
    return result, pca, projection_details


def projection_details_table(projection: pd.DataFrame) -> pd.DataFrame:
    """Return an Arrow-compatible frame for the Streamlit projection details table."""
    value_field = "base_value" if "base_value" in projection else "value"
    if value_field not in projection or not projection[value_field].eq("<averaged>").any():
        return projection

    table = projection.copy()
    table[value_field] = table[value_field].astype("string")
    return table


def reconstruction_residual_rms(
    activation_matrix: np.ndarray,
    prepared_matrix: np.ndarray | None,
    row_offsets: np.ndarray,
    scores: np.ndarray,
    model: PCA | IncrementalPCA | PLSRegression,
    *,
    batch_size: int = 2048,
) -> np.ndarray:
    """Return each point's activation-space reconstruction-residual RMS."""
    if batch_size < 1:
        raise ValueError("Reconstruction batch size must be positive.")

    scores = np.asarray(scores)
    row_count = len(scores)
    if prepared_matrix is not None:
        values = np.asarray(prepared_matrix)
        if values.ndim != 2 or len(values) != row_count:
            raise ValueError("Prepared activations and projection scores are misaligned.")
    else:
        if len(row_offsets) != row_count:
            raise ValueError("Activation row offsets and projection scores are misaligned.")
        values = None

    residual_rms = np.empty(row_count, dtype=np.float64)
    for batch in _bounded_batches(row_count, batch_size, 1):
        batch_values = (
            np.asarray(values[batch], dtype=np.float64)
            if values is not None
            else np.asarray(activation_matrix[row_offsets[batch]], dtype=np.float64)
        )
        reconstructed = np.asarray(model.inverse_transform(scores[batch]), dtype=np.float64)
        if reconstructed.shape != batch_values.shape:
            raise ValueError("Reconstructed activations have an invalid shape.")
        residual_rms[batch] = np.sqrt(
            np.mean(np.square(batch_values - reconstructed), axis=1)
        )
    return residual_rms


def reconstruction_residual_statistics(
    activation_matrix: np.ndarray,
    prepared_matrix: np.ndarray | None,
    row_offsets: np.ndarray,
    scores: np.ndarray,
    model: PCA | IncrementalPCA | PLSRegression,
    *,
    n_components: int = 3,
    batch_size: int = 2048,
) -> tuple[np.ndarray, np.ndarray, IncrementalPCA]:
    """Fit residual PCA and return RMS plus residual coordinates for every point."""
    scores = np.asarray(scores)
    row_count = len(scores)
    if batch_size < 1:
        raise ValueError("Reconstruction batch size must be positive.")
    if not 1 <= n_components <= row_count:
        raise ValueError("Residual PCA components must not exceed the projected row count.")
    feature_count = int(activation_matrix.shape[1])
    if n_components > feature_count:
        raise ValueError("Residual PCA components must not exceed the activation width.")
    if prepared_matrix is not None:
        values = np.asarray(prepared_matrix)
        if values.ndim != 2 or len(values) != row_count:
            raise ValueError("Prepared activations and projection scores are misaligned.")
    else:
        if len(row_offsets) != row_count:
            raise ValueError("Activation row offsets and projection scores are misaligned.")
        values = None

    residual_rms = np.empty(row_count, dtype=np.float64)
    residual_pca = IncrementalPCA(n_components=n_components, batch_size=batch_size)
    batches = list(_bounded_batches(row_count, batch_size, n_components))
    for batch in batches:
        batch_values = (
            np.asarray(values[batch], dtype=np.float64)
            if values is not None
            else np.asarray(activation_matrix[row_offsets[batch]], dtype=np.float64)
        )
        reconstructed = np.asarray(model.inverse_transform(scores[batch]), dtype=np.float64)
        if reconstructed.shape != batch_values.shape:
            raise ValueError("Reconstructed activations have an invalid shape.")
        residuals = batch_values - reconstructed
        residual_rms[batch] = np.sqrt(np.mean(np.square(residuals), axis=1))
        residual_pca.partial_fit(residuals)

    residual_scores = np.empty((row_count, n_components), dtype=np.float64)
    for batch in batches:
        batch_values = (
            np.asarray(values[batch], dtype=np.float64)
            if values is not None
            else np.asarray(activation_matrix[row_offsets[batch]], dtype=np.float64)
        )
        residuals = batch_values - np.asarray(
            model.inverse_transform(scores[batch]), dtype=np.float64
        )
        residual_scores[batch] = residual_pca.transform(residuals)
    return residual_rms, residual_scores, residual_pca


def reconstruction_residual_projection(
    activation_matrix: np.ndarray,
    prepared_matrix: np.ndarray | None,
    row_offsets: np.ndarray,
    scores: np.ndarray,
    model: PCA | IncrementalPCA | PLSRegression,
    *,
    residual_center: np.ndarray,
    residual_components: np.ndarray,
    batch_size: int = 2048,
) -> tuple[np.ndarray, np.ndarray]:
    """Project reconstruction residuals with residual-PCA parameters from a saved model."""
    scores = np.asarray(scores)
    center = np.asarray(residual_center, dtype=np.float64)
    components = np.asarray(residual_components, dtype=np.float64)
    if center.ndim != 1 or components.ndim != 2 or components.shape[1:] != center.shape:
        raise ValueError("Saved residual PCA parameters have incompatible shapes.")
    if center.shape[0] != int(activation_matrix.shape[1]):
        raise ValueError(
            "The spherical model residual PCA expects a different activation width."
        )
    row_count = len(scores)
    values = np.asarray(prepared_matrix) if prepared_matrix is not None else None
    if values is not None and (values.ndim != 2 or len(values) != row_count):
        raise ValueError("Prepared activations and projection scores are misaligned.")
    if values is None and len(row_offsets) != row_count:
        raise ValueError("Activation row offsets and projection scores are misaligned.")

    residual_rms = np.empty(row_count, dtype=np.float64)
    residual_scores = np.empty((row_count, components.shape[0]), dtype=np.float64)
    for batch in _bounded_batches(row_count, batch_size, 1):
        batch_values = (
            np.asarray(values[batch], dtype=np.float64)
            if values is not None
            else np.asarray(activation_matrix[row_offsets[batch]], dtype=np.float64)
        )
        residuals = batch_values - np.asarray(
            model.inverse_transform(scores[batch]), dtype=np.float64
        )
        residual_rms[batch] = np.sqrt(np.mean(np.square(residuals), axis=1))
        residual_scores[batch] = (residuals - center) @ components.T
    return residual_rms, residual_scores


def fit_pca_projection(
    activation_matrix: np.ndarray,
    prepared_matrix: np.ndarray | None,
    row_offsets: np.ndarray,
    analysis_metadata: pd.DataFrame,
    *,
    n_components: int,
    details: Mapping[str, Any],
    batch_size: int = 2048,
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

    return _build_projection_result(
        analysis_metadata,
        projections,
        pca,
        details,
        pca_source="fitted",
        pca_solver=solver,
    )


def transform_pca_projection(
    activation_matrix: np.ndarray,
    prepared_matrix: np.ndarray | None,
    row_offsets: np.ndarray,
    analysis_metadata: pd.DataFrame,
    *,
    pca: PCA | IncrementalPCA,
    details: Mapping[str, Any],
    batch_size: int = 2048,
) -> tuple[pd.DataFrame, PCA | IncrementalPCA, dict[str, Any]]:
    """Project prepared activations through a fitted PCA without refitting it."""
    component_count, expected_feature_count = validate_pca_model(pca)
    feature_count = int(activation_matrix.shape[1])
    if feature_count != expected_feature_count:
        raise ValueError(
            f"The loaded PCA expects {expected_feature_count:,} activation features, "
            f"but the selected component has {feature_count:,}."
        )
    if batch_size < 1:
        raise ValueError("PCA transform batch size must be positive.")

    analysis_rows = len(analysis_metadata)
    if prepared_matrix is not None:
        if prepared_matrix.ndim != 2 or prepared_matrix.shape != (
            analysis_rows,
            expected_feature_count,
        ):
            raise ValueError("Prepared activations and metadata are misaligned.")
        projections = pca.transform(prepared_matrix)
    else:
        if len(row_offsets) != analysis_rows:
            raise ValueError("Activation row offsets and metadata are misaligned.")
        projection_dtype = np.result_type(np.asarray(pca.components_).dtype, np.float32)
        projections = np.empty((analysis_rows, component_count), dtype=projection_dtype)
        for batch in _bounded_batches(analysis_rows, batch_size, 1):
            values = np.asarray(activation_matrix[row_offsets[batch]], dtype=np.float32)
            projections[batch] = pca.transform(values)
            del values

    return _build_projection_result(
        analysis_metadata,
        projections,
        pca,
        details,
        pca_source="loaded",
        pca_solver=type(pca).__name__,
    )


def prepare_projection_from_matrix(
    activation_matrix: np.ndarray,
    metadata_index: pd.DataFrame,
    *,
    cached_position: Any,
    metadata_filters: Mapping[str, Any] | None,
    aggregation_fields: Sequence[str] | None,
    n_components: int,
    max_samples: int | None = None,
    phrasing_fields: Sequence[str] = (
        "template_metadata.prompt_framing",
        "template_metadata.output_format",
    ),
) -> tuple[pd.DataFrame, PCA | IncrementalPCA, dict[str, Any]]:
    """Prepare filtered data and fit PCA using bounded memory."""
    validate_cached_position(cached_position)
    prepared = prepare_analysis_data(
        activation_matrix,
        metadata_index,
        cached_position=cached_position,
        metadata_filters=metadata_filters,
        aggregation_fields=aggregation_fields,
        max_samples=max_samples,
        phrasing_fields=phrasing_fields,
    )
    return fit_pca_projection(
        activation_matrix, *prepared[:3], n_components=n_components, details=prepared[3]
    )


def prepare_projection(
    sources: Sequence[str | Path | bytes | BinaryIO],
    *,
    layer_component: str,
    position_index: int,
    metadata_filters: Mapping[str, Any] | None,
    aggregation_fields: Sequence[str] | None,
    n_components: int,
    max_samples: int | None = None,
    phrasing_fields: Sequence[str] = (
        "template_metadata.prompt_framing",
        "template_metadata.output_format",
    ),
) -> tuple[pd.DataFrame, PCA, dict[str, Any]]:
    """Filter, aggregate, and refit PCA in the notebook's operation order."""
    validate_extraction_request(
        layer_component=layer_component,
        position_index=position_index,
    )
    feature_parts: list[torch.Tensor] = []
    metadata_records: list[dict[str, Any]] = []
    loaded_indices: list[int] = []
    absolute_positions: list[Any] = []
    cached_position: Any = None
    source_folders = activation_source_folder_names(sources)

    for source_index, (source, source_folder) in enumerate(
        zip(sources, source_folders, strict=True)
    ):
        payload = _load(source)
        tensor = validate_activation_payload(
            payload,
            source_name=f"Activation batch {source_index + 1}",
        )
        sample_indices, prompts = payload["sample_indices"], payload["prompts"]
        metadata_rows = payload["prompt_metadata"]
        if not (len(sample_indices) == len(prompts) == len(metadata_rows) == tensor.shape[0]):
            raise ValueError("A selected batch has misaligned activation and metadata rows.")
        positions = list(payload["positions"])
        if not 0 <= position_index < len(positions):
            raise ValueError(
                f"Position index {position_index} is unavailable in a selected batch."
            )
        position_value = PROMPT_TOKEN_POSITION
        if cached_position is None:
            cached_position = position_value
        elif position_value != cached_position:
            raise ValueError("Selected batches have inconsistent cached positions.")
        row_offsets = [
            offset
            for offset, metadata in enumerate(metadata_rows)
            if all(
                metadata_value_matches(
                    source_folder
                    if field == SOURCE_FOLDER_FIELD
                    else get_metadata_field(metadata, field),
                    expected,
                )
                for field, expected in (metadata_filters or {}).items()
            )
        ]
        if max_samples is not None:
            row_offsets = row_offsets[: max(max_samples - len(loaded_indices), 0)]
        if row_offsets:
            rows = torch.as_tensor(row_offsets, dtype=torch.long)
            feature_parts.append(
                tensor[:, CACHED_POSITION_INDEX, :]
                .index_select(0, rows)
                .to(torch.float32)
            )
            for offset in row_offsets:
                loaded_indices.append(int(sample_indices[offset]))
                metadata_records.append(
                    {
                        **flatten_scalar_metadata(metadata_rows[offset]),
                        SOURCE_FOLDER_FIELD: source_folder,
                    }
                )
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
            vectors.append(
                activation_matrix.index_select(
                    0, torch.as_tensor(row_offsets, dtype=torch.long)
                ).mean(dim=0)
            )
            row = metadata_df.iloc[row_offsets[0]].copy()
            row["source_sample_count"] = len(row_offsets)
            for field in [value_field, unit_field, *available_phrasing, SOURCE_FOLDER_FIELD]:
                if (
                    field in metadata_df
                    and metadata_df.iloc[row_offsets][field].nunique(dropna=True) > 1
                ):
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
        raise ValueError(
            f"PCA components must be between 1 and {max_components} for the prepared matrix."
        )
    pca = PCA(n_components=n_components, svd_solver="randomized", random_state=0)
    projections = pca.fit_transform(pca_input)
    result = analysis_metadata.copy().reset_index(drop=True)
    for index in range(n_components):
        result[f"PC{index + 1}"] = projections[:, index]
    result["log10_time_horizon_months"] = np.log10(result["time_horizon_months"])
    details = {
        "loaded_samples": len(loaded_indices),
        "analysis_rows": len(result),
        "feature_count": int(pca_input.shape[1]),
        "cached_position": cached_position,
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
    sources = list(sources)
    source_folders = activation_source_folder_names(sources)
    for source, source_folder in zip(sources, source_folders, strict=True):
        payload = _load(source)
        for metadata in payload["prompt_metadata"]:
            for field in fields:
                value = (
                    source_folder
                    if field == SOURCE_FOLDER_FIELD
                    else get_metadata_field(metadata, field)
                )
                if value is not MISSING and value not in seen[field]:
                    seen[field].add(value)
                    values[field].append(value)
    return {field: sorted(field_values, key=str) for field, field_values in values.items()}
