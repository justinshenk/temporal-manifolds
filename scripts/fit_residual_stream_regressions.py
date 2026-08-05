"""Fit regression probes from residual-stream activations to base-10 log-month horizon.

The input activation chunks are those produced by
``extract_residual_stream_positions_from_gcs.py``. Every fitted probe uses one
residual-stream layer at exactly one cached token position (0, 1, or 2).
"""

from __future__ import annotations

import argparse
import csv
import gc
import json
import math
import os
import re
from pathlib import Path
from typing import Any, Iterable, Iterator

import joblib
import numpy as np
import torch
from dotenv import load_dotenv
from google.cloud import storage
from sklearn.base import RegressorMixin
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


DEFAULT_ACTIVATIONS_URI = "gs://temporal-research-bucket/resid_only_0_2"
DEFAULT_COMPLETIONS_URI = (
    "gs://temporal-research-bucket/completions/completions_256.jsonl"
)
DEFAULT_DATA_DIR = Path("data/residual_stream_regression")
DEFAULT_OUTPUT_DIR = Path("results/residual_stream_regression")
POSITION_INDICES = (0, 1, 2)
MIN_LAYER_INDEX = 18
MONTHS_PER_UNIT = {
    "second": 1.0 / (30.4375 * 24.0 * 60.0 * 60.0),
    "minute": 1.0 / (30.4375 * 24.0 * 60.0),
    "hour": 1.0 / (30.4375 * 24.0),
    "day": 1.0 / 30.4375,
    "week": 7.0 / 30.4375,
    "month": 1.0,
    "year": 12.0,
    "decade": 120.0,
    "century": 1200.0,
    "millennium": 12_000.0,
}
UNIT_ALIASES = {"centuries": "century", "millennia": "millennium"}


def parse_gcs_uri(uri: str) -> tuple[str, str]:
    """Split a GCS URI into bucket and object/prefix."""
    match = re.fullmatch(r"gs://([^/]+)/(.+?)/*", uri)
    if match is None:
        raise ValueError(f"Expected gs://bucket/object-or-prefix, got {uri!r}.")
    return match.group(1), match.group(2)


def download_inputs(
    activations_uri: str,
    completions_uri: str,
    data_dir: Path,
    *,
    project_id: str | None = None,
    overwrite: bool = False,
) -> tuple[list[Path], Path]:
    """Download extracted activation chunks and the completion JSONL."""
    client = storage.Client(project=project_id)
    activation_bucket_name, activation_prefix = parse_gcs_uri(activations_uri)
    activation_bucket = client.bucket(activation_bucket_name)
    prefix = activation_prefix.rstrip("/") + "/"
    blobs = sorted(
        (
            blob
            for blob in activation_bucket.list_blobs(prefix=prefix)
            if blob.name.endswith(".pt")
        ),
        key=lambda blob: blob.name,
    )
    if not blobs:
        raise FileNotFoundError(f"No .pt activation chunks found below {activations_uri}.")

    activation_dir = data_dir / "activations"
    activation_dir.mkdir(parents=True, exist_ok=True)
    local_chunks: list[Path] = []
    for blob in blobs:
        destination = activation_dir / Path(blob.name).name
        if overwrite or not destination.exists():
            print(f"Downloading gs://{activation_bucket_name}/{blob.name} -> {destination}")
            blob.download_to_filename(str(destination))
        local_chunks.append(destination)

    completions_bucket_name, completions_object = parse_gcs_uri(completions_uri)
    completions_path = data_dir / Path(completions_object).name
    if overwrite or not completions_path.exists():
        completions_path.parent.mkdir(parents=True, exist_ok=True)
        print(f"Downloading {completions_uri} -> {completions_path}")
        client.bucket(completions_bucket_name).blob(completions_object).download_to_filename(
            str(completions_path)
        )
    return local_chunks, completions_path


def horizon_months(record: dict[str, Any]) -> float:
    """Return the canonical time horizon in average Gregorian months."""
    metadata = record.get("prompt_metadata")
    if not isinstance(metadata, dict):
        raise ValueError("Completion record has no prompt_metadata mapping.")
    value = metadata.get("base_value", metadata.get("value"))
    unit_value = metadata.get("base_unit", metadata.get("unit"))
    if not isinstance(value, (int, float)) or not isinstance(unit_value, str):
        raise ValueError("Completion metadata must contain numeric value and string unit.")
    raw_unit = unit_value.lower()
    unit = UNIT_ALIASES.get(raw_unit, raw_unit[:-1] if raw_unit.endswith("s") else raw_unit)
    if unit not in MONTHS_PER_UNIT:
        raise ValueError(f"Unsupported time unit {unit_value!r}.")
    months = float(value) * MONTHS_PER_UNIT[unit]
    if not math.isfinite(months) or months <= 0:
        raise ValueError(f"Time horizon must be positive and finite, got {months} months.")
    return months


def load_log_targets(completions_path: Path) -> np.ndarray:
    """Load base-10 log-month horizons, indexed by JSONL line/sample index."""
    targets: list[float] = []
    with completions_path.open(encoding="utf-8") as source:
        for line_number, line in enumerate(source, start=1):
            try:
                record = json.loads(line)
                targets.append(math.log10(horizon_months(record)))
            except (json.JSONDecodeError, ValueError) as error:
                raise ValueError(f"Invalid completion at line {line_number}: {error}") from error
    if not targets:
        raise ValueError(f"No completion records found in {completions_path}.")
    return np.asarray(targets, dtype=np.float64)


def _layer_sort_key(name: str) -> tuple[str, int | str]:
    prefix, separator, suffix = name.rpartition("/")
    return (prefix, int(suffix)) if separator and suffix.isdigit() else (name, name)


def layer_index(name: str) -> int:
    """Return the numeric suffix from a residual-stream layer name."""
    _prefix, separator, suffix = name.rpartition("/")
    if not separator or not suffix.isdigit():
        raise ValueError(f"Layer name must end in a numeric /index, got {name!r}.")
    return int(suffix)


def inspect_activation_chunks(chunk_paths: Iterable[Path]) -> tuple[list[str], np.ndarray]:
    """Validate chunk layouts and return layer names and ordered sample indices."""
    expected_layers: list[str] | None = None
    index_chunks: list[np.ndarray] = []
    for path in chunk_paths:
        payload = torch.load(path, map_location="cpu", weights_only=True)
        residuals = payload.get("residual_stream_activations")
        if not isinstance(residuals, dict) or not residuals:
            raise ValueError(f"{path} contains no residual_stream_activations.")
        layer_names = sorted(residuals, key=_layer_sort_key)
        if expected_layers is None:
            expected_layers = layer_names
        elif layer_names != expected_layers:
            raise ValueError(f"{path} has a different residual-stream layer set.")
        indices = np.asarray(payload.get("sample_indices"), dtype=np.int64)
        first_tensor = residuals[layer_names[0]]
        if first_tensor.ndim != 3 or first_tensor.shape[0] != len(indices):
            raise ValueError(f"{path} has an invalid activation or sample-index shape.")
        index_chunks.append(indices)
        del first_tensor, residuals, payload
        gc.collect()
    if expected_layers is None or not index_chunks:
        raise ValueError("No activation chunks were supplied.")
    all_indices = np.concatenate(index_chunks)
    if len(np.unique(all_indices)) != len(all_indices):
        raise ValueError("Activation chunks contain duplicate sample indices.")
    return expected_layers, np.sort(all_indices)


def load_layer_position_features(
    chunk_paths: Iterable[Path], layer_name: str, position_index: int
) -> tuple[np.ndarray, np.ndarray]:
    """Load one layer and one token position without retaining other layers."""
    if position_index not in POSITION_INDICES:
        raise ValueError(f"position_index must be one of {POSITION_INDICES}.")
    feature_chunks: list[np.ndarray] = []
    index_chunks: list[np.ndarray] = []

    for path in chunk_paths:
        payload = torch.load(path, map_location="cpu", weights_only=True)
        residuals = payload.get("residual_stream_activations")
        sample_indices = payload.get("sample_indices")
        cached_positions = payload.get("cached_position_indices", list(POSITION_INDICES))
        if not isinstance(residuals, dict) or not residuals:
            raise ValueError(f"{path} contains no residual_stream_activations.")
        if position_index not in cached_positions:
            raise ValueError(f"{path} does not contain cached position {position_index}.")
        position_offset = list(cached_positions).index(position_index)
        if layer_name not in residuals:
            raise ValueError(f"{path} does not contain layer {layer_name!r}.")
        tensor = residuals[layer_name]
        if tensor.ndim != 3:
            raise ValueError(f"{path}:{layer_name} must have shape samples x positions x width.")
        # contiguous() is essential: without the copy, NumPy keeps the entire
        # deserialized multi-layer payload storage alive through this view.
        features = tensor[:, position_offset, :].to(torch.float32).contiguous().numpy()
        indices = np.asarray(sample_indices, dtype=np.int64)
        if features.shape[0] != len(indices):
            raise ValueError(f"{path} has different activation and sample-index counts.")
        feature_chunks.append(features)
        index_chunks.append(indices)
        del tensor, residuals, payload
        gc.collect()

    if not feature_chunks:
        raise ValueError("No activation chunks were supplied.")
    all_features = np.concatenate(feature_chunks, axis=0)
    all_indices = np.concatenate(index_chunks)
    if len(np.unique(all_indices)) != len(all_indices):
        raise ValueError("Activation chunks contain duplicate sample indices.")
    if np.all(all_indices[:-1] <= all_indices[1:]):
        return all_features, all_indices
    order = np.argsort(all_indices)
    return all_features[order], all_indices[order]


def model_grid(seed: int, n_jobs: int) -> Iterator[tuple[str, RegressorMixin]]:
    """Yield models one at a time so fitted estimators can be released promptly."""
    del seed  # Kept in the interface for consistent experiment configuration.
    scaled = lambda model: make_pipeline(StandardScaler(), model)  # noqa: E731
    yield "ols", scaled(LinearRegression(n_jobs=n_jobs))
    for alpha in (0.01, 0.1, 1.0, 10.0, 100.0):
        yield f"ridge_alpha-{alpha:g}", scaled(Ridge(alpha=alpha))


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    """Compute RMSE, squared Pearson r, and coefficient of determination R2."""
    rmse = float(math.sqrt(mean_squared_error(y_true, y_pred)))
    if np.std(y_true) == 0 or np.std(y_pred) == 0:
        pearson_r2 = 0.0
    else:
        correlation = float(np.corrcoef(y_true, y_pred)[0, 1])
        pearson_r2 = correlation * correlation
    return {"rmse": rmse, "r2": pearson_r2, "R2": float(r2_score(y_true, y_pred))}


def fit_models(
    chunk_paths: list[Path],
    targets: np.ndarray,
    output_dir: Path,
    *,
    seed: int,
    test_fraction: float,
    n_jobs: int,
    save_models: bool,
    requested_layers: list[str] | None = None,
    position_indices: tuple[int, ...] = POSITION_INDICES,
) -> list[dict[str, Any]]:
    """Fit every model independently for each layer and requested position."""
    if not 0.0 < test_fraction < 1.0:
        raise ValueError("test_fraction must be strictly between 0 and 1.")
    output_dir.mkdir(parents=True, exist_ok=True)
    model_dir = output_dir / "models"
    if save_models:
        model_dir.mkdir(exist_ok=True)

    all_layer_names, sample_indices = inspect_activation_chunks(chunk_paths)
    layer_names = [name for name in all_layer_names if layer_index(name) >= MIN_LAYER_INDEX]
    if not layer_names:
        raise ValueError(f"The activation chunks contain no layers >= {MIN_LAYER_INDEX}.")
    if requested_layers:
        missing_layers = sorted(set(requested_layers) - set(all_layer_names))
        if missing_layers:
            raise ValueError(f"Requested layers are absent from the chunks: {missing_layers}")
        early_layers = sorted(
            name for name in requested_layers if layer_index(name) < MIN_LAYER_INDEX
        )
        if early_layers:
            raise ValueError(
                f"Requested layers must have indices >= {MIN_LAYER_INDEX}: {early_layers}"
            )
        layer_names = requested_layers
    invalid_positions = sorted(set(position_indices) - set(POSITION_INDICES))
    if invalid_positions:
        raise ValueError(f"Invalid position indices: {invalid_positions}")
    if sample_indices.min() < 0 or sample_indices.max() >= len(targets):
        raise IndexError("An activation sample_index is outside the completion JSONL.")
    rng = np.random.default_rng(seed)
    permutation = rng.permutation(len(sample_indices))
    test_count = max(1, int(round(test_fraction * len(sample_indices))))
    test_rows = permutation[:test_count]
    train_rows = permutation[test_count:]
    if not len(train_rows):
        raise ValueError("The split left no training samples.")
    y = targets[sample_indices]
    results: list[dict[str, Any]] = []

    split_payload = {
        "seed": seed,
        "test_fraction": test_fraction,
        "train_sample_indices": sample_indices[train_rows].tolist(),
        "test_sample_indices": sample_indices[test_rows].tolist(),
    }
    (output_dir / "split.json").write_text(json.dumps(split_payload, indent=2), encoding="utf-8")

    y_train, y_test = y[train_rows], y[test_rows]
    for layer_name in layer_names:
        safe_layer_name = re.sub(r"[^A-Za-z0-9_.-]+", "-", layer_name)
        for position in position_indices:
            features, position_sample_indices = load_layer_position_features(
                chunk_paths, layer_name, position
            )
            if not np.array_equal(position_sample_indices, sample_indices):
                raise ValueError(f"{layer_name}, position {position} has different indices.")
            n_features = features.shape[1]
            x_train, x_test = features[train_rows], features[test_rows]
            del features
            gc.collect()
            for specification, model in model_grid(seed, n_jobs):
                model_name = f"layer-{safe_layer_name}__position-{position}__{specification}"
                print(f"Fitting {model_name}", flush=True)
                model.fit(x_train, y_train)
                predictions = {
                    "train": model.predict(x_train),
                    "test": model.predict(x_test),
                }
                for split_name, split_targets in (("train", y_train), ("test", y_test)):
                    results.append(
                        {
                            "model": model_name,
                            "layer": layer_name,
                            "position": position,
                            "split": split_name,
                            "n_samples": len(split_targets),
                            "n_features": n_features,
                            **regression_metrics(split_targets, predictions[split_name]),
                        }
                    )
                if save_models:
                    joblib.dump(model, model_dir / f"{model_name}.joblib", compress=3)
                # Persist partial progress and release each fitted estimator;
                # forest models can otherwise accumulate several GB of state.
                write_metrics(results, output_dir)
                del model, predictions
                gc.collect()
            del x_train, x_test
            gc.collect()
    return results


def write_metrics(results: list[dict[str, Any]], output_dir: Path) -> None:
    """Write machine-readable CSV and JSON result tables."""
    fields = [
        "model",
        "layer",
        "position",
        "split",
        "n_samples",
        "n_features",
        "rmse",
        "r2",
        "R2",
    ]
    with (output_dir / "metrics.csv").open("w", newline="", encoding="utf-8") as output:
        writer = csv.DictWriter(output, fieldnames=fields)
        writer.writeheader()
        writer.writerows(results)
    (output_dir / "metrics.json").write_text(json.dumps(results, indent=2), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--activations-uri", default=DEFAULT_ACTIVATIONS_URI)
    parser.add_argument("--completions-uri", default=DEFAULT_COMPLETIONS_URI)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--project-id", default=os.getenv("GCP_PROJECT_ID"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--test-fraction", type=float, default=0.2)
    parser.add_argument(
        "--n-jobs",
        type=int,
        default=1,
        help="Parallel workers for supported estimators (default: 1 to limit peak RAM).",
    )
    parser.add_argument(
        "--layers",
        nargs="+",
        default=None,
        help=(
            f"Optional layer names to fit; indices must be >= {MIN_LAYER_INDEX} "
            "(for example: --layers layer_out/35)."
        ),
    )
    parser.add_argument(
        "--positions",
        nargs="+",
        type=int,
        choices=POSITION_INDICES,
        default=list(POSITION_INDICES),
    )
    parser.add_argument("--overwrite-downloads", action="store_true")
    parser.add_argument("--skip-download", action="store_true")
    parser.add_argument("--no-save-models", action="store_true")
    return parser.parse_args()


def main() -> None:
    load_dotenv()
    args = parse_args()
    if args.skip_download:
        chunks = sorted((args.data_dir / "activations").glob("*.pt"))
        completions_path = args.data_dir / Path(parse_gcs_uri(args.completions_uri)[1]).name
        if not chunks or not completions_path.exists():
            raise FileNotFoundError("--skip-download was used but local input files are missing.")
    else:
        chunks, completions_path = download_inputs(
            args.activations_uri,
            args.completions_uri,
            args.data_dir,
            project_id=args.project_id,
            overwrite=args.overwrite_downloads,
        )
    targets = load_log_targets(completions_path)
    results = fit_models(
        chunks,
        targets,
        args.output_dir,
        seed=args.seed,
        test_fraction=args.test_fraction,
        n_jobs=args.n_jobs,
        save_models=not args.no_save_models,
        requested_layers=args.layers,
        position_indices=tuple(args.positions),
    )
    write_metrics(results, args.output_dir)
    print(f"Wrote {len(results)} metric rows to {args.output_dir}", flush=True)


if __name__ == "__main__":
    main()
