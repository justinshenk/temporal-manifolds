"""Fit PLS and residual PCA on the expanded activation caches, one GCS folder at a time.

For every configured GCS folder this script:

1. Downloads the folder's activation batches to a local scratch directory.
2. Concatenates one layer's activations across both cached prompt positions
   (-2 and -1) into a single vector per prompt.
3. Fits a 6-component PLS regression against ``log10_time_horizon_months``.
4. Deflates the PLS component subspace out of the activations and fits a
   3-component PCA on the residuals.
5. Writes one CSV per folder holding every flattened prompt-metadata field, the
   PLS scores, the PLS prediction, and the residual PCA scores, plus a JSON
   sidecar with the fit diagnostics.
6. Deletes the folder's local batches before moving to the next folder.

Activations are streamed: the PLS and PCA models are fitted on a bounded random
sample of rows and then applied batch by batch, so a folder never has to fit in
memory all at once.
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import shutil
import sys
from collections.abc import Iterator, Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from dotenv import load_dotenv
from google.cloud import storage
from sklearn.cross_decomposition import PLSRegression
from sklearn.decomposition import PCA
from tqdm.auto import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent))

from cache_expanded_selected_acts import SCENARIOS, expanded_gcs_prefix  # noqa: E402

DEFAULT_BUCKET_NAME = "temporal-research-bucket"
DEFAULT_LAYER_COMPONENT = "layer_out/21"
DEFAULT_PLS_COMPONENTS = 6
DEFAULT_RESIDUAL_PCA_COMPONENTS = 3
DEFAULT_FIT_SAMPLE_SIZE = 50_000
DEFAULT_DOWNLOAD_WORKERS = 8
DEFAULT_DATA_ROOT = Path("data") / "expanded_pls"
DEFAULT_OUTPUT_DIR = Path("results") / "expanded_pls"
RANDOM_SEED = 0

# Month equivalents for every time unit the datasets emit, as used by the
# download_filtered_* notebooks.
UNIT_TO_MONTHS: dict[str, float] = {
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

NOT_APPLICABLE = "N/A"


@dataclass(frozen=True)
class Folder:
    """One GCS folder of cached activation batches."""

    name: str
    gcs_prefix: str


def default_folders() -> tuple[Folder, ...]:
    """Return the expanded caching folders written by cache_expanded_selected_acts.py."""
    return tuple(
        Folder(name=scenario.name, gcs_prefix=expanded_gcs_prefix(scenario.gcs_prefix))
        for scenario in SCENARIOS
    )


def flatten_scalar_metadata(metadata: Mapping[str, Any], prefix: str = "") -> dict[str, Any]:
    """Flatten nested prompt metadata into dotted scalar fields."""
    flattened: dict[str, Any] = {}
    for key, value in metadata.items():
        path = f"{prefix}.{key}" if prefix else key
        if isinstance(value, Mapping):
            flattened.update(flatten_scalar_metadata(value, path))
        elif not isinstance(value, (list, tuple, set)):
            flattened[path] = value
    return flattened


def time_horizon_months(metadata: Mapping[str, Any]) -> float | None:
    """Return a prompt's horizon in months, or None when it declares no horizon."""
    value = metadata.get("base_value", metadata.get("value"))
    unit = metadata.get("base_unit", metadata.get("unit"))
    if value in (None, NOT_APPLICABLE) or unit in (None, NOT_APPLICABLE):
        return None
    unit_key = str(unit).lower()
    if unit_key not in UNIT_TO_MONTHS:
        raise ValueError(f"Cannot convert time-horizon unit {unit!r} to months.")
    months = float(value) * UNIT_TO_MONTHS[unit_key]
    if months <= 0:
        raise ValueError(f"Time horizon must be positive; got {months} months.")
    return round(months, 12)


def list_batch_blobs(client: storage.Client, bucket_name: str, prefix: str) -> list[Any]:
    """Return every activation batch blob below a folder prefix, ordered by name."""
    blobs = [
        blob
        for blob in client.bucket(bucket_name).list_blobs(prefix=prefix.strip("/") + "/")
        if Path(blob.name).name.startswith("activations_batch_") and blob.name.endswith(".pt")
    ]
    return sorted(blobs, key=lambda blob: blob.name)


def download_folder(
    client: storage.Client,
    bucket_name: str,
    prefix: str,
    destination_dir: Path,
    *,
    workers: int,
    overwrite: bool,
) -> list[Path]:
    """Download one folder's activation batches and return the local paths."""
    blobs = list_batch_blobs(client, bucket_name, prefix)
    if not blobs:
        raise FileNotFoundError(f"No activation batches found below gs://{bucket_name}/{prefix}")

    destination_dir.mkdir(parents=True, exist_ok=True)

    def download(blob: Any) -> Path:
        destination = destination_dir / Path(blob.name).name
        if overwrite or not destination.exists():
            blob.download_to_filename(str(destination))
        return destination

    with ThreadPoolExecutor(max_workers=workers) as executor:
        paths = list(
            tqdm(
                executor.map(download, blobs),
                total=len(blobs),
                desc=f"Downloading {prefix}",
            )
        )
    return sorted(paths)


def load_batch(path: Path) -> dict[str, Any]:
    """Load one serialized activation batch."""
    return torch.load(path, map_location="cpu", weights_only=True, mmap=True)


def batch_features(payload: Mapping[str, Any], layer_component: str) -> torch.Tensor:
    """Return a batch's activations as rows of concatenated cached positions."""
    activations = payload.get("activations")
    if not isinstance(activations, Mapping) or layer_component not in activations:
        raise ValueError(f"Batch does not contain {layer_component!r}.")
    tensor = activations[layer_component]
    if tensor.ndim != 3:
        raise ValueError(
            f"{layer_component} must have shape batch x positions x hidden size; "
            f"got {tuple(tensor.shape)}."
        )
    # Positions stay in payload order, so column blocks are [position -2 | position -1].
    return tensor.reshape(tensor.shape[0], -1).to(torch.float32)


def iter_batches(paths: Sequence[Path], desc: str) -> Iterator[tuple[Path, dict[str, Any]]]:
    """Yield each batch file's payload with a progress bar."""
    for path in tqdm(paths, desc=desc):
        payload = load_batch(path)
        yield path, payload
        del payload
        gc.collect()


def collect_metadata(
    paths: Sequence[Path],
    layer_component: str,
    *,
    include_prompts: bool,
) -> tuple[pd.DataFrame, list[tuple[int, int]], int]:
    """Return the metadata table, per-batch row spans, and the feature width.

    Rows whose prompts declare no time horizon are dropped, because the PLS target
    ``log10_time_horizon_months`` is undefined for them. The returned spans give
    the ``(batch_offset, row_offset)`` pairs of the retained rows in file order.
    """
    records: list[dict[str, Any]] = []
    kept_rows: list[tuple[int, int]] = []
    feature_width = 0
    dropped_rows = 0

    for batch_offset, (path, payload) in enumerate(iter_batches(paths, "Reading metadata")):
        positions = list(payload.get("positions", []))
        metadata_rows = payload["prompt_metadata"]
        sample_indices = payload["sample_indices"]
        prompts = payload["prompts"]
        activations = payload.get("activations", {})
        if layer_component not in activations:
            raise ValueError(f"{path} does not contain {layer_component!r}.")
        tensor = activations[layer_component]
        if not (len(metadata_rows) == len(sample_indices) == len(prompts) == tensor.shape[0]):
            raise ValueError(f"Misaligned rows in {path}.")
        width = int(tensor.shape[1] * tensor.shape[2])
        if feature_width and width != feature_width:
            raise ValueError(f"{path} has feature width {width}; expected {feature_width}.")
        feature_width = width

        for row_offset, metadata in enumerate(metadata_rows):
            months = time_horizon_months(metadata)
            if months is None:
                dropped_rows += 1
                continue
            record = flatten_scalar_metadata(metadata)
            record["sample_index"] = int(sample_indices[row_offset])
            record["batch_index"] = int(payload.get("batch_index", batch_offset))
            record["batch_file"] = path.name
            record["dataset"] = payload.get("dataset", NOT_APPLICABLE)
            record["scenario"] = payload.get("scenario", NOT_APPLICABLE)
            record["model_name"] = payload.get("model_name", NOT_APPLICABLE)
            record["layer_component"] = layer_component
            record["cached_positions"] = "|".join(str(position) for position in positions)
            record["time_horizon_months"] = months
            record["log10_time_horizon_months"] = float(np.log10(months))
            if include_prompts:
                record["prompt"] = prompts[row_offset]
            records.append(record)
            kept_rows.append((batch_offset, row_offset))

    if dropped_rows:
        print(
            f"[expanded-pls] Dropped {dropped_rows:,} rows without a time horizon.",
            flush=True,
        )
    return pd.DataFrame(records), kept_rows, feature_width


def gather_fit_sample(
    paths: Sequence[Path],
    layer_component: str,
    kept_rows: Sequence[tuple[int, int]],
    sample_positions: np.ndarray,
) -> np.ndarray:
    """Load the sampled activation rows into one matrix."""
    rows_by_batch: dict[int, list[tuple[int, int]]] = {}
    for output_offset, kept_offset in enumerate(sample_positions):
        batch_offset, row_offset = kept_rows[int(kept_offset)]
        rows_by_batch.setdefault(batch_offset, []).append((row_offset, output_offset))

    sample_matrix: np.ndarray | None = None
    for batch_offset in tqdm(sorted(rows_by_batch), desc="Loading fit sample"):
        payload = load_batch(paths[batch_offset])
        features = batch_features(payload, layer_component)
        row_offsets, output_offsets = zip(*rows_by_batch[batch_offset])
        selected = features[list(row_offsets)].numpy()
        if sample_matrix is None:
            sample_matrix = np.empty((len(sample_positions), selected.shape[1]), dtype=np.float32)
        sample_matrix[list(output_offsets)] = selected
        del payload, features, selected
        gc.collect()

    if sample_matrix is None:
        raise ValueError("No rows were available for fitting.")
    return sample_matrix


def pls_residuals(pls: PLSRegression, features: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return PLS scores and the activation residuals left after deflation.

    ``inverse_transform`` maps the scores back into activation space, so the
    difference is exactly the part of each activation the PLS components do not
    explain.
    """
    scores = pls.transform(features)
    residuals = features - pls.inverse_transform(scores)
    return scores, residuals


def process_folder(
    folder: Folder,
    *,
    client: storage.Client,
    bucket_name: str,
    layer_component: str,
    pls_components: int,
    pca_components: int,
    fit_sample_size: int | None,
    data_root: Path,
    output_dir: Path,
    download_workers: int,
    overwrite_downloads: bool,
    include_prompts: bool,
    keep_local: bool,
) -> Path | None:
    """Download, fit, and write one folder's CSV, then delete its local batches."""
    folder_dir = data_root / folder.gcs_prefix
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / f"{folder.gcs_prefix}_{layer_component.replace('/', '_')}_pls.csv"
    json_path = csv_path.with_suffix(".json")

    try:
        paths = download_folder(
            client,
            bucket_name,
            folder.gcs_prefix,
            folder_dir,
            workers=download_workers,
            overwrite=overwrite_downloads,
        )

        metadata_df, kept_rows, feature_width = collect_metadata(
            paths, layer_component, include_prompts=include_prompts
        )
        row_count = len(kept_rows)
        if row_count == 0:
            print(
                f"[expanded-pls] Skipping {folder.name}: no prompt declares a time horizon, "
                "so log10_time_horizon_months has no values to regress against.",
                flush=True,
            )
            return None
        if row_count <= pls_components:
            raise ValueError(
                f"{folder.name} has {row_count} usable rows; at least {pls_components + 1} "
                "are required to fit the PLS model."
            )

        target = metadata_df["log10_time_horizon_months"].to_numpy(dtype=np.float64)
        rng = np.random.default_rng(RANDOM_SEED)
        if fit_sample_size is None or fit_sample_size >= row_count:
            sample_positions = np.arange(row_count)
        else:
            sample_positions = np.sort(rng.choice(row_count, size=fit_sample_size, replace=False))

        fit_features = gather_fit_sample(paths, layer_component, kept_rows, sample_positions)
        fit_target = target[sample_positions]

        pls = PLSRegression(n_components=pls_components, scale=False)
        pls.fit(fit_features, fit_target)
        fit_scores, fit_residuals = pls_residuals(pls, fit_features)
        pca = PCA(n_components=pca_components, svd_solver="randomized", random_state=RANDOM_SEED)
        pca.fit(fit_residuals)
        fit_r2 = float(pls.score(fit_features, fit_target))
        centered_energy = float(np.square(fit_features - fit_features.mean(axis=0)).sum())
        residual_energy_ratio = float(np.square(fit_residuals).sum() / centered_energy)
        del fit_features, fit_residuals, fit_scores
        gc.collect()

        score_columns = [f"pls_{index + 1}" for index in range(pls_components)]
        residual_columns = [f"residual_pc_{index + 1}" for index in range(pca_components)]
        scores_all = np.empty((row_count, pls_components), dtype=np.float32)
        residual_scores_all = np.empty((row_count, pca_components), dtype=np.float32)
        predictions_all = np.empty(row_count, dtype=np.float32)

        rows_by_batch: dict[int, list[tuple[int, int]]] = {}
        for kept_offset, (batch_offset, row_offset) in enumerate(kept_rows):
            rows_by_batch.setdefault(batch_offset, []).append((row_offset, kept_offset))

        for batch_offset in tqdm(sorted(rows_by_batch), desc=f"Projecting {folder.name}"):
            payload = load_batch(paths[batch_offset])
            features = batch_features(payload, layer_component)
            row_offsets, output_offsets = zip(*rows_by_batch[batch_offset])
            chunk = features[list(row_offsets)].numpy()
            chunk_scores, chunk_residuals = pls_residuals(pls, chunk)
            output_index = list(output_offsets)
            scores_all[output_index] = chunk_scores.astype(np.float32)
            residual_scores_all[output_index] = pca.transform(chunk_residuals).astype(np.float32)
            predictions_all[output_index] = pls.predict(chunk).reshape(-1).astype(np.float32)
            del payload, features, chunk, chunk_scores, chunk_residuals
            gc.collect()

        output_df = metadata_df.copy()
        output_df[score_columns] = scores_all
        output_df["pls_prediction_log10_months"] = predictions_all
        output_df["pls_residual_log10_months"] = (
            output_df["log10_time_horizon_months"] - output_df["pls_prediction_log10_months"]
        )
        output_df[residual_columns] = residual_scores_all
        output_df.to_csv(csv_path, index=False)

        diagnostics = {
            "folder": folder.name,
            "gcs_prefix": folder.gcs_prefix,
            "bucket": bucket_name,
            "layer_component": layer_component,
            "feature_width": feature_width,
            "batch_files": len(paths),
            "rows": row_count,
            "fit_rows": int(len(sample_positions)),
            "pls_components": pls_components,
            "residual_pca_components": pca_components,
            "target": "log10_time_horizon_months",
            "pls_r2_on_fit_sample": fit_r2,
            "residual_energy_fraction_after_pls": residual_energy_ratio,
            "residual_pca_explained_variance_ratio": [
                float(value) for value in pca.explained_variance_ratio_
            ],
            "random_seed": RANDOM_SEED,
            "csv_path": str(csv_path),
        }
        json_path.write_text(json.dumps(diagnostics, indent=2), encoding="utf-8")
        print(
            f"[expanded-pls] Wrote {csv_path} ({row_count:,} rows, "
            f"PLS R^2={fit_r2:.4f} on the fit sample).",
            flush=True,
        )
        return csv_path
    finally:
        if not keep_local and folder_dir.exists():
            shutil.rmtree(folder_dir, ignore_errors=True)
            print(f"[expanded-pls] Deleted local batches at {folder_dir}", flush=True)
        gc.collect()


def run(
    *,
    folders: Sequence[Folder],
    bucket_name: str,
    project_id: str | None,
    layer_component: str,
    pls_components: int,
    pca_components: int,
    fit_sample_size: int | None,
    data_root: Path,
    output_dir: Path,
    download_workers: int,
    overwrite_downloads: bool,
    include_prompts: bool,
    keep_local: bool,
) -> None:
    """Process every folder in turn, deleting each local copy before the next."""
    client = storage.Client(project=project_id)
    written: list[Path] = []
    for folder in folders:
        print(f"[expanded-pls] Processing {folder.name} (gs://{bucket_name}/{folder.gcs_prefix})")
        csv_path = process_folder(
            folder,
            client=client,
            bucket_name=bucket_name,
            layer_component=layer_component,
            pls_components=pls_components,
            pca_components=pca_components,
            fit_sample_size=fit_sample_size,
            data_root=data_root,
            output_dir=output_dir,
            download_workers=download_workers,
            overwrite_downloads=overwrite_downloads,
            include_prompts=include_prompts,
            keep_local=keep_local,
        )
        if csv_path is not None:
            written.append(csv_path)
    print(f"[expanded-pls] Wrote {len(written)} CSV file(s):", flush=True)
    for path in written:
        print(f"  {path}", flush=True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    known_folders = {folder.name: folder for folder in default_folders()}
    parser.add_argument(
        "--folder",
        dest="folder_names",
        action="append",
        choices=sorted(known_folders),
        default=None,
        help="Restrict the run to one expanded folder; repeatable. Defaults to every folder.",
    )
    parser.add_argument(
        "--gcs-prefix",
        dest="gcs_prefixes",
        action="append",
        default=None,
        help="Process an arbitrary bucket prefix instead of the known folders; repeatable.",
    )
    parser.add_argument("--bucket-name", default=None)
    parser.add_argument("--project-id", default=None)
    parser.add_argument("--layer-component", default=DEFAULT_LAYER_COMPONENT)
    parser.add_argument("--pls-components", type=int, default=DEFAULT_PLS_COMPONENTS)
    parser.add_argument(
        "--residual-pca-components", type=int, default=DEFAULT_RESIDUAL_PCA_COMPONENTS
    )
    parser.add_argument(
        "--fit-sample-size",
        type=int,
        default=DEFAULT_FIT_SAMPLE_SIZE,
        help="Rows used to fit PLS and the residual PCA. Use 0 to fit on every row.",
    )
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--download-workers", type=int, default=DEFAULT_DOWNLOAD_WORKERS)
    parser.add_argument("--overwrite-downloads", action="store_true")
    parser.add_argument(
        "--include-prompts",
        action="store_true",
        help="Also write the full prompt text into the CSV.",
    )
    parser.add_argument(
        "--keep-local",
        action="store_true",
        help="Keep each folder's downloaded batches instead of deleting them.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    load_dotenv()
    args = build_parser().parse_args(argv)

    known_folders = {folder.name: folder for folder in default_folders()}
    if args.gcs_prefixes:
        folders = tuple(
            Folder(name=prefix.strip("/"), gcs_prefix=prefix.strip("/"))
            for prefix in dict.fromkeys(args.gcs_prefixes)
        )
    elif args.folder_names:
        folders = tuple(known_folders[name] for name in dict.fromkeys(args.folder_names))
    else:
        folders = default_folders()

    if args.pls_components < 1:
        raise ValueError("--pls-components must be at least 1.")
    if args.residual_pca_components < 1:
        raise ValueError("--residual-pca-components must be at least 1.")
    if args.fit_sample_size < 0:
        raise ValueError("--fit-sample-size must be non-negative.")

    run(
        folders=folders,
        bucket_name=args.bucket_name or os.getenv("GCS_BUCKET_NAME") or DEFAULT_BUCKET_NAME,
        project_id=args.project_id or os.getenv("GCP_PROJECT_ID"),
        layer_component=args.layer_component,
        pls_components=args.pls_components,
        pca_components=args.residual_pca_components,
        fit_sample_size=args.fit_sample_size or None,
        data_root=args.data_root,
        output_dir=args.output_dir,
        download_workers=args.download_workers,
        overwrite_downloads=args.overwrite_downloads,
        include_prompts=args.include_prompts,
        keep_local=args.keep_local,
    )


if __name__ == "__main__":
    main()
