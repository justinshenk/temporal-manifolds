"""Fit a folder-balanced linear plane in a three-component activation PCA space."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import joblib
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import torch
from sklearn.decomposition import IncrementalPCA
from sklearn.metrics import accuracy_score, balanced_accuracy_score, classification_report
from sklearn.svm import LinearSVC

from temporal_manifolds.activations.extraction_policy import CACHED_POSITION_INDEX
from temporal_manifolds.viz.activation_explorer import (
    SOURCE_FOLDER_FIELD,
    discover_activation_batch_paths,
    inspect_sources,
    load_pca_model,
    prepare_analysis_data,
)

DEFAULT_FOLDERS = ("plain", "plain_long", "indirect", "new_conv")
DEFAULT_CLASS_BY_FOLDER = {
    "plain": 0,
    "plain_long": 0,
    "indirect": 1,
    "new_conv": 1,
}
GROUP_FIELDS = ("task", SOURCE_FOLDER_FIELD, "time_horizon_months")


def _bounded_slices(length: int, batch_size: int, minimum_size: int) -> list[slice]:
    if batch_size < minimum_size:
        raise ValueError(f"batch_size must be at least {minimum_size}.")
    batches: list[slice] = []
    start = 0
    while start < length:
        stop = min(start + batch_size, length)
        if 0 < length - stop < minimum_size:
            stop = length
        batches.append(slice(start, stop))
        start = stop
    return batches


def _plane_projection(points: np.ndarray, normal: np.ndarray, intercept: float) -> np.ndarray:
    normal_squared = float(normal @ normal)
    if not np.isfinite(normal_squared) or normal_squared <= 0:
        raise ValueError("The fitted plane has an invalid normal vector.")
    signed_values = points @ normal + intercept
    return points - np.outer(signed_values / normal_squared, normal)


def _stream_group_means(
    sources: Sequence[str | Path],
    source_row_counts: Sequence[int],
    component: str,
    group_codes: np.ndarray,
    group_counts: np.ndarray,
) -> np.ndarray:
    """Stream source tensors once and average rows carrying the same integer group code."""
    if len(sources) != len(source_row_counts):
        raise ValueError("Sources and source row counts are misaligned.")
    if len(group_codes) != int(np.sum(source_row_counts)):
        raise ValueError("Group codes and source activation rows are misaligned.")
    included_rows = group_codes >= 0
    if int(np.sum(group_counts)) != int(included_rows.sum()):
        raise ValueError("Group counts do not match the included activation rows.")
    group_sums: torch.Tensor | None = None
    row_start = 0
    for source, expected_rows in zip(sources, source_row_counts, strict=True):
        payload = torch.load(source, map_location="cpu", weights_only=True, mmap=True)
        activations = payload.get("activations")
        if not isinstance(activations, dict) or component not in activations:
            raise ValueError(f"{source} does not contain activation {component}.")
        tensor = activations[component]
        if tensor.ndim != 3 or tensor.shape[0] != expected_rows:
            raise ValueError(f"Unexpected activation shape in {source}: {tuple(tensor.shape)}.")
        if not 0 <= CACHED_POSITION_INDEX < tensor.shape[1]:
            raise ValueError(f"Cached position is unavailable in {source}.")
        values = tensor[:, CACHED_POSITION_INDEX, :].to(torch.float32)
        if group_sums is None:
            group_sums = torch.zeros(
                (len(group_counts), values.shape[1]), dtype=torch.float32
            )
        elif values.shape[1] != group_sums.shape[1]:
            raise ValueError(f"Inconsistent activation width in {source}.")
        row_stop = row_start + expected_rows
        batch_code_array = group_codes[row_start:row_stop]
        keep = batch_code_array >= 0
        if np.any(keep):
            batch_codes = torch.tensor(batch_code_array[keep], dtype=torch.int64)
            keep_tensor = torch.from_numpy(np.asarray(keep))
            group_sums.index_add_(0, batch_codes, values[keep_tensor])
        row_start = row_stop
        del payload, activations, tensor, values
    if group_sums is None or row_start != len(group_codes):
        raise ValueError("No aligned activation rows were loaded.")
    group_sums /= torch.from_numpy(group_counts.astype(np.float32))[:, None]
    return group_sums.numpy()


def _ignored_row_mask(
    metadata: pd.DataFrame,
    ignore_filters: Mapping[str, Sequence[Any]] | None,
) -> np.ndarray:
    """Return rows matching any configured field/value exclusion."""
    ignored = np.zeros(len(metadata), dtype=bool)
    for field, values in (ignore_filters or {}).items():
        if field not in metadata:
            raise ValueError(f"Ignore-filter field is unavailable: {field}.")
        if isinstance(values, (str, bytes)) or not isinstance(values, Sequence):
            raise ValueError(f"Ignore-filter values for {field} must be a list-like sequence.")
        ignored |= metadata[field].isin(list(values)).to_numpy(dtype=bool)
    return ignored


def _plane_figure(
    scores: np.ndarray,
    folders: np.ndarray,
    normal: np.ndarray,
    intercept: float,
    *,
    folder_order: Sequence[str],
    max_points_per_folder: int,
    random_state: int,
) -> go.Figure:
    rng = np.random.default_rng(random_state)
    palette = ("#1f77b4", "#17becf", "#d62728", "#ff7f0e")
    colors = {folder: palette[index % len(palette)] for index, folder in enumerate(folder_order)}
    figure = go.Figure()
    for folder in folder_order:
        indices = np.flatnonzero(folders == folder)
        if len(indices) > max_points_per_folder:
            indices = rng.choice(indices, max_points_per_folder, replace=False)
        figure.add_trace(
            go.Scatter3d(
                x=scores[indices, 0],
                y=scores[indices, 1],
                z=scores[indices, 2],
                mode="markers",
                name=folder,
                marker={"size": 2.5, "opacity": 0.45, "color": colors[folder]},
            )
        )

    solve_axis = int(np.argmax(np.abs(normal)))
    free_axes = [axis for axis in range(3) if axis != solve_axis]
    lower = np.quantile(scores, 0.01, axis=0)
    upper = np.quantile(scores, 0.99, axis=0)
    u, v = np.meshgrid(
        np.linspace(lower[free_axes[0]], upper[free_axes[0]], 30),
        np.linspace(lower[free_axes[1]], upper[free_axes[1]], 30),
    )
    coordinates: list[np.ndarray | None] = [None, None, None]
    coordinates[free_axes[0]], coordinates[free_axes[1]] = u, v
    coordinates[solve_axis] = -(
        intercept + normal[free_axes[0]] * u + normal[free_axes[1]] * v
    ) / normal[solve_axis]
    figure.add_trace(
        go.Surface(
            x=coordinates[0],
            y=coordinates[1],
            z=coordinates[2],
            name="SVM decision plane",
            opacity=0.35,
            showscale=False,
            colorscale=[[0, "#6a3d9a"], [1, "#6a3d9a"]],
        )
    )
    figure.update_layout(
        title="Three-component PCA with balanced maximum-margin plane",
        scene={"xaxis_title": "PC1", "yaxis_title": "PC2", "zaxis_title": "PC3"},
        legend_title="Source folder",
        height=750,
    )
    return figure


def fit_balanced_pca_plane(
    acts_dir: str | Path,
    output_dir: str | Path,
    *,
    cache_dir: str | Path,
    folders: Sequence[str] = DEFAULT_FOLDERS,
    class_by_folder: dict[str, int] | None = None,
    ignore_filters: Mapping[str, Sequence[Any]] | None = None,
    prefitted_pca_path: str | Path | None = None,
    pca_batch_size: int = 4096,
    svm_c: float = 1.0,
    random_state: int = 42,
    max_plot_points_per_folder: int = 5_000,
    pca_filename: str = "pca_3_components.joblib",
    classifier_filename: str = "balanced_linear_svm.joblib",
    points_filename: str = "original_and_projected_pc_points.csv",
    plot_filename: str = "pca_scores_and_plane.html",
) -> dict[str, Path]:
    """Fit the PCA and plane, then save the models, scores, and interactive plot."""
    acts_dir = Path(acts_dir).resolve()
    output_dir = Path(output_dir).resolve()
    cache_dir = Path(cache_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    folders = tuple(folders)
    labels = dict(DEFAULT_CLASS_BY_FOLDER if class_by_folder is None else class_by_folder)
    if set(folders) != set(labels):
        raise ValueError("class_by_folder must define exactly the selected folders.")
    if set(labels.values()) != {0, 1}:
        raise ValueError("class_by_folder must contain both binary labels 0 and 1.")

    sources, _ = discover_activation_batch_paths([acts_dir / folder for folder in folders])
    inspection = inspect_sources(sources, cache_dir=cache_dir)
    if len(inspection["components"]) != 1:
        raise ValueError(f"Expected one shared component, got {inspection['components']}.")
    component = inspection["components"][0]
    metadata_proxy = np.empty((inspection["row_count"], 0), dtype=np.float32)
    _, row_offsets, raw_metadata, _ = prepare_analysis_data(
        metadata_proxy,
        inspection["metadata_index"],
        cached_position=inspection["positions"][0],
        metadata_filters=None,
        aggregation_fields=None,
    )
    if not np.array_equal(row_offsets, np.arange(inspection["row_count"])):
        raise ValueError("Expected unfiltered metadata in source activation row order.")
    ignored_rows = _ignored_row_mask(raw_metadata, ignore_filters)
    fit_positions = np.flatnonzero(~ignored_rows)
    if len(fit_positions) < 3:
        raise ValueError("Ignore filters leave fewer than three rows available for fitting.")
    fit_metadata = raw_metadata.iloc[fit_positions].reset_index(drop=True)
    groups = fit_metadata.groupby(list(GROUP_FIELDS), dropna=False, sort=False).indices
    group_count = len(groups)
    group_codes = np.full(len(raw_metadata), -1, dtype=np.int64)
    group_counts = np.empty(group_count, dtype=np.int64)
    first_positions = np.empty(group_count, dtype=np.int64)
    for group_index, offsets in enumerate(groups.values()):
        offsets = np.asarray(offsets, dtype=np.int64)
        raw_offsets = fit_positions[offsets]
        group_codes[raw_offsets] = group_index
        group_counts[group_index] = len(offsets)
        first_positions[group_index] = offsets[0]
    if group_count == 0 or np.any(group_codes[fit_positions] < 0):
        raise ValueError("Could not assign every activation row to an aggregation group.")
    metadata = fit_metadata.iloc[first_positions][list(GROUP_FIELDS)].reset_index(drop=True)
    metadata["source_sample_count"] = group_counts
    analysis_matrix = _stream_group_means(
        sources,
        inspection["source_row_counts"],
        component,
        group_codes,
        group_counts,
    )
    if len(analysis_matrix) != len(metadata):
        raise ValueError("Aggregated activations and metadata are misaligned.")
    if len(analysis_matrix) < 3:
        raise ValueError("At least three aggregated rows are required.")

    batches = _bounded_slices(len(analysis_matrix), pca_batch_size, 3)
    if prefitted_pca_path is None:
        pca = IncrementalPCA(n_components=3, batch_size=pca_batch_size)
        for batch in batches:
            pca.partial_fit(np.asarray(analysis_matrix[batch]))
        pca_source = "fitted on current aggregated data"
    else:
        prefitted_pca_path = Path(prefitted_pca_path).resolve()
        if not prefitted_pca_path.is_file():
            raise FileNotFoundError(prefitted_pca_path)
        pca, _ = load_pca_model(prefitted_pca_path)
        if np.asarray(pca.components_).shape != (3, analysis_matrix.shape[1]):
            raise ValueError(
                "Pre-fitted PCA shape does not match the required "
                f"(3, {analysis_matrix.shape[1]}) components."
            )
        pca_source = f"loaded from {prefitted_pca_path}"
    scores = np.empty((len(analysis_matrix), 3), dtype=np.float64)
    for batch in batches:
        scores[batch] = pca.transform(np.asarray(analysis_matrix[batch]))

    source_folders = metadata[SOURCE_FOLDER_FIELD].astype(str).to_numpy()
    unexpected = sorted(set(source_folders) - set(folders))
    if unexpected:
        raise ValueError(f"Unexpected source-folder labels: {unexpected}.")
    target = np.asarray([labels[folder] for folder in source_folders], dtype=np.int8)
    folder_counts = pd.Series(source_folders).value_counts().reindex(folders)
    sample_weight = np.asarray([1.0 / folder_counts[folder] for folder in source_folders])
    sample_weight *= len(sample_weight) / sample_weight.sum()

    classifier = LinearSVC(
        C=svm_c, dual="auto", max_iter=10_000, random_state=random_state
    )
    classifier.fit(scores, target, sample_weight=sample_weight)
    normal = classifier.coef_[0].astype(np.float64)
    intercept = float(classifier.intercept_[0])
    projected = _plane_projection(scores, normal, intercept)
    if not np.allclose(projected @ normal + intercept, 0.0, atol=1e-8):
        raise AssertionError("Projected PCA scores do not lie on the fitted plane.")

    filenames = {
        "pca": pca_filename,
        "classifier": classifier_filename,
        "points": points_filename,
        "plot": plot_filename,
    }
    for name, filename in filenames.items():
        candidate = Path(filename)
        if candidate.name != filename or not filename:
            raise ValueError(f"{name}_filename must be a non-empty basename, got {filename!r}.")
    if len(set(filenames.values())) != len(filenames):
        raise ValueError("Artifact filenames must be unique.")
    paths = {name: output_dir / filename for name, filename in filenames.items()}
    joblib.dump(pca, paths["pca"])
    joblib.dump(classifier, paths["classifier"])

    points = metadata[list(GROUP_FIELDS) + ["source_sample_count"]].copy()
    points.insert(3, "class_id", target)
    points.insert(4, "class_name", np.where(target == 0, "class_1", "class_2"))
    points["sample_weight"] = sample_weight
    for axis in range(3):
        points[f"pc{axis + 1}"] = scores[:, axis]
        points[f"projected_pc{axis + 1}"] = projected[:, axis]
    decision_value = scores @ normal + intercept
    points["decision_value"] = decision_value
    points["distance_to_plane"] = np.abs(decision_value) / np.linalg.norm(normal)
    points.to_csv(paths["points"], index=False)
    _plane_figure(
        scores,
        source_folders,
        normal,
        intercept,
        folder_order=folders,
        max_points_per_folder=max_plot_points_per_folder,
        random_state=random_state,
    ).write_html(paths["plot"], include_plotlyjs="cdn")

    prediction = classifier.predict(scores)
    print(f"Metadata cache: {cache_dir}")
    print(f"PCA: {pca_source}")
    print(
        f"Ignored {int(ignored_rows.sum()):,} rows; aggregated {len(fit_positions):,} fit rows "
        f"into {len(metadata):,} points."
    )
    print(f"Classifier accuracy: {accuracy_score(target, prediction):.6f}")
    print(
        "Folder-weighted classifier accuracy: "
        f"{accuracy_score(target, prediction, sample_weight=sample_weight):.6f}"
    )
    print(
        "Folder-weighted balanced accuracy: "
        f"{balanced_accuracy_score(target, prediction, sample_weight=sample_weight):.6f}"
    )
    print(
        classification_report(
            target,
            prediction,
            target_names=["class_1", "class_2"],
            sample_weight=sample_weight,
        )
    )
    return paths


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--acts-dir", type=Path, default=Path(".acts"))
    parser.add_argument(
        "--output-dir", type=Path, default=Path("results/balanced_pca_plane")
    )
    parser.add_argument(
        "--cache-dir", type=Path, default=Path("data/activation_explorer_cache")
    )
    parser.add_argument("--folders", nargs="+", default=list(DEFAULT_FOLDERS))
    parser.add_argument("--pca-batch-size", type=int, default=4096)
    parser.add_argument(
        "--prefitted-pca-path",
        type=Path,
        default=None,
        help="Use this fitted three-component PCA model instead of fitting a new one.",
    )
    parser.add_argument("--svm-c", type=float, default=1.0)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--max-plot-points-per-folder", type=int, default=5_000)
    parser.add_argument(
        "--ignore-filters-json",
        default="{}",
        help='JSON mapping of metadata fields to excluded values, e.g. {"base_unit":["seconds"]}.',
    )
    parser.add_argument("--pca-filename", default="pca_3_components.joblib")
    parser.add_argument("--classifier-filename", default="balanced_linear_svm.joblib")
    parser.add_argument("--points-filename", default="original_and_projected_pc_points.csv")
    parser.add_argument("--plot-filename", default="pca_scores_and_plane.html")
    return parser


def main() -> None:
    args = _parser().parse_args()
    try:
        ignore_filters = json.loads(args.ignore_filters_json)
    except json.JSONDecodeError as exc:
        raise SystemExit(f"Invalid --ignore-filters-json: {exc}") from exc
    if not isinstance(ignore_filters, dict):
        raise SystemExit("--ignore-filters-json must decode to an object.")
    paths = fit_balanced_pca_plane(
        args.acts_dir,
        args.output_dir,
        cache_dir=args.cache_dir,
        folders=args.folders,
        ignore_filters=ignore_filters,
        prefitted_pca_path=args.prefitted_pca_path,
        pca_batch_size=args.pca_batch_size,
        svm_c=args.svm_c,
        random_state=args.random_state,
        max_plot_points_per_folder=args.max_plot_points_per_folder,
        pca_filename=args.pca_filename,
        classifier_filename=args.classifier_filename,
        points_filename=args.points_filename,
        plot_filename=args.plot_filename,
    )
    for name, path in paths.items():
        print(f"Saved {name}: {path}")


if __name__ == "__main__":
    main()
