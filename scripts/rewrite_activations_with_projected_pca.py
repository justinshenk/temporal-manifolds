"""Project every activation row onto a saved PCA-space plane and rewrite its batch."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Sequence

import joblib
import numpy as np
import torch

from temporal_manifolds.activations.extraction_policy import CACHED_POSITION_INDEX
from temporal_manifolds.viz.activation_explorer import discover_activation_batch_paths

DEFAULT_FOLDERS = ("plain", "plain_long", "indirect", "new_conv")


def project_onto_plane(
    scores: np.ndarray,
    normal: np.ndarray,
    intercept: float,
) -> np.ndarray:
    """Return the least-distance projection of each score onto a linear plane."""
    normal_squared = float(normal @ normal)
    if not np.isfinite(normal_squared) or normal_squared <= 0:
        raise ValueError("The fitted plane has an invalid normal vector.")
    signed_values = scores @ normal + intercept
    return scores - np.outer(signed_values / normal_squared, normal)


def pca_reconstruction_basis(pca: object) -> np.ndarray:
    """Return the matrix mapping PCA-score differences back to feature differences."""
    basis = np.asarray(pca.components_, dtype=np.float64).copy()
    if bool(getattr(pca, "whiten", False)):
        basis *= np.sqrt(np.asarray(pca.explained_variance_, dtype=np.float64))[:, None]
    return basis


def activation_displacement(
    original_scores: np.ndarray,
    projected_scores: np.ndarray,
    reconstruction_basis: np.ndarray,
) -> np.ndarray:
    """Compute inverse(projected)-inverse(original), with the shared mean cancelled."""
    return (projected_scores - original_scores) @ reconstruction_basis


def _fused_plane_parameters(
    pca: object,
    normal: np.ndarray,
    intercept: float,
) -> tuple[np.ndarray, float, np.ndarray, float]:
    """Collapse PCA scoring, plane projection, and inverse projection to rank-one form."""
    components = np.asarray(pca.components_, dtype=np.float64)
    scale = (
        np.sqrt(np.asarray(pca.explained_variance_, dtype=np.float64))
        if bool(getattr(pca, "whiten", False))
        else np.ones(components.shape[0], dtype=np.float64)
    )
    feature_direction = (normal / scale) @ components
    activation_direction = (normal * scale) @ components
    affine_intercept = float(intercept - np.asarray(pca.mean_) @ feature_direction)
    normal_squared = float(normal @ normal)
    if not np.isfinite(normal_squared) or normal_squared <= 0:
        raise ValueError("The fitted plane has an invalid normal vector.")
    return feature_direction, affine_intercept, activation_direction, normal_squared


def _validate_models(
    pca: object,
    classifier: object,
) -> tuple[np.ndarray, float, np.ndarray, float]:
    components = np.asarray(getattr(pca, "components_", None))
    if components.ndim != 2 or components.shape[0] < 3:
        raise ValueError(
            f"Expected a fitted PCA model with at least three components, got {components.shape}."
        )
    coefficient = np.asarray(getattr(classifier, "coef_", None))
    intercepts = np.asarray(getattr(classifier, "intercept_", None))
    if coefficient.shape != (1, components.shape[0]) or intercepts.shape != (1,):
        raise ValueError(
            "Expected a fitted binary linear classifier matching the PCA component count."
        )
    normal = coefficient[0].astype(np.float64)
    intercept = float(intercepts[0])
    basis = pca_reconstruction_basis(pca)
    fused = _fused_plane_parameters(pca, normal, intercept)

    probe_values = np.asarray(pca.mean_, dtype=np.float64)[None, :]
    probe_scores = np.asarray(pca.transform(probe_values), dtype=np.float64)
    projected_probe = project_onto_plane(probe_scores, normal, intercept)
    expected = activation_displacement(probe_scores, projected_probe, basis)
    feature_direction, affine_intercept, activation_direction, normal_squared = fused
    signed_value = probe_values @ feature_direction + affine_intercept
    actual = -np.outer(signed_value / normal_squared, activation_direction)
    if not np.allclose(actual, expected, atol=1e-10):
        raise ValueError("The rank-one rewrite is inconsistent with the PCA-space calculation.")
    return fused


def rewrite_activation_batches(
    acts_dir: str | Path,
    output_dir: str | Path,
    *,
    pca_path: str | Path,
    classifier_path: str | Path,
    folders: Sequence[str] = DEFAULT_FOLDERS,
    overwrite: bool = False,
) -> dict[str, int | float | Path]:
    """Rewrite batches with per-row projected PCA scores while preserving payload schemas."""
    acts_dir = Path(acts_dir).resolve()
    output_dir = Path(output_dir).resolve()
    pca_path = Path(pca_path).resolve()
    classifier_path = Path(classifier_path).resolve()
    for required in (pca_path, classifier_path):
        if not required.is_file():
            raise FileNotFoundError(required)
    if output_dir == acts_dir or acts_dir in output_dir.parents:
        raise ValueError("output_dir must not equal or be nested inside acts_dir.")

    pca = joblib.load(pca_path)
    classifier = joblib.load(classifier_path)
    feature_direction, affine_intercept, activation_direction, normal_squared = (
        _validate_models(pca, classifier)
    )
    sources, _ = discover_activation_batch_paths([acts_dir / folder for folder in folders])
    output_dir.mkdir(parents=True, exist_ok=True)

    component: str | None = None
    written_files = 0
    written_rows = 0
    written_paths: list[Path] = []
    maximum_plane_error = 0.0
    for source in sources:
        source_path = Path(source)
        folder = source_path.parent.name
        if folder not in folders:
            raise ValueError(f"Unexpected source folder: {folder}.")
        destination = output_dir / folder / source_path.name
        if destination.exists() and not overwrite:
            raise FileExistsError(f"Refusing to overwrite {destination}; pass overwrite=True.")

        payload = torch.load(source_path, map_location="cpu", weights_only=True, mmap=True)
        current_component = str(payload["layer_component"])
        component = current_component if component is None else component
        if current_component != component:
            raise ValueError(
                f"Inconsistent layer component in {source_path}: {current_component}."
            )
        activations = payload.get("activations")
        if not isinstance(activations, dict) or component not in activations:
            raise ValueError(f"{source_path} does not contain activation {component}.")
        original = activations[component]
        if original.ndim != 3 or original.shape[-1] != pca.n_features_in_:
            raise ValueError(f"Unexpected activation shape in {source_path}: {tuple(original.shape)}.")
        if not 0 <= CACHED_POSITION_INDEX < original.shape[1]:
            raise ValueError(f"Cached position is unavailable in {source_path}.")

        values = original[:, CACHED_POSITION_INDEX, :].to(torch.float32).numpy()
        signed_values = values @ feature_direction + affine_intercept
        projected_plane_values = signed_values - signed_values / normal_squared * normal_squared
        maximum_plane_error = max(
            maximum_plane_error,
            float(np.max(np.abs(projected_plane_values))),
        )
        delta = torch.from_numpy(
            (-np.outer(signed_values / normal_squared, activation_direction)).astype(np.float32)
        ).to(original.dtype)
        modified = original.clone()
        modified[:, CACHED_POSITION_INDEX, :] += delta
        payload["activations"][component] = modified

        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = destination.with_suffix(destination.suffix + ".tmp")
        try:
            torch.save(payload, temporary)
            os.replace(temporary, destination)
        except Exception:
            temporary.unlink(missing_ok=True)
            raise
        written_files += 1
        written_rows += len(original)
        written_paths.append(destination)

    missing_outputs = [path for path in written_paths if not path.is_file()]
    if missing_outputs:
        preview = ", ".join(str(path) for path in missing_outputs[:3])
        raise FileNotFoundError(
            f"{len(missing_outputs)} rewritten batches are missing after writing; first: {preview}"
        )
    verified_files = sum(
        1
        for folder in folders
        for _ in (output_dir / folder).glob("activations_batch_*.pt")
    )
    if verified_files != written_files:
        raise RuntimeError(
            f"Output verification found {verified_files} batches, expected {written_files}."
        )

    return {
        "output_dir": output_dir,
        "files": written_files,
        "verified_files": verified_files,
        "rows": written_rows,
        "maximum_plane_error": maximum_plane_error,
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--acts-dir", type=Path, default=Path(".acts"))
    parser.add_argument("--output-dir", type=Path, default=Path(".acts_new"))
    parser.add_argument(
        "--pca-path",
        type=Path,
        default=Path("results/balanced_pca_plane/pca_3_components.joblib"),
    )
    parser.add_argument(
        "--classifier-path",
        type=Path,
        default=Path("results/balanced_pca_plane/balanced_linear_svm.joblib"),
    )
    parser.add_argument("--folders", nargs="+", default=list(DEFAULT_FOLDERS))
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main() -> None:
    args = _parser().parse_args()
    result = rewrite_activation_batches(
        args.acts_dir,
        args.output_dir,
        pca_path=args.pca_path,
        classifier_path=args.classifier_path,
        folders=args.folders,
        overwrite=args.overwrite,
    )
    print(f"Wrote {result['rows']:,} rows in {result['files']:,} files.")
    print(f"Maximum plane-equation error: {result['maximum_plane_error']:.3e}")
    print(f"Output root: {result['output_dir']}")


if __name__ == "__main__":
    main()
