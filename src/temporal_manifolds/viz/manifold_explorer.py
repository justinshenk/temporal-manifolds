"""Kernel PCA embeddings of cached activations for the local manifold explorer.

The linear explorer (:mod:`temporal_manifolds.viz.activation_explorer`) fits PCA and
log-time-horizon-supervised PLS. This module adds the nonlinear counterpart: Kernel PCA,
which applies PCA in an implicit feature space defined by a kernel and so can unfold
curvature that a linear subspace flattens.

Every embedding is scored against the fixed target ``log10_time_horizon_months``. Kernel PCA
is unsupervised, so the target never influences the fit; it drives the reported quality
metrics only, which keeps different kernels and settings comparable on the same footing.
"""

from __future__ import annotations

import io
from collections.abc import Mapping
from dataclasses import dataclass, field
from hashlib import sha256
from pathlib import Path
from typing import Any, BinaryIO

import joblib
import numpy as np
import pandas as pd
from sklearn import __version__ as sklearn_version
from sklearn.decomposition import KernelPCA
from sklearn.metrics import pairwise_distances
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

MANIFOLD_MODEL_ARTIFACT_KIND = "temporal-manifolds.activation-manifold"
MANIFOLD_MODEL_ARTIFACT_VERSION = 1
MANIFOLD_EMBEDDING_FINGERPRINT_VERSION = 1

#: The optimization target is fixed for every embedding in this module.
TARGET_FEATURE = "log10_time_horizon_months"

#: The single supported method. Kernel PCA outperformed Isomap and UMAP on this data.
ALGORITHM = "kernel_pca"
ALGORITHM_LABEL = "Kernel PCA"
ALGORITHM_DESCRIPTION = (
    "Applies PCA in an implicit feature space defined by a kernel. It keeps a global, smooth "
    "structure and can unfold curvature that linear PCA flattens, while remaining a fixed "
    "projection that embeds out-of-sample points and reports an explained-variance spectrum."
)

KERNELS = {
    "rbf": "Radial basis function",
    "poly": "Polynomial",
    "sigmoid": "Sigmoid",
    "cosine": "Cosine",
    "linear": "Linear",
}


@dataclass(frozen=True)
class ManifoldModel:
    """A fitted Kernel PCA embedding plus everything needed to interpret its coordinates."""

    estimator: KernelPCA
    parameters: dict[str, Any]
    component_count: int
    feature_count: int
    scaler: StandardScaler | None
    #: Training embedding, retained so out-of-sample points can be compared against it.
    training_embedding: np.ndarray
    training_target: np.ndarray
    #: Share of kernel-space variance carried by each retained component.
    explained_variance_ratio: np.ndarray
    #: Share of the *total* spectrum the retained components carry, when it was computed.
    retained_variance_fraction: float | None
    metrics: dict[str, Any]
    warnings: tuple[str, ...] = ()

    @property
    def algorithm(self) -> str:
        return ALGORITHM

    def transform(self, values: np.ndarray) -> np.ndarray:
        """Embed new activation rows into the fitted coordinates."""

        prepared = np.asarray(values, dtype=np.float64)
        if prepared.ndim != 2 or prepared.shape[1] != self.feature_count:
            raise ValueError(
                f"Expected activations with {self.feature_count:,} features, "
                f"got shape {prepared.shape}."
            )
        if self.scaler is not None:
            prepared = self.scaler.transform(prepared)
        return np.asarray(self.estimator.transform(prepared), dtype=np.float64)


@dataclass
class ManifoldFitResult:
    """A fitted embedding together with its per-point coordinates and diagnostics."""

    model: ManifoldModel
    embedding: np.ndarray
    target: np.ndarray
    metrics: dict[str, Any]
    warnings: list[str] = field(default_factory=list)

    @property
    def algorithm(self) -> str:
        return ALGORITHM


def embedding_fingerprint(
    model: ManifoldModel, *, version: int = MANIFOLD_EMBEDDING_FINGERPRINT_VERSION
) -> str:
    """Fingerprint a fitted embedding so downstream artifacts can verify their basis."""

    if version != MANIFOLD_EMBEDDING_FINGERPRINT_VERSION:
        raise ValueError(f"Unsupported manifold fingerprint version: {version!r}.")
    signature = sha256()
    signature.update(b"temporal-manifolds.manifold-coordinate-system.v1\0")
    signature.update(ALGORITHM.encode("utf-8"))
    signature.update(repr(sorted(model.parameters.items())).encode("utf-8"))
    embedding = np.ascontiguousarray(model.training_embedding, dtype=np.float64)
    signature.update(str(embedding.shape).encode("ascii"))
    signature.update(embedding.tobytes())
    return signature.hexdigest()


def default_manifold_parameters() -> dict[str, Any]:
    """Return the Kernel PCA defaults."""

    return {"kernel": "rbf", "gamma": None, "degree": 3, "coef0": 1.0, "alpha": 1.0}


def _trustworthiness(
    original_distances: np.ndarray, embedding: np.ndarray, *, n_neighbors: int
) -> float:
    """Return trustworthiness, the fraction of embedded neighbors that are true neighbors.

    A score of 1.0 means every point's nearest neighbors in the embedding were also its
    nearest neighbors in activation space; lower values mean the embedding invented
    proximity that the original data does not support.
    """

    from sklearn.manifold import trustworthiness  # noqa: PLC0415 - keep import local

    return float(
        trustworthiness(
            original_distances,
            embedding,
            n_neighbors=n_neighbors,
            metric="precomputed",
        )
    )


def _continuity(
    original_distances: np.ndarray, embedding: np.ndarray, *, n_neighbors: int
) -> float:
    """Return continuity, trustworthiness computed in the opposite direction.

    It penalizes true neighbors that the embedding pushed apart, rather than false neighbors
    it pulled together.
    """

    from sklearn.manifold import trustworthiness  # noqa: PLC0415 - keep import local

    embedded_distances = pairwise_distances(embedding)
    return float(
        trustworthiness(
            embedded_distances,
            original_distances,
            n_neighbors=n_neighbors,
            metric="precomputed",
        )
    )


def target_alignment_metrics(
    embedding: np.ndarray,
    target: np.ndarray,
    *,
    n_neighbors: int = 10,
) -> dict[str, Any]:
    """Score how well an embedding organizes points by the fixed time-horizon target.

    These measure the target, never the method, so different kernels and component counts are
    judged identically.
    """

    embedding = np.asarray(embedding, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    finite = np.isfinite(target) & np.isfinite(embedding).all(axis=1)
    usable = int(finite.sum())
    metrics: dict[str, Any] = {
        "target_points": usable,
        "target_feature": TARGET_FEATURE,
    }
    if usable < 3:
        metrics.update(
            {
                "target_linear_r2": float("nan"),
                "target_spearman": float("nan"),
                "target_neighborhood_error": float("nan"),
                "target_axis_correlations": [],
            }
        )
        return metrics

    coordinates = embedding[finite]
    values = target[finite]

    # How much of the target a linear readout of the embedding coordinates explains. Fitted
    # and scored on the same points, so this measures geometric organization rather than
    # out-of-sample predictive accuracy.
    design = np.column_stack([coordinates, np.ones(len(coordinates))])
    solution, *_ = np.linalg.lstsq(design, values, rcond=None)
    predicted = design @ solution
    total = float(np.sum(np.square(values - values.mean())))
    residual = float(np.sum(np.square(values - predicted)))
    metrics["target_linear_r2"] = (
        float(1.0 - residual / total) if total > np.finfo(np.float64).eps else float("nan")
    )

    # Rank correlation of the single best axis, which catches monotone but curved layouts
    # that a linear R² understates.
    axis_correlations = []
    value_ranks = pd.Series(values).rank().to_numpy()
    for axis in range(coordinates.shape[1]):
        axis_ranks = pd.Series(coordinates[:, axis]).rank().to_numpy()
        if np.ptp(axis_ranks) <= 0:
            axis_correlations.append(0.0)
            continue
        correlation = float(np.corrcoef(axis_ranks, value_ranks)[0, 1])
        axis_correlations.append(0.0 if not np.isfinite(correlation) else correlation)
    metrics["target_axis_correlations"] = axis_correlations
    metrics["target_spearman"] = (
        float(max(axis_correlations, key=abs)) if axis_correlations else float("nan")
    )

    # Local consistency: among each point's embedded neighbors, how far off is the target?
    effective_neighbors = int(min(n_neighbors, len(coordinates) - 1))
    if effective_neighbors >= 1:
        finder = NearestNeighbors(n_neighbors=effective_neighbors + 1).fit(coordinates)
        _, indices = finder.kneighbors(coordinates)
        neighbor_targets = values[indices[:, 1:]]
        spread = float(np.mean(np.abs(neighbor_targets - values[:, np.newaxis])))
        target_scale = float(np.std(values))
        metrics["target_neighborhood_error"] = (
            spread / target_scale if target_scale > np.finfo(np.float64).eps else float("nan")
        )
    else:
        metrics["target_neighborhood_error"] = float("nan")
    return metrics


def explained_variance_table(
    result: ManifoldFitResult | ManifoldModel,
) -> pd.DataFrame:
    """Return the per-component and cumulative explained-variance ratios.

    Mirrors the linear explorer's variance table. The ratios are shares of kernel-space
    variance, which is not the same quantity as activation-space variance: it describes how
    the kernel's geometry is distributed, not how much of the original signal was kept.
    """

    model = result.model if isinstance(result, ManifoldFitResult) else result
    ratios = np.asarray(model.explained_variance_ratio, dtype=np.float64)
    return pd.DataFrame(
        {
            "component": embedding_fields(len(ratios)),
            "explained_variance": ratios,
            "cumulative_variance": np.cumsum(ratios),
        }
    )


def _variance_ratios(
    estimator: KernelPCA,
    prepared: np.ndarray,
    *,
    n_components: int,
    full_spectrum: bool,
    parameters: Mapping[str, Any],
) -> tuple[np.ndarray, float | None, list[str]]:
    """Return per-component ratios and, optionally, the retained share of total variance.

    ``KernelPCA`` only keeps the eigenvalues it was asked for, so the ratios of a truncated
    fit are shares *among retained components* and sum to 1.0 by construction. Recovering the
    true denominator needs the whole spectrum, which is an N×N eigendecomposition, so it is
    opt-in rather than automatic.
    """

    eigenvalues = np.asarray(estimator.eigenvalues_, dtype=np.float64)
    # Guard against tiny negative eigenvalues from an indefinite centered kernel matrix.
    eigenvalues = np.clip(eigenvalues, 0.0, None)
    retained_total = float(eigenvalues.sum())
    warnings: list[str] = []
    if retained_total <= np.finfo(np.float64).eps:
        return np.zeros(n_components, dtype=np.float64), None, warnings

    ratios = eigenvalues / retained_total
    if not full_spectrum:
        return ratios, None, warnings

    full = KernelPCA(
        n_components=None,
        kernel=parameters["kernel"],
        gamma=parameters["gamma"],
        degree=parameters["degree"],
        coef0=parameters["coef0"],
        alpha=parameters["alpha"],
        fit_inverse_transform=False,
        eigen_solver="dense",
        copy_X=False,
    )
    try:
        full.fit(prepared)
    except (ValueError, MemoryError, np.linalg.LinAlgError) as exc:
        warnings.append(f"The full eigenvalue spectrum could not be computed: {exc}")
        return ratios, None, warnings

    spectrum = np.clip(np.asarray(full.eigenvalues_, dtype=np.float64), 0.0, None)
    spectrum_total = float(spectrum.sum())
    if spectrum_total <= np.finfo(np.float64).eps:
        return ratios, None, warnings
    # Express each retained component against the full spectrum's total.
    ordered = np.sort(spectrum)[::-1][:n_components]
    return ordered / spectrum_total, float(ordered.sum() / spectrum_total), warnings


def fit_manifold(
    values: np.ndarray,
    target: np.ndarray,
    *,
    n_components: int,
    parameters: Mapping[str, Any] | None = None,
    standardize: bool = True,
    random_state: int = 42,
    quality_neighbors: int = 10,
    max_quality_points: int = 2_000,
    full_variance_spectrum: bool = False,
) -> ManifoldFitResult:
    """Fit a Kernel PCA embedding and score it against ``log10_time_horizon_months``.

    ``values`` are the prepared activation rows and ``target`` the aligned log-time-horizon
    values. Rows whose target is missing (unconstrained prompts state no horizon) still take
    part in the embedding, because the geometry is defined by the activations; they are
    excluded only from the target-alignment metrics.
    """

    values = np.asarray(values, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    if values.ndim != 2:
        raise ValueError("Activation values must be a two-dimensional matrix.")
    row_count, feature_count = values.shape
    if len(target) != row_count:
        raise ValueError("Activations and time-horizon targets are misaligned.")
    if row_count < 3:
        raise ValueError("At least three points are required to fit a manifold embedding.")
    max_components = min(row_count - 1, feature_count)
    if not 1 <= n_components <= max_components:
        raise ValueError(
            f"Components must be between 1 and {max_components} for the prepared matrix."
        )
    if not np.isfinite(values).all():
        raise ValueError("Activation values contain non-finite entries.")

    options = default_manifold_parameters()
    options.update(dict(parameters or {}))
    if options["kernel"] not in KERNELS:
        raise ValueError(f"Unknown kernel: {options['kernel']!r}.")
    warnings: list[str] = []

    scaler: StandardScaler | None = None
    prepared = values
    if standardize:
        # Kernel methods are distance-based, so unequal feature scales would silently let a
        # few high-variance activation dimensions dominate the geometry.
        scaler = StandardScaler()
        prepared = scaler.fit_transform(values)

    gamma = options.get("gamma")
    fitted_parameters = {
        "kernel": str(options["kernel"]),
        "gamma": None if gamma is None else float(gamma),
        "degree": int(options.get("degree", 3)),
        "coef0": float(options.get("coef0", 1.0)),
        "alpha": float(options.get("alpha", 1.0)),
    }
    estimator = KernelPCA(
        n_components=n_components,
        kernel=fitted_parameters["kernel"],
        gamma=fitted_parameters["gamma"],
        degree=fitted_parameters["degree"],
        coef0=fitted_parameters["coef0"],
        alpha=fitted_parameters["alpha"],
        fit_inverse_transform=False,
        eigen_solver="auto",
        random_state=random_state,
        copy_X=False,
    )
    embedding = np.asarray(estimator.fit_transform(prepared), dtype=np.float64)

    if embedding.shape != (row_count, n_components):
        raise ValueError(
            f"{ALGORITHM_LABEL} returned an embedding of shape {embedding.shape}, "
            f"expected {(row_count, n_components)}."
        )
    if not np.isfinite(embedding).all():
        raise ValueError(
            f"{ALGORITHM_LABEL} produced non-finite coordinates. Try a different kernel, "
            "gamma, or enable standardization."
        )

    variance_ratios, retained_fraction, variance_warnings = _variance_ratios(
        estimator,
        prepared,
        n_components=n_components,
        full_spectrum=full_variance_spectrum,
        parameters=fitted_parameters,
    )
    warnings.extend(variance_warnings)

    metrics = target_alignment_metrics(
        embedding, target, n_neighbors=min(quality_neighbors, max(row_count - 1, 1))
    )
    metrics.update(
        _structure_metrics(
            prepared,
            embedding,
            quality_neighbors=quality_neighbors,
            max_quality_points=max_quality_points,
            random_state=random_state,
        )
    )
    metrics.update(
        {
            "algorithm": ALGORITHM,
            "input_points": row_count,
            "feature_count": feature_count,
            "component_count": n_components,
            "standardized": bool(standardize),
            "explained_variance_ratio": variance_ratios,
            "cumulative_variance_ratio": np.cumsum(variance_ratios),
            "retained_variance_fraction": retained_fraction,
            "full_variance_spectrum": bool(full_variance_spectrum),
        }
    )

    model = ManifoldModel(
        estimator=estimator,
        parameters=fitted_parameters,
        component_count=n_components,
        feature_count=feature_count,
        scaler=scaler,
        training_embedding=embedding,
        training_target=target,
        explained_variance_ratio=variance_ratios,
        retained_variance_fraction=retained_fraction,
        metrics=metrics,
        warnings=tuple(warnings),
    )
    return ManifoldFitResult(
        model=model,
        embedding=embedding,
        target=target,
        metrics=metrics,
        warnings=warnings,
    )


def _structure_metrics(
    prepared: np.ndarray,
    embedding: np.ndarray,
    *,
    quality_neighbors: int,
    max_quality_points: int,
    random_state: int,
) -> dict[str, Any]:
    """Score how faithfully the embedding preserves activation-space neighborhoods.

    Trustworthiness and continuity are O(n²) in memory, so a large point cloud is subsampled
    to a bounded number of rows. The subsample is deterministic for a given seed.
    """

    row_count = len(prepared)
    if row_count > max_quality_points:
        generator = np.random.default_rng(random_state)
        selected = np.sort(generator.choice(row_count, size=max_quality_points, replace=False))
        sample_prepared = prepared[selected]
        sample_embedding = embedding[selected]
        subsampled = True
    else:
        sample_prepared = prepared
        sample_embedding = embedding
        subsampled = False

    # Trustworthiness and continuity are only defined while the neighborhood is smaller than
    # half the sample, since beyond that every point is "nearby" and the score is degenerate.
    sample_count = len(sample_embedding)
    neighbors = int(min(quality_neighbors, (sample_count - 1) // 2))
    if neighbors < 1:
        return {
            "trustworthiness": float("nan"),
            "continuity": float("nan"),
            "quality_neighbors": 0,
            "quality_points": sample_count,
            "quality_subsampled": subsampled,
        }

    original_distances = pairwise_distances(sample_prepared)
    return {
        "trustworthiness": _trustworthiness(
            original_distances, sample_embedding, n_neighbors=neighbors
        ),
        "continuity": _continuity(original_distances, sample_embedding, n_neighbors=neighbors),
        "quality_neighbors": neighbors,
        "quality_points": sample_count,
        "quality_subsampled": subsampled,
    }


def manifold_projection_frame(
    analysis_metadata: pd.DataFrame,
    result: ManifoldFitResult,
    *,
    details: Mapping[str, Any],
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Attach embedding coordinates to the prepared metadata for plotting and download."""

    analysis_rows = len(analysis_metadata)
    if len(result.embedding) != analysis_rows:
        raise ValueError(
            "Embedding coordinates and prepared metadata are misaligned: "
            f"expected {analysis_rows}, got {len(result.embedding)}."
        )
    projection = analysis_metadata.copy().reset_index(drop=True)
    for index, field_name in enumerate(embedding_fields(result.model.component_count)):
        projection[field_name] = result.embedding[:, index]
    projection[TARGET_FEATURE] = result.target

    projection_details = {
        **details,
        **result.metrics,
        "direction_method": ALGORITHM_LABEL,
        "direction_source": "fitted",
        "direction_target": TARGET_FEATURE,
        "manifold_parameters": dict(result.model.parameters),
    }
    return projection, projection_details


def embedding_fields(component_count: int) -> list[str]:
    """Return the ordered coordinate column names for a fitted embedding."""

    return [f"KPC{index + 1}" for index in range(component_count)]


def validate_manifold_model(model: Any) -> tuple[int, int]:
    """Validate a fitted manifold model and return its component and feature counts."""

    if not isinstance(model, ManifoldModel):
        raise ValueError("The selected file does not contain a manifold embedding model.")
    embedding = np.asarray(model.training_embedding)
    if embedding.ndim != 2 or not all(embedding.shape):
        raise ValueError("The manifold model has an invalid training embedding.")
    if embedding.shape[1] != model.component_count:
        raise ValueError("The manifold model has inconsistent component metadata.")
    if model.feature_count < 1:
        raise ValueError("The manifold model has an invalid activation width.")
    return int(model.component_count), int(model.feature_count)


def serialize_manifold_model(
    model: ManifoldModel,
    *,
    metadata: Mapping[str, Any] | None = None,
) -> bytes:
    """Serialize a fitted manifold model and provenance as a versioned joblib artifact."""

    component_count, feature_count = validate_manifold_model(model)
    artifact = {
        "kind": MANIFOLD_MODEL_ARTIFACT_KIND,
        "version": MANIFOLD_MODEL_ARTIFACT_VERSION,
        "model": model,
        "sklearn_version": sklearn_version,
        "algorithm": ALGORITHM,
        "component_count": component_count,
        "feature_count": feature_count,
        "metadata": {
            **dict(metadata or {}),
            "manifold_algorithm": ALGORITHM,
            "manifold_parameters": dict(model.parameters),
            "direction_target": TARGET_FEATURE,
            "explained_variance_ratio": np.asarray(model.explained_variance_ratio).tolist(),
        },
    }
    buffer = io.BytesIO()
    joblib.dump(artifact, buffer, compress=3)
    return buffer.getvalue()


def load_manifold_model(
    source: str | Path | bytes | BinaryIO,
) -> tuple[ManifoldModel, dict[str, Any]]:
    """Load a trusted manifold artifact and return its provenance.

    Joblib and pickle files can execute arbitrary code while loading. Callers must only pass
    files from trusted sources.
    """

    if isinstance(source, bytes):
        source = io.BytesIO(source)
    elif hasattr(source, "seek"):
        source.seek(0)
    try:
        payload = joblib.load(source)
    except Exception as exc:  # noqa: BLE001 - normalize artifact errors for UI callers
        raise ValueError(f"The manifold model file could not be loaded: {exc}") from exc

    if not isinstance(payload, Mapping) or payload.get("kind") != MANIFOLD_MODEL_ARTIFACT_KIND:
        raise ValueError("The selected file is not a supported manifold model artifact.")
    if payload.get("version") != MANIFOLD_MODEL_ARTIFACT_VERSION:
        raise ValueError(f"Unsupported manifold artifact version: {payload.get('version')!r}.")
    model = payload.get("model")
    component_count, feature_count = validate_manifold_model(model)
    if payload.get("component_count") != component_count:
        raise ValueError("The manifold artifact's component count does not match its model.")
    if payload.get("feature_count") != feature_count:
        raise ValueError("The manifold artifact's feature count does not match its model.")
    metadata = payload.get("metadata", {})
    if not isinstance(metadata, Mapping):
        raise ValueError("The manifold artifact contains invalid provenance metadata.")
    provenance = {
        **dict(metadata),
        "artifact_kind": MANIFOLD_MODEL_ARTIFACT_KIND,
        "artifact_version": MANIFOLD_MODEL_ARTIFACT_VERSION,
        "sklearn_version": payload.get("sklearn_version"),
        "algorithm": ALGORITHM,
        "component_count": component_count,
        "feature_count": feature_count,
    }
    return model, provenance
