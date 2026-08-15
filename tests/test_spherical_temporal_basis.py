from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn.decomposition import PCA

from temporal_manifolds.geometry.spherical_temporal_basis import (
    FEATURES,
    OUTPUT_COLUMNS,
    fit_spherical_temporal_basis,
    load_spherical_temporal_basis,
    serialize_spherical_temporal_basis,
    transform_spherical_temporal_basis,
)
from temporal_manifolds.viz.activation_explorer import (
    reconstruction_residual_projection,
    reconstruction_residual_statistics,
)


def _training_frame() -> pd.DataFrame:
    rows = []
    rng = np.random.default_rng(42)
    for source_index, source in enumerate(("abst", "base", "conv")):
        for horizon in (1.0, 3.0, 12.0, 48.0):
            for task_index, task in enumerate(("alpha", "beta", "gamma")):
                log_time = np.log10(horizon)
                signal = np.array(
                    [
                        log_time,
                        log_time**2,
                        np.sin(log_time),
                        source_index * 0.4,
                        task_index * 0.2,
                        log_time - source_index * 0.15,
                    ]
                )
                values = signal + rng.normal(scale=0.04, size=6)
                rows.append(
                    {
                        "source_folder": source,
                        "time_horizon_months": horizon,
                        "task": task,
                        **dict(zip(FEATURES, values, strict=True)),
                    }
                )
    return pd.DataFrame(rows)


def test_fit_transform_uses_every_source_and_emits_three_coordinates() -> None:
    frame = _training_frame()
    residual_center = np.linspace(-0.2, 0.2, 11)
    residual_components = np.eye(11)[:3]
    model = fit_spherical_temporal_basis(
        frame,
        residual_pca_center=residual_center,
        residual_pca_components=residual_components,
    )
    transformed = transform_spherical_temporal_basis(frame, model)

    assert model["training_sources"] == ["abst", "base", "conv"]
    assert model["parameters"] == {
        "within_weight": 0.5,
        "pair_weight": 5.0,
        "local_weight": 0.02,
        "ridge_fraction": 0.10,
    }
    assert set(OUTPUT_COLUMNS).issubset(transformed)
    assert transformed.loc[:, OUTPUT_COLUMNS].shape == (len(frame), 3)
    assert np.isfinite(transformed.loc[:, OUTPUT_COLUMNS].to_numpy()).all()
    assert np.allclose(model["basis"].T @ model["basis"], np.eye(3), atol=1e-10)


def test_model_artifact_round_trip_preserves_coordinates_and_parameters() -> None:
    frame = _training_frame()
    model = fit_spherical_temporal_basis(
        frame,
        residual_pca_center=np.zeros(9),
        residual_pca_components=np.eye(9)[:3],
        within_weight=0.75,
        pair_weight=2.5,
        local_weight=0.08,
        ridge_fraction=0.2,
    )
    loaded = load_spherical_temporal_basis(serialize_spherical_temporal_basis(model))

    expected = transform_spherical_temporal_basis(frame, model)
    actual = transform_spherical_temporal_basis(frame, loaded)
    assert loaded["parameters"] == model["parameters"]
    assert np.allclose(
        actual.loc[:, OUTPUT_COLUMNS], expected.loc[:, OUTPUT_COLUMNS], atol=1e-12
    )
    assert np.array_equal(
        loaded["residual_pca_components"], model["residual_pca_components"]
    )


def test_model_loader_rejects_non_artifact_bytes() -> None:
    with pytest.raises(ValueError, match="could not be loaded"):
        load_spherical_temporal_basis(b"not an npz model")


def test_saved_residual_projection_matches_fitted_incremental_pca() -> None:
    rng = np.random.default_rng(7)
    values = rng.normal(size=(30, 10))
    projection_model = PCA(n_components=3, random_state=0).fit(values)
    scores = projection_model.transform(values)
    offsets = np.arange(len(values))

    rms, residual_scores, residual_pca = reconstruction_residual_statistics(
        values,
        values,
        offsets,
        scores,
        projection_model,
        batch_size=9,
    )
    loaded_rms, loaded_scores = reconstruction_residual_projection(
        values,
        values,
        offsets,
        scores,
        projection_model,
        residual_center=residual_pca.mean_,
        residual_components=residual_pca.components_,
        batch_size=8,
    )

    assert np.allclose(loaded_rms, rms)
    assert np.allclose(loaded_scores, residual_scores)
