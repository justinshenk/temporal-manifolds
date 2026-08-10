from __future__ import annotations

import io
from dataclasses import replace

import joblib
import numpy as np
import pytest

from temporal_manifolds.viz.surface_fitting import (
    ALGORITHM_POINT_CAPS,
    SURFACE_ALGORITHMS,
    SURFACE_DESCRIPTIONS,
    SURFACE_MODEL_ARTIFACT_KIND,
    SURFACE_MODEL_ARTIFACT_VERSION,
    default_surface_parameters,
    evaluate_surface_model,
    fit_surface,
    load_surface_model,
    serialize_surface_model,
    surface_point_cap,
)


def _curved_points(size: int = 8) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    axis = np.linspace(-1.5, 1.5, size)
    x_grid, y_grid = np.meshgrid(axis, axis)
    x = x_grid.ravel()
    y = y_grid.ravel()
    z = np.sin(1.2 * x) + 0.35 * y**2 - 0.2 * x * y
    return x, y, z


def _joblib_bytes(payload) -> bytes:
    buffer = io.BytesIO()
    joblib.dump(payload, buffer)
    return buffer.getvalue()


class _HalfInfinitePredictor:
    def predict(self, coordinates: np.ndarray) -> np.ndarray:
        prediction = coordinates[:, 0] + coordinates[:, 1]
        prediction[coordinates[:, 0] > 0] = np.inf
        return prediction


class _SingleValuePredictor:
    def predict(self, coordinates: np.ndarray) -> np.ndarray:
        return np.array([0.0])


def test_only_extrapolatable_surface_algorithms_are_advertised() -> None:
    expected = {"plane", "polynomial", "rbf", "svr", "gaussian_process"}
    assert set(SURFACE_ALGORITHMS) == expected
    assert set(SURFACE_DESCRIPTIONS) == expected
    assert set(ALGORITHM_POINT_CAPS) == expected


def test_plane_recovers_exact_height_field() -> None:
    x, y, _ = _curved_points(6)
    z = 1.5 + 2.0 * x - 3.0 * y

    result = fit_surface(
        x,
        y,
        z,
        algorithm="plane",
        grid_resolution=25,
        domain_mask="full_grid",
        validation_fraction=0.25,
    )

    assert result.grid_x.shape == result.grid_y.shape == result.grid_z.shape == (25, 25)
    assert np.all(np.isfinite(result.grid_z))
    assert result.metrics["train_rmse"] == pytest.approx(0.0, abs=1e-12)
    assert result.metrics["validation_rmse"] == pytest.approx(0.0, abs=1e-12)
    assert result.metrics["validation_r2"] == pytest.approx(1.0)
    assert result.metrics["validation_predicted_count"] == result.metrics["validation_total_count"]
    outside = np.array([[4.0, -5.0], [-3.0, 2.5]])
    assert result.model.predict(outside) == pytest.approx(
        1.5 + 2.0 * outside[:, 0] - 3.0 * outside[:, 1]
    )
    assert result.model.extrapolation_mask(outside).tolist() == [True, True]
    assert result.model.equation is not None


def test_polynomial_degree_two_recovers_quadratic() -> None:
    x, y, _ = _curved_points(7)
    z = 0.4 + 1.2 * x - 0.8 * y + 0.7 * x**2 + 0.3 * x * y - 0.5 * y**2

    result = fit_surface(
        x,
        y,
        z,
        algorithm="polynomial",
        parameters={"degree": 2, "alpha": 0.0},
        grid_resolution=21,
        validation_fraction=0.2,
    )

    assert result.metrics["train_r2"] == pytest.approx(1.0)
    assert result.metrics["validation_r2"] == pytest.approx(1.0)
    assert result.metrics["validation_rmse"] == pytest.approx(0.0, abs=1e-10)


def test_saved_surface_preserves_normalization_for_raw_extrapolation() -> None:
    x = np.array([1_000.0, 1_010.0, 1_000.0, 1_010.0, 1_005.0, 1_002.0])
    y = np.array([0.001, 0.001, 0.003, 0.003, 0.002, 0.0025])
    z = 4.0 + 0.002 * x - 500.0 * y
    result = fit_surface(
        x,
        y,
        z,
        algorithm="plane",
        grid_resolution=15,
        validation_fraction=0,
        input_features=("PC1", "PC2"),
        output_feature="PC3",
    )
    outside = np.array([[1_050.0, 0.01], [980.0, -0.002]])
    expected = 4.0 + 0.002 * outside[:, 0] - 500.0 * outside[:, 1]

    assert result.model.predict(outside) == pytest.approx(expected, abs=1e-10)
    restored, provenance = load_surface_model(serialize_surface_model(result.model))
    assert restored.predict(outside) == pytest.approx(expected, abs=1e-10)
    assert provenance["equation"] == result.model.equation


def test_loaded_surface_uses_a_new_chart_grid_and_accepts_swapped_axes() -> None:
    pc1, pc2, _ = _curved_points(6)
    pc3 = 1.5 + 2.0 * pc1 - 3.0 * pc2
    fitted = fit_surface(
        pc1,
        pc2,
        pc3,
        algorithm="plane",
        grid_resolution=15,
        validation_fraction=0,
        input_features=("PC1", "PC2"),
        output_feature="PC3",
    )
    restored, _ = load_surface_model(serialize_surface_model(fitted.model))

    evaluated = evaluate_surface_model(
        restored,
        pc2,
        pc1,
        pc3,
        input_features=("PC2", "PC1"),
        output_feature="PC3",
        grid_resolution=23,
        padding_fraction=0.25,
        domain_mask="full_grid",
    )

    assert evaluated.grid_x.shape == evaluated.grid_y.shape == evaluated.grid_z.shape == (23, 23)
    assert evaluated.point_x == pytest.approx(pc2)
    assert evaluated.point_y == pytest.approx(pc1)
    assert evaluated.point_prediction == pytest.approx(pc3, abs=1e-12)
    assert evaluated.grid_z == pytest.approx(
        1.5 + 2.0 * evaluated.grid_y - 3.0 * evaluated.grid_x,
        abs=1e-12,
    )
    assert np.min(evaluated.grid_x) == pytest.approx(np.min(pc2) - 0.25 * np.ptp(pc2))
    assert np.max(evaluated.grid_y) == pytest.approx(np.max(pc1) + 0.25 * np.ptp(pc1))
    assert evaluated.metrics["evaluation_rmse"] == pytest.approx(0.0, abs=1e-12)
    assert evaluated.metrics["grid_extrapolation_fraction"] > 0


def test_loaded_full_grid_can_use_tightly_filtered_current_points() -> None:
    pc1, pc2, _ = _curved_points(5)
    pc3 = 1.5 + 2.0 * pc1 - 3.0 * pc2
    fitted = fit_surface(
        pc1,
        pc2,
        pc3,
        algorithm="plane",
        grid_resolution=15,
        validation_fraction=0,
        input_features=("PC1", "PC2"),
        output_feature="PC3",
    )

    evaluated = evaluate_surface_model(
        fitted.model,
        [0.0],
        [0.0],
        [1.5],
        input_features=("PC1", "PC2"),
        output_feature="PC3",
        grid_resolution=19,
        domain_mask="full_grid",
    )

    assert evaluated.grid_z.shape == (19, 19)
    assert np.all(np.isfinite(evaluated.grid_z))
    assert np.min(evaluated.grid_x) == pytest.approx(np.min(pc1))
    assert np.max(evaluated.grid_y) == pytest.approx(np.max(pc2))
    assert evaluated.metrics["evaluation_total_count"] == 1
    assert evaluated.metrics["evaluation_r2"] is None
    assert any("saved training interval" in warning for warning in evaluated.warnings)

    with pytest.raises(ValueError, match="three unique"):
        evaluate_surface_model(
            fitted.model,
            [0.0, 1.0],
            [0.0, 1.0],
            [1.5, 0.5],
            input_features=("PC1", "PC2"),
            output_feature="PC3",
            domain_mask="convex_hull",
        )


def test_loaded_grid_uses_finite_xy_extent_and_does_not_clip_infinity() -> None:
    x, y, _ = _curved_points(5)
    z = x + y
    fitted = fit_surface(
        x,
        y,
        z,
        algorithm="plane",
        grid_resolution=15,
        validation_fraction=0,
        input_features=("PC1", "PC2"),
        output_feature="PC3",
    )
    nonfinite_model = replace(fitted.model, estimator=_HalfInfinitePredictor())

    evaluated = evaluate_surface_model(
        nonfinite_model,
        [-1.0, 1.0, 10.0],
        [-1.0, 1.0, 2.0],
        [-2.0, 2.0, np.nan],
        input_features=("PC1", "PC2"),
        output_feature="PC3",
        grid_resolution=15,
        domain_mask="full_grid",
        clip_mode="observed_range",
    )

    assert np.max(evaluated.grid_x) == pytest.approx(10.0)
    assert np.max(evaluated.grid_y) == pytest.approx(2.0)
    assert 0 < evaluated.metrics["grid_coverage"] < 1
    assert not np.isinf(evaluated.grid_z).any()
    assert np.isnan(evaluated.grid_z).any()
    assert np.nanmax(evaluated.grid_z) <= 2.0
    assert any("preview-grid" in warning for warning in evaluated.warnings)


def test_loaded_gaussian_process_builds_uncertainty_on_the_new_grid() -> None:
    x, y, z = _curved_points(5)
    fitted = fit_surface(
        x,
        y,
        z,
        algorithm="gaussian_process",
        parameters={**default_surface_parameters("gaussian_process", len(x)), "optimize": False},
        grid_resolution=15,
        validation_fraction=0,
        input_features=("PC1", "PC2"),
        output_feature="PC3",
    )

    evaluated = evaluate_surface_model(
        fitted.model,
        x,
        y,
        z,
        input_features=("PC1", "PC2"),
        output_feature="PC3",
        grid_resolution=17,
        padding_fraction=0.2,
    )

    assert evaluated.grid_uncertainty is not None
    assert evaluated.grid_uncertainty.shape == (17, 17)
    assert np.all(np.isfinite(evaluated.grid_uncertainty))
    assert np.all(evaluated.grid_uncertainty >= 0)


def test_loaded_surface_rejects_incompatible_chart_components() -> None:
    x, y, z = _curved_points(4)
    fitted = fit_surface(
        x,
        y,
        z,
        algorithm="plane",
        grid_resolution=15,
        validation_fraction=0,
        input_features=("PC1", "PC2"),
        output_feature="PC3",
    )

    with pytest.raises(ValueError, match="current X/Y components"):
        evaluate_surface_model(
            fitted.model,
            x,
            y,
            z,
            input_features=("PC1", "PC4"),
            output_feature="PC3",
        )
    with pytest.raises(ValueError, match="current Z component"):
        evaluate_surface_model(
            fitted.model,
            x,
            y,
            z,
            input_features=("PC1", "PC2"),
            output_feature="PC4",
        )


@pytest.mark.parametrize("algorithm", list(SURFACE_ALGORITHMS))
def test_every_surface_algorithm_produces_a_grid(algorithm: str) -> None:
    x, y, z = _curved_points()
    parameters = default_surface_parameters(algorithm, len(x))
    if algorithm == "gaussian_process":
        parameters["optimize"] = False
    result = fit_surface(
        x,
        y,
        z,
        algorithm=algorithm,
        parameters=parameters,
        grid_resolution=18,
        validation_fraction=0.2,
        random_state=9,
        input_features=("PC1", "PC2"),
        output_feature="PC3",
    )

    assert result.algorithm == algorithm
    assert result.grid_z.shape == (18, 18)
    assert np.isfinite(result.grid_z).any()
    assert np.isfinite(result.point_prediction).any()
    assert result.metrics["fit_points"] == len(x)
    assert 0 < result.metrics["grid_coverage"] <= 1
    assert result.metrics["train_rmse"] is not None

    outside = np.array([[-3.0, 0.0], [0.0, 3.0], [2.5, -2.5]])
    expected = result.model.predict(outside)
    assert np.all(np.isfinite(expected))
    assert result.model.extrapolation_mask(outside).all()
    artifact = serialize_surface_model(result.model, metadata={"fit_scope": "Visible"})
    restored, provenance = load_surface_model(artifact)
    assert restored.predict(outside) == pytest.approx(expected)
    if algorithm == "gaussian_process":
        restored_mean, restored_std = restored.predict(outside, return_std=True)
        assert restored_mean == pytest.approx(expected)
        assert np.all(np.isfinite(restored_std))
        assert np.all(restored_std >= 0)
    assert restored.input_features == ("PC1", "PC2")
    assert restored.output_feature == "PC3"
    assert provenance["fit_scope"] == "Visible"
    assert provenance["artifact_kind"] == SURFACE_MODEL_ARTIFACT_KIND
    assert provenance["artifact_version"] == SURFACE_MODEL_ARTIFACT_VERSION


def test_stochastic_fit_is_deterministic_for_a_fixed_seed() -> None:
    x, y, z = _curved_points(10)
    arguments = {
        "algorithm": "plane",
        "parameters": {
            "estimator": "ransac",
            "ransac_min_samples": 0.4,
            "ransac_max_trials": 50,
        },
        "grid_resolution": 20,
        "max_fit_points": 55,
        "random_state": 123,
    }

    first = fit_surface(x, y, z, **arguments)
    second = fit_surface(x, y, z, **arguments)

    assert np.allclose(first.grid_z, second.grid_z, equal_nan=True)
    assert first.metrics["validation_rmse"] == pytest.approx(second.metrics["validation_rmse"])


def test_duplicate_and_nonfinite_points_are_reported() -> None:
    x, y, z = _curved_points(5)
    result = fit_surface(
        np.append(x, [x[0], np.nan]),
        np.append(y, [y[0], y[1]]),
        np.append(z, [z[0] + 2.0, z[1]]),
        algorithm="plane",
        duplicate_reducer="median",
        grid_resolution=15,
    )

    assert result.metrics["input_points"] == 27
    assert result.metrics["dropped_nonfinite"] == 1
    assert result.metrics["duplicates_merged"] == 1
    assert any("non-finite" in warning for warning in result.warnings)
    assert any("duplicate" in warning for warning in result.warnings)


def test_convex_hull_mask_and_prediction_clipping() -> None:
    x = np.array([0.0, 1.0, 0.0, 0.25, 0.6, 0.2])
    y = np.array([0.0, 0.0, 1.0, 0.2, 0.2, 0.6])
    z = x + y
    result = fit_surface(
        x,
        y,
        z,
        algorithm="plane",
        grid_resolution=30,
        padding_fraction=0.2,
        domain_mask="convex_hull",
        clip_mode="observed_range",
        validation_fraction=0,
    )

    finite = np.isfinite(result.grid_z)
    assert 0 < finite.mean() < 1
    assert np.nanmin(result.grid_z) >= np.min(z)
    assert np.nanmax(result.grid_z) <= np.max(z)
    assert result.model.predict([[2.0, 2.0]])[0] == pytest.approx(4.0)


@pytest.mark.parametrize(
    ("x", "y", "z", "message"),
    [
        ([0, 1], [0], [0, 1], "same number"),
        ([0, 1, 2, 3], [0, 1, 2, 3], [0, 1, 4, 9], "collinear"),
    ],
)
def test_invalid_point_sets_raise_friendly_errors(x, y, z, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        fit_surface(x, y, z)


def test_unknown_algorithm_and_invalid_grid_are_rejected() -> None:
    x, y, z = _curved_points(4)
    with pytest.raises(ValueError, match="Unknown surface algorithm"):
        fit_surface(x, y, z, algorithm="magic")
    with pytest.raises(ValueError, match="grid_resolution"):
        fit_surface(x, y, z, grid_resolution=10)


def test_interaction_only_polynomial_uses_its_actual_feature_count() -> None:
    x = np.array([0.0, 1.0, 0.0, 1.0])
    y = np.array([0.0, 0.0, 1.0, 1.0])
    z = 1.0 + 2.0 * x - y + 0.5 * x * y

    result = fit_surface(
        x,
        y,
        z,
        algorithm="polynomial",
        parameters={"degree": 6, "alpha": 0.0, "interaction_only": True},
        grid_resolution=15,
        validation_fraction=0,
    )

    assert result.metrics["train_rmse"] == pytest.approx(0.0, abs=1e-12)


def test_invalid_rbf_tail_and_neighbor_combinations_are_rejected() -> None:
    x, y, z = _curved_points(4)
    with pytest.raises(ValueError, match="needs at least 6 local neighbors"):
        fit_surface(
            x,
            y,
            z,
            algorithm="rbf",
            parameters={"kernel": "quintic", "degree": None, "neighbors": 3},
        )
    with pytest.raises(ValueError, match="degree 2 or greater"):
        fit_surface(
            x,
            y,
            z,
            algorithm="rbf",
            parameters={"kernel": "quintic", "degree": 1, "neighbors": 10},
        )


def test_invalid_reducer_and_density_factor_are_rejected_with_unique_points() -> None:
    x, y, z = _curved_points(4)
    with pytest.raises(ValueError, match="duplicate_reducer"):
        fit_surface(x, y, z, duplicate_reducer="bogus")
    with pytest.raises(ValueError, match="density_factor"):
        fit_surface(x, y, z, domain_mask="dense_support", density_factor=np.nan)


def test_unique_shuffled_points_keep_coordinates_paired_with_values() -> None:
    x = np.array([1.2, -0.4, 0.8, -1.1, 0.1, 1.7])
    y = np.array([-0.7, 1.5, 0.3, -0.2, 0.9, 1.1])
    z = np.array([4.2, -1.0, 0.3, 2.7, 8.1, -3.4])

    result = fit_surface(
        x,
        y,
        z,
        algorithm="plane",
        grid_resolution=15,
        validation_fraction=0,
    )

    assert np.array_equal(result.point_x, x)
    assert np.array_equal(result.point_y, y)
    assert np.array_equal(result.point_z, z)


def test_global_rbf_has_a_stricter_safety_cap() -> None:
    assert surface_point_cap("rbf", {"neighbors": None}) == 1_000
    assert surface_point_cap("rbf", {"neighbors": 80}) == 8_000


def test_surface_artifact_rejects_invalid_payloads_and_preprocessing() -> None:
    x, y, z = _curved_points(4)
    result = fit_surface(
        x,
        y,
        z,
        algorithm="plane",
        grid_resolution=15,
        validation_fraction=0,
    )

    with pytest.raises(ValueError, match="coordinate scale"):
        serialize_surface_model(replace(result.model, coordinate_scale=np.array([1.0, 0.0])))
    with pytest.raises(ValueError, match="invalid prediction count"):
        replace(result.model, estimator=_SingleValuePredictor()).predict([[0.0, 0.0], [1.0, 1.0]])
    with pytest.raises(ValueError, match="not a supported surface model artifact"):
        load_surface_model(_joblib_bytes(result.model.estimator))
    with pytest.raises(ValueError, match="not a supported surface model artifact"):
        load_surface_model(_joblib_bytes({"kind": "another-artifact"}))

    payload = joblib.load(io.BytesIO(serialize_surface_model(result.model)))
    payload["version"] = SURFACE_MODEL_ARTIFACT_VERSION + 1
    with pytest.raises(ValueError, match="Unsupported surface artifact version"):
        load_surface_model(_joblib_bytes(payload))
    with pytest.raises(ValueError, match="could not be loaded"):
        load_surface_model(b"not a joblib artifact")
