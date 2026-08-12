from __future__ import annotations

import io

import joblib
import numpy as np
import pytest
from scipy.interpolate import CubicSpline

from temporal_manifolds.viz.extruded_spline_surface import (
    EXTRUDED_SURFACE_ARTIFACT_KIND,
    ExtrudedSplineSurface,
    evaluate_extruded_surface,
    fit_extrusion_direction,
    load_extruded_surface,
    load_surface,
    serialize_extruded_surface,
)
from temporal_manifolds.viz.curve_fitting import CurveModel


def _artifact() -> dict:
    parameter = np.linspace(-1.0, 1.0, 7)
    return {
        "artifact_kind": EXTRUDED_SURFACE_ARTIFACT_KIND,
        "curve_predictors": (
            CubicSpline(parameter, parameter),
            CubicSpline(parameter, 2.0 * parameter),
            CubicSpline(parameter, -parameter),
        ),
        "parameter_center": 0.0,
        "parameter_scale": 1.0,
        "training_parameter_bounds": np.array([-1.0, 1.0]),
        "extrusion_direction": np.array([0.0, 1.0, 0.0]),
        "parameter_feature": "log10_time_horizon_months",
        "coordinate_features": ("PLS1", "PLS2", "PLS3"),
        "metadata": {"direction_method": "PLS"},
    }


def _joblib_bytes(payload: dict) -> bytes:
    buffer = io.BytesIO()
    joblib.dump(payload, buffer)
    return buffer.getvalue()


def _curve_model(*, degree: int = 3) -> CurveModel:
    artifact = _artifact()
    return CurveModel(
        algorithm="spline",
        predictors=artifact["curve_predictors"],
        parameter_feature=artifact["parameter_feature"],
        coordinate_features=artifact["coordinate_features"],
        parameter_center=artifact["parameter_center"],
        parameter_scale=artifact["parameter_scale"],
        training_parameter_bounds=artifact["training_parameter_bounds"],
        training_coordinate_bounds=np.array([[-1.0, 1.0], [-2.0, 2.0], [-1.0, 1.0]]),
        parameters={"degree": degree, "smoothing": 0.0},
    )


def test_loads_portable_reference_artifact_and_preserves_compatibility_api() -> None:
    payload = _joblib_bytes(_artifact())

    model, metadata = load_extruded_surface(payload)

    assert isinstance(model, ExtrudedSplineSurface)
    assert metadata["direction_method"] == "PLS"
    np.testing.assert_allclose(
        model.curve([0.0, 0.5]),
        [[0.0, 0.0, 0.0], [0.5, 1.0, -0.5]],
    )
    assert load_surface(payload).coordinate_features == ("PLS1", "PLS2", "PLS3")


def test_extruded_surface_round_trip_preserves_model_and_metadata() -> None:
    original = ExtrudedSplineSurface.from_artifact(_artifact())

    payload = serialize_extruded_surface(
        original,
        metadata={"direction_method": "PLS", "fit_scope": "Visible"},
    )
    restored, metadata = load_extruded_surface(payload)

    parameter = np.array([-0.75, 0.0, 0.8])
    extrusion = np.array([-2.0, 1.0, 3.5])
    np.testing.assert_allclose(
        restored.evaluate(parameter, extrusion),
        original.evaluate(parameter, extrusion),
    )
    assert restored.direction == pytest.approx(original.direction)
    assert metadata["artifact_version"] == 2
    assert metadata["direction_method"] == "PLS"
    assert metadata["fit_scope"] == "Visible"


def test_projects_points_and_samples_parametric_sheet() -> None:
    model = ExtrudedSplineSurface.from_artifact(_artifact())
    points = np.array(
        [
            [0.25, 9.0, -0.25],
            [-0.5, -4.0, 0.5],
        ]
    )

    result = evaluate_extruded_surface(
        model,
        points,
        parameter_samples=40,
        extrusion_samples=12,
        parameter_padding_fraction=0.1,
        extrusion_padding_fraction=0.0,
    )

    assert result.grid_xyz.shape == (12, 40, 3)
    assert result.point_parameter == pytest.approx([0.25, -0.5], abs=1e-5)
    assert result.point_projection == pytest.approx(points, abs=1e-5)
    assert result.metrics["current_rmse_3d"] == pytest.approx(0.0, abs=1e-5)
    assert result.metrics["display_parameter_bounds"] == pytest.approx([-1.2, 1.2])


def test_fits_global_least_squares_extrusion_direction_from_cubic_spline() -> None:
    curve = _curve_model()
    parameter = np.linspace(-0.9, 0.9, 30)
    curve_xyz = curve.predict(parameter)
    expected_direction = np.array([1.0, -2.0, 3.0])
    expected_direction /= np.linalg.norm(expected_direction)
    extrusion = np.linspace(-4.0, 5.0, len(parameter))
    points = curve_xyz + extrusion[:, np.newaxis] * expected_direction

    fitted = fit_extrusion_direction(curve, parameter, points)

    assert fitted.model.direction == pytest.approx(expected_direction)
    assert fitted.metrics["surface_rmse_3d"] == pytest.approx(0.0, abs=1e-12)
    assert fitted.metrics["curve_rmse_3d"] > 0
    assert fitted.metrics["captured_residual_variance"] == pytest.approx(1.0)

    evaluated = evaluate_extruded_surface(
        fitted.model,
        points,
        point_parameter=parameter,
        parameter_samples=40,
        extrusion_samples=12,
    )
    assert evaluated.metrics["parameter_aligned_projection"]
    assert evaluated.metrics["current_rmse_3d"] == pytest.approx(0.0, abs=1e-12)


def test_extrusion_direction_has_minimum_rmse_over_candidate_directions() -> None:
    rng = np.random.default_rng(7)
    curve = _curve_model()
    parameter = np.linspace(-1.0, 1.0, 80)
    curve_xyz = curve.predict(parameter)
    residuals = rng.normal(size=(len(parameter), 3)) @ np.diag([0.2, 2.0, 0.6])
    points = curve_xyz + residuals

    fitted = fit_extrusion_direction(curve, parameter, points)
    fitted_error = np.sum(np.square(points - fitted.point_projection))
    for candidate in rng.normal(size=(200, 3)):
        candidate /= np.linalg.norm(candidate)
        candidate_projection = curve_xyz + (residuals @ candidate)[:, None] * candidate
        candidate_error = np.sum(np.square(points - candidate_projection))
        assert fitted_error <= candidate_error + 1e-10


def test_quadratic_extrusion_recovers_curved_cross_section() -> None:
    curve = _curve_model()
    parameter = np.linspace(-1.0, 1.0, 120)
    curve_xyz = curve.predict(parameter)
    extrusion = np.linspace(-2.0, 2.0, len(parameter))
    linear = np.array([1.4, -0.3, 0.2])
    quadratic = np.array([-0.2, 0.7, 1.1])
    points = (
        curve_xyz
        + extrusion[:, None] * linear
        + np.square(extrusion)[:, None] * quadratic
    )

    fitted_linear = fit_extrusion_direction(curve, parameter, points, degree=1)
    fitted_quadratic = fit_extrusion_direction(curve, parameter, points, degree=2)

    assert fitted_quadratic.model.extrusion_degree == 2
    assert fitted_quadratic.model.quadratic_direction is not None
    assert fitted_quadratic.metrics["surface_rmse_3d"] < 1e-5
    assert fitted_quadratic.metrics["surface_rmse_3d"] < (
        fitted_linear.metrics["surface_rmse_3d"] * 1e-4
    )
    evaluated = evaluate_extruded_surface(
        fitted_quadratic.model,
        points,
        point_parameter=parameter,
        parameter_samples=40,
        extrusion_samples=20,
    )
    assert evaluated.metrics["current_rmse_3d"] < 1e-5


def test_quadratic_surface_round_trip_preserves_both_vector_coefficients() -> None:
    surface = ExtrudedSplineSurface(
        predictors=_curve_model().predictors,
        parameter_center=0.0,
        parameter_scale=1.0,
        parameter_bounds=np.array([-1.0, 1.0]),
        direction=np.array([1.0, 0.0, -0.25]),
        quadratic_direction=np.array([0.1, 0.8, 0.4]),
        parameter_feature="log10_time_horizon_months",
        coordinate_features=("PLS1", "PLS2", "PLS3"),
    )

    restored, metadata = load_extruded_surface(serialize_extruded_surface(surface))

    assert metadata["artifact_version"] == 2
    assert restored.extrusion_degree == 2
    np.testing.assert_allclose(restored.extrusion_coefficients, surface.extrusion_coefficients)
    np.testing.assert_allclose(
        restored.evaluate([0.2, 0.7], [-1.5, 2.0]),
        surface.evaluate([0.2, 0.7], [-1.5, 2.0]),
    )


def test_extrusion_rejects_non_cubic_curve() -> None:
    with pytest.raises(ValueError, match="degree-3 cubic spline"):
        fit_extrusion_direction(_curve_model(degree=4), [0.0], [[0.0, 0.0, 0.0]])


def test_rejects_invalid_direction_and_artifact_kind() -> None:
    invalid = _artifact()
    invalid["extrusion_direction"] = np.zeros(3)
    with pytest.raises(ValueError, match="non-zero"):
        ExtrudedSplineSurface.from_artifact(invalid)

    with pytest.raises(ValueError, match="not a supported"):
        load_extruded_surface(_joblib_bytes({"artifact_kind": "other"}))
