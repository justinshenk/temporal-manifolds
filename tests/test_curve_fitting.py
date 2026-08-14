from __future__ import annotations

import io
from dataclasses import replace

import joblib
import numpy as np
import pytest
from temporal_manifolds.viz.curve_fitting import (
    CURVE_MODEL_ARTIFACT_VERSION,
    TangentExtrapolatingBSpline,
    evaluate_curve_model,
    fit_geometric_spline,
    load_curve_model,
    parse_spline_quantiles,
    project_onto_curve_parameter,
    serialize_curve_model,
)


def _trajectory(size: int = 80) -> np.ndarray:
    parameter = np.linspace(0.0, 1.0, size)
    return np.column_stack(
        [
            np.cos(np.pi * parameter),
            np.sin(np.pi * parameter),
            2.0 * parameter - 1.0,
        ]
    )


def _fit(coordinates: np.ndarray, **kwargs: object):
    return fit_geometric_spline(
        coordinates,
        start_point=coordinates[0],
        end_point=coordinates[-1],
        **kwargs,
    )


def test_geometric_spline_fits_a_three_dimensional_curve() -> None:
    coordinates = _trajectory()
    result = _fit(
        coordinates,
        parameters={"degree": 3, "smoothing": 0.01},
        sample_count=101,
        coordinate_features=("PC1", "PC2", "PC3"),
    )

    assert result.curve_xyz.shape == (101, 3)
    assert result.point_prediction.shape == coordinates.shape
    assert result.model.parameter_feature == "t"
    assert result.model.coordinate_features == ("PC1", "PC2", "PC3")
    assert result.model.parameters["parameterization"] == "geometric_principal_curve"
    assert np.all((result.point_parameter >= 0) & (result.point_parameter <= 1))
    assert result.metrics["geometric_rmse_3d"] < 0.02
    np.testing.assert_allclose(result.model.predict([0.0])[0], coordinates[0], atol=1e-12)
    np.testing.assert_allclose(result.model.predict([1.0])[0], coordinates[-1], atol=1e-12)


def test_cubic_spline_places_knots_at_requested_quantiles() -> None:
    coordinates = _trajectory()
    quantiles = [0.25, 0.5, 0.75]
    result = _fit(
        coordinates,
        parameters={"degree": 3, "smoothing": 0.02, "knot_quantiles": quantiles},
        refinement_iterations=1,
    )

    assert result.model.parameters["knot_quantiles"] == quantiles
    assert len(result.model.parameters["resolved_knots"]) == len(quantiles)
    assert all(
        isinstance(predictor, TangentExtrapolatingBSpline)
        for predictor in result.model.predictors
    )


def test_cubic_spline_accepts_total_interior_knot_count() -> None:
    result = _fit(
        _trajectory(),
        parameters={"degree": 3, "smoothing": 0.02, "knot_count": 4},
        refinement_iterations=1,
    )

    assert result.model.parameters["knot_count"] == 4
    assert len(result.model.parameters["resolved_knots"]) == 4


@pytest.mark.parametrize(
    ("value", "message"),
    [
        ("(0.25, 0.5, 0.75)", "Python list"),
        ("[0.25, 0.5, 0.5]", "strictly increasing"),
        ("[0.25, 'half', 0.75]", "finite number"),
        ("[0, 0.5, 0.75]", "strictly between"),
    ],
)
def test_spline_quantile_parser_rejects_invalid_lists(value: str, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        parse_spline_quantiles(value)


def test_geometric_spline_is_independent_of_input_row_order() -> None:
    rng = np.random.default_rng(5)
    coordinates = _trajectory(120)
    start, end = coordinates[0].copy(), coordinates[-1].copy()
    coordinates += rng.normal(scale=0.01, size=coordinates.shape)
    rng.shuffle(coordinates)

    result = fit_geometric_spline(
        coordinates,
        start_point=start,
        end_point=end,
        parameters={"degree": 3, "smoothing": 0.02, "knot_count": 6},
        sample_count=200,
    )

    assert result.metrics["geometric_rmse_3d"] < 0.03


def test_saved_geometric_curve_round_trip_and_projection() -> None:
    coordinates = _trajectory()
    fitted = _fit(coordinates)
    artifact = serialize_curve_model(
        fitted.model,
        metadata={"direction_method": "PCA", "pca_sha256": "abc"},
    )

    restored, provenance = load_curve_model(artifact)
    parameter = project_onto_curve_parameter(restored, coordinates)
    evaluated = evaluate_curve_model(restored, parameter, coordinates, sample_count=151)

    assert evaluated.curve_xyz.shape == (151, 3)
    assert evaluated.metrics["current_rmse_3d"] < 0.02
    assert provenance["artifact_version"] == CURVE_MODEL_ARTIFACT_VERSION


def test_legacy_curve_models_and_artifact_versions_are_rejected() -> None:
    fitted = _fit(_trajectory())
    legacy_model = replace(fitted.model, parameters={"degree": 3, "smoothing": 0.1})
    with pytest.raises(ValueError, match="Only geometric spline"):
        serialize_curve_model(legacy_model)

    payload = joblib.load(io.BytesIO(serialize_curve_model(fitted.model)))
    payload["artifact_version"] = 1
    buffer = io.BytesIO()
    joblib.dump(payload, buffer)
    with pytest.raises(ValueError, match="Unsupported curve artifact version"):
        load_curve_model(buffer.getvalue())


def test_geometric_spline_requires_distinct_finite_geometry() -> None:
    with pytest.raises(ValueError, match="distinct 3D points"):
        fit_geometric_spline(
            np.zeros((5, 3)),
            start_point=np.zeros(3),
            end_point=np.ones(3),
        )


def test_geometric_spline_honors_arbitrary_endpoint_constraints() -> None:
    coordinates = _trajectory()
    start = np.array([-2.0, 0.5, -1.5])
    end = np.array([1.5, -0.25, 2.0])

    result = fit_geometric_spline(
        coordinates,
        start_point=start,
        end_point=end,
        parameters={"degree": 3, "smoothing": 0.5, "knot_count": 5},
    )

    np.testing.assert_allclose(result.curve_xyz[0], start, atol=1e-12)
    np.testing.assert_allclose(result.curve_xyz[-1], end, atol=1e-12)


def test_curve_padding_uses_linear_endpoint_tangent_extension() -> None:
    coordinates = _trajectory()
    result = _fit(coordinates, padding_fraction=0.25, sample_count=101)

    assert result.curve_parameter[[0, -1]] == pytest.approx([-0.25, 1.25])
    np.testing.assert_allclose(result.model.predict([0.0])[0], coordinates[0], atol=1e-12)
    np.testing.assert_allclose(result.model.predict([1.0])[0], coordinates[-1], atol=1e-12)
    left = result.model.predict([-0.2, -0.1, 0.0])
    right = result.model.predict([1.0, 1.1, 1.2])
    np.testing.assert_allclose(np.diff(left, axis=0)[0], np.diff(left, axis=0)[1])
    np.testing.assert_allclose(np.diff(right, axis=0)[0], np.diff(right, axis=0)[1])
