from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from scipy.interpolate import CubicSpline

from temporal_manifolds.geometry.extrusion.rms_spline_surface_transformer import (
    RMSSplineSurfaceTransformer,
    add_surface_parameter_columns,
    evaluate_rms_spline_surface,
    serialize_rms_spline_surface,
)
from temporal_manifolds.viz.curve_fitting import CurveModel


def _curve_model() -> CurveModel:
    parameter = np.linspace(-1.0, 1.0, 9)
    coordinates = np.column_stack(
        [parameter, parameter**2 - 0.25, np.sin(parameter)]
    )
    return CurveModel(
        algorithm="spline",
        predictors=tuple(
            CubicSpline(parameter, coordinates[:, axis]) for axis in range(3)
        ),
        parameter_feature="log10_time_horizon_months",
        coordinate_features=("PLS1", "PLS2", "PLS3"),
        parameter_center=0.0,
        parameter_scale=1.0,
        training_parameter_bounds=np.array([-1.0, 1.0]),
        training_coordinate_bounds=np.column_stack(
            [coordinates.min(axis=0), coordinates.max(axis=0)]
        ),
        parameters={"degree": 3, "smoothing": 0.0},
    )


def test_rms_transformer_fit_evaluate_and_artifact_round_trip() -> None:
    curve = _curve_model()
    parameter = np.linspace(-0.9, 0.9, 40)
    residual = np.geomspace(0.4, 2.5, len(parameter))
    residual_reference = float(residual.min())
    log_scale = float(np.std(np.log10(residual)))
    extrusion = (np.log10(residual) - np.log10(residual_reference)) / log_scale
    coefficients = np.array(
        [
            [1.2, -0.15, 0.04],
            [-0.4, 0.3, -0.02],
            [0.25, 0.1, 0.05],
        ]
    )
    polynomial = np.column_stack([extrusion, extrusion**2, extrusion**3])
    coordinates = curve.predict(parameter) + polynomial @ coefficients.T
    frame = pd.DataFrame(coordinates, columns=curve.coordinate_features)
    frame[curve.parameter_feature] = parameter
    frame["reconstruction_residual_rms"] = residual
    frame["source_folder"] = "training"

    model = RMSSplineSurfaceTransformer.fit(
        frame,
        {"model": curve},
        base_source="training",
        fit_sources=("training",),
        residual_feature="reconstruction_residual_rms",
        degree=3,
        ridge_alpha=0.0,
    )

    np.testing.assert_allclose(model.coefficients, coefficients, atol=1e-10)
    result = evaluate_rms_spline_surface(
        model,
        coordinates,
        residual,
        true_parameter=parameter,
        parameter_samples=40,
        residual_samples=12,
    )
    assert result.metrics["current_rmse_3d"] == pytest.approx(0.0, abs=1e-7)
    assert result.metrics["parameter_rmse"] == pytest.approx(0.0, abs=1e-7)
    assert result.grid_xyz.shape == (12, 40, 3)

    transformed = add_surface_parameter_columns(frame, model)
    np.testing.assert_allclose(transformed["t"], parameter, atol=1e-7)
    np.testing.assert_allclose(transformed["u"], extrusion, atol=1e-10)

    restored, metadata = RMSSplineSurfaceTransformer.load_artifact(
        serialize_rms_spline_surface(model, {"fit_scope": "Visible"})
    )
    assert metadata["fit_scope"] == "Visible"
    np.testing.assert_allclose(restored.coefficients, model.coefficients)
    np.testing.assert_allclose(
        restored.surface(parameter[:3], residual[:3]),
        model.surface(parameter[:3], residual[:3]),
    )


def test_rms_transformer_rejects_an_old_extruded_surface_artifact() -> None:
    with pytest.raises(ValueError, match="Not an RMS-conditioned"):
        RMSSplineSurfaceTransformer.from_artifact(
            {"artifact_kind": "temporal_manifolds_extruded_surface"}
        )
