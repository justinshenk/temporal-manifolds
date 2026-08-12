from __future__ import annotations

import io
from dataclasses import replace

import joblib
import numpy as np
import pytest

from temporal_manifolds.viz.curve_fitting import (
    CURVE_ALGORITHMS,
    evaluate_curve_model,
    fit_curve,
    load_curve_model,
    serialize_curve_model,
)


def _trajectory(size: int = 80) -> tuple[np.ndarray, np.ndarray]:
    parameter = np.linspace(-2.0, 2.0, size)
    coordinates = np.column_stack(
        [parameter, parameter**2 - 1.0, np.sin(parameter * 1.5)]
    )
    return parameter, coordinates


@pytest.mark.parametrize("algorithm", list(CURVE_ALGORITHMS))
def test_every_curve_algorithm_fits_a_three_dimensional_line(algorithm: str) -> None:
    parameter, coordinates = _trajectory()
    parameters = (
        {"degree": 3, "alpha": 1e-5}
        if algorithm == "polynomial"
        else {"degree": 3, "smoothing": 0.01}
    )

    result = fit_curve(
        parameter,
        coordinates,
        algorithm=algorithm,
        parameters=parameters,
        sample_count=101,
        validation_fraction=0.2,
        parameter_feature="time",
        coordinate_features=("PC1", "PC2", "PC3"),
    )

    assert result.curve_xyz.shape == (101, 3)
    assert result.point_prediction.shape == coordinates.shape
    assert result.model.parameter_feature == "time"
    assert result.model.coordinate_features == ("PC1", "PC2", "PC3")
    assert result.metrics["validation_points"] == 16
    assert result.metrics["validation_rmse_3d"] is not None


def test_duplicate_parameters_are_reduced_before_curve_fit() -> None:
    parameter, coordinates = _trajectory(12)
    duplicated_parameter = np.repeat(parameter, 2)
    duplicated_coordinates = np.repeat(coordinates, 2, axis=0)
    duplicated_coordinates[1::2] += 0.1

    result = fit_curve(
        duplicated_parameter,
        duplicated_coordinates,
        algorithm="polynomial",
        parameters={"degree": 3, "alpha": 0.0},
        validation_fraction=0.0,
    )

    assert result.metrics["duplicates_merged"] == 12
    assert result.metrics["unique_parameter_values"] == 12


def test_curve_extends_to_display_bounds_and_padding_extends_further() -> None:
    parameter = np.linspace(-1.0, 1.0, 81)
    base = parameter**3 - parameter
    coordinates = np.column_stack([base, 2.0 * base, -0.5 * base])
    options = {
        "algorithm": "polynomial",
        "parameters": {"degree": 3, "alpha": 0.0},
        "sample_count": 121,
        "validation_fraction": 0.0,
    }

    default_result = fit_curve(parameter, coordinates, **options)
    padded_result = fit_curve(parameter, coordinates, padding_fraction=0.5, **options)

    assert default_result.curve_parameter[0] < parameter.min()
    assert default_result.curve_parameter[-1] > parameter.max()
    assert padded_result.curve_parameter[0] < default_result.curve_parameter[0]
    assert padded_result.curve_parameter[-1] > default_result.curve_parameter[-1]
    assert default_result.metrics["display_padding_fraction"] == 0.0
    assert padded_result.metrics["display_padding_fraction"] == 0.5


def test_saved_curve_round_trip_and_current_data_evaluation() -> None:
    parameter, coordinates = _trajectory()
    fitted = fit_curve(
        parameter,
        coordinates,
        algorithm="polynomial",
        parameters={"degree": 5, "alpha": 1e-6},
        parameter_feature="log_time",
        coordinate_features=("PLS1", "PLS2", "PLS3"),
    )
    artifact = serialize_curve_model(
        fitted.model,
        metadata={"direction_method": "PLS", "pca_sha256": "abc"},
    )

    restored, provenance = load_curve_model(artifact)
    evaluated = evaluate_curve_model(
        restored,
        parameter,
        coordinates,
        sample_count=151,
        padding_fraction=0.1,
    )

    assert np.allclose(restored.predict(parameter), fitted.model.predict(parameter))
    assert evaluated.curve_xyz.shape == (151, 3)
    assert provenance["direction_method"] == "PLS"
    assert provenance["artifact_version"] == 1
    assert evaluated.metrics["current_rmse_3d"] is not None


def test_curve_artifact_validation_rejects_unversioned_and_invalid_models() -> None:
    parameter, coordinates = _trajectory()
    fitted = fit_curve(parameter, coordinates, validation_fraction=0.0)
    raw = io.BytesIO()
    joblib.dump(fitted.model, raw)
    with pytest.raises(ValueError, match="not a supported curve model artifact"):
        load_curve_model(raw.getvalue())

    invalid = replace(fitted.model, parameter_scale=0.0)
    with pytest.raises(ValueError, match="scale must be positive"):
        serialize_curve_model(invalid)

    with pytest.raises(ValueError, match="At least three distinct"):
        fit_curve(
            np.array([1.0, 1.0, 1.0]),
            np.zeros((3, 3)),
            validation_fraction=0.0,
        )
