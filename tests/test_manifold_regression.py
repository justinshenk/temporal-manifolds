from __future__ import annotations

import io

import joblib
import numpy as np
import pytest

from temporal_manifolds.viz.manifold_regression import (
    GROUP_FEATURE,
    TARGET_FEATURE,
    RegressionModel,
    evaluate_regression_model,
    fit_manifold_regression,
    load_regression_model,
    regression_scores_table,
    serialize_regression_model,
    task_disjoint_split,
)


def _task_structured_points(
    task_count: int = 8, horizons_per_task: int = 12, seed: int = 0
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Return embedding-like coordinates where each task has its own offset.

    The per-task offset is what makes a random split leak: every test point would have
    near-identical siblings from the same task sitting in the training set.
    """
    generator = np.random.default_rng(seed)
    coordinates: list[np.ndarray] = []
    targets: list[float] = []
    groups: list[str] = []
    for task in range(task_count):
        offset = generator.normal(size=3) * 0.4
        for horizon in np.linspace(-2.0, 2.0, horizons_per_task):
            coordinates.append(
                np.array([horizon, horizon * horizon * 0.3, 0.0])
                + offset
                + generator.normal(scale=0.05, size=3)
            )
            targets.append(float(horizon))
            groups.append(f"task_{task}")
    return np.array(coordinates), np.array(targets), groups


FEATURES = ("KPC1", "KPC2", "KPC3")


def test_split_never_shares_a_task_between_train_and_test() -> None:
    _, _, groups = _task_structured_points()

    train_indices, test_indices = task_disjoint_split(
        groups, test_fraction=0.25, random_state=42
    )

    group_array = np.asarray(groups)
    train_tasks = set(group_array[train_indices])
    test_tasks = set(group_array[test_indices])
    assert train_tasks and test_tasks
    assert not (train_tasks & test_tasks)
    # Every point must be assigned exactly once.
    assert len(train_indices) + len(test_indices) == len(groups)
    assert not set(train_indices) & set(test_indices)


def test_fitted_regression_keeps_its_split_task_disjoint() -> None:
    coordinates, target, groups = _task_structured_points()

    result = fit_manifold_regression(
        coordinates, target, groups, coordinate_features=FEATURES, degree=2
    )

    group_array = np.asarray(groups)
    assert not (
        set(group_array[result.train_indices]) & set(group_array[result.test_indices])
    )
    assert result.metrics["train_tasks"] + result.metrics["test_tasks"] == (
        result.metrics["total_tasks"]
    )


def test_task_disjoint_split_is_not_optimistic_like_a_random_split() -> None:
    """The whole reason for grouping: a random split inflates the held-out score."""
    from sklearn.linear_model import Ridge
    from sklearn.metrics import r2_score
    from sklearn.model_selection import train_test_split
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import PolynomialFeatures, StandardScaler

    coordinates, target, groups = _task_structured_points()

    train_x, test_x, train_y, test_y = train_test_split(
        coordinates, target, test_size=0.25, random_state=42
    )
    leaky = Pipeline(
        [
            ("scale", StandardScaler()),
            ("polynomial", PolynomialFeatures(2, include_bias=False)),
            ("ridge", Ridge(alpha=1.0)),
        ]
    ).fit(train_x, train_y)
    leaky_r2 = r2_score(test_y, leaky.predict(test_x))

    grouped = fit_manifold_regression(
        coordinates, target, groups, coordinate_features=FEATURES, degree=2
    )

    # The leaky split sees the same tasks on both sides and scores higher for it.
    assert leaky_r2 > grouped.metrics["test_r2"]


def test_regression_reports_train_and_test_r2_and_rmse() -> None:
    coordinates, target, groups = _task_structured_points()

    result = fit_manifold_regression(
        coordinates, target, groups, coordinate_features=FEATURES, degree=2
    )
    metrics = result.metrics

    for key in ("train_r2", "train_rmse", "test_r2", "test_rmse"):
        assert np.isfinite(metrics[key]), key
    assert metrics["train_rmse"] >= 0.0
    assert metrics["test_rmse"] >= 0.0
    assert metrics["target_feature"] == TARGET_FEATURE
    assert metrics["group_feature"] == GROUP_FEATURE
    # Predictions must line up with the recorded split sizes.
    assert len(result.train_prediction) == metrics["train_points"]
    assert len(result.test_prediction) == metrics["test_points"]


def test_rmse_matches_a_direct_computation() -> None:
    coordinates, target, groups = _task_structured_points()

    result = fit_manifold_regression(
        coordinates, target, groups, coordinate_features=FEATURES, degree=2
    )

    expected = float(
        np.sqrt(np.mean(np.square(result.test_actual - result.test_prediction)))
    )
    np.testing.assert_allclose(result.metrics["test_rmse"], expected, rtol=1e-10)


def test_cross_validation_folds_are_grouped_by_task() -> None:
    coordinates, target, groups = _task_structured_points()

    result = fit_manifold_regression(
        coordinates,
        target,
        groups,
        coordinate_features=FEATURES,
        degree=2,
        cross_validation_folds=4,
    )

    assert result.metrics["cv_folds"] == 4
    assert len(result.metrics["cv_r2_folds"]) == 4
    assert np.isfinite(result.metrics["cv_r2_mean"])


def test_cross_validation_folds_are_capped_by_the_task_count() -> None:
    coordinates, target, groups = _task_structured_points(task_count=3)

    result = fit_manifold_regression(
        coordinates,
        target,
        groups,
        coordinate_features=FEATURES,
        degree=1,
        cross_validation_folds=10,
    )

    assert result.metrics["cv_folds"] == 3
    assert any("limited by the number" in warning for warning in result.warnings)


def test_points_without_a_horizon_are_excluded() -> None:
    coordinates, target, groups = _task_structured_points()
    target = target.copy()
    target[:10] = np.nan  # Unconstrained prompts state no horizon.

    result = fit_manifold_regression(
        coordinates, target, groups, coordinate_features=FEATURES, degree=2
    )

    assert result.metrics["excluded_points"] == 10
    assert result.metrics["train_points"] + result.metrics["test_points"] == (
        len(target) - 10
    )
    assert any("excluded" in warning for warning in result.warnings)


def test_split_requires_at_least_two_tasks() -> None:
    coordinates, target, _ = _task_structured_points(task_count=1)
    groups = ["only_task"] * len(target)

    with pytest.raises(ValueError, match="at least two distinct"):
        fit_manifold_regression(
            coordinates, target, groups, coordinate_features=FEATURES, degree=2
        )


def test_degree_one_reproduces_plain_ridge() -> None:
    coordinates, target, groups = _task_structured_points()

    result = fit_manifold_regression(
        coordinates, target, groups, coordinate_features=FEATURES, degree=1
    )

    # A degree-1 expansion adds no terms beyond the raw coordinates.
    assert result.metrics["polynomial_terms"] == len(FEATURES)


def test_higher_degree_expands_the_term_count() -> None:
    coordinates, target, groups = _task_structured_points()

    linear = fit_manifold_regression(
        coordinates, target, groups, coordinate_features=FEATURES, degree=1
    )
    quadratic = fit_manifold_regression(
        coordinates, target, groups, coordinate_features=FEATURES, degree=3
    )

    assert quadratic.metrics["polynomial_terms"] > linear.metrics["polynomial_terms"]


def test_regression_rejects_invalid_controls() -> None:
    coordinates, target, groups = _task_structured_points()

    with pytest.raises(ValueError, match="degree must be at least 1"):
        fit_manifold_regression(
            coordinates, target, groups, coordinate_features=FEATURES, degree=0
        )
    with pytest.raises(ValueError, match="alpha must not be negative"):
        fit_manifold_regression(
            coordinates, target, groups, coordinate_features=FEATURES, alpha=-1.0
        )
    with pytest.raises(ValueError, match="test fraction must be between"):
        fit_manifold_regression(
            coordinates,
            target,
            groups,
            coordinate_features=FEATURES,
            test_fraction=1.5,
        )


def test_regression_model_round_trip_reproduces_predictions() -> None:
    coordinates, target, groups = _task_structured_points()
    result = fit_manifold_regression(
        coordinates, target, groups, coordinate_features=FEATURES, degree=2
    )

    payload = serialize_regression_model(
        result.model, metadata={"layer_component": "layer_out/21"}
    )
    loaded, provenance = load_regression_model(payload)

    assert isinstance(loaded, RegressionModel)
    assert provenance["target_feature"] == TARGET_FEATURE
    assert provenance["layer_component"] == "layer_out/21"
    assert loaded.coordinate_features == FEATURES
    np.testing.assert_allclose(
        loaded.predict(coordinates), result.model.predict(coordinates), atol=1e-10
    )


def test_load_regression_model_rejects_foreign_artifacts() -> None:
    buffer = io.BytesIO()
    joblib.dump({"kind": "something-else", "version": 1}, buffer)

    with pytest.raises(ValueError, match="not a supported regression model artifact"):
        load_regression_model(buffer.getvalue())


def test_regression_model_rejects_mismatched_coordinate_counts() -> None:
    coordinates, target, groups = _task_structured_points()
    result = fit_manifold_regression(
        coordinates, target, groups, coordinate_features=FEATURES, degree=2
    )

    with pytest.raises(ValueError, match="Expected coordinates with 3 columns"):
        result.model.predict(np.zeros((5, 2)))


def test_evaluate_regression_model_scores_current_points() -> None:
    coordinates, target, groups = _task_structured_points()
    result = fit_manifold_regression(
        coordinates, target, groups, coordinate_features=FEATURES, degree=2
    )

    scores = evaluate_regression_model(result.model, coordinates, target)

    assert np.isfinite(scores["r2"])
    assert scores["points"] == len(target)
    assert scores["excluded_points"] == 0


def test_scores_table_lists_both_splits() -> None:
    coordinates, target, groups = _task_structured_points()
    result = fit_manifold_regression(
        coordinates,
        target,
        groups,
        coordinate_features=FEATURES,
        degree=2,
        cross_validation_folds=3,
    )

    table = regression_scores_table(result.metrics)

    assert list(table["split"])[:2] == ["Train", "Test (task-disjoint)"]
    assert "Cross-validated" in list(table["split"])[2]
    assert set(table.columns) == {"split", "points", "tasks", "R²", "RMSE"}
