"""Reusable grouped regression utilities for the Streamlit regression explorer."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge
from sklearn.metrics import (
    explained_variance_score,
    max_error,
    mean_absolute_error,
    mean_absolute_percentage_error,
    mean_squared_error,
    median_absolute_error,
    r2_score,
)
from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler


@dataclass(frozen=True)
class RegressionResult:
    """A fitted model and its split predictions and diagnostics."""

    model: Pipeline
    train_indices: np.ndarray
    test_indices: np.ndarray
    train_predictions: np.ndarray
    test_predictions: np.ndarray
    metrics: pd.DataFrame


def grouped_train_test_indices(
    groups: pd.Series | np.ndarray,
    *,
    test_size: float,
    random_state: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Split rows while keeping every occurrence of a group on one side."""

    group_values = pd.Series(groups).astype("string").fillna("<missing>").to_numpy()
    if pd.unique(group_values).size < 2:
        raise ValueError("The grouping column must contain at least two distinct tasks.")
    splitter = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
    train_indices, test_indices = next(splitter.split(group_values, groups=group_values))
    return train_indices, test_indices


def build_regression_pipeline(
    *,
    degree: int,
    alpha: float,
    use_pca: bool,
    pca_components: int | float | None,
    fit_intercept: bool,
    interaction_only: bool,
) -> Pipeline:
    """Build a scaled polynomial ridge model, optionally preceded by PCA."""

    steps: list[tuple[str, object]] = [("input_scaler", StandardScaler())]
    if use_pca:
        steps.append(("pca", PCA(n_components=pca_components, svd_solver="full")))
    steps.extend(
        [
            (
                "polynomial",
                PolynomialFeatures(
                    degree=degree,
                    include_bias=False,
                    interaction_only=interaction_only,
                ),
            ),
            ("feature_scaler", StandardScaler()),
            ("ridge", Ridge(alpha=alpha, fit_intercept=fit_intercept)),
        ]
    )
    return Pipeline(steps)


def regression_metrics(y_true: np.ndarray, predictions: np.ndarray) -> dict[str, float]:
    """Return a broad set of standard regression diagnostics."""

    mse = mean_squared_error(y_true, predictions)
    nonzero = np.abs(y_true) > np.finfo(float).eps
    return {
        "R²": r2_score(y_true, predictions),
        "RMSE": float(np.sqrt(mse)),
        "MAE": mean_absolute_error(y_true, predictions),
        "MSE": mse,
        "Median AE": median_absolute_error(y_true, predictions),
        "Max error": max_error(y_true, predictions),
        "Explained variance": explained_variance_score(y_true, predictions),
        "MAPE": (
            mean_absolute_percentage_error(y_true[nonzero], predictions[nonzero])
            if nonzero.any()
            else np.nan
        ),
    }


def fit_grouped_regression(
    features: pd.DataFrame,
    target: pd.Series,
    groups: pd.Series,
    *,
    test_size: float,
    random_state: int,
    degree: int,
    alpha: float,
    use_pca: bool,
    pca_components: int | float | None,
    fit_intercept: bool = True,
    interaction_only: bool = False,
) -> RegressionResult:
    """Fit and evaluate polynomial ridge regression on a leakage-safe group split."""

    train_indices, test_indices = grouped_train_test_indices(
        groups, test_size=test_size, random_state=random_state
    )
    model = build_regression_pipeline(
        degree=degree,
        alpha=alpha,
        use_pca=use_pca,
        pca_components=pca_components,
        fit_intercept=fit_intercept,
        interaction_only=interaction_only,
    )
    x_train, x_test = features.iloc[train_indices], features.iloc[test_indices]
    y_train, y_test = target.iloc[train_indices], target.iloc[test_indices]
    model.fit(x_train, y_train)
    train_predictions = model.predict(x_train)
    test_predictions = model.predict(x_test)
    metrics = pd.DataFrame(
        {
            "Train": regression_metrics(y_train.to_numpy(), train_predictions),
            "Test": regression_metrics(y_test.to_numpy(), test_predictions),
        }
    ).rename_axis("Metric").reset_index()
    return RegressionResult(
        model=model,
        train_indices=train_indices,
        test_indices=test_indices,
        train_predictions=train_predictions,
        test_predictions=test_predictions,
        metrics=metrics,
    )
