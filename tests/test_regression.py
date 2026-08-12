import numpy as np
import pandas as pd

from temporal_manifolds.regression import fit_grouped_regression, grouped_train_test_indices


def test_grouped_split_has_no_group_overlap():
    groups = pd.Series(np.repeat(["a", "b", "c", "d", "e"], 3))
    train, test = grouped_train_test_indices(groups, test_size=0.4, random_state=7)
    assert set(groups.iloc[train]).isdisjoint(set(groups.iloc[test]))


def test_polynomial_ridge_fits_grouped_data_with_pca():
    rng = np.random.default_rng(4)
    features = pd.DataFrame(rng.normal(size=(40, 4)), columns=list("abcd"))
    target = 2 * features["a"] - features["b"] + rng.normal(scale=0.05, size=40)
    groups = pd.Series(np.repeat(np.arange(10), 4))
    result = fit_grouped_regression(
        features,
        target,
        groups,
        test_size=0.2,
        random_state=2,
        degree=2,
        alpha=1.0,
        use_pca=True,
        pca_components=0.95,
    )
    assert set(result.metrics["Metric"]) >= {"R²", "RMSE", "MAE", "MSE", "MAPE"}
    assert len(result.test_predictions) == len(result.test_indices)
