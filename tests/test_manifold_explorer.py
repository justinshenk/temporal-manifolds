from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch
from streamlit.testing.v1 import AppTest

from temporal_manifolds.activations.extraction_policy import (
    PROMPT_TOKEN_POSITION,
    TARGET_LAYER_COMPONENT,
)
from temporal_manifolds.viz.manifold_regression import GROUP_FEATURE
from temporal_manifolds.viz.manifold_explorer import (
    ALGORITHM_LABEL,
    TARGET_FEATURE,
    ManifoldModel,
    default_manifold_parameters,
    embedding_fields,
    embedding_fingerprint,
    explained_variance_table,
    fit_manifold,
    load_manifold_model,
    serialize_manifold_model,
    target_alignment_metrics,
)

APP_PATH = Path(__file__).resolve().parents[1] / "apps" / "manifold_explorer.py"


def _swiss_roll(
    point_count: int = 200, feature_count: int = 8, seed: int = 0
) -> tuple[np.ndarray, np.ndarray]:
    """Return a Swiss roll rotated into high dimensions, with the target running along it.

    The target is the position *along* the rolled sheet, so points that are far apart on the
    manifold can be close in ambient space. A linear projection cannot undo that — PCA scores
    roughly R²=0.32 here — while a method that follows the manifold recovers the ordering.
    This is the case the app exists for, so the fixture must genuinely exhibit it.
    """
    generator = np.random.default_rng(seed)
    target = np.linspace(0.6 * np.pi, 3.0 * np.pi, point_count)
    height = generator.uniform(-1.0, 1.0, point_count)
    base = np.column_stack([target * np.cos(target), height * 4.0, target * np.sin(target)])
    # An orthonormal lift keeps the manifold's geometry intact while hiding it from the axes.
    rotation, _ = np.linalg.qr(generator.normal(size=(feature_count, 3)))
    values = base @ rotation.T + generator.normal(
        scale=0.02, size=(point_count, feature_count)
    )
    return values, target


def _manifold_batch(path: Path, point_count: int = 24) -> None:
    values = [2.0**index for index in range(point_count)]
    metadata = [
        {
            "task": f"task_{index % 3}",
            "base_value": value,
            "base_unit": "months",
            "template_metadata": {
                "prompt_framing": "task_available_time",
            },
        }
        for index, value in enumerate(values)
    ]
    generator = torch.Generator().manual_seed(23)
    tensor = torch.randn(point_count, 1, 8, generator=generator)
    torch.save(
        {
            "sample_indices": list(range(point_count)),
            "prompts": [f"prompt-{index}" for index in range(point_count)],
            "prompt_metadata": metadata,
            "layer_component": TARGET_LAYER_COMPONENT,
            "positions": [PROMPT_TOKEN_POSITION],
            "activations": {TARGET_LAYER_COMPONENT: tensor},
        },
        path,
    )


def test_fit_manifold_recovers_a_curved_target_ordering() -> None:
    values, target = _swiss_roll()

    result = fit_manifold(values, target, n_components=3)

    assert result.embedding.shape == (len(values), 3)
    assert np.isfinite(result.embedding).all()
    # The roll is monotone along its length, so a faithful embedding must order it.
    assert abs(result.metrics["target_spearman"]) > 0.55
    assert result.metrics["trustworthiness"] > 0.9
    assert result.metrics["target_feature"] == TARGET_FEATURE


def test_kernel_pca_beats_linear_pca_on_a_rolled_manifold() -> None:
    """The premise of this app: a linear projection cannot unroll a curved manifold."""
    from sklearn.decomposition import PCA

    values, target = _swiss_roll()

    linear_score = target_alignment_metrics(
        PCA(n_components=3).fit_transform(values), target
    )["target_linear_r2"]
    kernel_score = fit_manifold(values, target, n_components=3).metrics["target_linear_r2"]

    assert linear_score < 0.5
    assert kernel_score > linear_score


def test_fit_manifold_scores_only_points_with_a_finite_horizon() -> None:
    values, target = _swiss_roll(point_count=40)
    target = target.copy()
    target[:10] = np.nan  # Unconstrained prompts state no horizon.

    result = fit_manifold(values, target, n_components=2)

    assert result.embedding.shape == (40, 2)
    assert result.metrics["target_points"] == 30
    assert np.isfinite(result.metrics["target_linear_r2"])


def test_fit_manifold_rejects_out_of_range_components() -> None:
    values, target = _swiss_roll(point_count=10, feature_count=4)

    with pytest.raises(ValueError, match="Components must be between 1 and 4"):
        fit_manifold(values, target, n_components=9)


def test_fit_manifold_rejects_unknown_kernels() -> None:
    values, target = _swiss_roll(point_count=10)

    with pytest.raises(ValueError, match="Unknown kernel"):
        fit_manifold(values, target, n_components=2, parameters={"kernel": "tanh"})


def test_target_alignment_metrics_reward_an_ordered_embedding() -> None:
    target = np.linspace(0.0, 1.0, 50)
    ordered = np.column_stack([target, np.zeros_like(target)])
    shuffled = np.column_stack([np.random.default_rng(1).permutation(target), target * 0])

    ordered_metrics = target_alignment_metrics(ordered, target)
    shuffled_metrics = target_alignment_metrics(shuffled, target)

    assert ordered_metrics["target_linear_r2"] > 0.99
    assert ordered_metrics["target_linear_r2"] > shuffled_metrics["target_linear_r2"]
    assert ordered_metrics["target_neighborhood_error"] < (
        shuffled_metrics["target_neighborhood_error"]
    )


def test_manifold_model_round_trip_reproduces_the_embedding() -> None:
    values, target = _swiss_roll()
    result = fit_manifold(values, target, n_components=3)

    payload = serialize_manifold_model(result.model, metadata={"layer_component": "layer_out/21"})
    loaded, provenance = load_manifold_model(payload)

    assert isinstance(loaded, ManifoldModel)
    assert provenance["algorithm"] == "kernel_pca"
    assert provenance["direction_target"] == TARGET_FEATURE
    assert provenance["layer_component"] == "layer_out/21"
    assert embedding_fingerprint(loaded) == embedding_fingerprint(result.model)
    np.testing.assert_allclose(loaded.transform(values), result.embedding, atol=1e-8)


def test_load_manifold_model_rejects_foreign_artifacts() -> None:
    import io

    import joblib

    buffer = io.BytesIO()
    joblib.dump({"kind": "something-else", "version": 1}, buffer)

    with pytest.raises(ValueError, match="not a supported manifold model artifact"):
        load_manifold_model(buffer.getvalue())


def test_manifold_model_rejects_mismatched_activation_width() -> None:
    values, target = _swiss_roll()
    result = fit_manifold(values, target, n_components=2)

    with pytest.raises(ValueError, match="Expected activations with 8 features"):
        result.model.transform(np.zeros((4, 3)))


def test_explained_variance_ratios_are_ordered_and_cumulative() -> None:
    values, target = _swiss_roll()

    result = fit_manifold(values, target, n_components=4)
    table = explained_variance_table(result)

    assert list(table["component"]) == ["KPC1", "KPC2", "KPC3", "KPC4"]
    ratios = table["explained_variance"].to_numpy()
    # Kernel PCA returns eigenvalues in descending order, so the ratios must not increase.
    assert np.all(np.diff(ratios) <= 1e-12)
    assert np.all(ratios >= 0.0)
    np.testing.assert_allclose(table["cumulative_variance"].to_numpy(), np.cumsum(ratios))
    # Without the full spectrum the ratios are shares among retained components.
    assert result.model.retained_variance_fraction is None
    np.testing.assert_allclose(table["cumulative_variance"].to_numpy()[-1], 1.0, atol=1e-9)


def test_linear_kernel_variance_ratios_match_pca() -> None:
    """A linear kernel is ordinary PCA, so the full spectrum must reproduce its ratios."""
    from sklearn.decomposition import PCA

    values, target = _swiss_roll(point_count=60, feature_count=8)

    result = fit_manifold(
        values,
        target,
        n_components=5,
        standardize=False,
        parameters={"kernel": "linear"},
        full_variance_spectrum=True,
    )
    expected = PCA(n_components=5).fit(values).explained_variance_ratio_

    np.testing.assert_allclose(result.model.explained_variance_ratio, expected, atol=1e-8)


def test_full_spectrum_reports_the_retained_share_of_total_variance() -> None:
    values, target = _swiss_roll()

    truncated = fit_manifold(values, target, n_components=2)
    full = fit_manifold(values, target, n_components=2, full_variance_spectrum=True)

    # The truncated fit can only express shares among what it kept.
    assert truncated.model.retained_variance_fraction is None
    # The full spectrum knows the real denominator, so two components cannot reach everything.
    assert full.model.retained_variance_fraction is not None
    assert 0.0 < full.model.retained_variance_fraction < 1.0
    assert full.metrics["full_variance_spectrum"] is True


def test_embedding_fields_are_named_for_kernel_pca() -> None:
    assert embedding_fields(2) == ["KPC1", "KPC2"]
    assert embedding_fields(3) == ["KPC1", "KPC2", "KPC3"]


def test_default_parameters_describe_the_kernel() -> None:
    assert set(default_manifold_parameters()) == {
        "kernel",
        "gamma",
        "degree",
        "coef0",
        "alpha",
    }
    assert default_manifold_parameters()["kernel"] == "rbf"


def test_app_fits_a_kernel_pca_embedding(tmp_path: Path) -> None:
    path = tmp_path / "activations_batch_000.pt"
    _manifold_batch(path)
    app = AppTest.from_file(APP_PATH)
    app.session_state["sources"] = [str(path)]
    app.session_state["source_label"] = str(tmp_path)
    app.session_state["source_is_local"] = True
    app.session_state["source_revision"] = 1

    app.run(timeout=60)

    assert not app.exception
    method_control = next(
        control for control in app.segmented_control if control.label == "Embedding method"
    )
    assert method_control.options == ["Kernel PCA", "PCA", "PLS"]
    assert method_control.value == "Kernel PCA"
    assert "Kernel" in [selectbox.label for selectbox in app.selectbox]
    assert any(ALGORITHM_LABEL in markdown.value for markdown in app.markdown)

    fit_button = next(
        button for button in app.button if button.label == "Fit / update embedding"
    )
    fit_button.click().run(timeout=60)

    assert not app.exception, [error.value for error in app.error]
    assert app.session_state["manifold_result"] is not None
    projection = app.session_state["projection"]
    assert {"KPC1", "KPC2", "KPC3"} <= set(projection.columns)
    assert TARGET_FEATURE in projection.columns


@pytest.mark.parametrize(("method", "prefix"), [("PCA", "PC"), ("PLS", "PLS")])
def test_app_fits_linear_embedding_methods(
    tmp_path: Path, method: str, prefix: str
) -> None:
    path = tmp_path / "activations_batch_000.pt"
    _manifold_batch(path)
    app = AppTest.from_file(APP_PATH)
    app.session_state["sources"] = [str(path)]
    app.session_state["source_label"] = str(tmp_path)
    app.session_state["source_is_local"] = True
    app.session_state["source_revision"] = 1
    app.run(timeout=60)

    next(
        control for control in app.segmented_control if control.label == "Embedding method"
    ).set_value(method).run(timeout=60)
    next(
        button for button in app.button if button.label == "Fit / update embedding"
    ).click().run(timeout=60)

    assert not app.exception, [error.value for error in app.error]
    projection = app.session_state["projection"]
    assert {f"{prefix}1", f"{prefix}2", f"{prefix}3"} <= set(projection.columns)
    assert app.session_state["details"]["direction_method"] == method
    assert app.session_state["manifold_result"].model.parameters["method"] == method
    assert TARGET_FEATURE in projection.columns


def test_app_reports_cumulative_explained_variance(tmp_path: Path) -> None:
    app = _fitted_app(tmp_path)

    details = app.session_state["details"]
    cumulative = np.asarray(details["cumulative_variance_ratio"], dtype=float)
    ratios = np.asarray(details["explained_variance_ratio"], dtype=float)

    assert len(cumulative) == len(ratios)
    np.testing.assert_allclose(cumulative, np.cumsum(ratios))
    assert np.all(np.diff(cumulative) >= -1e-12)  # Cumulative must be non-decreasing.
    assert any(metric.label == "Variance captured" for metric in app.metric)


def test_app_switching_kernels_invalidates_the_previous_embedding(tmp_path: Path) -> None:
    """A kernel change alters the embedding, so stale coordinates must be withdrawn."""
    app = _fitted_app(tmp_path)
    assert "projection" in app.session_state

    next(selectbox for selectbox in app.selectbox if selectbox.label == "Kernel").set_value(
        "cosine"
    ).run(timeout=60)

    assert not app.exception
    assert "projection" not in app.session_state

    next(
        button for button in app.button if button.label == "Fit / update embedding"
    ).click().run(timeout=60)

    assert not app.exception, [error.value for error in app.error]
    assert app.session_state["manifold_result"].model.parameters["kernel"] == "cosine"


def test_app_exposes_target_range_and_visual_filters(tmp_path: Path) -> None:
    path = tmp_path / "activations_batch_000.pt"
    _manifold_batch(path)
    app = AppTest.from_file(APP_PATH)
    app.session_state["sources"] = [str(path)]
    app.session_state["source_label"] = str(tmp_path)
    app.session_state["source_is_local"] = True
    app.session_state["source_revision"] = 1
    app.run(timeout=60)
    next(
        button for button in app.button if button.label == "Fit / update embedding"
    ).click().run(timeout=60)

    assert not app.exception
    toggle_labels = [toggle.label for toggle in app.toggle]
    assert "Restrict visible time horizons" in toggle_labels
    assert "Show points" in toggle_labels

    restrict = next(
        toggle for toggle in app.toggle if toggle.label == "Restrict visible time horizons"
    )
    restrict.set_value(True).run(timeout=60)

    assert not app.exception
    assert f"Visible {TARGET_FEATURE}" in [slider.label for slider in app.slider]

    horizon_slider = next(
        slider for slider in app.slider if slider.label == f"Visible {TARGET_FEATURE}"
    )
    low, high = horizon_slider.value
    horizon_slider.set_value((low, (low + high) / 2)).run(timeout=60)

    assert not app.exception
    visible_captions = [caption.value for caption in app.caption if "Visible" in caption.value]
    assert visible_captions


def _fitted_app(tmp_path: Path) -> AppTest:
    """Return an app with one embedding already fitted."""
    path = tmp_path / "activations_batch_000.pt"
    _manifold_batch(path)
    app = AppTest.from_file(APP_PATH)
    app.session_state["sources"] = [str(path)]
    app.session_state["source_label"] = str(tmp_path)
    app.session_state["source_is_local"] = True
    app.session_state["source_revision"] = 1
    app.run(timeout=60)
    next(
        button for button in app.button if button.label == "Fit / update embedding"
    ).click().run(timeout=60)
    assert not app.exception, [error.value for error in app.error]
    return app


def test_recovering_from_an_empty_filter_reprepares_the_analysis(tmp_path: Path) -> None:
    """A failed preparation must not leave a key that makes stale data look current."""
    app = _fitted_app(tmp_path)
    keep_filters = [
        multiselect
        for multiselect in app.multiselect
        if multiselect.label.startswith("Keep · ")
    ]
    assert keep_filters, "the fixture should expose at least one metadata filter"
    original = list(keep_filters[0].value)

    # Editing the draft does not disturb the active analysis.
    keep_filters[0].set_value([]).run(timeout=60)
    assert not app.exception
    assert "projection" in app.session_state

    # Applying no selected values matches no rows, so preparation fails.
    next(button for button in app.button if button.label == "Apply filters").click().run(
        timeout=60
    )
    assert not app.exception
    assert any("could not be prepared" in error.value for error in app.error)

    # Restoring the previous selection must rebuild the analysis, not raise a KeyError.
    restored = next(
        multiselect
        for multiselect in app.multiselect
        if multiselect.label.startswith("Keep · ")
    )
    restored.set_value(original).run(timeout=60)
    next(button for button in app.button if button.label == "Apply filters").click().run(
        timeout=60
    )

    assert not app.exception, [error.value for error in app.error]
    assert "prepared_metadata" in app.session_state


def test_aggregation_groups_by_exactly_the_selected_fields(tmp_path: Path) -> None:
    """Preparation must group by what the user picked, with nothing added on its behalf."""
    app = _fitted_app(tmp_path)

    aggregate_by = next(
        multiselect for multiselect in app.multiselect if multiselect.label == "Aggregate by"
    )
    assert aggregate_by.value == ["time_horizon_months"]
    # The second element of the preparation key is the grouping preparation actually used.
    assert app.session_state["prepared_key"][1] == ("time_horizon_months",)
    assert app.session_state["details"]["aggregation_applied"] is True
    # The regression needs one task per point, but it must say so rather than regroup.
    assert any(GROUP_FEATURE in warning.value for warning in app.warning)

    aggregate_by.set_value(["time_horizon_months", GROUP_FEATURE]).run(timeout=60)
    assert app.session_state["prepared_key"][1] == ("time_horizon_months",)
    next(button for button in app.button if button.label == "Apply filters").click().run(
        timeout=60
    )

    assert not app.exception, [error.value for error in app.error]
    assert app.session_state["prepared_key"][1] == ("time_horizon_months", GROUP_FEATURE)


def test_changing_filters_withdraws_the_previous_metrics(tmp_path: Path) -> None:
    """Metrics and plot must not survive a preparation change that invalidates them."""
    app = _fitted_app(tmp_path)
    assert "projection" in app.session_state

    next(
        toggle for toggle in app.toggle if toggle.label == "Limit source samples"
    ).set_value(True).run(timeout=60)

    assert not app.exception
    assert "projection" in app.session_state
    assert "details" in app.session_state
    assert "manifold_result" in app.session_state

    next(button for button in app.button if button.label == "Apply filters").click().run(
        timeout=60
    )

    assert not app.exception
    assert "projection" not in app.session_state
    assert "details" not in app.session_state
    assert "manifold_result" not in app.session_state


def test_app_reports_the_fixed_optimization_target(tmp_path: Path) -> None:
    path = tmp_path / "activations_batch_000.pt"
    _manifold_batch(path)
    app = AppTest.from_file(APP_PATH)
    app.session_state["sources"] = [str(path)]
    app.session_state["source_label"] = str(tmp_path)
    app.session_state["source_is_local"] = True
    app.session_state["source_revision"] = 1

    app.run(timeout=60)

    assert not app.exception
    assert any(TARGET_FEATURE in info.value for info in app.info)
    # The target is fixed, so no widget may offer an alternative.
    assert not any(
        "target" in (selectbox.label or "").casefold()
        and selectbox.label != "Method"
        for selectbox in app.selectbox
    )


def test_app_fits_a_task_disjoint_regression(tmp_path: Path) -> None:
    app = _fitted_app(tmp_path)

    fit_regression = next(
        button for button in app.button if button.label == "Fit / update regression"
    )
    fit_regression.click().run(timeout=60)

    assert not app.exception, [error.value for error in app.error]
    result = app.session_state["regression_result"]
    metrics = result.metrics
    # Train and test tasks must not overlap, which is the point of the grouped split.
    assert metrics["train_tasks"] + metrics["test_tasks"] == metrics["total_tasks"]
    for key in ("train_r2", "train_rmse", "test_r2", "test_rmse"):
        assert np.isfinite(metrics[key]), key
    labels = [metric.label for metric in app.metric]
    assert {"Train R²", "Train RMSE", "Test R²", "Test RMSE"} <= set(labels)


def test_app_offers_uploads_for_both_saved_models(tmp_path: Path) -> None:
    app = _fitted_app(tmp_path)

    # The regression section can load a saved regression artifact.
    next(
        control
        for control in app.segmented_control
        if control.label == "Regression model"
    ).set_value("Use saved").run(timeout=60)
    assert not app.exception
    assert "Saved regression model" in [
        uploader.label for uploader in app.file_uploader
    ]

    # The embedding section can load a saved Kernel PCA artifact.
    next(
        control
        for control in app.segmented_control
        if control.label == "Embedding model"
    ).set_value("Use saved").run(timeout=60)
    assert not app.exception
    assert "Saved Kernel PCA model" in [
        uploader.label for uploader in app.file_uploader
    ]


def test_app_round_trips_a_saved_embedding(tmp_path: Path) -> None:
    from temporal_manifolds.viz.manifold_explorer import serialize_manifold_model

    app = _fitted_app(tmp_path)
    fitted_coordinates = app.session_state["projection"][["KPC1", "KPC2"]].to_numpy()
    payload = serialize_manifold_model(app.session_state["manifold_result"].model)

    next(
        control
        for control in app.segmented_control
        if control.label == "Embedding model"
    ).set_value("Use saved").run(timeout=60)
    uploader = next(
        uploader
        for uploader in app.file_uploader
        if uploader.label == "Saved Kernel PCA model"
    )
    uploader.set_value(("manifold.joblib", payload, "application/octet-stream")).run(
        timeout=60
    )
    next(
        button for button in app.button if button.label == "Load uploaded embedding"
    ).click().run(timeout=60)

    assert not app.exception, [error.value for error in app.error]
    reloaded = app.session_state["projection"][["KPC1", "KPC2"]].to_numpy()
    # Re-projecting the same points through the saved model must reproduce the coordinates.
    np.testing.assert_allclose(reloaded, fitted_coordinates, atol=1e-8)


def test_app_exposes_both_component_controls(tmp_path: Path) -> None:
    """The embedding width and the regression's feature count are separate choices."""
    app = _fitted_app(tmp_path)

    labels = [number_input.label for number_input in app.number_input]
    assert "Components" in labels  # How many embedding coordinates to compute.
    assert "Feature components" in labels  # How many of them the regression consumes.


def test_app_regression_honours_the_feature_count(tmp_path: Path) -> None:
    app = _fitted_app(tmp_path)

    next(
        button for button in app.button if button.label == "Fit / update regression"
    ).click().run(timeout=60)
    assert not app.exception, [error.value for error in app.error]
    assert app.session_state["regression_result"].model.coordinate_features == (
        "KPC1",
        "KPC2",
        "KPC3",
    )

    next(
        number_input
        for number_input in app.number_input
        if number_input.label == "Feature components"
    ).set_value(2).run(timeout=60)
    next(
        button for button in app.button if button.label == "Fit / update regression"
    ).click().run(timeout=60)

    assert not app.exception, [error.value for error in app.error]
    # Only the leading coordinates feed the model when the count is reduced.
    assert app.session_state["regression_result"].model.coordinate_features == (
        "KPC1",
        "KPC2",
    )
