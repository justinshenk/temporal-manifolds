from pathlib import Path

import pytest
import yaml

import temporal_manifolds.workflows.activation_caching_workflow as workflow_module
from temporal_manifolds.workflows.activation_caching_workflow import (
    REPO_ROOT,
    WorkflowConfig,
    ensure_input_artifact_available,
    ensure_nodes_file_available,
    parse_args,
    run_workflow,
)


def config_values() -> dict:
    return {
        "dataset": "conversational",
        "prompt_records_path": "data/prompts.json",
        "dataset_gcs_prefix": "dataset-artifacts",
        "completions_path": "data/completions.jsonl",
        "completions_gcs_prefix": "completion-artifacts",
        "nodes_path": "data/nodes.pkl",
        "nodes_gcs_prefix": "node-artifacts",
        "output_dir": "results/activations",
        "model_name": "test/model",
        "batch_size": 4,
        "max_new_tokens": 32,
        "temperature": 0.0,
        "top_k": 20,
        "dtype": None,
        "device": "cpu",
        "attn_type": "sdpa",
        "position_selection_policy": "default",
        "max_samples": 2,
        "overwrite": False,
        "save_to_gcp": False,
        "upload_worker_count": 16,
        "upload_queue_capacity": 8,
        "modules": ["completions", "activations"],
    }


def test_loads_yaml_options_and_gcp_settings_from_environment(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(workflow_module, "GCP_PROJECT_ID", "test-project")
    monkeypatch.setattr(workflow_module, "GCS_BUCKET_NAME", "test-bucket")
    config_path = tmp_path / "activation_caching.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "workflow": "activation-caching",
                "gcs_prefix": "test-prefix",
                "args": config_values(),
            }
        ),
        encoding="utf-8",
    )

    config = WorkflowConfig.from_yaml(config_path)

    assert config.model_name == "test/model"
    assert config.prompt_records_path == REPO_ROOT / "data/prompts.json"
    assert config.gcp_project_id == "test-project"
    assert config.gcs_bucket_name == "test-bucket"
    assert config.gcs_prefix == "test-prefix"
    assert config.dataset_gcs_prefix == "dataset-artifacts"
    assert config.completions_gcs_prefix == "completion-artifacts"
    assert config.nodes_gcs_prefix == "node-artifacts"
    assert config.modules == ("completions", "activations")
    assert config.upload_worker_count == 16
    assert config.upload_queue_capacity == 8
    assert not config.includes_stage("dataset")
    assert config.includes_stage("completions")


def test_config_requires_every_option() -> None:
    values = config_values()
    del values["batch_size"]

    with pytest.raises(ValueError, match="Missing.*batch_size"):
        WorkflowConfig.from_mapping(values)


def test_modules_must_be_a_non_empty_list_of_known_stages() -> None:
    values = config_values()
    values["modules"] = ["dataset", "unknown"]
    with pytest.raises(ValueError, match="modules entries"):
        WorkflowConfig.from_mapping(values)

    values["modules"] = []
    with pytest.raises(TypeError, match="non-empty YAML list"):
        WorkflowConfig.from_mapping(values)


def test_after_assistant_position_policy_is_supported() -> None:
    values = config_values()
    values["position_selection_policy"] = "after_assistant"

    config = WorkflowConfig.from_mapping(values)

    assert config.position_selection_policy == "after_assistant"


def test_after_assistant_residual_stream_scenario_config() -> None:
    config = WorkflowConfig.from_yaml(
        REPO_ROOT
        / "configs"
        / "activation_caching"
        / "conversational_after_assistant_residual_stream.yaml"
    )

    assert config.position_selection_policy == "after_assistant"
    assert config.completions_path == REPO_ROOT / "completions_256.jsonl"
    assert config.completions_gcs_prefix == "completions"
    assert config.modules == ("activations",)
    assert config.output_dir == (
        REPO_ROOT / "results" / "feature_geometry_after_assistant_residual_stream"
    )


@pytest.mark.parametrize("key", ["upload_worker_count", "upload_queue_capacity"])
def test_upload_concurrency_values_must_be_positive(key: str) -> None:
    values = config_values()
    values[key] = 0

    with pytest.raises(ValueError, match=key):
        WorkflowConfig.from_mapping(values)


def test_cli_only_accepts_scenario_config() -> None:
    args = parse_args(["--scenario-config", "config.yaml"])
    assert args.scenario_config == Path("config.yaml")

    with pytest.raises(SystemExit):
        parse_args(["--batch-size", "4"])


def test_dataset_and_completion_artifacts_are_uploaded(monkeypatch: pytest.MonkeyPatch) -> None:
    values = config_values()
    values["modules"] = ["dataset", "completions", "activations"]
    values["save_to_gcp"] = True
    config = WorkflowConfig.from_mapping(values, gcs_prefix="test-prefix")
    uploaded: list[Path] = []
    activation_kwargs: dict = {}

    monkeypatch.setattr(
        workflow_module,
        "run_dataset_stage",
        lambda **_kwargs: config.prompt_records_path,
    )
    monkeypatch.setattr(
        workflow_module,
        "run_completions_stage",
        lambda **_kwargs: config.completions_path,
    )

    def capture_activation_stage(**kwargs: object) -> Path:
        activation_kwargs.update(kwargs)
        return config.output_dir

    monkeypatch.setattr(workflow_module, "run_activations_stage", capture_activation_stage)
    monkeypatch.setattr(
        workflow_module,
        "upload_stage_artifacts",
        lambda artifacts, **_kwargs: uploaded.extend(artifacts),
    )
    monkeypatch.setattr(
        workflow_module,
        "ensure_input_artifact_available",
        lambda *_args, **_kwargs: None,
    )
    monkeypatch.setattr(
        workflow_module,
        "download_cached_artifact",
        lambda *_args, **_kwargs: False,
    )

    run_workflow(config)

    assert uploaded == [config.prompt_records_path, config.completions_path]
    assert activation_kwargs["save_to_gcp"] is True
    assert activation_kwargs["gcs_prefix"] == "test-prefix"
    assert activation_kwargs["nodes_gcs_prefix"] == "node-artifacts"
    assert activation_kwargs["upload_worker_count"] == 16
    assert activation_kwargs["upload_queue_capacity"] == 8


def test_cached_dataset_and_completions_skip_generation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    values = config_values()
    values["modules"] = ["dataset", "completions", "activations"]
    values["save_to_gcp"] = True
    config = WorkflowConfig.from_mapping(values, gcs_prefix="test-prefix")
    checked: list[tuple[Path, str | None]] = []

    def download_cached(path: Path, **kwargs: object) -> bool:
        checked.append((path, kwargs["gcs_prefix"]))
        return True

    monkeypatch.setattr(workflow_module, "download_cached_artifact", download_cached)
    monkeypatch.setattr(
        workflow_module,
        "ensure_input_artifact_available",
        lambda *_args, **_kwargs: None,
    )
    monkeypatch.setattr(
        workflow_module,
        "run_dataset_stage",
        lambda **_kwargs: pytest.fail("dataset should not be regenerated"),
    )
    monkeypatch.setattr(
        workflow_module,
        "run_completions_stage",
        lambda **_kwargs: pytest.fail("completions should not be regenerated"),
    )
    monkeypatch.setattr(
        workflow_module,
        "upload_stage_artifacts",
        lambda *_args, **_kwargs: pytest.fail("cached artifacts should not be uploaded"),
    )
    monkeypatch.setattr(workflow_module, "run_activations_stage", lambda **_kwargs: None)

    run_workflow(config)

    assert checked == [
        (config.prompt_records_path, "dataset-artifacts"),
        (config.completions_path, "completion-artifacts"),
    ]


@pytest.mark.parametrize(
    ("artifact_name", "prefix"),
    [("Dataset", "dataset-artifacts"), ("Completions", "completion-artifacts")],
)
def test_missing_input_artifact_is_downloaded_from_its_gcs_prefix(
    artifact_name: str,
    prefix: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    artifact_path = tmp_path / f"{artifact_name.lower()}.jsonl"
    requested: list[tuple[str, Path]] = []

    def build_fetcher(**kwargs: object):
        assert kwargs["prefix"] == prefix

        def fetch(object_name: str, destination: Path) -> bool:
            requested.append((object_name, destination))
            destination.write_text("artifact", encoding="utf-8")
            return True

        return fetch

    monkeypatch.setattr(
        "temporal_manifolds.utils.gcs_upload.maybe_build_gcs_existing_object_fetcher",
        build_fetcher,
    )

    ensure_input_artifact_available(
        artifact_path,
        artifact_name=artifact_name,
        save_to_gcp=True,
        gcp_project_id="test-project",
        gcs_bucket_name="test-bucket",
        gcs_prefix=prefix,
    )

    assert requested == [(artifact_path.name, artifact_path)]
    assert artifact_path.read_text(encoding="utf-8") == "artifact"


def test_missing_nodes_file_is_downloaded_from_gcs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    nodes_path = tmp_path / "data" / "selected_nodes" / "nodes.pkl"
    requested: list[tuple[str, Path]] = []

    def build_fetcher(**kwargs: object):
        assert kwargs == {
            "enabled": True,
            "project_id": "test-project",
            "bucket_name": "test-bucket",
            "prefix": "node-artifacts",
        }

        def fetch(object_name: str, destination: Path) -> bool:
            requested.append((object_name, destination))
            destination.parent.mkdir(parents=True)
            destination.write_bytes(b"nodes")
            return True

        return fetch

    monkeypatch.setattr(
        "temporal_manifolds.utils.gcs_upload.maybe_build_gcs_existing_object_fetcher",
        build_fetcher,
    )

    ensure_nodes_file_available(
        nodes_path,
        save_to_gcp=True,
        gcp_project_id="test-project",
        gcs_bucket_name="test-bucket",
        nodes_gcs_prefix="node-artifacts",
    )

    assert requested == [("nodes.pkl", nodes_path)]
    assert nodes_path.read_bytes() == b"nodes"


def test_existing_nodes_file_does_not_check_gcs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    nodes_path = tmp_path / "nodes.pkl"
    nodes_path.write_bytes(b"nodes")
    monkeypatch.setattr(
        "temporal_manifolds.utils.gcs_upload.maybe_build_gcs_existing_object_fetcher",
        lambda **_kwargs: pytest.fail("GCS should not be checked"),
    )

    ensure_nodes_file_available(
        nodes_path,
        save_to_gcp=True,
        gcp_project_id="test-project",
        gcs_bucket_name="test-bucket",
        nodes_gcs_prefix="node-artifacts",
    )
