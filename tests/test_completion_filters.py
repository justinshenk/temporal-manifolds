"""Tests for filtering completion records by prompt metadata."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest
from google.auth.exceptions import DefaultCredentialsError

from temporal_manifolds.utils import completion_filters
from temporal_manifolds.utils.completion_filters import (
    download_activation_files,
    find_activation_paths,
    find_completion_indices,
)


def _write_completions(path: Path) -> None:
    metadata_rows = [
        (
            {"prompt_framing": "task_available_time", "output_format": "strategy_steps"},
            {"difficulty": "low", "domain": "communication"},
        ),
        (
            {"prompt_framing": "task_deadline", "output_format": "strategy_steps"},
            {"difficulty": "high", "domain": "communication"},
        ),
        (
            {"prompt_framing": "task_available_time", "output_format": "summary_checklist"},
            {"difficulty": "low", "domain": "analysis"},
        ),
    ]
    with path.open("w", encoding="utf-8") as output_file:
        for template_metadata, task_metadata in metadata_rows:
            record = {
                "prompt_metadata": {
                    "template_metadata": template_metadata,
                    "task_metadata": task_metadata,
                }
            }
            output_file.write(json.dumps(record) + "\n")


def test_find_activation_paths_filters_each_metadata_key(tmp_path: Path) -> None:
    completions_path = tmp_path / "completions.jsonl"
    activations_dir = tmp_path / "activations"
    _write_completions(completions_path)

    assert find_activation_paths(
        prompt_framing="task_available_time",
        completions_path=completions_path,
        activations_dir=activations_dir,
    ) == [
        activations_dir / "activations_sample_00000.pt",
        activations_dir / "activations_sample_00002.pt",
    ]
    assert find_activation_paths(
        output_format="strategy_steps",
        completions_path=completions_path,
        activations_dir=activations_dir,
    ) == [
        activations_dir / "activations_sample_00000.pt",
        activations_dir / "activations_sample_00001.pt",
    ]


def test_find_activation_paths_combines_filters(tmp_path: Path) -> None:
    completions_path = tmp_path / "completions.jsonl"
    activations_dir = tmp_path / "activations"
    _write_completions(completions_path)

    assert find_activation_paths(
        prompt_framing="task_available_time",
        output_format="summary_checklist",
        task_metadata={"difficulty": "low", "domain": "analysis"},
        completions_path=completions_path,
        activations_dir=activations_dir,
    ) == [activations_dir / "activations_sample_00002.pt"]


def test_find_activation_paths_filters_task_metadata(tmp_path: Path) -> None:
    completions_path = tmp_path / "completions.jsonl"
    activations_dir = tmp_path / "activations"
    _write_completions(completions_path)

    assert find_activation_paths(
        task_metadata={"difficulty": "low"},
        completions_path=completions_path,
        activations_dir=activations_dir,
    ) == [
        activations_dir / "activations_sample_00000.pt",
        activations_dir / "activations_sample_00002.pt",
    ]
    assert find_activation_paths(
        task_metadata={"stakes": "high"},
        completions_path=completions_path,
        activations_dir=activations_dir,
    ) == []


def test_find_activation_paths_without_filters_returns_every_path(tmp_path: Path) -> None:
    completions_path = tmp_path / "completions.jsonl"
    activations_dir = tmp_path / "activations"
    _write_completions(completions_path)

    assert find_activation_paths(
        completions_path=completions_path,
        activations_dir=activations_dir,
    ) == [
        activations_dir / "activations_sample_00000.pt",
        activations_dir / "activations_sample_00001.pt",
        activations_dir / "activations_sample_00002.pt",
    ]


def test_original_function_name_now_returns_activation_paths(tmp_path: Path) -> None:
    completions_path = tmp_path / "completions.jsonl"
    activations_dir = tmp_path / "activations"
    _write_completions(completions_path)

    assert find_completion_indices(
        prompt_framing="task_deadline",
        completions_path=completions_path,
        activations_dir=activations_dir,
    ) == [activations_dir / "activations_sample_00001.pt"]


class _FakeBlob:
    def __init__(self, object_name: str, requested_objects: list[str]) -> None:
        self.object_name = object_name
        self.requested_objects = requested_objects

    def download_to_filename(self, destination: str) -> None:
        self.requested_objects.append(self.object_name)
        Path(destination).write_bytes(b"activation")


class _FakeBucket:
    def __init__(self, requested_objects: list[str]) -> None:
        self.requested_objects = requested_objects

    def blob(self, object_name: str) -> _FakeBlob:
        return _FakeBlob(object_name, self.requested_objects)


class _FakeClient:
    def __init__(self, requested_objects: list[str]) -> None:
        self.requested_objects = requested_objects
        self.requested_bucket: str | None = None

    def bucket(self, bucket_name: str) -> _FakeBucket:
        self.requested_bucket = bucket_name
        return _FakeBucket(self.requested_objects)


def test_download_activation_files_uses_env_and_pipeline_prefix(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    requested_objects: list[str] = []
    fake_client = _FakeClient(requested_objects)
    monkeypatch.setenv("GCP_PROJECT_ID", "test-project")
    monkeypatch.setenv("GCS_BUCKET_NAME", "test-bucket")
    monkeypatch.setattr(completion_filters, "_authenticated_gcs_client", lambda _project: fake_client)
    paths = [
        tmp_path / "results" / "activations" / "activations_sample_00000.pt",
        tmp_path / "results" / "activations" / "activations_sample_00002.pt",
    ]

    assert download_activation_files(
        paths, gcs_prefix="test-prefix", upload_root=tmp_path
    ) == paths
    assert fake_client.requested_bucket == "test-bucket"
    assert requested_objects == [
        "test-prefix/results/activations/activations_sample_00000.pt",
        "test-prefix/results/activations/activations_sample_00002.pt",
    ]
    assert all(path.read_bytes() == b"activation" for path in paths)


def test_download_activation_files_skips_existing_files(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    requested_objects: list[str] = []
    fake_client = _FakeClient(requested_objects)
    monkeypatch.setenv("GCP_PROJECT_ID", "test-project")
    monkeypatch.setenv("GCS_BUCKET_NAME", "test-bucket")
    monkeypatch.setattr(completion_filters, "_authenticated_gcs_client", lambda _project: fake_client)
    existing_path = tmp_path / "activations_sample_00000.pt"
    existing_path.write_bytes(b"existing")

    assert download_activation_files([existing_path]) == [existing_path]
    assert requested_objects == []
    assert existing_path.read_bytes() == b"existing"


def test_missing_credentials_triggers_gcloud_application_default_login(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_client = _FakeClient([])
    client_attempts = iter([DefaultCredentialsError("missing"), fake_client])
    commands: list[list[str]] = []

    def fake_storage_client(*, project: str) -> _FakeClient:
        attempt = next(client_attempts)
        if isinstance(attempt, Exception):
            raise attempt
        assert project == "test-project"
        return attempt

    def fake_run(command: list[str], *, check: bool) -> subprocess.CompletedProcess[str]:
        assert check is True
        commands.append(command)
        return subprocess.CompletedProcess(command, 0)

    monkeypatch.setattr(completion_filters.storage, "Client", fake_storage_client)
    monkeypatch.setattr(completion_filters.shutil, "which", lambda _command: "gcloud")
    monkeypatch.setattr(completion_filters.subprocess, "run", fake_run)

    assert completion_filters._authenticated_gcs_client("test-project") is fake_client
    assert commands == [
        ["gcloud", "auth", "application-default", "login", "--project", "test-project"]
    ]
