"""Helpers for selecting completion records by prompt template metadata."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
from collections.abc import Iterable
from pathlib import Path

from dotenv import load_dotenv
from google.auth.exceptions import DefaultCredentialsError
from google.cloud import storage

from temporal_manifolds.utils.gcs_upload import apply_gcs_prefix, gcs_object_name_for_file


PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_COMPLETIONS_PATH = PROJECT_ROOT / "data" / "completions_completions_256.jsonl"
DEFAULT_ACTIVATIONS_DIR = PROJECT_ROOT / "results" / "feature_geometry_new_activations"
DEFAULT_GCS_PREFIX = "conversational_default"


def find_activation_paths(
    *,
    prompt_framing: str | None = None,
    output_format: str | None = None,
    completions_path: str | Path = DEFAULT_COMPLETIONS_PATH,
    activations_dir: str | Path = DEFAULT_ACTIVATIONS_DIR,
) -> list[Path]:
    """Return activation-cache paths whose template metadata matches all filters.

    A filter set to ``None`` is ignored. Consequently, calling this function
    without metadata filters returns the expected activation path for every
    record in the JSONL file. Paths follow the naming convention used by
    ``cache_completion_activations``.

    Args:
        prompt_framing: Required value of ``template_metadata.prompt_framing``.
        output_format: Required value of ``template_metadata.output_format``.
        completions_path: JSONL completions file to search.
        activations_dir: Directory containing the cached ``.pt`` files.
    """
    filters = {
        "prompt_framing": prompt_framing,
        "output_format": output_format,
    }
    active_filters = {key: value for key, value in filters.items() if value is not None}

    activation_root = Path(activations_dir)
    matching_paths: list[Path] = []
    with Path(completions_path).open(encoding="utf-8") as completion_file:
        for index, line in enumerate(completion_file):
            record = json.loads(line)
            template_metadata = record["prompt_metadata"]["template_metadata"]
            if all(template_metadata.get(key) == value for key, value in active_filters.items()):
                matching_paths.append(activation_root / f"activations_sample_{index:05d}.pt")

    return matching_paths


def find_completion_indices(
    *,
    prompt_framing: str | None = None,
    output_format: str | None = None,
    completions_path: str | Path = DEFAULT_COMPLETIONS_PATH,
    activations_dir: str | Path = DEFAULT_ACTIVATIONS_DIR,
) -> list[Path]:
    """Return matching activation paths, retaining the original public function name."""
    return find_activation_paths(
        prompt_framing=prompt_framing,
        output_format=output_format,
        completions_path=completions_path,
        activations_dir=activations_dir,
    )


def _authenticated_gcs_client(project_id: str) -> storage.Client:
    """Build a GCS client, launching gcloud's ADC login when credentials are absent."""
    try:
        return storage.Client(project=project_id)
    except DefaultCredentialsError:
        gcloud = shutil.which("gcloud")
        if gcloud is None:
            raise RuntimeError(
                "Google Cloud credentials were not found and the gcloud CLI is not installed "
                "or is not available on PATH."
            ) from None

        subprocess.run(
            [
                gcloud,
                "auth",
                "application-default",
                "login",
                "--project",
                project_id,
            ],
            check=True,
        )
        return storage.Client(project=project_id)


def download_activation_files(
    activation_paths: Iterable[str | Path],
    *,
    gcs_prefix: str | None = DEFAULT_GCS_PREFIX,
    overwrite: bool = False,
    upload_root: str | Path = PROJECT_ROOT,
) -> list[Path]:
    """Download activation-cache paths from GCS and return their local paths.

    ``GCP_PROJECT_ID`` and ``GCS_BUCKET_NAME`` are loaded from the repository's
    ``.env`` file. If Application Default Credentials are unavailable, the
    function launches ``gcloud auth application-default login`` interactively.

    Args:
        activation_paths: Paths returned by :func:`find_activation_paths`.
        gcs_prefix: GCS prefix used by the activation-caching pipeline.
        overwrite: Download files that already exist locally when true.
        upload_root: Local root used to derive pipeline-relative GCS object names.
    """
    load_dotenv(PROJECT_ROOT / ".env")
    project_id = os.getenv("GCP_PROJECT_ID")
    if not project_id:
        raise ValueError("GCP_PROJECT_ID must be set in the repository .env file.")
    bucket_name = os.getenv("GCS_BUCKET_NAME")
    if not bucket_name:
        raise ValueError("GCS_BUCKET_NAME must be set in the repository .env file.")

    local_paths = [Path(path) for path in activation_paths]
    if not local_paths:
        return []

    bucket = _authenticated_gcs_client(project_id).bucket(bucket_name)
    for local_path in local_paths:
        if local_path.is_file() and not overwrite:
            continue

        object_name = gcs_object_name_for_file(local_path, upload_root=Path(upload_root))
        object_name = apply_gcs_prefix(object_name, gcs_prefix)
        local_path.parent.mkdir(parents=True, exist_ok=True)
        bucket.blob(object_name).download_to_filename(str(local_path))

    return local_paths
