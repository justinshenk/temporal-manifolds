"""Shared Google Cloud Storage upload helpers."""

import json
import os
import queue
import threading
from collections.abc import Iterable
from pathlib import Path
from typing import Callable

from dotenv import load_dotenv
from google.cloud import storage
from google.oauth2 import service_account

load_dotenv()

UploadQueue = queue.Queue[tuple[Path, str] | None]
EnqueueUpload = Callable[[Path, str], None]
FetchGCSObjectIfExists = Callable[[str, Path], bool]


def gcs_object_name_for_file(local_file: Path, upload_root: Path | None = None) -> str:
    """Return a stable GCS object name for a local artifact path."""
    local_file_abs = local_file.resolve()
    root = (upload_root or Path.cwd()).resolve()
    try:
        return local_file_abs.relative_to(root).as_posix()
    except ValueError:
        return local_file.name


def _build_gcs_client(project_id: str) -> storage.Client:
    """Build a GCS client from credentials configured in the environment."""
    credentials_json = os.getenv("GOOGLE_APPLICATION_CREDENTIALS_JSON")
    if credentials_json:
        credentials_info = json.loads(credentials_json)
        credentials = service_account.Credentials.from_service_account_info(credentials_info)
        return storage.Client(project=project_id, credentials=credentials)

    return storage.Client(project=project_id)


def _resolve_gcs_config(
    project_id: str | None,
    bucket_name: str | None,
) -> tuple[str, str]:
    """Resolve GCS project and bucket settings from args or environment."""
    resolved_project_id = project_id or os.getenv("GCP_PROJECT_ID")
    if not resolved_project_id:
        raise ValueError("GCP_PROJECT_ID environment variable is required for GCS uploads.")

    resolved_bucket_name = bucket_name or os.getenv("GCS_BUCKET_NAME")
    if not resolved_bucket_name:
        raise ValueError("GCS_BUCKET_NAME environment variable is required for GCS uploads.")

    return resolved_project_id, resolved_bucket_name


def apply_gcs_prefix(object_name: str, prefix: str | None = None) -> str:
    """Apply a GCS prefix to an object name using the upload worker convention."""
    resolved_prefix = (prefix or "").strip("/")
    if resolved_prefix:
        return f"{resolved_prefix}/{object_name.lstrip('/')}"
    return object_name


def maybe_build_gcs_existing_object_fetcher(
    enabled: bool,
    project_id: str | None = None,
    bucket_name: str | None = None,
    prefix: str | None = None,
    *,
    download_existing: bool = True,
) -> FetchGCSObjectIfExists:
    """Return a fetcher that detects and optionally downloads existing GCS objects."""
    if not enabled:
        return lambda _object_name, _destination: False

    resolved_project_id, resolved_bucket_name = _resolve_gcs_config(project_id, bucket_name)
    resolved_prefix = (prefix or "").strip("/")
    gcs_client = _build_gcs_client(resolved_project_id)
    bucket = gcs_client.bucket(resolved_bucket_name)

    def _fetch_if_exists(object_name: str, destination: Path) -> bool:
        prefixed_object_name = apply_gcs_prefix(object_name, resolved_prefix)
        blob = bucket.blob(prefixed_object_name)
        if not blob.exists():
            return False

        if download_existing and not destination.exists():
            destination.parent.mkdir(parents=True, exist_ok=True)
            blob.download_to_filename(str(destination))
            print(
                "[GCS resume] Downloaded existing "
                f"object={prefixed_object_name} local_path={destination}",
                flush=True,
            )
        elif not download_existing:
            print(
                "[GCS resume] Found existing "
                f"object={prefixed_object_name}; skipping local download",
                flush=True,
            )
        return True

    return _fetch_if_exists


def maybe_start_gcs_upload_worker(
    enabled: bool,
    project_id: str | None = None,
    bucket_name: str | None = None,
    prefix: str | None = None,
    *,
    delete_local_after_upload: bool = False,
) -> tuple[UploadQueue | None, threading.Thread | None, EnqueueUpload]:
    """Start a background GCS uploader when enabled."""
    if not enabled:
        return None, None, lambda _local_file, _object_name: None

    resolved_project_id, resolved_bucket_name = _resolve_gcs_config(project_id, bucket_name)
    resolved_prefix = (prefix or "").strip("/")
    gcs_client = _build_gcs_client(resolved_project_id)
    bucket = gcs_client.bucket(resolved_bucket_name)
    print(
        "[GCS upload] Using "
        f"project_id={resolved_project_id} "
        f"bucket={resolved_bucket_name} "
        f"prefix={resolved_prefix or '<none>'}",
        flush=True,
    )

    upload_queue: UploadQueue = queue.Queue()

    def _upload_worker() -> None:
        while True:
            item = upload_queue.get()
            if item is None:
                upload_queue.task_done()
                break
            local_file, object_name = item
            object_name = apply_gcs_prefix(object_name, resolved_prefix)
            file_size = local_file.stat().st_size
            print(
                f"[GCS upload] Starting file={local_file.name} "
                f"local_path={local_file} object_name={object_name} "
                f"size_gb={file_size / (1024**3):.2f}",
                flush=True,
            )
            blob = bucket.blob(object_name)
            blob.upload_from_filename(str(local_file))
            print(
                f"[GCS upload] Completed file={local_file.name}",
                flush=True,
            )
            if delete_local_after_upload:
                local_file.unlink(missing_ok=True)
                print(
                    f"[GCS upload] Deleted local file={local_file.name}",
                    flush=True,
                )
            upload_queue.task_done()

    upload_thread = threading.Thread(target=_upload_worker, daemon=True)
    upload_thread.start()

    def _enqueue_upload(local_file: Path, object_name: str) -> None:
        upload_queue.put((local_file, object_name))

    return upload_queue, upload_thread, _enqueue_upload


def upload_files_to_gcs(
    files: Iterable[Path],
    *,
    enabled: bool,
    project_id: str | None = None,
    bucket_name: str | None = None,
    prefix: str | None = None,
    upload_root: Path | None = None,
) -> None:
    """Upload generated artifact files to GCS when enabled."""
    upload_queue, upload_thread, enqueue_upload = maybe_start_gcs_upload_worker(
        enabled=enabled,
        project_id=project_id,
        bucket_name=bucket_name,
        prefix=prefix,
    )
    try:
        for file_path in files:
            if file_path.exists():
                enqueue_upload(
                    file_path.resolve(),
                    gcs_object_name_for_file(file_path, upload_root=upload_root),
                )
    finally:
        if upload_queue is not None and upload_thread is not None:
            upload_queue.put(None)
            upload_thread.join()
