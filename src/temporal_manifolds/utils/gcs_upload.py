"""Shared Google Cloud Storage upload helpers."""

import json
import os
import queue
import threading
from pathlib import Path
from typing import Callable

from dotenv import load_dotenv
from google.cloud import storage
from google.oauth2 import service_account

load_dotenv()

UploadQueue = queue.Queue[tuple[Path, str] | None]
EnqueueUpload = Callable[[Path, str], None]


def _build_gcs_client(project_id: str) -> storage.Client:
    """Build a GCS client from credentials configured in the environment."""
    credentials_json = os.getenv("GOOGLE_APPLICATION_CREDENTIALS_JSON")
    if credentials_json:
        credentials_info = json.loads(credentials_json)
        credentials = service_account.Credentials.from_service_account_info(credentials_info)
        return storage.Client(project=project_id, credentials=credentials)

    return storage.Client(project=project_id)


def maybe_start_gcs_upload_worker(
    enabled: bool,
    project_id: str | None = None,
    bucket_name: str | None = None,
    prefix: str | None = None,
) -> tuple[UploadQueue | None, threading.Thread | None, EnqueueUpload]:
    """Start a background GCS uploader when enabled."""
    if not enabled:
        return None, None, lambda _local_file, _object_name: None

    resolved_project_id = project_id or os.getenv("GCP_PROJECT_ID")
    if not resolved_project_id:
        raise ValueError("GCP_PROJECT_ID environment variable is required for GCS uploads.")

    resolved_bucket_name = bucket_name or os.getenv("GCS_BUCKET_NAME")
    if not resolved_bucket_name:
        raise ValueError("GCS_BUCKET_NAME environment variable is required for GCS uploads.")

    resolved_prefix = (prefix if prefix is not None else os.getenv("GCS_PREFIX", "")).strip("/")
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
            if resolved_prefix:
                object_name = f"{resolved_prefix}/{object_name.lstrip('/')}"
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
            upload_queue.task_done()

    upload_thread = threading.Thread(target=_upload_worker, daemon=True)
    upload_thread.start()

    def _enqueue_upload(local_file: Path, object_name: str) -> None:
        upload_queue.put((local_file, object_name))

    return upload_queue, upload_thread, _enqueue_upload
