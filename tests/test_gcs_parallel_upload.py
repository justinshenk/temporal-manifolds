"""Tests for parallel GCS upload and post-upload deletion."""

from __future__ import annotations

import threading
import io
from pathlib import Path

from temporal_manifolds.utils import gcs_upload


def test_parallel_upload_deletes_only_after_success(
    monkeypatch, tmp_path: Path
) -> None:
    events: list[tuple[str, str]] = []
    event_lock = threading.Lock()

    class FakeBlob:
        def __init__(self, object_name: str) -> None:
            self.object_name = object_name

        def upload_from_filename(self, filename: str) -> None:
            with event_lock:
                events.append(("uploaded", self.object_name))
            assert Path(filename).exists()

    class FakeBucket:
        def blob(self, object_name: str) -> FakeBlob:
            return FakeBlob(object_name)

    class FakeClient:
        def bucket(self, _bucket_name: str) -> FakeBucket:
            return FakeBucket()

    monkeypatch.setattr(gcs_upload, "_build_gcs_client", lambda _project_id: FakeClient())
    files = [tmp_path / f"activation_{index}.pt" for index in range(3)]
    for file_path in files:
        file_path.write_bytes(b"cache")

    upload_queue, upload_threads, delete_queue, delete_thread, enqueue = (
        gcs_upload.maybe_start_parallel_gcs_upload_workers(
            enabled=True,
            project_id="project",
            bucket_name="bucket",
            prefix="scenario",
            worker_count=2,
        )
    )
    assert upload_queue is not None
    assert delete_queue is not None
    assert delete_thread is not None

    for file_path in files:
        enqueue(file_path, file_path.name)
    for _ in upload_threads:
        upload_queue.put(None)
    for upload_thread in upload_threads:
        upload_thread.join()
    delete_queue.put(None)
    delete_thread.join()

    assert sorted(events) == [
        ("uploaded", f"scenario/activation_{index}.pt") for index in range(3)
    ]
    assert all(not file_path.exists() for file_path in files)


def test_parallel_upload_is_noop_when_disabled() -> None:
    upload_queue, upload_threads, delete_queue, delete_thread, enqueue = (
        gcs_upload.maybe_start_parallel_gcs_upload_workers(enabled=False)
    )

    enqueue(Path("unused"), "unused")
    assert upload_queue is None
    assert upload_threads == []
    assert delete_queue is None
    assert delete_thread is None


def test_memory_upload_uses_bounded_queue_and_closes_buffer(
    monkeypatch,
) -> None:
    uploaded: list[tuple[str, bytes]] = []

    class FakeBlob:
        def __init__(self, object_name: str) -> None:
            self.object_name = object_name

        def upload_from_file(self, buffer, *, rewind: bool) -> None:
            assert rewind is True
            uploaded.append((self.object_name, buffer.read()))

    class FakeBucket:
        def blob(self, object_name: str) -> FakeBlob:
            return FakeBlob(object_name)

    class FakeClient:
        def bucket(self, _bucket_name: str) -> FakeBucket:
            return FakeBucket()

    monkeypatch.setattr(gcs_upload, "_build_gcs_client", lambda _project_id: FakeClient())
    upload_queue, upload_threads, enqueue = gcs_upload.maybe_start_memory_gcs_upload_workers(
        enabled=True,
        project_id="project",
        bucket_name="bucket",
        prefix="scenario",
        worker_count=1,
        queue_capacity=2,
    )
    assert upload_queue is not None
    assert upload_queue.maxsize == 2
    buffer = io.BytesIO(b"activation-cache")

    enqueue(buffer, "results/activations/activations_sample_00000.pt", "sample.pt")
    for _ in upload_threads:
        upload_queue.put(None)
    for upload_thread in upload_threads:
        upload_thread.join()

    assert uploaded == [
        (
            "scenario/results/activations/activations_sample_00000.pt",
            b"activation-cache",
        )
    ]
    assert buffer.closed
