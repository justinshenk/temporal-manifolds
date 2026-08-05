"""Tests for the standalone residual-stream exploration script."""

from __future__ import annotations

import io
import importlib.util
from pathlib import Path

import pytest
import torch

SCRIPT_PATH = Path(__file__).parents[1] / "scripts" / "extract_residual_stream_positions_from_gcs.py"
SCRIPT_SPEC = importlib.util.spec_from_file_location(
    "extract_residual_stream_positions_from_gcs",
    SCRIPT_PATH,
)
assert SCRIPT_SPEC is not None and SCRIPT_SPEC.loader is not None
SCRIPT_MODULE = importlib.util.module_from_spec(SCRIPT_SPEC)
SCRIPT_SPEC.loader.exec_module(SCRIPT_MODULE)

download_and_extract = SCRIPT_MODULE.download_and_extract
download_extract_pipeline = SCRIPT_MODULE.download_extract_pipeline
chunk_object_name = SCRIPT_MODULE.chunk_object_name
chunked = SCRIPT_MODULE.chunked
combine_samples = SCRIPT_MODULE.combine_samples
extract_residual_stream_positions = SCRIPT_MODULE.extract_residual_stream_positions
parse_gcs_uri = SCRIPT_MODULE.parse_gcs_uri
upload_chunk = SCRIPT_MODULE.upload_chunk


def _payload() -> dict[str, object]:
    return {
        "position_selection_policy": "after_assistant",
        "positions": [10, 11, 12, 13],
        "metadata": [{"sample_index": 7}],
        "residual_stream_activations": {
            "layer_out/0": torch.arange(24).reshape(1, 4, 6),
            "layer_out/1": torch.arange(24, 48).reshape(1, 4, 6),
        },
    }


class FakeBlob:
    def __init__(self, sample_index: int = 7) -> None:
        self.sample_index = sample_index
        self.name = f"prefix/activations_sample_{sample_index:05d}.pt"
        self.destination: io.BytesIO | None = None

    def download_to_file(self, destination: io.BytesIO) -> None:
        self.destination = destination
        payload = _payload()
        payload["metadata"] = [{"sample_index": self.sample_index}]
        torch.save(payload, destination)


def test_parse_gcs_uri() -> None:
    assert parse_gcs_uri("gs://bucket/some/prefix/") == ("bucket", "some/prefix")
    with pytest.raises(ValueError, match="gs://"):
        parse_gcs_uri("bucket/some/prefix")


def test_extracts_first_three_cached_positions_from_every_layer() -> None:
    result = extract_residual_stream_positions(_payload(), source_object="source.pt")

    assert result["sample_index"] == 7
    assert result["cached_position_indices"] == [0, 1, 2]
    assert result["absolute_token_positions"] == [10, 11, 12]
    residuals = result["residual_stream_activations"]
    assert isinstance(residuals, dict)
    assert torch.equal(residuals["layer_out/0"], torch.arange(18).reshape(1, 3, 6))
    assert torch.equal(residuals["layer_out/1"], torch.arange(24, 42).reshape(1, 3, 6))


def test_rejects_payload_with_too_few_positions() -> None:
    payload = _payload()
    payload["positions"] = [10, 11]
    payload["residual_stream_activations"] = {
        "layer_out/0": torch.zeros(1, 2, 6)
    }

    with pytest.raises(ValueError, match="only 2 cached positions"):
        extract_residual_stream_positions(payload, source_object="source.pt")


def test_downloads_and_loads_payload_in_memory() -> None:
    blob = FakeBlob()
    result = download_and_extract(blob)  # type: ignore[arg-type]

    assert result["source_object"] == blob.name
    assert result["absolute_token_positions"] == [10, 11, 12]


def test_combines_samples_as_num_samples_by_three_by_model_width() -> None:
    first = extract_residual_stream_positions(_payload(), source_object="sample_0.pt")
    second_payload = _payload()
    second_payload["metadata"] = [{"sample_index": 8}]
    second_payload["positions"] = [20, 21, 22, 23]
    second = extract_residual_stream_positions(
        second_payload,
        source_object="sample_1.pt",
    )

    combined = combine_samples([first, second])

    assert combined["sample_indices"] == [7, 8]
    assert combined["absolute_token_positions"] == [[10, 11, 12], [20, 21, 22]]
    residuals = combined["residual_stream_activations"]
    assert residuals["layer_out/0"].shape == (2, 3, 6)
    assert residuals["layer_out/1"].shape == (2, 3, 6)
    assert torch.equal(residuals["layer_out/0"][0], torch.arange(18).reshape(3, 6))
    assert first["residual_stream_activations"] == {}
    assert second["residual_stream_activations"] == {}


def test_pipeline_preserves_order_and_closes_raw_download_buffers() -> None:
    blobs = [FakeBlob(7), FakeBlob(8), FakeBlob(9)]

    results = download_extract_pipeline(
        blobs,
        download_workers=2,
        processing_workers=2,
        queue_capacity=1,
    )

    assert [result["sample_index"] for result in results] == [7, 8, 9]
    assert all(blob.destination is not None and blob.destination.closed for blob in blobs)


def test_pipeline_can_assemble_directly_into_final_chunk_tensors() -> None:
    blobs = [FakeBlob(7), FakeBlob(8), FakeBlob(9)]

    combined = download_extract_pipeline(
        blobs,
        download_workers=2,
        processing_workers=2,
        queue_capacity=1,
        combine_results=True,
    )

    assert isinstance(combined, dict)
    assert combined["sample_indices"] == [7, 8, 9]
    assert combined["source_objects"] == [blob.name for blob in blobs]
    residuals = combined["residual_stream_activations"]
    assert residuals["layer_out/0"].shape == (3, 3, 6)
    assert torch.equal(residuals["layer_out/0"][0], torch.arange(18).reshape(3, 6))
    assert all(blob.destination is not None and blob.destination.closed for blob in blobs)


def test_chunks_files_and_builds_destination_object_names() -> None:
    blobs = [FakeBlob(index) for index in range(10)]

    chunks = list(chunked(blobs, 4))

    assert [len(chunk) for chunk in chunks] == [4, 4, 2]
    assert chunk_object_name("/resid_only_0_2/", 3) == (
        "resid_only_0_2/residual_stream_positions_0_2_chunk_00003.pt"
    )


def test_uploads_one_serialized_chunk_to_gcs() -> None:
    class UploadBlob:
        uploaded: bytes | None = None

        def exists(self) -> bool:
            return False

        def upload_from_file(
            self,
            source: object,
            *,
            rewind: bool,
            content_type: str,
        ) -> None:
            assert rewind is True
            assert content_type == "application/octet-stream"
            source.seek(0)
            self.uploaded = source.read()

    class UploadBucket:
        name = "test-bucket"

        def __init__(self) -> None:
            self.upload_blob = UploadBlob()

        def blob(self, _name: str) -> UploadBlob:
            return self.upload_blob

    bucket = UploadBucket()
    payload = {"residual_stream_activations": {"layer_out/0": torch.ones(2, 3, 6)}}

    upload_chunk(bucket, "resid_only_0_2/chunk.pt", payload, overwrite=False)

    assert bucket.upload_blob.uploaded is not None
    loaded = torch.load(
        io.BytesIO(bucket.upload_blob.uploaded),
        map_location="cpu",
        weights_only=True,
    )
    assert loaded["residual_stream_activations"]["layer_out/0"].shape == (2, 3, 6)
