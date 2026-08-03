"""Tests for bounded-memory matched activation aggregation."""

from __future__ import annotations

import io
import json
import pickle
from pathlib import Path
from typing import Any

import pytest
import torch

from temporal_manifolds.activations.stream_matched_activations import (
    ProcessingState,
    _load_existing_state,
    activation_object_name,
    build_completion_index,
    build_output_payload,
    download_selected_node_groups,
    extract_assistant_feature_vector,
    limited_output_object,
    process_groups,
    requested_nodes,
    upload_output_payload,
)


def _completion(
    task: str,
    base_value: int,
    base_unit: str,
    prompt_framing: str,
    output_format: str,
) -> dict[str, Any]:
    return {
        "prompt": "large prompt that is deliberately not retained in the index",
        "full_text": "large completion that is deliberately not retained in the index",
        "prompt_metadata": {
            "task": task,
            "base_value": base_value,
            "base_unit": base_unit,
            "template_metadata": {
                "prompt_framing": prompt_framing,
                "output_format": output_format,
            },
        },
    }


def _write_completions(path: Path, records: list[dict[str, Any]]) -> None:
    path.write_text(
        "".join(json.dumps(record) + "\n" for record in records),
        encoding="utf-8",
    )


def _activation_payload(sample_index: int, first_value: float) -> dict[str, Any]:
    return {
        "position_selection_policy": "after_assistant",
        "positions": [12, 13],
        "metadata": [{"sample_index": sample_index}],
        "activations": {
            "mlp_hidden/1": {
                "node_indices": [4, 9],
                "values": torch.tensor(
                    [[[first_value, 900.0], [first_value + 100.0, 901.0]]]
                ),
            },
        },
        "residual_stream_activations": {
            "layer_out/0": torch.full((1, 2, 3), 999.0),
        },
    }


def _serialized_payload(sample_index: int, first_value: float) -> bytes:
    buffer = io.BytesIO()
    torch.save(_activation_payload(sample_index, first_value), buffer)
    return buffer.getvalue()


class FakeBlob:
    def __init__(self, name: str, objects: dict[str, bytes], opened_files: list[io.BytesIO]):
        self.name = name
        self._objects = objects
        self._opened_files = opened_files

    def download_to_file(self, destination: io.BytesIO) -> None:
        self._opened_files.append(destination)
        destination.write(self._objects[self.name])

    def upload_from_file(self, source: io.BytesIO, *, rewind: bool) -> None:
        if rewind:
            source.seek(0)
        self._objects[self.name] = source.read()

    def exists(self) -> bool:
        return self.name in self._objects


class FakeBucket:
    name = "test-bucket"

    def __init__(self, objects: dict[str, bytes]):
        self.objects = objects
        self.opened_files: list[io.BytesIO] = []

    def blob(self, name: str) -> FakeBlob:
        return FakeBlob(name, self.objects, self.opened_files)


def test_completion_index_preserves_first_seen_noncontiguous_groups(tmp_path: Path) -> None:
    completions_path = tmp_path / "completions.jsonl"
    _write_completions(
        completions_path,
        [
            _completion("task-a", 1, "seconds", "framing-a", "steps"),
            _completion("task-b", 2, "minutes", "framing-a", "steps"),
            _completion("task-a", 1, "seconds", "framing-b", "steps"),
        ],
    )

    index = build_completion_index(completions_path)

    assert index.record_count == 3
    assert len(index.sha256) == 64
    assert [group.semantic_values for group in index.groups] == [
        ("task-a", 1, "seconds"),
        ("task-b", 2, "minutes"),
    ]
    assert index.groups[0].sample_indices == [0, 2]
    assert index.groups[0].output_metadata() == {
        "task": "task-a",
        "base_value": 1,
        "base_unit": "seconds",
        "template_metadata.prompt_framing": "<averaged>",
        "template_metadata.output_format": "steps",
        "sample_index": 0,
        "source_sample_count": 2,
    }


def test_missing_file_metadata_uses_only_included_samples(tmp_path: Path) -> None:
    completions_path = tmp_path / "completions.jsonl"
    _write_completions(
        completions_path,
        [
            _completion("task-a", 1, "seconds", "framing-a", "steps"),
            _completion("task-a", 1, "seconds", "framing-b", "checklist"),
        ],
    )
    group = build_completion_index(
        completions_path,
        retain_sample_metadata=True,
    ).groups[0]

    assert group.output_metadata([1]) == {
        "task": "task-a",
        "base_value": 1,
        "base_unit": "seconds",
        "template_metadata.prompt_framing": "framing-b",
        "template_metadata.output_format": "checklist",
        "sample_index": 1,
        "source_sample_count": 1,
        "expected_source_sample_count": 2,
    }


def test_limited_runs_get_a_noncanonical_output_name() -> None:
    assert limited_output_object("results/activations.pt", 3) == (
        "results/activations_first_3_matched_groups.pt"
    )


def test_extracts_only_requested_nodes_at_first_assistant_position() -> None:
    selected_node_groups = {
        "positive": [((1, "mlp_hidden"), 4), ((2, "z"), 7)],
    }
    allowed_nodes = requested_nodes(selected_node_groups)
    payload = _activation_payload(sample_index=5, first_value=3.0)
    payload["activations"]["z/2"] = {
        "node_indices": [2, 7],
        "values": torch.tensor(
            [
                [
                    [[20.0, 21.0], [7.0, 8.0]],
                    [[120.0, 121.0], [107.0, 108.0]],
                ]
            ]
        ),
    }

    vector, schema = extract_assistant_feature_vector(
        payload,
        allowed_nodes,
        expected_sample_index=5,
    )

    assert torch.equal(vector, torch.tensor([3.0, 7.0, 8.0]))
    assert schema == [
        {
            "activation_type": "mlp",
            "name": "mlp_hidden/1",
            "node_indices": [4],
            "value_shape": [1],
            "feature_start": 0,
            "feature_stop": 1,
        },
        {
            "activation_type": "attn",
            "name": "z/2",
            "node_indices": [7],
            "value_shape": [1, 2],
            "feature_start": 1,
            "feature_stop": 3,
        },
    ]


def test_missing_requested_node_is_not_silently_ignored() -> None:
    payload = _activation_payload(sample_index=0, first_value=1.0)

    with pytest.raises(ValueError, match="missing requested nodes"):
        extract_assistant_feature_vector(
            payload,
            {"mlp_hidden/1": {4, 123}},
        )


def test_selected_node_definition_is_downloaded_without_a_local_file() -> None:
    object_name = "nodes/final_node_list.pkl"
    bucket = FakeBucket(
        {object_name: pickle.dumps({"positive": {("mlp_hidden/1", 4)}})}
    )

    groups, digest = download_selected_node_groups(  # type: ignore[arg-type]
        bucket,
        object_name,
    )

    assert groups == {"positive": [((1, "mlp_hidden"), 4)]}
    assert len(digest) == 64
    assert bucket.opened_files[0].closed


def test_process_groups_matches_noncontiguous_samples_without_local_files(
    tmp_path: Path,
) -> None:
    completions_path = tmp_path / "completions.jsonl"
    _write_completions(
        completions_path,
        [
            _completion("task-a", 1, "seconds", "framing-a", "steps"),
            _completion("task-b", 2, "minutes", "framing-a", "steps"),
            _completion("task-a", 1, "seconds", "framing-b", "checklist"),
        ],
    )
    completion_index = build_completion_index(completions_path)
    activation_prefix = "source/results"
    objects = {
        activation_object_name(activation_prefix, 0): _serialized_payload(0, 1.0),
        activation_object_name(activation_prefix, 1): _serialized_payload(1, 10.0),
        activation_object_name(activation_prefix, 2): _serialized_payload(2, 5.0),
    }
    bucket = FakeBucket(objects)
    selected_node_groups = {"selected": [((1, "mlp_hidden"), 4)]}

    state = process_groups(
        bucket,  # type: ignore[arg-type]
        activation_prefix,
        completion_index.groups,
        requested_nodes(selected_node_groups),
        ProcessingState(),
        selected_node_groups=selected_node_groups,
        provenance={},
        run_fingerprint="test",
        output_object="unused.pt",
        download_workers=2,
        checkpoint_every_groups=0,
        skip_missing=False,
    )

    assert torch.equal(torch.stack(state.vectors), torch.tensor([[3.0], [10.0]]))
    assert state.source_sample_indices == [[0, 2], [1]]
    assert [row["source_sample_count"] for row in state.metadata] == [2, 1]
    assert state.metadata[0]["template_metadata.prompt_framing"] == "<averaged>"
    assert state.metadata[0]["template_metadata.output_format"] == "<averaged>"
    assert bucket.opened_files
    assert all(opened_file.closed for opened_file in bucket.opened_files)
    assert "unused.pt" not in bucket.objects


def test_in_memory_checkpoint_round_trip_is_resumable(tmp_path: Path) -> None:
    completions_path = tmp_path / "completions.jsonl"
    _write_completions(
        completions_path,
        [_completion("task-a", 1, "seconds", "framing-a", "steps")],
    )
    groups = build_completion_index(completions_path).groups
    feature_schema = [
        {
            "activation_type": "mlp",
            "name": "mlp_hidden/1",
            "node_indices": [4],
            "value_shape": [1],
            "feature_start": 0,
            "feature_stop": 1,
        }
    ]
    state = ProcessingState(
        vectors=[torch.tensor([2.5])],
        metadata=[groups[0].output_metadata()],
        source_sample_indices=[[0]],
        feature_schema=feature_schema,
    )
    selected_node_groups = {"selected": [((1, "mlp_hidden"), 4)]}
    output_object = "results/checkpoint.pt"
    bucket = FakeBucket({})
    payload = build_output_payload(
        state,
        selected_node_groups,
        provenance={"test": True},
        run_fingerprint="fingerprint",
        status="in_progress",
        target_group_count=1,
    )

    upload_output_payload(bucket, output_object, payload)  # type: ignore[arg-type]
    resumed, is_complete = _load_existing_state(  # type: ignore[arg-type]
        bucket,
        output_object,
        "fingerprint",
        groups,
    )

    assert is_complete is False
    assert torch.equal(torch.stack(resumed.vectors), torch.tensor([[2.5]]))
    assert resumed.metadata == state.metadata
    assert resumed.source_sample_indices == [[0]]
    assert resumed.feature_schema == feature_schema
    assert all(opened_file.closed for opened_file in bucket.opened_files)
