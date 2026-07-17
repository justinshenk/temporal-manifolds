"""Tests for the content-addressed DataManager (no model required)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.datasets.data_manager import (
    DataConfig,
    DataManager,
    config_fingerprint,
    sanitize_model_name,
)

CONFIGS_DIR = Path(__file__).parent.parent / "configs" / "prompt_datasets"


@pytest.fixture
def dataset_cfg() -> dict:
    with open(CONFIGS_DIR / "test_minimal.json") as f:
        return json.load(f)


@pytest.fixture
def config(dataset_cfg: dict) -> DataConfig:
    return DataConfig(
        dataset_cfg=dataset_cfg,
        model="Qwen/Qwen3-4B-Instruct",
        seed=42,
        layers=[0],
        components=["resid_post"],
        positions=["time_horizon"],
    )


def test_fingerprint_is_deterministic_and_key_order_invariant(dataset_cfg: dict) -> None:
    reordered = dict(reversed(list(dataset_cfg.items())))
    assert config_fingerprint(dataset_cfg, 42) == config_fingerprint(reordered, 42)
    assert config_fingerprint(dataset_cfg, 42) != config_fingerprint(dataset_cfg, 43)
    assert len(config_fingerprint(dataset_cfg, 42)) == 12


def test_sanitize_model_name() -> None:
    assert sanitize_model_name("Qwen/Qwen3-4B-Instruct") == "Qwen__Qwen3-4B-Instruct"
    assert "/" not in sanitize_model_name("org/model:latest")


def test_directory_layout(config: DataConfig, tmp_path: Path) -> None:
    manager = DataManager(data_root=tmp_path)
    assert manager.prompt_dir(config) == tmp_path / config.prompt_fingerprint
    assert manager.model_dir(config) == (
        tmp_path / config.prompt_fingerprint / "Qwen__Qwen3-4B-Instruct"
    )


def test_config_json_roundtrip(config: DataConfig, tmp_path: Path) -> None:
    path = tmp_path / "experiment.json"
    with open(path, "w") as f:
        json.dump(config.to_dict(), f)
    loaded = DataConfig.from_json(path)
    assert loaded.prompt_fingerprint == config.prompt_fingerprint
    assert loaded.model == config.model
    assert [t.key for t in loaded.targets] == [t.key for t in config.targets]


def test_from_dict_accepts_dataset_path(tmp_path: Path) -> None:
    cfg = DataConfig.from_dict(
        {"dataset": str(CONFIGS_DIR / "test_minimal.json"), "model": "m"}
    )
    assert cfg.dataset_cfg["name"] == "test_minimal"


def test_prompt_dataset_is_generated_once_then_reused(
    config: DataConfig, tmp_path: Path
) -> None:
    manager = DataManager(data_root=tmp_path)

    dataset = manager.get_prompt_dataset(config)
    assert len(dataset.samples) > 0
    dataset_path = manager.prompt_dir(config) / "prompt_dataset.json"
    assert dataset_path.exists()
    assert (manager.prompt_dir(config) / "dataset_config.json").exists()

    # Second call must load the cached file, not regenerate it
    mtime = dataset_path.stat().st_mtime_ns
    reloaded = manager.get_prompt_dataset(config)
    assert dataset_path.stat().st_mtime_ns == mtime
    assert len(reloaded.samples) == len(dataset.samples)


def test_is_complete_gating(config: DataConfig, tmp_path: Path) -> None:
    manager = DataManager(data_root=tmp_path)
    assert not manager.is_complete(config)

    manager.model_dir(config).mkdir(parents=True)
    manager._write_manifest(config, status="in_progress")
    assert not manager.is_complete(config)

    manager._write_manifest(config, status="complete", n_samples=3)
    assert manager.is_complete(config)

    # Requesting targets beyond what the cache covers must invalidate it
    bigger = DataConfig(
        dataset_cfg=config.dataset_cfg,
        model=config.model,
        seed=config.seed,
        layers=[0, 1],
        components=config.components,
        positions=config.positions,
    )
    assert not manager.is_complete(bigger)

    # Different max_samples must invalidate too
    fewer = DataConfig(
        dataset_cfg=config.dataset_cfg,
        model=config.model,
        seed=config.seed,
        max_samples=2,
        layers=config.layers,
        components=config.components,
        positions=config.positions,
    )
    assert not manager.is_complete(fewer)
