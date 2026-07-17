"""Content-addressed management of generated samples + activations.

Experiments are configured with a JSON file (see ``DataConfig``) and their
outputs are cached on disk keyed by *what produced them*, so re-running with
the same config is a no-op load instead of a regeneration:

    data/
      <prompt_hash>/                  # fingerprint of dataset config + seed
        dataset_config.json           # exact config that produced the prompts
        prompt_dataset.json           # generated prompts (model-independent,
                                      #   shared by every model below)
        <model_name>/                 # sanitized model id, e.g. Qwen__Qwen3-4B
          manifest.json               # completion marker + provenance
          data/
            metadata.json             # ActivationData metadata
            samples/
              sample_0/               # per-sample activations + json
                position_mapping.json
                prompt_sample.json
                preference_sample.json
                choice.json
                L{layer}/{component}_{abs_pos}.npy

The fingerprint is a sha256 over the canonical (sorted-key) JSON of the
dataset config and seed — the same content-addressing idea used by
HuggingFace datasets' fingerprinting. Anything that changes the prompts
changes the hash; anything that doesn't (e.g. which model we extract with)
lives one level below it.

Crash-safety: ``extract_activations`` resumes per-sample, and the manifest is
only marked "complete" after extraction finishes, so an interrupted run picks
up where it left off on the next call.
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

from ..common.project_paths import get_project_root
from ..geometry.geometry_config import GeometryConfig, TargetSpec
from ..geometry.geometry_data import (
    ActivationData,
    extract_activations,
    load_cached_data,
)
from ..geometry.geometry_utils import COMPONENTS, LAYERS, POSITIONS
from .prompt import PromptDataset, PromptDatasetConfig, PromptDatasetGenerator

logger = logging.getLogger(__name__)

MANIFEST_NAME = "manifest.json"
DATASET_CONFIG_NAME = "dataset_config.json"
PROMPT_DATASET_NAME = "prompt_dataset.json"
FINGERPRINT_LENGTH = 12


# =============================================================================
# Fingerprinting
# =============================================================================


def canonical_json(obj: dict) -> str:
    """Serialize a dict deterministically (sorted keys, no whitespace)."""
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), default=str)


def config_fingerprint(dataset_cfg: dict, seed: int) -> str:
    """Content hash identifying the prompt samples a config+seed produces."""
    payload = canonical_json({"dataset_cfg": dataset_cfg, "seed": seed})
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:FINGERPRINT_LENGTH]


def sanitize_model_name(model: str) -> str:
    """Turn a model identifier into a safe directory name.

    "Qwen/Qwen3-4B-Instruct" -> "Qwen__Qwen3-4B-Instruct"
    """
    name = model.replace("/", "__")
    return re.sub(r"[^A-Za-z0-9._-]+", "-", name)


# =============================================================================
# Experiment data configuration (JSON-friendly)
# =============================================================================


@dataclass
class DataConfig:
    """Everything that determines one samples+activations dataset.

    JSON format (all fields except ``dataset`` and ``model`` optional):

        {
          "dataset": {...inline dataset config...}  // or a path string to a
                                                    // dataset-config JSON
          "model": "Qwen/Qwen3-4B-Instruct",
          "seed": 42,
          "max_samples": null,
          "layers": [0, 4, 8],          // default: geometry_utils.LAYERS
          "components": ["resid_post"], // default: geometry_utils.COMPONENTS
          "positions": ["time_horizon"] // default: geometry_utils.POSITIONS
        }
    """

    dataset_cfg: dict
    model: str
    seed: int = 42
    max_samples: int | None = None
    layers: list[int] = field(default_factory=lambda: list(LAYERS))
    components: list[str] = field(default_factory=lambda: list(COMPONENTS))
    positions: list[str] = field(default_factory=lambda: list(POSITIONS))

    @property
    def targets(self) -> list[TargetSpec]:
        """All layer/component/position combinations to extract."""
        return [
            TargetSpec(layer=layer, component=component, position=position)
            for layer in self.layers
            for component in self.components
            for position in self.positions
        ]

    @property
    def prompt_fingerprint(self) -> str:
        return config_fingerprint(self.dataset_cfg, self.seed)

    @property
    def model_dir_name(self) -> str:
        return sanitize_model_name(self.model)

    @classmethod
    def from_dict(cls, d: dict, **overrides) -> "DataConfig":
        """Build from a JSON-style dict; ``overrides`` win over dict values."""
        d = {**d, **{k: v for k, v in overrides.items() if v is not None}}

        dataset = d["dataset"]
        if isinstance(dataset, (str, Path)):
            with open(dataset) as f:
                dataset = json.load(f)

        kwargs = dict(dataset_cfg=dataset, model=d["model"])
        for key in ("seed", "max_samples", "layers", "components", "positions"):
            if d.get(key) is not None:
                kwargs[key] = d[key]
        return cls(**kwargs)

    @classmethod
    def from_json(cls, path: str | Path, **overrides) -> "DataConfig":
        with open(path) as f:
            return cls.from_dict(json.load(f), **overrides)

    def to_dict(self) -> dict:
        return {
            "dataset": self.dataset_cfg,
            "model": self.model,
            "seed": self.seed,
            "max_samples": self.max_samples,
            "layers": list(self.layers),
            "components": list(self.components),
            "positions": list(self.positions),
        }


# =============================================================================
# DataManager
# =============================================================================


class DataManager:
    """Generate-or-load samples and activations, content-addressed on disk.

    Usage:
        manager = DataManager()
        config = DataConfig.from_json("configs/experiments/my_run.json")
        data = manager.get_or_generate(config)   # generates only if missing
    """

    def __init__(self, data_root: str | Path | None = None):
        self.data_root = (
            Path(data_root) if data_root is not None else get_project_root() / "data"
        )

    # -- paths ---------------------------------------------------------------

    def prompt_dir(self, config: DataConfig) -> Path:
        """data/<prompt_hash>/ — shared across models."""
        return self.data_root / config.prompt_fingerprint

    def model_dir(self, config: DataConfig) -> Path:
        """data/<prompt_hash>/<model_name>/ — samples + activations."""
        return self.prompt_dir(config) / config.model_dir_name

    def manifest_path(self, config: DataConfig) -> Path:
        return self.model_dir(config) / MANIFEST_NAME

    # -- status --------------------------------------------------------------

    def read_manifest(self, config: DataConfig) -> dict | None:
        path = self.manifest_path(config)
        if not path.exists():
            return None
        with open(path) as f:
            return json.load(f)

    def is_complete(self, config: DataConfig) -> bool:
        """True if a finished extraction exists covering all requested targets."""
        manifest = self.read_manifest(config)
        if manifest is None or manifest.get("status") != "complete":
            return False
        requested = {t.key for t in config.targets}
        available = set(manifest.get("targets", []))
        missing = requested - available
        if missing:
            logger.info(
                f"Cache at {self.model_dir(config)} is complete but missing "
                f"{len(missing)} requested targets; will regenerate."
            )
            return False
        if manifest.get("max_samples") != config.max_samples:
            logger.info(
                f"Cache exists with max_samples={manifest.get('max_samples')} "
                f"(requested {config.max_samples}); will regenerate."
            )
            return False
        return True

    # -- prompt dataset (model-independent) -----------------------------------

    def get_prompt_dataset(self, config: DataConfig) -> PromptDataset:
        """Load the shared prompt dataset for this fingerprint, or generate it."""
        prompt_dir = self.prompt_dir(config)
        dataset_path = prompt_dir / PROMPT_DATASET_NAME

        if dataset_path.exists():
            logger.info(f"Loading cached prompt dataset: {dataset_path}")
            return PromptDataset.from_json(dataset_path)

        logger.info(
            f"Generating prompt dataset "
            f"(config: {config.dataset_cfg.get('name', 'unnamed')}, "
            f"fingerprint: {config.prompt_fingerprint})"
        )
        dataset_config = PromptDatasetConfig.from_dict(config.dataset_cfg)
        dataset = PromptDatasetGenerator(dataset_config).generate()

        prompt_dir.mkdir(parents=True, exist_ok=True)
        with open(prompt_dir / DATASET_CONFIG_NAME, "w") as f:
            json.dump(
                {"dataset_cfg": config.dataset_cfg, "seed": config.seed},
                f,
                indent=2,
                default=str,
            )
        dataset.save_as_json(dataset_path)
        logger.info(f"Saved {len(dataset.samples)} prompts to {dataset_path}")
        return dataset

    # -- main entry point ------------------------------------------------------

    def get_or_generate(self, config: DataConfig, force: bool = False) -> ActivationData:
        """Return activations for this config, generating them only if needed.

        Args:
            config: What to generate (dataset config, model, targets, ...).
            force: Regenerate even if a complete cache exists.

        Returns:
            ActivationData backed by data/<prompt_hash>/<model_name>/data/.
        """
        model_dir = self.model_dir(config)
        geometry_config = self._geometry_config(config)

        if not force and self.is_complete(config):
            logger.info(f"Cache hit: {model_dir} (skipping generation)")
            data = load_cached_data(geometry_config)
            if data is not None:
                return data
            logger.warning(
                f"Manifest says complete but data is unreadable at {model_dir}; "
                f"regenerating."
            )

        dataset = self.get_prompt_dataset(config)

        model_dir.mkdir(parents=True, exist_ok=True)
        self._write_manifest(config, status="in_progress")

        data = extract_activations(dataset, config.targets, geometry_config)

        self._write_manifest(
            config,
            status="complete",
            n_samples=len(data.samples),
            targets_with_data=sorted(data.get_target_keys()),
        )
        logger.info(f"Generated {len(data.samples)} samples at {model_dir}")
        return data

    def load(self, config: DataConfig) -> ActivationData | None:
        """Load cached activations without ever generating. None if missing."""
        return load_cached_data(self._geometry_config(config))

    # -- internals -------------------------------------------------------------

    def _geometry_config(self, config: DataConfig) -> GeometryConfig:
        return GeometryConfig(
            targets=config.targets,
            output_dir=self.model_dir(config),
            model=config.model,
            seed=config.seed,
            max_samples=config.max_samples,
            dataset_cfg=config.dataset_cfg,
        )

    def _write_manifest(
        self,
        config: DataConfig,
        status: str,
        n_samples: int | None = None,
        targets_with_data: list[str] | None = None,
    ) -> None:
        manifest = {
            "status": status,
            "prompt_fingerprint": config.prompt_fingerprint,
            "model": config.model,
            "seed": config.seed,
            "max_samples": config.max_samples,
            "layers": list(config.layers),
            "components": list(config.components),
            "positions": list(config.positions),
            "targets": [t.key for t in config.targets],
            "n_samples": n_samples,
            "targets_with_data": targets_with_data,
            "updated_at": datetime.now().isoformat(timespec="seconds"),
        }
        with open(self.manifest_path(config), "w") as f:
            json.dump(manifest, f, indent=2)
