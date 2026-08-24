"""Fit PCA + logistic regression to predict the four stakes levels.

Evaluation is grouped by task: every activation for a task is kept in the same
fold. This avoids measuring memorisation of task wording as generalisation.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

import joblib
import numpy as np
import torch
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix
from sklearn.model_selection import StratifiedGroupKFold, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


LABEL_ORDER = ["low", "medium", "high", "existential"]


def load_sample(
    input_dir: Path, max_rows_per_task: int, seed: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, int]]:
    """Reservoir-sample rows per task while streaming PyTorch batches."""
    rng = np.random.default_rng(seed)
    reservoirs: dict[str, list[np.ndarray]] = defaultdict(list)
    seen: Counter[str] = Counter()
    task_label: dict[str, str] = {}
    layer_component: str | None = None

    paths = sorted(input_dir.rglob("activations_batch_*.pt"))
    if not paths:
        raise ValueError(f"No activation batches found below {input_dir}")

    for file_index, path in enumerate(paths, start=1):
        payload = torch.load(path, map_location="cpu", weights_only=False)
        current_layer = str(payload["layer_component"])
        if layer_component is None:
            layer_component = current_layer
        elif current_layer != layer_component:
            raise ValueError(f"Mixed layer components: {layer_component!r} and {current_layer!r}")
        activations = payload["activations"][current_layer]
        if activations.ndim == 3 and activations.shape[1] == 1:
            activations = activations[:, 0, :]
        if activations.ndim != 2:
            raise ValueError(f"Expected [rows, features] activations in {path}, got {activations.shape}")

        values = activations.float().numpy()
        metadata = payload["prompt_metadata"]
        if len(values) != len(metadata):
            raise ValueError(f"Activation/metadata length mismatch in {path}")
        for row, item in zip(values, metadata, strict=True):
            task = str(item["task"])
            label = str(item["task_metadata"]["stakes"]).lower()
            if label not in LABEL_ORDER:
                continue
            previous = task_label.setdefault(task, label)
            if previous != label:
                raise ValueError(f"Task {task!r} has both {previous!r} and {label!r} labels")
            seen[task] += 1
            bucket = reservoirs[task]
            if len(bucket) < max_rows_per_task:
                bucket.append(row.copy())
            else:
                replacement = int(rng.integers(seen[task]))
                if replacement < max_rows_per_task:
                    bucket[replacement] = row.copy()
        if file_index % 200 == 0 or file_index == len(paths):
            print(f"Loaded {file_index}/{len(paths)} batches", flush=True)

    tasks = sorted(reservoirs)
    X = np.concatenate([np.stack(reservoirs[task]) for task in tasks]).astype(np.float32)
    groups = np.concatenate([np.repeat(task, len(reservoirs[task])) for task in tasks])
    y = np.concatenate([np.repeat(task_label[task], len(reservoirs[task])) for task in tasks])
    return X, y, groups, dict(seen)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=Path(".acts_filter_2"))
    parser.add_argument("--output-dir", type=Path, default=Path("results/stakes_classifier"))
    parser.add_argument("--rows-per-task", type=int, default=128)
    parser.add_argument("--components", type=int, default=32)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if args.rows_per_task < 1 or args.components < 1 or args.folds < 2:
        parser.error("rows-per-task and components must be positive; folds must be at least 2")

    X, y, groups, rows_seen = load_sample(args.input_dir, args.rows_per_task, args.seed)
    task_labels = {task: y[np.flatnonzero(groups == task)[0]] for task in np.unique(groups)}
    tasks_per_class = Counter(task_labels.values())
    if set(tasks_per_class) != set(LABEL_ORDER):
        raise ValueError(f"Expected all four stakes labels, found {dict(tasks_per_class)}")
    n_splits = min(args.folds, min(tasks_per_class.values()))
    n_components = min(args.components, X.shape[1], len(X) - 1)

    model = Pipeline(
        [
            ("pca", PCA(n_components=n_components, svd_solver="randomized", random_state=args.seed)),
            ("scale", StandardScaler()),
            ("classifier", LogisticRegression(max_iter=2_000, class_weight="balanced")),
        ]
    )
    splitter = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=args.seed)
    predicted = cross_val_predict(model, X, y, groups=groups, cv=splitter, method="predict")
    majority = Counter(y).most_common(1)[0][1] / len(y)
    metrics = {
        "accuracy": float(accuracy_score(y, predicted)),
        "balanced_accuracy": float(balanced_accuracy_score(y, predicted)),
        "majority_baseline_accuracy": float(majority),
        "labels": LABEL_ORDER,
        "confusion_matrix": confusion_matrix(y, predicted, labels=LABEL_ORDER).tolist(),
        "sampled_rows": int(len(X)),
        "activation_features": int(X.shape[1]),
        "pca_components": int(n_components),
        "cross_validation_folds": int(n_splits),
        "split_unit": "task",
        "tasks": int(len(task_labels)),
        "tasks_per_class": dict(tasks_per_class),
        "sampled_rows_per_class": dict(Counter(y)),
        "total_rows_seen": int(sum(rows_seen.values())),
        "max_rows_sampled_per_task": int(args.rows_per_task),
        "random_seed": int(args.seed),
    }
    print(json.dumps(metrics, indent=2))

    model.fit(X, y)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, args.output_dir / "pca_logistic.joblib")
    (args.output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
    np.savez_compressed(
        args.output_dir / "cross_validated_predictions.npz",
        actual=y,
        predicted=predicted,
        task=groups,
    )


if __name__ == "__main__":
    main()
