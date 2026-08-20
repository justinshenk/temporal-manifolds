"""Scorecards for a fitted horizon model, sliced the way the experiment is read.

A single headline RMSE hides the thing that matters here: because each task occupies its own
window of the horizon range, a model can look strong overall while being useless on the short
tasks. Reporting per task and per source folder makes that visible, and drives the decision
about which tasks to drop as outliers.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .evaluation import grouped_metrics, outlier_groups, regression_metrics


@dataclass(frozen=True)
class Scorecards:
    """Overall, per-folder and per-task metrics for one set of out-of-fold predictions."""

    overall: dict[str, float]
    by_folder: pd.DataFrame
    by_task: pd.DataFrame
    by_fold: pd.DataFrame
    by_unit: pd.DataFrame
    predictions: pd.DataFrame

    def summary(self, *, top_tasks: int = 10) -> str:
        overall = ", ".join(
            f"{key}={value:.4f}" if isinstance(value, float) else f"{key}={value}"
            for key, value in self.overall.items()
        )
        return "\n".join(
            [
                f"OVERALL  {overall}",
                "",
                "PER FOLDER",
                self.by_folder.to_string(index=False),
                "",
                "PER FOLD",
                self.by_fold.to_string(index=False),
                "",
                f"PER TASK (worst {top_tasks} of {len(self.by_task)})",
                self.by_task.head(top_tasks).to_string(index=False),
            ]
        )


def build_scorecards(
    metadata: pd.DataFrame, actual: np.ndarray, predicted: np.ndarray, folds: np.ndarray
) -> Scorecards:
    """Assemble every slice of the error report from out-of-fold predictions."""
    frame = metadata.assign(
        actual=np.asarray(actual, dtype=np.float64),
        predicted=np.asarray(predicted, dtype=np.float64),
        fold=np.asarray(folds),
    )
    frame["error"] = frame["predicted"] - frame["actual"]
    return Scorecards(
        overall=regression_metrics(frame["actual"], frame["predicted"]),
        by_folder=grouped_metrics(frame, "source_folder"),
        by_task=grouped_metrics(frame, "task"),
        by_fold=grouped_metrics(frame, "fold").sort_values("fold").reset_index(drop=True),
        by_unit=grouped_metrics(frame, "base_unit"),
        predictions=frame,
    )


def task_folder_matrix(scorecards: Scorecards, metric: str = "rmse") -> pd.DataFrame:
    """Cross-tabulate one metric by task and source folder."""
    per_cell = grouped_metrics(scorecards.predictions, ["task", "source_folder"])
    return per_cell.pivot(index="task", columns="source_folder", values=metric)


def flag_outlier_tasks(scorecards: Scorecards, *, tolerance: float = 3.0) -> list[str]:
    """Tasks whose RMSE is a robust outlier against the rest of the task scorecard."""
    return outlier_groups(scorecards.by_task, key="task", tolerance=tolerance)
