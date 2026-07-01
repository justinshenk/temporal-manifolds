"""Artifact builders for the Q&A EAP-IG workflow."""

from __future__ import annotations

import pickle
import re
from collections import defaultdict
from pathlib import Path
from typing import Any, TypeAlias

import numpy as np

from temporal_manifolds.utils.artifact_io import write_json, write_pickle

MergedArrayGroup: TypeAlias = dict[str, np.ndarray]
PromptFormatResults: TypeAlias = dict[str, MergedArrayGroup]
CaseResults: TypeAlias = dict[str, PromptFormatResults]
TopComponent: TypeAlias = tuple[str, int, float]
SelectedNode: TypeAlias = tuple[str, int]

DEFAULT_CASES = ("explicit", "implicit")
COMPONENT_ARRAY_NAMES = frozenset({"z", "mlp_hidden"})
LOGIT_ARRAY_NAMES = frozenset({"clean_logits", "corrupted_logits"})
NPZ_FILENAME_PATTERN = re.compile(
    r"^(?P<stem>.+)_(?P<option_order>short_first|long_first)_(?P<metric>logit_[AB])_batch_\d+\.npz$"
)


def extract_merged_array_name(array_key: str) -> str | None:
    """Map a saved NPZ key to the merged output key consumed downstream."""
    parts = array_key.split("__")
    if parts[-1] in LOGIT_ARRAY_NAMES:
        return parts[-1]
    if len(parts) >= 3 and parts[-2] in COMPONENT_ARRAY_NAMES:
        return f"{parts[-2]}/{parts[-1]}"
    return None


def process_eap_ig_result_folder(folder: Path) -> PromptFormatResults:
    """Merge batched NPZ outputs for a single prompt-format folder."""
    grouped_files: dict[str, list[Path]] = defaultdict(list)
    for file_path in sorted(folder.glob("*.npz")):
        match = NPZ_FILENAME_PATTERN.match(file_path.name)
        if not match:
            continue
        group_name = f"{match.group('option_order')}_{match.group('metric')}"
        grouped_files[group_name].append(file_path)

    merged_results: PromptFormatResults = {}
    for group_name, file_paths in grouped_files.items():
        merged_arrays: dict[str, list[np.ndarray]] = defaultdict(list)
        for file_path in file_paths:
            with np.load(file_path, allow_pickle=False) as data:
                for array_key in data.files:
                    merged_name = extract_merged_array_name(array_key)
                    if merged_name is None:
                        # Metadata is useful at the raw file level, but the
                        # downstream selection code only consumes merged logits
                        # and component tensors.
                        continue
                    merged_arrays[merged_name].append(data[array_key])

        if merged_arrays:
            merged_results[group_name] = {
                merged_name: np.concatenate(arrays, axis=0)
                for merged_name, arrays in merged_arrays.items()
            }

    return merged_results


def load_case_results(
    results_dir: Path,
    case_name: str,
) -> CaseResults:
    """Load all Q&A prompt-format folders for a case like ``explicit``."""
    case_dir = results_dir / case_name
    if not case_dir.exists():
        raise FileNotFoundError(f"Missing case directory: {case_dir}")

    case_results: CaseResults = {}
    for folder in sorted(case_dir.iterdir(), key=lambda path: path.name.lower()):
        if not folder.is_dir() or folder.name == "short":
            continue
        folder_results = process_eap_ig_result_folder(folder)
        if folder_results:
            case_results[folder.name] = folder_results

    if not case_results:
        raise FileNotFoundError(f"No Q&A result folders found in {case_dir}")

    return case_results


def get_denom(group: MergedArrayGroup, logit_index: int) -> np.ndarray:
    """Match the notebook's normalization denominator."""
    denom = np.abs(
        group["clean_logits"][:, logit_index] - group["corrupted_logits"][:, logit_index]
    )
    return np.where(denom < 0.5, np.nan, denom)


def get_attr_normalized(
    group: MergedArrayGroup,
    logit_index: int,
) -> dict[str, np.ndarray]:
    """Normalize component attribution arrays by the logit delta magnitude."""
    denom = get_denom(group, logit_index).reshape(-1, 1)
    with np.errstate(invalid="ignore", divide="ignore"):
        return {
            key: values / denom for key, values in group.items() if key not in LOGIT_ARRAY_NAMES
        }


def aggregate_case_scores(
    case_results: CaseResults,
) -> dict[str, dict[str, np.ndarray]]:
    """Aggregate attribution scores exactly as in ``datasetwise_top_n.ipynb``."""
    aggregated: dict[str, dict[str, np.ndarray]] = {}

    for horizon in ("ST", "LT"):
        net_scores: dict[str, list[np.ndarray]] = defaultdict(list)

        for variant_results in case_results.values():
            if horizon == "ST":
                # The original notebook defines short-term salience by averaging
                # long-first answers scored on logit_B and short-first answers
                # scored on logit_A.
                long_first = get_attr_normalized(
                    variant_results["long_first_logit_B"],
                    logit_index=1,
                )
                short_first = get_attr_normalized(
                    variant_results["short_first_logit_A"],
                    logit_index=0,
                )
            else:
                long_first = get_attr_normalized(
                    variant_results["long_first_logit_A"],
                    logit_index=0,
                )
                short_first = get_attr_normalized(
                    variant_results["short_first_logit_B"],
                    logit_index=1,
                )

            for component_name in long_first:
                net_scores[component_name].append(
                    (long_first[component_name] + short_first[component_name]) / 2
                )

        aggregated[horizon] = {
            component_name: np.nanmean(
                np.stack(component_values, axis=-1),
                axis=(0, -1),
            )
            for component_name, component_values in net_scores.items()
        }

    return aggregated


def topk_components(
    component_dict: dict[str, np.ndarray],
    k: int,
) -> list[TopComponent]:
    """Return the top-k components by absolute attribution magnitude."""
    if not component_dict or k <= 0:
        return []

    component_names: list[str] = []
    flattened_arrays: list[np.ndarray] = []
    for component_name, values in component_dict.items():
        component_names.append(component_name)
        flattened_arrays.append(np.asarray(values).ravel())

    concatenated = np.concatenate(flattened_arrays)
    component_ids = np.concatenate(
        [
            np.full(len(component_values), index, dtype=np.int32)
            for index, component_values in enumerate(flattened_arrays)
        ]
    )
    local_indices = np.concatenate(
        [np.arange(len(component_values), dtype=np.int32) for component_values in flattened_arrays]
    )

    k = min(k, len(concatenated))
    absolute_values = np.abs(concatenated)
    topk_indices = np.argpartition(absolute_values, -k)[-k:]
    topk_indices = topk_indices[np.argsort(absolute_values[topk_indices])[::-1]]

    return [
        (
            component_names[component_ids[index]],
            int(local_indices[index]),
            float(concatenated[index]),
        )
        for index in topk_indices
    ]


def serialize_top_components(
    top_components: dict[str, list[TopComponent]],
) -> dict[str, list[dict[str, Any]]]:
    """Convert top-component tuples to JSON-friendly records."""
    return {
        horizon: [
            {"component": component, "index": index, "score": score}
            for component, index, score in entries
        ]
        for horizon, entries in top_components.items()
    }


def expected_top_components_path(
    top_components_dir: Path,
    top_n: int,
    case_name: str,
) -> Path:
    """Return the expected pickle output path for a case."""
    return top_components_dir / f"top_{top_n}_components_{case_name}.pkl"


def build_top_components_for_case(
    *,
    case_name: str,
    results_dir: Path,
    top_components_dir: Path,
    completeness_figures_dir: Path | None,
    top_n: int,
) -> tuple[Path, Path | None]:
    """Compute, plot, and save top components for a single case."""
    case_results = load_case_results(results_dir, case_name)

    if completeness_figures_dir is not None:
        from temporal_manifolds.viz.eap_ig_plots import plot_case_completeness

        figure_path = plot_case_completeness(
            case_name,
            case_results,
            completeness_figures_dir / f"{case_name}_completeness.png",
        )
    else:
        figure_path = None

    aggregated_scores = aggregate_case_scores(case_results)
    top_components = {
        horizon: topk_components(aggregated_scores[horizon], top_n) for horizon in aggregated_scores
    }

    pickle_path = expected_top_components_path(top_components_dir, top_n, case_name)
    write_pickle(pickle_path, top_components)
    write_json(
        pickle_path.with_suffix(".json"),
        serialize_top_components(top_components),
    )

    return pickle_path, figure_path


def build_top_components(
    *,
    results_dir: Path,
    top_components_dir: Path,
    completeness_figures_dir: Path,
    top_n: int,
    compute_completeness: bool = True,
) -> tuple[dict[str, Path], dict[str, Path]]:
    """Build top-component artifacts for both explicit and implicit cases."""
    pickles: dict[str, Path] = {}
    figures: dict[str, Path] = {}
    for case_name in DEFAULT_CASES:
        pickle_path, figure_path = build_top_components_for_case(
            case_name=case_name,
            results_dir=results_dir,
            top_components_dir=top_components_dir,
            completeness_figures_dir=completeness_figures_dir if compute_completeness else None,
            top_n=top_n,
        )
        pickles[case_name] = pickle_path
        if figure_path is not None:
            figures[case_name] = figure_path
    return pickles, figures


def load_top_components_for_selection(
    top_components_dir: Path,
    top_n: int,
) -> dict[str, dict[str, list[TopComponent]]]:
    """Load explicit and implicit top-component pickles for node selection."""
    loaded: dict[str, dict[str, list[TopComponent]]] = {}
    for case_name in DEFAULT_CASES:
        path = expected_top_components_path(top_components_dir, top_n, case_name)
        with path.open("rb") as f:
            loaded[case_name] = pickle.load(f)
    return loaded


def select_signed_nodes(
    components: list[TopComponent],
    selection_limit: int,
    *,
    positive: bool,
) -> set[SelectedNode]:
    """Take the notebook's first-``selection_limit`` components filtered by sign."""
    score_predicate = (lambda score: score > 0) if positive else (lambda score: score < 0)
    return {
        (component, index)
        for component, index, score in components[:selection_limit]
        if score_predicate(score)
    }


def build_selected_node_groups(
    top_components_dir: Path,
    *,
    top_n: int,
    selection_limit: int,
) -> dict[str, set[SelectedNode]]:
    """Reproduce the node grouping from ``node_selection.ipynb``."""
    top_components = load_top_components_for_selection(top_components_dir, top_n)

    def selected(case_name: str, horizon: str, *, positive: bool) -> set[SelectedNode]:
        return select_signed_nodes(
            top_components[case_name][horizon],
            selection_limit,
            positive=positive,
        )

    p_nodes_exp_lt = selected("explicit", "LT", positive=True)
    p_nodes_exp_st = selected("explicit", "ST", positive=True)
    p_nodes_imp_lt = selected("implicit", "LT", positive=True)
    p_nodes_imp_st = selected("implicit", "ST", positive=True)

    n_nodes_exp_lt = selected("explicit", "LT", positive=False)
    n_nodes_exp_st = selected("explicit", "ST", positive=False)
    n_nodes_imp_lt = selected("implicit", "LT", positive=False)
    n_nodes_imp_st = selected("implicit", "ST", positive=False)

    p_generic = p_nodes_exp_lt & p_nodes_exp_st & p_nodes_imp_lt & p_nodes_imp_st
    n_generic = n_nodes_exp_lt & n_nodes_exp_st & n_nodes_imp_lt & n_nodes_imp_st

    p_nodes_exp_lt = p_nodes_exp_lt - p_generic
    p_nodes_imp_lt = p_nodes_imp_lt - p_generic
    n_nodes_exp_lt = n_nodes_exp_lt - n_generic
    n_nodes_imp_lt = n_nodes_imp_lt - n_generic
    p_nodes_exp_st = p_nodes_exp_st - p_generic
    p_nodes_imp_st = p_nodes_imp_st - p_generic
    n_nodes_exp_st = n_nodes_exp_st - n_generic
    n_nodes_imp_st = n_nodes_imp_st - n_generic

    p_common_lt = p_nodes_exp_lt & p_nodes_imp_lt
    n_common_lt = n_nodes_exp_lt & n_nodes_imp_lt
    p_common_st = p_nodes_exp_st & p_nodes_imp_st
    n_common_st = n_nodes_exp_st & n_nodes_imp_st

    sym_lt_p = p_common_lt & n_common_st
    sym_st_p = p_common_st & n_common_lt

    p_common_lt = p_common_lt - sym_lt_p
    n_common_st = n_common_st - sym_lt_p

    p_common_st = p_common_st - sym_st_p
    n_common_lt = n_common_lt - sym_st_p

    return {
        "p_generic": p_generic,
        "n_generic": n_generic,
        "sym_LT_p": sym_lt_p,
        "sym_ST_p": sym_st_p,
        "p_common_LT": p_common_lt,
        "p_common_ST": p_common_st,
        "n_common_LT": n_common_lt,
        "n_common_ST": n_common_st,
    }


def serialize_selected_node_groups(
    selected_nodes: dict[str, set[SelectedNode]],
) -> dict[str, list[dict[str, Any]]]:
    """Convert selected-node sets to JSON-friendly records."""
    return {
        node_class: [{"component": component, "index": index} for component, index in sorted(nodes)]
        for node_class, nodes in selected_nodes.items()
    }


def expected_selected_nodes_path(
    selected_nodes_dir: Path,
    top_n: int,
    *,
    artifact_label: str = "eap_ig",
) -> Path:
    """Return the expected selected-nodes pickle path."""
    return selected_nodes_dir / f"final_{top_n}_{artifact_label}.pkl"


def legacy_selected_nodes_path(selected_nodes_dir: Path, top_n: int) -> Path:
    """Return the legacy selected-nodes pickle path."""
    return selected_nodes_dir / f"final_{top_n}_QnA.pkl"


def resolve_selected_nodes_path(
    selected_nodes_dir: Path,
    top_n: int,
    *,
    artifact_label: str = "eap_ig",
) -> Path:
    """Return the preferred selected-node artifact, falling back to the legacy name."""
    preferred_path = expected_selected_nodes_path(
        selected_nodes_dir,
        top_n,
        artifact_label=artifact_label,
    )
    if preferred_path.exists():
        return preferred_path

    legacy_path = legacy_selected_nodes_path(selected_nodes_dir, top_n)
    if legacy_path.exists():
        return legacy_path

    return preferred_path


def write_selected_nodes(
    *,
    top_components_dir: Path,
    selected_nodes_dir: Path,
    top_n: int,
    selection_limit: int,
    artifact_label: str = "eap_ig",
) -> tuple[Path, Path]:
    """Save the notebook-equivalent node-selection artifacts."""
    selected_nodes = build_selected_node_groups(
        top_components_dir,
        top_n=top_n,
        selection_limit=selection_limit,
    )

    pickle_path = expected_selected_nodes_path(
        selected_nodes_dir,
        top_n,
        artifact_label=artifact_label,
    )
    json_path = pickle_path.with_suffix(".json")
    write_pickle(pickle_path, selected_nodes)
    write_json(json_path, serialize_selected_node_groups(selected_nodes))
    return pickle_path, json_path
