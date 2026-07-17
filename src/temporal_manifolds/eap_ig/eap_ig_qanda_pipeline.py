"""
Execution pipeline for the Q&A EAP-IG workflow.
"""

import gc
import json
from pathlib import Path
from typing import Any, Literal

import numpy as np
import torch
from tqdm import tqdm

try:
    from ..utils.gcs_upload import (
        gcs_object_name_for_file,
        maybe_build_gcs_existing_object_fetcher,
        maybe_start_gcs_upload_worker,
    )
    from .eap_ig_qanda_common import (
        GCP_PROJECT_ID,
        GCS_BUCKET_NAME,
        build_layer_components,
        chunk_list,
        extract_alnum,
        load_and_merge_pairs,
        load_config,
        resolve_config_path,
        resolve_quadrature,
        tensor_to_numpy,
    )
except ImportError:
    from temporal_manifolds.eap_ig.eap_ig_qanda_common import (
        GCP_PROJECT_ID,
        GCS_BUCKET_NAME,
        build_layer_components,
        chunk_list,
        extract_alnum,
        load_and_merge_pairs,
        load_config,
        resolve_config_path,
        resolve_quadrature,
        tensor_to_numpy,
    )
    from temporal_manifolds.utils.gcs_upload import (
        gcs_object_name_for_file,
        maybe_build_gcs_existing_object_fetcher,
        maybe_start_gcs_upload_worker,
    )


def build_metrics(token_a: int, token_b: int) -> dict[str, Any]:
    """Build metric functions from the resolved option tokens."""
    return {
        "logit_A": lambda logits: logits[:, -1, token_a].mean(),
        "logit_B": lambda logits: logits[:, -1, token_b].mean(),
    }


def resolve_option_tokens(tokenizer: Any, option_keys: list[str]) -> tuple[int, int]:
    """Resolve option keys into single-token ids."""
    token_a_str = extract_alnum(option_keys[0])
    token_b_str = extract_alnum(option_keys[1])

    token_a = tokenizer.tokenizer.encode(token_a_str, add_special_tokens=False)
    token_b = tokenizer.tokenizer.encode(token_b_str, add_special_tokens=False)

    if len(token_a) != 1 or len(token_b) != 1:
        raise ValueError(
            f"Token A tokenizes to {token_a}\nToken B tokenizes to {token_b}\n"
            "Optionkeys must tokenize to single token each."
        )
    return token_a[0], token_b[0]


def build_option_orders(
    input_file_path: Path,
    template: str,
    option_keys: list[str],
    batch_size: int,
) -> list[tuple[str, list[list[str]], list[list[str]]]]:
    """Build prompt batches for both option orders."""
    all_clean_prompts, all_corrupted_prompts = load_and_merge_pairs(
        input_file_path,
        template=template,
        option_keys=option_keys,
        text_order=["question", "immediate", "long_term"],
    )

    all_corrupted_prompts_swapped, all_clean_prompts_swapped = load_and_merge_pairs(
        input_file_path,
        template=template,
        option_keys=option_keys,
        text_order=["question", "long_term", "immediate"],
    )

    return [
        (
            "short_first",
            chunk_list(all_clean_prompts, batch_size),
            chunk_list(all_corrupted_prompts, batch_size),
        ),
        (
            "long_first",
            chunk_list(all_clean_prompts_swapped, batch_size),
            chunk_list(all_corrupted_prompts_swapped, batch_size),
        ),
    ]


def expected_output_files(
    save_loc: Path,
    filename: str,
    option_orders: list[tuple[str, list[list[str]], list[list[str]]]],
) -> list[Path]:
    """Return the output files produced by a Q&A attribution run."""
    output_files = []
    for order_label, chunked_clean_prompts, _chunked_corrupted_prompts in option_orders:
        for metric_label in ("logit_A", "logit_B"):
            for i in range(len(chunked_clean_prompts)):
                output_files.append(
                    save_loc / f"{filename}_{order_label}_{metric_label}_batch_{i:05d}.npz"
                )
    return output_files


def existing_output_matches_run(
    output_file: Path,
    *,
    attribution_method: Literal["eap_ig", "eap"],
    compute_gradient_at: Literal["clean", "corrupted"],
) -> bool:
    """Return whether an existing output can be reused for this run."""
    if not output_file.exists():
        return False
    if attribution_method != "eap":
        return True

    try:
        with np.load(output_file, allow_pickle=False) as data:
            saved_method = str(data["metadata__attribution_method"].item())
            saved_gradient_side = str(data["metadata__compute_gradient_at"].item())
    except (KeyError, OSError, ValueError):
        # Older vanilla-EAP artifacts did not record the gradient side. They
        # were produced with the historical default, so only clean runs may
        # safely resume from them.
        return compute_gradient_at == "clean"

    return saved_method == attribution_method and saved_gradient_side == compute_gradient_at


def process_batch(
    *,
    model: Any,
    expand_mask: Any,
    attribution_fn: Any,
    attribution_method: Literal["eap_ig", "eap"],
    tokenized_clean_batch: dict[str, Any],
    tokenized_corrupted_batch: dict[str, Any],
    metric_fn: Any,
    layer_components: list[tuple[int, str]],
    steps: list[int],
    quadrature: str,
    system_prompt_length: int,
    token_a: int,
    token_b: int,
    config: dict,
    order_label: str,
    metric_type: str,
    compute_gradient_at: Literal["clean", "corrupted"] = "clean",
) -> dict[str, np.ndarray]:
    """Run all configured step counts for one tokenized prompt batch."""
    batch_output: dict[str, np.ndarray] = {}
    batch_output["metadata__config_json"] = np.array(json.dumps(config), dtype=np.str_)
    batch_output["metadata__option_order"] = np.array(order_label, dtype=np.str_)
    batch_output["metadata__metric_type"] = np.array(metric_type, dtype=np.str_)
    batch_output["metadata__attribution_method"] = np.array(attribution_method, dtype=np.str_)
    batch_output["metadata__compute_gradient_at"] = np.array(compute_gradient_at, dtype=np.str_)

    step_labels = steps if attribution_method == "eap_ig" else steps[:1]

    for num_steps in tqdm(step_labels, desc="Step counts", leave=False):
        clean_inputs = {
            k: v.clone() if isinstance(v, torch.Tensor) else v
            for k, v in tokenized_clean_batch.items()
        }
        corrupted_inputs = {
            k: v.clone() if isinstance(v, torch.Tensor) else v
            for k, v in tokenized_corrupted_batch.items()
        }

        if attribution_method == "eap_ig":
            eap_scores, (clean_logits, corrupted_logits) = attribution_fn(
                model,
                clean_inputs,
                corrupted_inputs,
                metric_fn,
                layer_components,
                steps=num_steps,
                include_block_outputs=True,
                quadrature=quadrature,
            )
        else:
            eap_scores, (clean_logits, corrupted_logits) = attribution_fn(
                model,
                clean_inputs,
                corrupted_inputs,
                compute_grad_at=compute_gradient_at,
                metric_fn=metric_fn,
                layer_components=layer_components,
                include_block_outputs=True,
            )

        clean_logits_cpu = clean_logits[:, -1, [token_a, token_b]].detach().cpu()
        corrupted_logits_cpu = corrupted_logits[:, -1, [token_a, token_b]].detach().cpu()
        del clean_logits, corrupted_logits

        eap_scores.attention_mask = expand_mask(
            eap_scores.attention_mask, system_prompt_length
        )
        token_position_counts = eap_scores.attention_mask.sum(dim=1).detach().cpu()
        eap_scores = eap_scores.apply(torch.nansum, dim=1, mask_aware=True)
        eap_scores = eap_scores.apply(lambda x: x.detach().cpu())

        for key, value in eap_scores.items():
            batch_output[f"step_{num_steps}__{key[1]}__{key[0]}"] = tensor_to_numpy(value)

        batch_output[f"step_{num_steps}__clean_logits"] = clean_logits_cpu.float().numpy()
        batch_output[f"step_{num_steps}__corrupted_logits"] = corrupted_logits_cpu.float().numpy()
        batch_output[f"step_{num_steps}__token_positions_considered"] = tensor_to_numpy(
            token_position_counts
        )

        del (
            eap_scores,
            token_position_counts,
            clean_inputs,
            corrupted_inputs,
            clean_logits_cpu,
            corrupted_logits_cpu,
        )

    return batch_output


def resolve_save_loc(config_save_loc: Path, results_root: Path | None) -> Path:
    """Resolve a config save path, optionally remapping it under a workflow root."""
    if results_root is None:
        return config_save_loc

    parts = config_save_loc.parts
    lowered_parts = [part.lower() for part in parts]
    if "results" in lowered_parts:
        results_index = lowered_parts.index("results")
        relative_save_loc = Path(*parts[results_index + 1 :])
    else:
        relative_save_loc = Path(config_save_loc.name)
    return results_root / relative_save_loc


def run_qanda_attribution(
    config_path: Path,
    model=None,
    tokenizer=None,
    *,
    save_to_gcp: bool = True,
    results_root: Path | None = None,
    attribution_method: Literal["eap_ig", "eap"] = "eap_ig",
    compute_gradient_at: Literal["clean", "corrupted"] = "clean",
    gcs_prefix: str = "",
    delete_local_after_gcs_upload: bool = False,
    download_existing_gcs_outputs: bool = True,
) -> tuple[Any, Any]:
    """Run Q&A EAP-style attribution from Python or notebooks."""
    config = load_config(resolve_config_path(config_path))

    model_name: str = config["setup"]["model"]
    seed: int = config["setup"]["seed"]
    batch_size: int = config["setup"]["batch_size"]
    layer_components = config["setup"].get("layer_components", None)
    granularity = config["setup"].get("granularity", "coarse")
    quadrature = resolve_quadrature(config)

    dtype = config["setup"].get("dtype", None)

    data_loc = Path(config["paths"]["data_loc"])
    save_loc = resolve_save_loc(Path(config["paths"]["save_loc"]), results_root)

    data_file: str = config["input"]["data_file"]
    template = config["input"]["template"]
    option_keys: list[str] = config["input"]["option_keys"]
    prompt_suffix: str = config["input"]["prompt_suffix"]

    filename: str = config["output"]["filename"]
    gcp_project_id: str | None = config["output"].get("gcp_project_id", GCP_PROJECT_ID)
    gcs_bucket_name: str | None = config["output"].get("gcs_bucket_name", GCS_BUCKET_NAME)

    system_prompt: str = config["parameters"]["system_prompt"]
    metric_type: str = config["parameters"]["metric_type"]
    steps: list[int] = config["parameters"]["steps"]

    input_file_path = data_loc / data_file
    save_loc.mkdir(parents=True, exist_ok=True)
    option_orders = build_option_orders(
        input_file_path=input_file_path,
        template=template,
        option_keys=option_keys,
        batch_size=batch_size,
    )
    output_files = expected_output_files(
        save_loc=save_loc,
        filename=filename,
        option_orders=option_orders,
    )
    if output_files and all(
        existing_output_matches_run(
            output_file,
            attribution_method=attribution_method,
            compute_gradient_at=compute_gradient_at,
        )
        for output_file in output_files
    ):
        print(
            "[resume] All attribution outputs already exist locally; "
            f"skipping config={config_path}",
            flush=True,
        )
        return model, tokenizer

    fetch_existing_gcs_object = maybe_build_gcs_existing_object_fetcher(
        enabled=save_to_gcp,
        project_id=gcp_project_id,
        bucket_name=gcs_bucket_name,
        prefix=gcs_prefix,
        download_existing=download_existing_gcs_outputs,
    )
    upload_queue, upload_thread, enqueue_upload = maybe_start_gcs_upload_worker(
        enabled=save_to_gcp,
        project_id=gcp_project_id,
        bucket_name=gcs_bucket_name,
        prefix=gcs_prefix,
        delete_local_after_upload=delete_local_after_gcs_upload,
    )

    from ..utils.mech_interp_toolkit.activation_dict import expand_mask
    from ..utils.mech_interp_toolkit.gradient_based_attribution import (
        edge_attribution_patching,
        eap_integrated_gradients,
    )
    from ..utils.mech_interp_toolkit.utils import (
        load_model_tokenizer_config,
        set_global_seed,
    )

    set_global_seed(seed)

    if model is None or tokenizer is None:
        model, tokenizer, _ = load_model_tokenizer_config(
            model_name=model_name,
            suffix=prompt_suffix,
            system_prompt=system_prompt,
            attn_type="eager",
            dtype=dtype,
        )

    n_layers = model.config.num_hidden_layers
    layer_components = build_layer_components(
        n_layers=n_layers,
        granularity=granularity,
        layer_components=layer_components,
    )

    system_prompt_length = (
        len(tokenizer.tokenizer.encode(system_prompt, add_special_tokens=False)) + 1
    )
    token_a, token_b = resolve_option_tokens(tokenizer, option_keys)
    metrics = build_metrics(token_a, token_b)

    for order_label, chunked_clean_prompts, chunked_corrupted_prompts in option_orders:
        tokenized_clean = [tokenizer(batch) for batch in chunked_clean_prompts]
        tokenized_corrupted = [tokenizer(batch) for batch in chunked_corrupted_prompts]

        for metric_label, metric_fn in metrics.items():
            for i in tqdm(
                range(len(tokenized_clean)),
                desc=f"[{order_label}/{metric_label}] Batches",
            ):
                output_file = save_loc / (
                    f"{filename}_{order_label}_{metric_label}_batch_{i:05d}.npz"
                )
                output_file_abs = output_file.resolve()
                object_name = gcs_object_name_for_file(output_file_abs)
                if existing_output_matches_run(
                    output_file_abs,
                    attribution_method=attribution_method,
                    compute_gradient_at=compute_gradient_at,
                ):
                    print(
                        "[resume] Skipping existing local output "
                        f"local_path={output_file_abs}",
                        flush=True,
                    )
                    continue
                if output_file_abs.exists():
                    print(
                        "[resume] Recomputing output with mismatched EAP metadata "
                        f"local_path={output_file_abs}",
                        flush=True,
                    )
                    output_file_abs.unlink()
                if fetch_existing_gcs_object(object_name, output_file_abs):
                    if not existing_output_matches_run(
                        output_file_abs,
                        attribution_method=attribution_method,
                        compute_gradient_at=compute_gradient_at,
                    ):
                        print(
                            "[GCS resume] Recomputing output with mismatched EAP metadata "
                            f"object={object_name} local_path={output_file_abs}",
                            flush=True,
                        )
                        output_file_abs.unlink(missing_ok=True)
                    else:
                        print(
                            "[GCS resume] Skipping existing "
                            f"object={object_name} local_path={output_file_abs}",
                            flush=True,
                        )
                        continue
                if output_file_abs.exists():
                    print(
                        "[resume] Skipping existing local output "
                        f"local_path={output_file_abs}",
                        flush=True,
                    )
                    continue

                batch_output = process_batch(
                    model=model,
                    expand_mask=expand_mask,
                    attribution_fn=(
                        eap_integrated_gradients
                        if attribution_method == "eap_ig"
                        else edge_attribution_patching
                    ),
                    attribution_method=attribution_method,
                    tokenized_clean_batch=tokenized_clean[i],
                    tokenized_corrupted_batch=tokenized_corrupted[i],
                    metric_fn=metric_fn,
                    layer_components=layer_components,
                    steps=steps,
                    quadrature=quadrature,
                    system_prompt_length=system_prompt_length,
                    token_a=token_a,
                    token_b=token_b,
                    config=config,
                    order_label=order_label,
                    metric_type=metric_type,
                    compute_gradient_at=compute_gradient_at,
                )
                output_file.parent.mkdir(parents=True, exist_ok=True)
                try:
                    np.savez_compressed(output_file, **batch_output)
                except OSError:
                    output_file.unlink(missing_ok=True)
                    raise
                enqueue_upload(output_file_abs, object_name)

                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                gc.collect()

    if upload_queue is not None and upload_thread is not None:
        upload_queue.put(None)
        upload_thread.join()

    return model, tokenizer


def run_eap_ig(
    config_path: Path,
    model=None,
    tokenizer=None,
    *,
    save_to_gcp: bool = True,
    results_root: Path | None = None,
    gcs_prefix: str = "",
    delete_local_after_gcs_upload: bool = False,
    download_existing_gcs_outputs: bool = True,
) -> tuple[Any, Any]:
    """Run Q&A EAP-IG from Python or notebooks."""
    return run_qanda_attribution(
        config_path,
        model=model,
        tokenizer=tokenizer,
        save_to_gcp=save_to_gcp,
        results_root=results_root,
        attribution_method="eap_ig",
        gcs_prefix=gcs_prefix,
        delete_local_after_gcs_upload=delete_local_after_gcs_upload,
        download_existing_gcs_outputs=download_existing_gcs_outputs,
    )


def run_eap(
    config_path: Path,
    model=None,
    tokenizer=None,
    *,
    save_to_gcp: bool = True,
    results_root: Path | None = None,
    compute_gradient_at: Literal["clean", "corrupted"] = "clean",
    gcs_prefix: str = "",
    delete_local_after_gcs_upload: bool = False,
    download_existing_gcs_outputs: bool = True,
) -> tuple[Any, Any]:
    """Run Q&A vanilla EAP from Python or notebooks."""
    return run_qanda_attribution(
        config_path,
        model=model,
        tokenizer=tokenizer,
        save_to_gcp=save_to_gcp,
        results_root=results_root,
        attribution_method="eap",
        compute_gradient_at=compute_gradient_at,
        gcs_prefix=gcs_prefix,
        delete_local_after_gcs_upload=delete_local_after_gcs_upload,
        download_existing_gcs_outputs=download_existing_gcs_outputs,
    )
