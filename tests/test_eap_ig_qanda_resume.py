from __future__ import annotations

import json

import yaml

from temporal_manifolds.eap_ig.eap_ig_qanda_pipeline import run_qanda_attribution


def test_run_qanda_attribution_skips_when_all_local_outputs_exist(tmp_path) -> None:
    data_dir = tmp_path / "data"
    results_dir = tmp_path / "results"
    data_dir.mkdir()
    results_dir.mkdir()

    data_path = data_dir / "pairs.json"
    data_path.write_text(
        json.dumps(
            {
                "pairs": [
                    {
                        "question": "Question 1",
                        "immediate": "Immediate 1",
                        "long_term": "Long term 1",
                    },
                    {
                        "question": "Question 2",
                        "immediate": "Immediate 2",
                        "long_term": "Long term 2",
                    },
                ]
            }
        ),
        encoding="utf-8",
    )

    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "setup": {
                    "model": "would-fail-if-loaded",
                    "seed": 0,
                    "batch_size": 1,
                },
                "paths": {
                    "data_loc": str(data_dir),
                    "save_loc": str(results_dir),
                },
                "input": {
                    "data_file": data_path.name,
                    "template": "{} {} {}",
                    "option_keys": ["(A)", "(B)"],
                    "prompt_suffix": "",
                },
                "output": {
                    "filename": "artifact",
                },
                "parameters": {
                    "system_prompt": "",
                    "metric_type": "logit",
                    "steps": [1],
                },
            }
        ),
        encoding="utf-8",
    )

    for order_label in ("short_first", "long_first"):
        for metric_label in ("logit_A", "logit_B"):
            for batch_index in range(2):
                output_file = (
                    results_dir
                    / f"artifact_{order_label}_{metric_label}_batch_{batch_index:05d}.npz"
                )
                output_file.write_bytes(b"present")

    model, tokenizer = run_qanda_attribution(config_path, save_to_gcp=False)

    assert model is None
    assert tokenizer is None
