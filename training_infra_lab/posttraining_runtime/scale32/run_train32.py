"""Three diagnostic GRPO updates on 16 learner and 16 rollout GPUs."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys

MILES_SHA = "2233d795dcf167be489d1859971c9233393f04c3"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=("prepare", "train"))
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root, output = args.root.resolve(), args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    model = root / "models/Qwen3-8B"
    code = root / "code"
    manifest = json.loads((code / "model-manifest-8b.json").read_text())
    data = code / "explicit-prompts.jsonl"
    format_manifest = json.loads((code / "format-manifest.json").read_text())
    if hashlib.sha256(data.read_bytes()).hexdigest() != format_manifest["explicit_sha256"]:
        raise ValueError("Frozen training data mismatch")
    if subprocess.check_output(["git", "-C", "/root/miles", "rev-parse", "HEAD"], text=True).strip() != MILES_SHA:
        raise ValueError("Miles revision mismatch")
    if (code / "math_dapo_utils.py").read_bytes() != Path(
            "/root/miles/miles/rollout/rm_hub/math_dapo_utils.py").read_bytes():
        raise ValueError("Frozen DAPO scorer differs from the running revision")
    os.chdir("/root/miles")
    from miles.utils.external_utils import command_utils
    config = command_utils.default_config()
    config.output_dir = str(output)
    config.extra_env_vars = json.dumps({
        "PYTHONPATH": f"{code}:/root/miles:/root/Megatron-LM",
        "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1",
        "OMP_NUM_THREADS": "1", "RAY_DIAGNOSTIC_DIR": str(output / "ray-diagnostics"),
        "GRPO_PUBLICATION_BENCHMARK": os.environ.get("GRPO_PUBLICATION_BENCHMARK", "0"),
    })
    backend = config.create_backend()
    if args.stage == "prepare":
        backend.convert_checkpoint(model_name="Qwen3-8B", megatron_model_type="qwen3-8B",
                                   num_gpus_per_node=2, dir_dst=str(model.parent),
                                   hf_checkpoint=str(model))
        return
    if os.environ.get("MILES_SCRIPT_EXTERNAL_RAY") != "1":
        raise ValueError("Training requires the already verified external four-node Ray cluster")
    eval_data = output / "format-eval.jsonl"
    eval_data.write_text("\n".join(data.read_text().splitlines()[:16]) + "\n")
    options = [
        "--hf-checkpoint", str(model), "--load", str(model.parent / "Qwen3-8B_torch_dist"),
        "--ref-load", str(model.parent / "Qwen3-8B_torch_dist"),
        "--finetune", "--no-load-optim", "--no-load-rng",
        "--prompt-data", str(data), "--input-key", "prompt", "--label-key", "label",
        "--apply-chat-template", "--apply-chat-template-kwargs", '{"enable_thinking":false}',
        "--rm-type", "dapo", "--reward-key", "score",
        "--num-rollout", "3", "--rollout-batch-size", "16", "--n-samples-per-prompt", "4",
        "--global-batch-size", "64", "--rollout-max-response-len", "4096",
        "--rollout-temperature", "1", "--seed", "20261025",
        "--rollout-function-path", "bounded_rollout.BoundedRolloutFn",
        "--custom-rollout-log-function-path", "rollout_gate.check_before_train",
        "--custom-megatron-before-train-step-hook-path", "train32_checks.before_step",
        "--advantage-estimator", "grpo", "--kl-loss-coef", "0.00", "--kl-loss-type", "low_var_kl",
        "--entropy-coef", "0.00", "--eps-clip", "0.2", "--eps-clip-high", "0.28",
        "--optimizer", "adam", "--lr", "1e-6", "--lr-decay-style", "constant",
        "--weight-decay", "0.1", "--adam-beta1", "0.9", "--adam-beta2", "0.98",
        "--tensor-model-parallel-size", "2", "--sequence-parallel",
        "--pipeline-model-parallel-size", "1", "--context-parallel-size", "1",
        "--recompute-granularity", "full", "--recompute-method", "uniform", "--recompute-num-layers", "1",
        "--use-dynamic-batch-size", "--max-tokens-per-gpu", "4096", "--balance-data",
        "--attention-dropout", "0.0", "--hidden-dropout", "0.0",
        "--accumulate-allreduce-grads-in-fp32", "--attention-softmax-in-fp32", "--attention-backend", "flash",
        "--actor-num-nodes", "2", "--num-gpus-per-node", "8", "--actor-num-gpus-per-node", "8",
        "--rollout-num-gpus", "16", "--rollout-num-gpus-per-engine", "2",
        "--sglang-mem-fraction-static", "0.6", "--update-weight-transfer-mode", "broadcast",
        "--check-weight-update-equal", "--save-local-weight-checksum",
        "--enable-sample-ownership-checker", "--sample-ownership-grace-steps", "0",
        "--save-debug-event-data", str(output / "events"),
        "--save-debug-train-data", str(output / "train-{rollout_id}-rank-{rank}.pt"),
        "--save-debug-rollout-data", str(output / "rollout-{rollout_id}.pt"),
        "--save-debug-trajectory-data", str(output / "trajectory-{rollout_id}.jsonl"),
        "--eval-interval", "3", "--skip-eval-before-train",
        "--eval-prompt-data", "format_probe", str(eval_data),
        "--n-samples-per-eval-prompt", "1", "--eval-max-response-len", "4096", "--eval-temperature", "0",
    ]
    contract = {
        "model_identity": manifest["revision"], "hf_revision_verified": manifest["hf_revision_verified"],
        "miles_commit": MILES_SHA, "train_data_sha256": format_manifest["explicit_sha256"],
        "learner_gpus": 16, "rollout_gpus": 16, "engine_count": 8, "tp": 2, "dp": 8,
        "requested_rollouts": [0, 1, 2], "expected_weight_versions": [1, 2, 3],
        "batch_samples": 64, "prompts": 16, "answers_per_prompt": 4,
        "full_checksum_diagnostic": True, "performance_claim": False,
        "capability_claim": False, "evaluation_scope": "Exposed prompts; output protocol only",
        "publication_benchmark_requested": os.environ.get("GRPO_PUBLICATION_BENCHMARK") == "1",
        "eos_token_ids": json.loads((model / "generation_config.json").read_text())["eos_token_id"],
    }
    (output / "train32-contract.json").write_text(json.dumps(contract, indent=2))
    (output / "requested-args.json").write_text(json.dumps(options, indent=2))

    def before_submit():
        subprocess.run([sys.executable, str(code / "ray_readiness.py"),
                        "--output", str(output / "ray-jobs-readiness.json")], check=True, timeout=135)

    backend.execute_train(
        train_args=shlex.join(options), num_gpus_per_node=8, megatron_model_type="qwen3-8B",
        train_script="train.py", job_lifetime="launcher", config=config,
        before_ray_job_submit=before_submit,
    )
    (output / "driver-completed.json").write_text(json.dumps(
        {"driver_returned": True, "semantic_audit_pending": True, "performance_claim": False}))


if __name__ == "__main__":
    main()
