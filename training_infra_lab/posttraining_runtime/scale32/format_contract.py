"""Freeze prompts and audit unconstrained SGLang output against the original DAPO scorer."""

import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import re
import subprocess
import sys
import time
from urllib import request

DATA_SHA = "67173a6074167cb280f1cf70542ccd1c1f91583defa8698f2428d268858e4fff"
MODEL_SHA = "1cfa9a7208912126459214e8b04321603b3df60c"
FINAL_LINE = re.compile(r"Answer: \\boxed\{(-?\d+)\}")
FORMAT_INSTRUCTION = (
    "Solve this math problem with concise reasoning. The answer is an integer.\n\n"
    "{question}\n\n"
    "Finish with exactly one final line in the form Answer: \\boxed{integer}, "
    "replacing integer with your result. Keep Answer: and the boxed integer on "
    "the same line. Use no dollar signs, Markdown bold, heading, or text after "
    "that final line. Then end the response."
)


def chat_token_ids(tokenizer, messages):
    """Normalize Transformers 4/5 chat-template returns to one JSON token list."""
    value = tokenizer.apply_chat_template(
        messages, tokenize=True, add_generation_prompt=True, enable_thinking=False)
    if hasattr(value, "keys"):
        value = value["input_ids"]
    if hasattr(value, "tolist"):
        value = value.tolist()
    if value and isinstance(value[0], list):
        if len(value) != 1:
            raise ValueError("Expected one chat prompt")
        value = value[0]
    if not isinstance(value, list) or not value or any(type(i) is not int or i < 0 for i in value):
        raise ValueError("Chat template did not return a flat integer token sequence")
    return value


def freeze(source, destination):
    blob = source.read_bytes()
    if hashlib.sha256(blob).hexdigest() != DATA_SHA:
        raise ValueError("Original data hash mismatch")
    destination.mkdir(exist_ok=False, parents=True)
    original = [json.loads(line) for line in blob.decode().splitlines()]
    rows = []
    for index, row in enumerate(original):
        messages = row["prompt"]
        if len(messages) != 1 or messages[0]["role"] != "user":
            raise ValueError("Unexpected original message structure")
        text = messages[0]["content"]
        prefix, question = text.split("\n\n", 1)
        suffix = '\n\nRemember to put your answer on its own line after "Answer:".'
        if not prefix.startswith("Solve the following math problem") or not question.endswith(suffix):
            raise ValueError("Unexpected original prompt wrapper")
        question = question[:-len(suffix)]
        rows.append({
            "source_index": index, "question_sha256": hashlib.sha256(question.encode()).hexdigest(),
            "prompt": [{"role": "user", "content": FORMAT_INSTRUCTION.replace("{question}", question)}],
            "label": row["label"],
        })
    corrected = b"".join((json.dumps(row) + "\n").encode() for row in rows)
    (destination / "explicit-prompts.jsonl").write_bytes(corrected)
    (destination / "original-prompts.jsonl").write_bytes(blob)
    manifest = {
        "original_sha256": DATA_SHA, "explicit_sha256": hashlib.sha256(corrected).hexdigest(),
        "rows": len(rows), "selection": "all original 64 rows; original order and labels",
        "deterministic_probe_indices": list(range(16)), "stochastic_probe_indices": list(range(4)),
        "samples_per_stochastic_prompt": 4, "model_revision": MODEL_SHA,
        "max_response_tokens": 4096, "format_instruction": FORMAT_INSTRUCTION,
        "purpose": "format contract; previously exposed infrastructure prompts, not held-out capability",
        "scorer_modified": False, "constrained_decoding": False,
    }
    (destination / "format-manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def audit_response(result, eos_ids):
    text, meta = result["text"], result["meta_info"]
    lines = text.strip().splitlines()
    match = FINAL_LINE.fullmatch(lines[-1]) if lines else None
    finish = meta.get("finish_reason", {})
    logprobs = meta.get("output_token_logprobs", [])
    ids = [row[1] for row in logprobs]
    matched = finish.get("matched")
    natural_eos = finish.get("type") == "stop" and (
        matched in eos_ids or (bool(ids) and ids[-1] in eos_ids)
    )
    probability_ok = bool(logprobs) and len(logprobs) == meta.get("completion_tokens") and all(
        isinstance(row[0], (float, int)) and math.isfinite(row[0]) and row[0] <= 1e-5
        for row in logprobs
    )
    pred = result["reward"]["pred"]
    format_ok = bool(match) and pred == match[1]
    return {
        "natural_eos": natural_eos, "finish_reason": finish, "format_ok": format_ok,
        "probability_ok": probability_ok, "completion_tokens": meta.get("completion_tokens"),
        "scorer_pred": pred, "reward": result["reward"]["score"],
        "passed": natural_eos and format_ok and probability_ok,
    }


def generate(root, data, output):
    from transformers import AutoTokenizer

    output.mkdir(exist_ok=False, parents=True)
    manifest = json.loads((data / "format-manifest.json").read_text())
    model = root / "models/Qwen3-4B"
    rows = {}
    for variant, filename, expected in [
        ("original", "original-prompts.jsonl", manifest["original_sha256"]),
        ("explicit", "explicit-prompts.jsonl", manifest["explicit_sha256"]),
    ]:
        blob = (data / filename).read_bytes()
        if hashlib.sha256(blob).hexdigest() != expected:
            raise ValueError("Frozen prompt hash mismatch")
        rows[variant] = [json.loads(line) for line in blob.decode().splitlines()]
    scorer_path = Path("/root/miles/miles/rollout/rm_hub/math_dapo_utils.py")
    scorer_blob = scorer_path.read_bytes()
    frozen_scorer = (data / "math_dapo_utils.py").read_bytes()
    if scorer_blob != frozen_scorer:
        raise ValueError("Scorer differs from the frozen revision")
    spec = importlib.util.spec_from_file_location("frozen_dapo", scorer_path)
    scorer = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(scorer)
    tokenizer = AutoTokenizer.from_pretrained(model, local_files_only=True)
    config = json.loads((model / "generation_config.json").read_text())
    eos = config["eos_token_id"]
    eos_ids = set(eos if isinstance(eos, list) else [eos]) | {tokenizer.eos_token_id}
    tasks = []
    for variant in ("original", "explicit"):
        for index in manifest["deterministic_probe_indices"]:
            tasks.append((variant, index, 0, 0.0))
    for index in manifest["stochastic_probe_indices"]:
        for repeat in range(4):
            tasks.append(("explicit", index, repeat + 1, 1.0))
    contract = {
        **manifest, "scorer_sha256": hashlib.sha256(scorer_blob).hexdigest(),
        "tasks": tasks, "eos_token_ids": sorted(eos_ids), "engine_tp": 2,
        "enable_thinking": False, "forced_answer_suffix": False, "grammar": None,
    }
    (output / "generation-contract.json").write_text(json.dumps(contract, indent=2))
    address = "http://127.0.0.1:30000"
    opener = request.build_opener(request.ProxyHandler({}))

    def call(path, payload=None):
        req = request.Request(address + path, data=json.dumps(payload).encode() if payload else None,
                              headers={"Content-Type": "application/json"})
        with opener.open(req, timeout=600 if payload else 5) as response:
            return json.load(response)

    with (output / "sglang.log").open("w") as log:
        server = subprocess.Popen([
            sys.executable, "-m", "sglang.launch_server", "--model-path", str(model),
            "--tp-size", "2", "--host", "127.0.0.1", "--port", "30000",
            "--mem-fraction-static", "0.6", "--random-seed", "20261025",
        ], stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        try:
            ready = False
            deadline = time.monotonic() + 240
            while time.monotonic() < deadline:
                if server.poll() is not None:
                    raise RuntimeError("SGLang exited during startup")
                try:
                    call("/get_model_info")
                    ready = True
                    break
                except (OSError, ValueError):
                    time.sleep(1)
            if not ready:
                raise TimeoutError("SGLang readiness deadline")

            def one(task):
                variant, index, repeat, temperature = task
                row = rows[variant][index]
                ids = chat_token_ids(tokenizer, row["prompt"])
                start = time.monotonic()
                response = call("/generate", {
                    "input_ids": ids, "return_logprob": True, "logprob_start_len": -1,
                    "sampling_params": {"temperature": temperature, "max_new_tokens": 4096},
                })
                response.update(variant=variant, source_index=index, repeat=repeat,
                                temperature=temperature, prompt_token_ids=ids,
                                elapsed_seconds=time.monotonic() - start, label=row["label"])
                response["reward"] = scorer.compute_score(response["text"], row["label"])
                response["audit"] = audit_response(response, eos_ids)
                name = f"{variant}-{index}-{repeat}.json"
                (output / name).write_text(json.dumps(response, indent=2))
                print(json.dumps({"event": "format_probe", "variant": variant, "index": index,
                                  "repeat": repeat, **response["audit"]}), flush=True)
                return response

            with ThreadPoolExecutor(max_workers=16) as pool:
                results = list(pool.map(one, tasks))
            summary = {"passed": False, "capability_claim": False, "groups": {}}
            for name, selected in [
                ("original_deterministic", [r for r in results if r["variant"] == "original"]),
                ("explicit_deterministic", [r for r in results if r["variant"] == "explicit" and r["repeat"] == 0]),
                ("explicit_stochastic", [r for r in results if r["repeat"] > 0]),
            ]:
                summary["groups"][name] = {
                    "samples": len(selected),
                    **{key: sum(r["audit"][key] for r in selected) for key in
                       ("passed", "natural_eos", "format_ok", "probability_ok")},
                    "correct": sum(r["reward"]["score"] == 1 for r in selected),
                }
            summary["passed"] = all(r["audit"]["passed"] for r in results if r["variant"] == "explicit")
            (output / "summary.json").write_text(json.dumps(summary, indent=2))
            print(json.dumps(summary), flush=True)
        finally:
            import os
            import signal
            if server.poll() is None:
                os.killpg(server.pid, signal.SIGTERM)
                try:
                    server.wait(timeout=20)
                except subprocess.TimeoutExpired:
                    os.killpg(server.pid, signal.SIGKILL)
                    server.wait(timeout=5)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=("freeze", "generate"))
    parser.add_argument("--source", type=Path)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--root", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.stage == "freeze":
        print(json.dumps(freeze(args.source, args.data)))
    else:
        generate(args.root, args.data, args.output)


if __name__ == "__main__":
    main()
