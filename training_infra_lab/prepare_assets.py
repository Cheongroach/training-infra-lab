"""Download pinned public artifacts and construct auditable assistant-only labels."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import requests

MODEL = "Qwen/Qwen2.5-1.5B-Instruct"
MODEL_REV = "989aa7980e4cf806f80c7fef2b1adb7bc71aa306"
DATASET = "HuggingFaceH4/no_robots"
DATASET_REV = "e6f9a4ac5c37faeb744ba9ecf0473184d7f8105b"
DATASET_MIRROR_REV = "c10ce599dc8beb6be40b7473ecf28b05d80302ba"
DATASET_SHA256 = {
    "train": "76d64120a2ab20eb85d6b22bdced5b1ec638a6213a97004741006745c92489ef",
    "test": "60707b2636a46e37bb0c1e9ca263a18553f430317b7a53c691676d6a492fc0f2",
}
# Qwen's official ModelScope mirror. Verify bytes against the pinned HF Git/LFS
# identities, so switching the download host cannot silently switch the model.
MIRROR_BASE_REV = "a3c2dc17129625b1e51caf21ab486d32d1f12982"
MODEL_FILES = {
    "config.json": ("f81ead14ab072d65a07817f83a3ee0e5a1890d10", MIRROR_BASE_REV),
    "generation_config.json": ("dfc11073787daf1b0f9c0f1499487ab5f4c93738", MIRROR_BASE_REV),
    "merges.txt": ("20024bfe7c83998e9aeaf98a0cd6a2ce6306c2f0", MIRROR_BASE_REV),
    "tokenizer.json": ("443909a61d429dff23010e5bddd28ff530edda00", MIRROR_BASE_REV),
    "vocab.json": ("4783fe10ac3adce15ac8f358ef5462739852c569", MIRROR_BASE_REV),
    "tokenizer_config.json": (
        "07bfe0640cb5a0037f9322287fbfc682806cf672",
        "bc68ae58812462d8bf5d9ba5131317ed7fb71665",
    ),
    "LICENSE": (
        "6634c8cc3133b3848ec74b9f275acaaa1ea618ab",
        "800496101fe8b9995a7a3ed260ef7a7288d68472",
    ),
    "model.safetensors": (
        "dd924a11b4c220f385b51ffa522daea7c9f3d850e31b162bb5661df483c6d3ee",
        MIRROR_BASE_REV,
    ),
}


def download_weight(url, path, size, sha256):
    """Resume an existing prefix; range-fetch the suffix to local disk.

    HDFS FUSE need not support append or random writes: only the verified,
    assembled file is copied back sequentially.
    """
    if path.exists():
        return
    partial = path.with_suffix(path.suffix + ".partial")
    prefix = partial.stat().st_size if partial.exists() else 0
    assert 0 <= prefix <= size
    with tempfile.TemporaryDirectory(prefix="sft-download-") as directory:
        directory = Path(directory)
        block = 32 * 1024 * 1024
        ranges = [(start, min(size, start + block) - 1) for start in range(prefix, size, block)]

        def fetch(bounds):
            start, end = bounds
            output = directory / str(start)
            for attempt in range(4):
                try:
                    with requests.get(
                        url, headers={"Range": f"bytes={start}-{end}"},
                        timeout=(30, 90), stream=True,
                    ) as response:
                        response.raise_for_status()
                        assert response.status_code == 206
                        assert response.headers["Content-Range"] == f"bytes {start}-{end}/{size}"
                        with output.open("wb") as file:
                            for chunk in response.iter_content(1024 * 1024):
                                file.write(chunk)
                    assert output.stat().st_size == end - start + 1
                    print(json.dumps({"event": "weight_range_ready", "start": start, "end": end}), flush=True)
                    return output
                except (requests.RequestException, TimeoutError):
                    if attempt == 3:
                        raise
                    time.sleep(2 ** attempt)

        print(json.dumps({"event": "weight_resume", "existing_bytes": prefix, "total_bytes": size}), flush=True)
        with ThreadPoolExecutor(max_workers=8) as pool:
            pieces = list(pool.map(fetch, ranges))
        assembled = directory / "model.safetensors"
        with assembled.open("wb") as out:
            if prefix:
                with partial.open("rb") as inp:
                    shutil.copyfileobj(inp, out, 8 * 1024 * 1024)
            for piece in pieces:
                with piece.open("rb") as inp:
                    shutil.copyfileobj(inp, out, 8 * 1024 * 1024)
        with assembled.open("rb") as file:
            assert hashlib.file_digest(file, "sha256").hexdigest() == sha256
        upload = path.with_suffix(path.suffix + ".verified")
        shutil.copyfile(assembled, upload)
        upload.replace(path)
    print(json.dumps({"event": "weight_download_verified", "bytes": size}), flush=True)


def download(url, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        return
    tmp = path.with_suffix(path.suffix + ".partial")
    for attempt in range(6):
        try:
            with requests.get(url, stream=True, timeout=(30, 180)) as response:
                response.raise_for_status()
                with tmp.open("wb") as out:
                    for chunk in response.iter_content(8 * 1024 * 1024):
                        out.write(chunk)
            tmp.replace(path)
            break
        except (requests.RequestException, TimeoutError) as error:
            code = getattr(getattr(error, "response", None), "status_code", None)
            if attempt == 5 or (code is not None and code not in (408, 429, 500, 502, 503, 504)):
                raise
            delay = min(60, 5 * 2**attempt)
            if code == 429:
                retry_after = error.response.headers.get("Retry-After", "")
                if retry_after.isdigit():
                    delay = max(delay, min(300, int(retry_after) + 1))
            print(json.dumps({
                "event": "download_retry", "file": path.name,
                "status": code, "attempt": attempt + 1, "delay_seconds": delay,
            }), flush=True)
            end = time.monotonic() + delay
            while time.monotonic() < end:
                time.sleep(min(30, end - time.monotonic()))
    print(json.dumps({"event": "downloaded", "file": str(path), "bytes": path.stat().st_size}), flush=True)


def encode_chat(tokenizer, messages, max_length):
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
    encoded = tokenizer(text, add_special_tokens=False, return_offsets_mapping=True)
    ids = encoded["input_ids"]
    labels = [-100] * len(ids)
    spans = []
    for i, message in enumerate(messages):
        if message["role"] != "assistant":
            continue
        prefix = tokenizer.apply_chat_template(messages[:i], tokenize=False, add_generation_prompt=True)
        end = tokenizer.apply_chat_template(messages[: i + 1], tokenize=False, add_generation_prompt=False)
        assert text.startswith(prefix) and text.startswith(end)
        spans.append((len(prefix), len(end)))
    boundary_tokens_ignored = 0
    for index, (start, end) in enumerate(encoded["offset_mapping"]):
        if end <= start:
            continue
        if any(start >= left and end <= right for left, right in spans):
            labels[index] = ids[index]
        elif any(start < right and end > left for left, right in spans):
            # BPE can merge the assistant header's trailing newline with answer
            # whitespace. Such a token partly belongs to the prompt: mask it.
            boundary_tokens_ignored += 1
    ids, labels = ids[:max_length], labels[:max_length]
    if not any(x != -100 for x in labels[1:]):
        return None
    return {"input_ids": ids, "labels": labels, "boundary_tokens_ignored": boundary_tokens_ignored}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--max-length", type=int, default=4096)
    args = parser.parse_args()
    root = args.root
    model_path = root / "model"
    for name, (identity, mirror_rev) in MODEL_FILES.items():
        path = model_path / name
        url = f"https://modelscope.cn/models/{MODEL}/resolve/{mirror_rev}/{name}"
        if name == "model.safetensors":
            download_weight(url, path, 3087467144, identity)
        else:
            download(url, path)
        if name.endswith(".safetensors"):
            with path.open("rb") as file:
                actual = hashlib.file_digest(file, "sha256").hexdigest()
        else:
            content = path.read_bytes()
            actual = hashlib.sha1(b"blob " + str(len(content)).encode() + b"\0" + content).hexdigest()
        assert actual == identity, f"Model file identity mismatch: {name}"
        print(json.dumps({"event": "model_file_verified", "file": name}), flush=True)
    for split in ("train", "test"):
        download(
            f"https://modelscope.cn/datasets/{DATASET}/resolve/{DATASET_MIRROR_REV}/data/{split}-00000-of-00001.parquet",
            root / f"{split}.parquet",
        )
        with (root / f"{split}.parquet").open("rb") as file:
            assert hashlib.file_digest(file, "sha256").hexdigest() == DATASET_SHA256[split]
    from transformers import AutoTokenizer
    import pyarrow.parquet as pq
    import random

    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
    counts = {}
    for split, limit in (("train", 2048), ("test", 128)):
        rows = pq.read_table(root / f"{split}.parquet").to_pylist()
        random.Random(20261019).shuffle(rows)
        records = []
        for row in rows:
            record = encode_chat(tokenizer, row["messages"], args.max_length)
            if record:
                record["id"] = str(row.get("prompt_id", row.get("id", len(records))))
                records.append(record)
            if len(records) == limit:
                break
        path = root / f"{split}.jsonl"
        path.write_text("".join(json.dumps(r) + "\n" for r in records))
        counts[split] = {
            "examples": len(records),
            "tokens": sum(len(r["input_ids"]) for r in records),
            "supervised_tokens": sum(sum(t != -100 for t in r["labels"][1:]) for r in records),
            "boundary_tokens_ignored": sum(r["boundary_tokens_ignored"] for r in records),
        }
    manifest = {
        "model": MODEL, "model_revision": MODEL_REV,
        "model_mirror": "https://modelscope.cn/models/" + MODEL,
        "model_file_identities": MODEL_FILES,
        "dataset": DATASET, "dataset_revision": DATASET_REV,
        "dataset_mirror_revision": DATASET_MIRROR_REV,
        "dataset_file_sha256": DATASET_SHA256,
        "dataset_license": "cc-by-nc-4.0",
        "selection_seed": 20261019, "max_length": args.max_length,
        "counts": counts,
        "files": {
            str(p.relative_to(root)): hashlib.file_digest(p.open("rb"), "sha256").hexdigest()
            for p in root.rglob("*")
            if p.is_file() and p.name != "manifest.json" and p.suffix not in (".partial", ".verified")
        },
    }
    (root / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(json.dumps({"event": "assets_ready", **manifest}), flush=True)


if __name__ == "__main__":
    main()
