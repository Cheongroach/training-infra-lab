"""Stage the Qwen3-8B mirror using frozen per-file revisions, sizes and SHA256."""

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import shutil
import subprocess
import time
from urllib import request


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--inventory", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--upload", action="store_true")
    args = parser.parse_args()
    inventory = json.loads(args.inventory.read_text())
    if inventory["Code"] != 200:
        raise ValueError("Mirror inventory failed")
    files = {}
    for item in inventory["Data"]["Files"]:
        name = item["Path"]
        path = PurePosixPath(name)
        if path.is_absolute() or ".." in path.parts:
            raise ValueError("Unexpected model path")
        if item["Type"] != "blob" or not (
            name.endswith((".json", ".safetensors", ".txt", ".model", ".jinja")) or name == "LICENSE"
        ):
            continue
        files[name] = {"bytes": item["Size"], "sha256": item["Sha256"], "revision": item["Revision"]}
    if len([name for name in files if name.endswith(".safetensors")]) != 5:
        raise ValueError("Expected five frozen weight shards")
    identity = hashlib.sha256(json.dumps(files, sort_keys=True).encode()).hexdigest()
    manifest = {
        "repo": "Qwen/Qwen3-8B", "source_endpoint": "https://modelscope.cn",
        "revision": "files-sha256-" + identity, "files": files,
        "hf_revision_verified": False, "pinning": "Each file has its own immutable mirror revision and content SHA256",
    }
    args.output.mkdir(parents=True, exist_ok=True)
    model = args.output / "Qwen3-8B"
    model.mkdir(exist_ok=True)
    need = sum(meta["bytes"] for name, meta in files.items() if not (model / name).exists())
    if shutil.disk_usage(args.output).free < need + 2 * 1024**3:
        raise ValueError("Insufficient local disk")
    (args.output / "model-manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")

    def verify(path, meta):
        if not path.is_file() or path.stat().st_size != meta["bytes"]:
            return False
        with path.open("rb") as handle:
            return hashlib.file_digest(handle, "sha256").hexdigest() == meta["sha256"]

    def download(name, meta):
        target = model / name
        if verify(target, meta):
            return {"name": name, "verified": True, "cached": True}
        url = f"https://modelscope.cn/models/Qwen/Qwen3-8B/resolve/{meta['revision']}/{name}"
        target.parent.mkdir(parents=True, exist_ok=True)
        partial = target.with_name(target.name + ".partial")
        last_error = None
        for attempt in range(3):
            try:
                digest, size = hashlib.sha256(), 0
                with request.urlopen(url, timeout=45) as response, partial.open("wb") as output:
                    while chunk := response.read(4 * 1024**2):
                        output.write(chunk)
                        digest.update(chunk)
                        size += len(chunk)
                if size != meta["bytes"] or digest.hexdigest() != meta["sha256"]:
                    raise ValueError("Downloaded file does not match frozen mirror hash")
                partial.replace(target)
                return {"name": name, "verified": True, "cached": False, "bytes": size}
            except (OSError, ValueError) as exc:
                last_error = exc
                partial.unlink(missing_ok=True)
                if attempt < 2:
                    time.sleep(2)
        raise RuntimeError(f"Failed to stage {name}: {last_error}")

    records = []
    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = [pool.submit(download, name, meta) for name, meta in files.items()]
        for future in as_completed(futures):
            record = future.result()
            records.append(record)
            print(json.dumps(record), flush=True)
            (args.output / "download-progress.json").write_text(json.dumps(records, indent=2))
    proof = {"verified": True, "manifest_identity": identity, "files": len(files),
             "bytes": sum(meta["bytes"] for meta in files.values()), "uploaded": False}
    if args.upload:
        uri = ("hdfs://haruna/home/byte_search_lifeservice/zhangyang.exp/"
               "training_infra_lab/assets/Qwen3-8B/" + manifest["revision"])
        env = {**os.environ, "HADOOP_CONF_SAFE_REFRESH_RUNNING": "true", "PYTHONDONTWRITEBYTECODE": "1"}
        subprocess.run(["hdfs", "dfs", "-mkdir", "-p", uri], check=True, timeout=45, env=env)
        for name in files:
            subprocess.run(["hdfs", "dfs", "-put", "-f", str(model / name), uri + "/" + name],
                           check=True, timeout=600, env=env)
            print(json.dumps({"uploaded_file": name}), flush=True)
        subprocess.run(["hdfs", "dfs", "-put", "-f", str(args.output / "model-manifest.json"),
                        uri + "/model-manifest.json"], check=True, timeout=45, env=env)
        proof.update(uploaded=True, remote_uri=uri,
                     remote_mount=uri.replace("hdfs://haruna/home/byte_search_lifeservice/zhangyang.exp",
                                              "/mnt/hdfs/lab_owner", 1))
    (args.output / "assets-verified.json").write_text(json.dumps(proof, indent=2) + "\n")
    print(json.dumps(proof), flush=True)


if __name__ == "__main__":
    main()
