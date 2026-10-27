"""Join the platform's four workers to Ray, verify them, then launch one diagnostic."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import socket
import subprocess
import sys
import time


def digest(path):
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2))
    temporary.replace(path)


def read_json(path):
    try:
        return json.loads(path.read_text())
    except (FileNotFoundError, json.JSONDecodeError):
        return None


def worker_port_range(gcs_port):
    """Keep Ray workers outside its reserved components and the platform GCS port."""
    for minimum in (20000, 21000):
        maximum = minimum + 999
        if not minimum <= gcs_port <= maximum:
            return minimum, maximum
    raise ValueError("No disjoint worker port range")


def wait_for(condition, remote, timeout):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        failed = list((remote / "control").glob("failed-*.json"))
        if failed:
            raise RuntimeError("A cluster member failed: " + failed[0].read_text())
        result = condition()
        if result:
            return result
        time.sleep(2)
    raise TimeoutError("Cluster phase did not finish within its bound")


def probe_node(root_string):
    """One Ray-scheduled GPU operation per physical node, plus local asset checks."""
    import torch
    import ray

    root = Path(root_string)
    manifest = json.loads((root / "code/model-manifest-8b.json").read_text())
    config = root / "models/Qwen3-8B/config.json"
    if digest(config) != manifest["files"]["config.json"]["sha256"]:
        raise ValueError("Model config mismatch on Ray worker")
    value = torch.ones((128, 128), dtype=torch.bfloat16, device="cuda", requires_grad=True)
    loss = (value @ value).float().mean()
    loss.backward()
    torch.cuda.synchronize()
    if loss.item() != 128 or not torch.equal(value.grad, torch.full_like(value, 2 / 128)):
        raise ValueError("Ray worker BF16 operation failed")
    return {"arnold_id": int(os.environ["ARNOLD_ID"]), "hostname": socket.gethostname(),
            "ray_node_id": ray.get_runtime_context().get_node_id(),
            "gpu_ids": ray.get_gpu_ids(), "cuda_device": torch.cuda.get_device_name(),
            "bf16_passed": True}


def copy_verified(source, destination, manifest):
    destination.mkdir(parents=True, exist_ok=True)
    for name, meta in manifest.items():
        target = destination / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source / name, target)
        if target.stat().st_size != meta["bytes"] or digest(target) != meta["sha256"]:
            raise ValueError(f"Copied file mismatch: {name}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--remote", type=Path, required=True)
    parser.add_argument("--asset", type=Path, required=True)
    args = parser.parse_args()
    root, remote, asset = args.root, args.remote, args.asset
    benchmark = os.environ.get("GRPO_PUBLICATION_BENCHMARK") == "1"
    node = int(os.environ["ARNOLD_ID"])
    if int(os.environ["ARNOLD_WORKER_NUM"]) != 4 or not 0 <= node < 4:
        raise ValueError("Expected four platform workers")
    (remote / "control").mkdir(parents=True, exist_ok=True)
    local_results = root / "results"
    local_results.mkdir(exist_ok=True)
    master = os.environ["METIS_WORKER_0_HOST"].strip("[]")
    port = int(os.environ["METIS_WORKER_0_PORT"])
    worker_min, worker_max = worker_port_range(port)
    family = socket.AF_INET6 if ":" in master else socket.AF_INET
    with socket.socket(family, socket.SOCK_DGRAM) as sock:
        sock.connect((master, port))
        local_ip = sock.getsockname()[0]
    gcs = f"[{master}]:{port}" if ":" in master else f"{master}:{port}"
    os.environ["MASTER_ADDR"] = master
    os.environ["MILES_SCRIPT_EXTERNAL_RAY"] = "1"
    os.environ["RAY_ADDRESS"] = "http://127.0.0.1:8265"
    os.environ["no_proxy"] = ",".join(("127.0.0.1", "localhost", master, local_ip))
    os.environ["RAY_DIAGNOSTIC_DIR"] = str(local_results / "ray-diagnostics")
    os.environ["PYTHONPATH"] = f"{root / 'code'}:/root/miles:/root/Megatron-LM"
    write_json(local_results / "node-bootstrap.json", {
        "arnold_id": node, "nodes": 4, "hostname": socket.gethostname(),
        "master": master, "local_ip": local_ip, "gcs": gcs,
        "worker_min_port": worker_min, "worker_max_port": worker_max,
    })
    ray_started = False
    try:
        manifest = json.loads((root / "code/model-manifest-8b.json").read_text())
        cache = read_json(root / "code/converted-cache.json")
        if cache and (not cache["verified"] or not cache["source_conversion_verified"]
                      or cache["model_identity"] != manifest["revision"]):
            raise ValueError("Converted cache model or terminal provenance mismatch")
        copy_verified(asset, root / "models/Qwen3-8B", manifest["files"])
        write_json(local_results / "assets-verified.json",
                   {"verified": True, "model_identity": manifest["revision"]})
        for script, report in [
            ("patch_broadcast_ipv6.py", "broadcast-patch.json"),
            ("patch_publication_probe.py", "publication-patch.json"),
            ("patch_ray_launcher.py", "launcher-patch.json"),
        ]:
            subprocess.run(["sudo", sys.executable, str(root / "code" / script),
                            "--report", str(local_results / report)], check=True, timeout=30)
        if benchmark:
            subprocess.run(["sudo", sys.executable, str(root / "code/publication_benchmark.py"),
                            "--report", str(local_results / "benchmark-patch.json")],
                           check=True, timeout=30)
        common = ["--node-ip-address", local_ip, "--num-gpus", "8", "--num-cpus", "60",
                  "--min-worker-port", str(worker_min), "--max-worker-port", str(worker_max),
                  "--disable-usage-stats"]
        ray_command = [sys.executable, "-m", "ray.scripts.scripts", "start"]
        if node == 0:
            ray_command += ["--head", "--port", str(port), "--include-dashboard", "true",
                            "--dashboard-host", "127.0.0.1", "--dashboard-port", "8265"]
        else:
            def connected():
                try:
                    with socket.create_connection((master, port), timeout=2):
                        return True
                except OSError:
                    return False
            wait_for(connected, remote, 300)
            ray_command += ["--address", gcs]
        subprocess.run(ray_command + common, check=True, timeout=180)
        ray_started = True
        write_json(remote / f"control/ray-started-{node}.json",
                   {"node": node, "hostname": socket.gethostname(), "model_identity": manifest["revision"]})
        if node == 0:
            import ray
            from ray.util.scheduling_strategies import NodeAffinitySchedulingStrategy

            wait_for(lambda: all((remote / f"control/ray-started-{i}.json").is_file() for i in range(4)),
                     remote, 300)
            ray.init(address=gcs)
            nodes = [n for n in ray.nodes() if n["Alive"]]
            if len(nodes) != 4 or any(n["Resources"].get("GPU") != 8 for n in nodes):
                raise ValueError("Ray does not have four nodes with eight GPUs each")
            probe = ray.remote(num_cpus=1, num_gpus=1)(probe_node)
            futures = [
                probe.options(scheduling_strategy=NodeAffinitySchedulingStrategy(n["NodeID"], soft=False))
                .remote(str(root)) for n in nodes
            ]
            reports = ray.get(futures, timeout=120)
            if {r["arnold_id"] for r in reports} != set(range(4)):
                raise ValueError("Ray member identities do not cover all platform workers")
            write_json(remote / "results/ray-cluster-verified.json",
                       {"verified": True, "gpu_resources": 32, "nodes": reports})
            ray.shutdown()
            if cache:
                converted_manifest = cache["files"]
            else:
                prepare_output = local_results / "conversion"
                subprocess.run([sys.executable, str(root / "code/run_train32.py"), "prepare",
                                "--root", str(root), "--output", str(prepare_output)], check=True, timeout=900)
                converted = root / "models/Qwen3-8B_torch_dist"
                converted_manifest = {
                    str(p.relative_to(converted)): {"bytes": p.stat().st_size, "sha256": digest(p)}
                    for p in sorted(converted.rglob("*")) if p.is_file() and not p.is_symlink()
                }
                if not converted_manifest or (converted / "latest_checkpointed_iteration.txt").read_text().strip() != "release":
                    raise ValueError("No completed converted checkpoint")
                copy_verified(converted, remote / "converted-checkpoint", converted_manifest)
            write_json(remote / "control/converted-manifest.json", {"files": converted_manifest})
        else:
            converted_manifest = wait_for(
                lambda: read_json(remote / "control/converted-manifest.json"), remote, 1500)
            converted_manifest = converted_manifest["files"]
        if cache or node != 0:
            source = Path(cache["remote_mount"]) if cache else remote / "converted-checkpoint"
            copy_verified(source, root / "models/Qwen3-8B_torch_dist", converted_manifest)
        if cache:
            write_json(local_results / "converted-cache-use.json", {
                "verified": True, "source_run": cache["source_run"],
                "source_audit_sha256": cache["source_audit_sha256"],
                "files": converted_manifest,
            })
        write_json(remote / f"control/models-ready-{node}.json", {"node": node, "ready": True})
        if node == 0:
            wait_for(lambda: all((remote / f"control/models-ready-{i}.json").is_file() for i in range(4)),
                     remote, 600)
            subprocess.run([sys.executable, str(root / "code/run_train32.py"), "train",
                            "--root", str(root), "--output", str(remote / "results")],
                           check=True, timeout=3000 if benchmark else 1800)
            write_json(remote / "control/driver-finished.json", {"success": True})
        else:
            finished = wait_for(lambda: read_json(remote / "control/driver-finished.json"),
                                remote, 4200 if benchmark else 2400)
            if not finished["success"]:
                raise RuntimeError("Training coordinator failed")
        write_json(local_results / "node-completed.json", {"node": node, "success": True})
    except Exception as exc:
        write_json(remote / f"control/failed-{node}.json",
                   {"node": node, "error": f"{type(exc).__name__}: {exc}"})
        raise
    finally:
        if ray_started:
            subprocess.run([sys.executable, str(root / "code/ray_readiness.py"), "--snapshot-only"],
                           timeout=15, check=False)
            subprocess.run([sys.executable, "-m", "ray.scripts.scripts", "stop", "--force"],
                           timeout=45, check=False)


if __name__ == "__main__":
    main()
