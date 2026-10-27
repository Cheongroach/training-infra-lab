"""Epoch-fenced synchronous checkpoints on shared POSIX storage.

flock/replace/fsync guarantees are NOT assumed on HDFS FUSE or object stores.
Exporting an artifact archive to HDFS is separate from this commit protocol.
"""
import fcntl
import hashlib
import json
import os
from contextlib import contextmanager
from pathlib import Path


class StaleEpoch(RuntimeError):
    pass


def file_sha(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def sync_directory(path):
    fd = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def atomic_json(path, value):
    temporary = path.with_name(path.name + f".{os.getpid()}.tmp")
    with temporary.open("xb") as stream:
        stream.write(json.dumps(value, sort_keys=True, allow_nan=False).encode())
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)
    sync_directory(path.parent)


class LocalCommitStore:
    def __init__(self, root):
        self.root = Path(root).resolve()
        if "://" in str(root) or str(self.root).startswith("/mnt/hdfs/"):
            raise ValueError("Requires verified POSIX durability, not HDFS FUSE")
        self.root.mkdir(parents=True, exist_ok=True)

    @contextmanager
    def locked(self):
        with (self.root / ".lock").open("a+b") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            try:
                yield
            finally:
                fcntl.flock(lock, fcntl.LOCK_UN)

    def epoch(self):
        path = self.root / "epoch.json"
        return json.loads(path.read_text())["epoch"] if path.exists() else 0

    def acquire_epoch(self):
        with self.locked():
            epoch = self.epoch() + 1
            atomic_json(self.root / "epoch.json", {"epoch": epoch})
            return epoch

    def stage(self, epoch, next_update):
        with self.locked():
            if epoch != self.epoch():
                raise StaleEpoch(epoch)
            if next_update < 0:
                raise ValueError("Invalid update cursor")
            path = self.root / f"epoch-{epoch}-update-{next_update}"
            path.mkdir()
            sync_directory(self.root)
            return path

    def commit(self, path, epoch, metadata):
        required = {"next_update", "policy_version", "data_cursor", "plan_sha256",
                    "batch_sha256", "world_size", "contract"}
        if not required <= metadata.keys() or metadata["world_size"] < 1:
            raise ValueError("Incomplete update state")
        path = Path(path).resolve()
        if path.parent != self.root or path.name != f"epoch-{epoch}-update-{metadata['next_update']}":
            raise ValueError("Snapshot path/identity mismatch")
        files = {}
        for file in sorted(path.rglob("*")):
            if file.is_symlink():
                raise ValueError("Checkpoint symlink unsupported")
            if file.is_file():
                with file.open("rb") as stream:
                    os.fsync(stream.fileno())
                files[str(file.relative_to(path))] = {"sha256": file_sha(file), "bytes": file.stat().st_size}
        if "tensors/.metadata" not in files or any(
            f"rank-{r}.pt" not in files for r in range(metadata["world_size"])
        ):
            raise ValueError("Missing tensor metadata or per-rank state")
        from torch.distributed.checkpoint import FileSystemReader
        tensor_metadata = FileSystemReader(path / "tensors").read_metadata()
        for location in tensor_metadata.storage_data.values():
            name = "tensors/" + location.relative_path
            if name not in files or files[name]["bytes"] < location.offset + location.length:
                raise ValueError(f"Missing or truncated DCP shard: {name}")
        manifest = {"schema": 1, "epoch": epoch, "metadata": metadata, "files": files}
        with self.locked():
            if epoch != self.epoch():
                raise StaleEpoch(epoch)
            latest = self.root / "latest.json"
            if latest.exists() and metadata["next_update"] <= self.read()["metadata"]["next_update"]:
                raise ValueError("Durable update cursor must advance")
            if (path / "manifest.json").exists():
                raise ValueError("Already committed")
            sync_directory(path / "tensors")
            atomic_json(path / "manifest.json", manifest)
            atomic_json(latest, {"directory": path.name, "manifest_sha256": file_sha(path / "manifest.json")})
        return manifest

    def read(self):
        latest = json.loads((self.root / "latest.json").read_text())
        if Path(latest["directory"]).name != latest["directory"]:
            raise ValueError("Unsafe snapshot path")
        path = self.root / latest["directory"]
        if file_sha(path / "manifest.json") != latest["manifest_sha256"]:
            raise ValueError("Corrupt commit manifest")
        manifest = json.loads((path / "manifest.json").read_text())
        for name, expected in manifest["files"].items():
            file = path / name
            if not file.resolve().is_relative_to(path.resolve()):
                raise ValueError("Unsafe checkpoint member")
            if file.stat().st_size != expected["bytes"] or file_sha(file) != expected["sha256"]:
                raise ValueError(f"Corrupt checkpoint member: {name}")
        return dict(manifest, path=str(path))
