"""Checkpoint the bounded sampler journal; the Miles actor tracker commits training."""
import hashlib
import json
import os
from pathlib import Path
import shutil
from uuid import uuid4

from qualification import MAX_CANDIDATES, assess_candidate

CURSOR_KEYS = ("epoch_id", "sample_offset", "sample_index", "sample_group_index")


def digest(content):
    return hashlib.sha256(content).hexdigest()


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def read_regular(path):
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"Missing or non-regular journal file: {path.name}")
    return path.read_bytes()


def journal_prefix(directory, completed, cursor, expected_seen=None):
    """Recompute every admitted/discarded identity and behavior-version span."""
    if type(completed) is not int or not 0 <= completed < 3:
        raise ValueError("Recovery is limited to the three-update synchronous smoke")
    if set(cursor) != set(CURSOR_KEYS) or any(type(v) is not int or v < 0 for v in cursor.values()):
        raise ValueError("Invalid datasource cursor")
    files, seen = {}, set()
    for rid in range(completed + 1):
        name = f"step-{rid}.json"
        blob = read_regular(directory / name)
        files[name] = blob
        manifest = json.loads(blob)
        attempts = manifest["attempts"]
        if (type(manifest["rollout_id"]) is not int or type(manifest["weight_version"]) is not int
                or manifest["rollout_id"] != rid or manifest["weight_version"] != rid
                or manifest["max_candidates"] != MAX_CANDIDATES
                or manifest["passed"] is not True or not 1 <= len(attempts) <= MAX_CANDIDATES):
            raise ValueError(f"Invalid committed qualification step {rid}")
        for number, record in enumerate(attempts):
            name = f"candidate-{rid}-{number}.json"
            if record["attempt"] != number or record["file"] != name:
                raise ValueError("Journal sequence or path mismatch")
            blob = read_regular(directory / name)
            if digest(blob) != record["sha256"]:
                raise ValueError("Candidate hash mismatch")
            batch = json.loads(blob)
            report = assess_candidate(batch["samples"], batch["issued"], seen,
                                      rollout_id=rid, weight_version=rid)
            last = number == len(attempts) - 1
            if (report != record["assessment"]
                    or record["decision"] != ("admitted" if last else "rejected")
                    or not (report["passed"] if last else report["retryable"])
                    or record["sample_indices"] != [s["index"] for s in batch["samples"]]):
                raise ValueError("Journal decision or identity mismatch")
            seen.update(record["sample_indices"])
            files[name] = blob
    if seen != set(range(cursor["sample_index"])):
        raise ValueError("Datasource sample cursor and journal disagree")
    if cursor["sample_index"] != 4 * cursor["sample_group_index"]:
        raise ValueError("Datasource group cursor and journal disagree")
    if expected_seen is not None and seen != set(expected_seen):
        raise ValueError("Sampler identity state and journal disagree")
    if {p.name for p in directory.iterdir()} != set(files):
        raise ValueError("Unaccounted journal files")
    return files, seen


def save_journal(directory, checkpoint, *, completed, cursor, seen, contract):
    """Called inside Miles' atomic rollout folder, before actor tracker publication."""
    files, _ = journal_prefix(directory, completed, cursor, seen)
    checkpoint.mkdir(parents=True, exist_ok=False)
    target = checkpoint / "journal"
    target.mkdir()
    for name, blob in files.items():
        (target / name).write_bytes(blob)
    state = {"schema": 1, "completed_rollout_id": completed,
             "next_rollout_id": completed + 1, "next_weight_version": completed + 1,
             "cursor": cursor, "seen_indices": sorted(seen), "contract": contract,
             "files": {name: digest(blob) for name, blob in files.items()}}
    blob = canonical(state)
    (checkpoint / "state.json").write_bytes(blob)
    (checkpoint / "state.sha256").write_text(digest(blob))
    return state


def load_journal(checkpoint, *, cursor, contract):
    """Validate before changing the active journal or accepting any new samples."""
    blob = read_regular(checkpoint / "state.json")
    if digest(blob) != read_regular(checkpoint / "state.sha256").decode().strip():
        raise ValueError("Qualification checkpoint hash mismatch")
    state = json.loads(blob)
    if state["schema"] != 1 or state["contract"] != contract or state["cursor"] != cursor:
        raise ValueError("Recovery contract or restored datasource cursor mismatch")
    if (not isinstance(state["seen_indices"], list)
            or any(type(i) is not int or i < 0 for i in state["seen_indices"])
            or state["seen_indices"] != sorted(set(state["seen_indices"]))):
        raise ValueError("Invalid persisted sampler identities")
    completed = state["completed_rollout_id"]
    if state["next_rollout_id"] != completed + 1 or state["next_weight_version"] != completed + 1:
        raise ValueError("Recovery rollout/version mismatch")
    files, seen = journal_prefix(checkpoint / "journal", completed, cursor, state["seen_indices"])
    if state["files"] != {name: digest(content) for name, content in files.items()}:
        raise ValueError("Qualification checkpoint inventory mismatch")
    return state, files, seen


def restore_journal(checkpoint, results, *, cursor, contract):
    state, files, seen = load_journal(checkpoint, cursor=cursor, contract=contract)
    epoch = uuid4().hex
    staging = results / f".qualification-restore-{epoch}"
    active = results / "qualification"
    history = results / f"qualification-before-restore-{epoch}"
    staging.mkdir()
    try:
        for name, blob in files.items():
            (staging / name).write_bytes(blob)
        if active.is_symlink():
            raise ValueError("Active journal must not be a symlink")
        moved = active.exists()
        if moved:
            os.replace(active, history)
        try:
            os.replace(staging, active)
        except Exception:
            if moved:
                os.replace(history, active)
            raise
    finally:
        if staging.exists():
            shutil.rmtree(staging)
    report = {"restored": True, "completed_rollout_id": state["completed_rollout_id"],
              "next_rollout_id": state["next_rollout_id"], "cursor": cursor,
              "seen_samples": len(seen), "checkpoint_sha256": digest(canonical(state)),
              "preserved_previous_journal": history.name if moved else None}
    (results / "qualification-restored.json").write_text(json.dumps(report, indent=2))
    return state, seen
