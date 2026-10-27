"""Stage authored scale32 code and selected evidence with source/publication hashes."""

import argparse
import hashlib
import json
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parents[3]
SOURCE = Path(__file__).resolve().parent
SECRET = re.compile(
    r"\b(?:gh[pousr]_[A-Za-z0-9]{25,}|github_pat_[A-Za-z0-9_]{25,}|"
    r"eyJ[A-Za-z0-9_-]{12,}\.[A-Za-z0-9_-]{12,}\.[A-Za-z0-9_-]{12,})\b"
    r"|Bearer\s+[A-Za-z0-9_.+/=-]{20,}")
KEY = re.compile(r"(?:^|_)(?:password|secret|authorization|api_key|access_key|"
                 r"auth_token|access_token|refresh_token|jwt|credential)(?:$|_)", re.I)
RUN_FILES = {
    "verified.json", "training-audit.json", "publication-benchmark-audit.json",
    "status.json", "source-manifest.json", "manifest.json", "local-validation.json", "stage-baseline.json",
    "precheck-observed.json", "failure-analysis.json",
}


def clean(value):
    if isinstance(value, dict):
        result = {key: "[credential redacted]" if KEY.search(key) else clean(item)
                  for key, item in value.items()}
        # Environment dumps can contain credentials under arbitrary provider-specific names.
        if value.get("type") in ("env_report", "engine_env_report"):
            for key in ("report", "server_info"):
                if key in result:
                    result[key] = {"publication_note": "Environment payload omitted; raw source hash retained"}
        return result
    if isinstance(value, list):
        return [clean(item) for item in value]
    return SECRET.sub("[credential redacted]", value) if isinstance(value, str) else value


def selected():
    yield from (p for p in SOURCE.iterdir()
                if p.is_file() and p.suffix in (".py", ".md", ".json", ".patch"))
    yield from (SOURCE / "format-data").glob("*.json*")
    yield from (p for p in (SOURCE / "figures").glob("*")
                if p.is_file() and p.suffix in (".png", ".svg", ".json"))
    for directory in sorted((SOURCE / "runs").iterdir()):
        # Unsubmitted drafts are development history, not runtime evidence.
        if not any((directory / name).is_file()
                   for name in ("create-response.json", "create-result.json")):
            continue
        yield from (directory / name for name in sorted(RUN_FILES) if (directory / name).is_file())
        for path in (directory / "grafana").glob("*.json"):
            if path.name.endswith(".summary.json"):
                yield path
        for parent in (directory / "shared-results", directory / "artifacts/results",
                       *(directory / f"node-{rank}/results" for rank in range(4))):
            if parent.is_dir():
                for path in sorted(parent.rglob("*")):
                    if not path.is_file() or "__pycache__" in path.parts:
                        continue
                    if path.suffix in (".json", ".jsonl") or (
                        path.suffix == ".pt" and path.name.startswith(("train-", "rollout-"))
                    ):
                        yield path
        for parent in (directory / "artifacts/code",
                       *(directory / f"node-{rank}/code" for rank in range(4))):
            if parent.is_dir():
                yield from (p for p in parent.rglob("*") if p.is_file()
                            and p.suffix in (".py", ".json", ".jsonl", ".patch"))


def stage(publication):
    publication = publication.resolve()
    if not publication.is_relative_to(ROOT) or not (publication / "provenance/source-inventory.json").is_file():
        raise ValueError("Use an existing publication checkout inside this workspace")
    inventory_path = publication / "provenance/source-inventory.json"
    inventory = json.loads(inventory_path.read_text())
    latest = {item["destination"]: item for item in inventory}
    records = []
    for source in sorted(set(selected())):
        relative = source.relative_to(ROOT)
        raw = source.read_bytes()
        if len(raw) > 95_000_000:
            raise ValueError(f"Evidence exceeds publication file limit: {relative}")
        if source.suffix in (".json", ".jsonl"):
            if source.suffix == ".json":
                original = json.loads(raw)
                redacted = clean(original)
                content = raw if redacted == original else (
                    json.dumps(redacted, ensure_ascii=False, indent=2) + "\n").encode()
            else:
                original = [json.loads(line) for line in raw.decode().splitlines() if line.strip()]
                redacted = [clean(row) for row in original]
                content = raw if redacted == original else "".join(
                    json.dumps(row, ensure_ascii=False) + "\n" for row in redacted).encode()
        else:
            content = raw
        if source.suffix != ".pt" and SECRET.search(content.decode(errors="replace")):
            raise ValueError(f"Credential marker in selected source: {relative}")
        destination = publication / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(content)
        record = {
            "source": str(relative), "destination": str(relative),
            "source_sha256": hashlib.sha256(raw).hexdigest(),
            "published_sha256": hashlib.sha256(content).hexdigest(),
        }
        latest[str(relative)] = record
        records.append(record)
    inventory_path.write_text(json.dumps(list(latest.values()), ensure_ascii=False, indent=2) + "\n")
    scope = {
        "files_staged": len(records), "source": str(SOURCE.relative_to(ROOT)),
        "excluded": ["model weights and converted checkpoints", "raw startup/stdout/environment logs",
                     "signed log URLs and raw task requests", "unsubmitted draft runs"],
        "environment_event_policy": "Preserve event identity and omit environment payload",
        "source_records": records,
    }
    (publication / "provenance/scale32-stage.json").write_text(json.dumps(scope, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"files_staged": len(records), "publication": str(publication)}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--publication", type=Path, required=True)
    args = parser.parse_args()
    stage(args.publication)
