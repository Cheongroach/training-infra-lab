"""Collect immutable result evidence and compare matched variants."""

import argparse
import gzip
import json
import statistics
import tarfile
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--store", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("tags", nargs="+")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    runs = []
    for tag in args.tags:
        folder = args.store / "runs" / tag
        archives = sorted(folder.glob("**/artifacts.tar.gz"))
        for archive in archives:
            node = archive.parent.relative_to(folder)
            with tarfile.open(archive) as tar:
                for member in tar.getmembers():
                    path = Path(member.name)
                    if not member.isfile() or path.is_absolute() or ".." in path.parts:
                        continue
                    if path.parts[0] != "results":
                        continue
                    # Collect user-facing data and gates, not installation logs
                    # or potentially sensitive platform environment dumps.
                    if path.suffix not in (".json", ".txt") or path.name == "environment.txt":
                        continue
                    content = tar.extractfile(member).read()
                    destination = args.output / tag / node / path
                    destination.parent.mkdir(parents=True, exist_ok=True)
                    if path.name.startswith("trace-rank-"):
                        with gzip.open(destination.with_suffix(".json.gz"), "wb", compresslevel=5) as file:
                            file.write(content)
                        if destination.exists():
                            destination.unlink()
                    else:
                        destination.write_bytes(content)
    # Aggregate the complete local evidence set, including earlier collected
    # immutable jobs. New tags must not erase those from the index.
    for destination in sorted(args.output.glob("**/result.json")):
        result = json.loads(destination.read_text())
        if "steps" not in result:
            continue
        result["tag"] = destination.relative_to(args.output).parts[0]
        result["variant"] = destination.parent.name
        result["artifact"] = str(destination.relative_to(args.output))
        total = sum(s["padded_tokens"] for s in result["steps"] if s["measured"])
        useful = sum(s["nonpad_tokens"] for s in result["steps"] if s["measured"])
        result["padding_fraction"] = 1 - useful / total
        runs.append(result)
    (args.output / "all_results.json").write_text(json.dumps(runs, indent=2))
    columns = [
        "tag", "variant", "world_size", "nonpad_tokens_per_second",
        "supervised_tokens_per_second", "peak_allocated_gib", "padding_fraction",
    ]
    print("\t".join(columns))
    for result in runs:
        print("\t".join(str(round(result[c], 3) if isinstance(result[c], float) else result[c]) for c in columns))
    groups = {}
    for result in runs:
        key = (result["tag"], result["device"], result["world_size"], result["global_batch"],
               result["seq_length"], result.get("padding", "fixed"),
               result["mode"], result["activation_checkpoint"],
               result.get("defer_gradient_sync", False), result.get("deterministic", False),
               tuple(s["step"] for s in result["steps"] if s["measured"]))
        groups.setdefault(key, []).append(result)
    summary = [{
        "configuration": key, "repeats": len(values),
        "median_nonpad_tokens_per_second": statistics.median(v["nonpad_tokens_per_second"] for v in values),
        "min_nonpad_tokens_per_second": min(v["nonpad_tokens_per_second"] for v in values),
        "max_nonpad_tokens_per_second": max(v["nonpad_tokens_per_second"] for v in values),
        "max_peak_allocated_gib": max(v["peak_allocated_gib"] for v in values),
    } for key, values in groups.items()]
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
