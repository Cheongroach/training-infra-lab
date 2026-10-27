"""Create a pinned MATH train subset; selection never observes model outputs."""
import argparse
import hashlib
import json
import re
from pathlib import Path


def boxed_integer(solution):
    begin = solution.rfind("\\boxed{")
    if begin < 0:
        return None
    begin += len("\\boxed{")
    depth = 1
    for end in range(begin, len(solution)):
        depth += (solution[end] == "{") - (solution[end] == "}")
        if depth == 0:
            answer = solution[begin:end].replace(",", "").strip()
            return str(int(answer)) if re.fullmatch(r"-?\d+", answer) else None
    return None


def prepare(source, output, revision):
    import pyarrow.parquet as pq

    groups = {"Level 4": [], "Level 5": []}
    for index, row in enumerate(pq.read_table(source).to_pylist()):
        answer = boxed_integer(row["solution"])
        if (answer is None or row["level"] not in groups or
                len(row["problem"]) > 1400 or "[asy]" in row["problem"]):
            continue
        key = hashlib.sha256(row["problem"].encode()).hexdigest()
        groups[row["level"]].append((key, {
            "question": row["problem"], "answer": "#### " + answer,
            "source_id": f"EleutherAI/hendrycks_math/{revision}/algebra/train/{index}",
            "level": row["level"], "problem_sha256": key,
        }))
    for values in groups.values():
        values.sort(key=lambda value: value[0])
        if len(values) < 32:
            raise ValueError("Insufficient eligible training problems")
    rows = [groups[level][i][1] for i in range(32) for level in groups]
    if len({row["problem_sha256"] for row in rows}) != len(rows):
        raise ValueError("Duplicate selected problems")
    output.mkdir(parents=True, exist_ok=True)
    data = "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows).encode()
    (output / "data.jsonl").write_bytes(data)
    manifest = {
        "dataset": "EleutherAI/hendrycks_math", "revision": revision,
        "split": "algebra/train", "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "sha256": hashlib.sha256(data).hexdigest(), "rows": len(rows),
        "selection": "balanced Level 4/5; last boxed answer is integer; no Asymptote; <=1400 chars; question SHA256 order",
        "eligible_per_level": {key: len(value) for key, value in groups.items()},
        "usage": "training infrastructure workload; not held-out evaluation; no observed-output selection",
        "reward": "last_decimal_exact_v1; numeric-only subset, not a general MATH equivalence evaluator",
    }
    (output / "data_manifest.json").write_text(json.dumps(manifest, indent=2))
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("source", type=Path)
    p.add_argument("output", type=Path)
    p.add_argument("--revision", required=True)
    args = p.parse_args()
    prepare(args.source, args.output, args.revision)
