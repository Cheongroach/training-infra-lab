"""Aggregate frozen real-batch diagnostics without converting them to speed claims."""
import argparse
import json
import math
from pathlib import Path


def analyze(directory, *, allow_partial_profiles=False):
    rows = [json.loads(p.read_text()) for p in sorted(directory.glob("diagnostic-rank*.json"))]
    if len(rows) != 4:
        raise ValueError("The prescribed diagnostic requires all four rank files")
    prescribed = [(fp32, case) for fp32 in (False, True) for case in range(5)]
    for rank, records in enumerate(rows):
        if [(r["fp32_head"], r["case"]) for r in records] != prescribed:
            raise ValueError("Incomplete diagnostic matrix")
        if any(r["result"]["rank"] != rank for r in records):
            raise ValueError("Missing or misidentified rank")
    identities = {r["batch_sha256"] for records in rows for r in records}
    if len(identities) != 1:
        raise ValueError("Diagnostic workload changed")
    comparisons = []
    for index, (head, case) in enumerate(prescribed):
        if case == 0:
            continue
        records = [row[index] for row in rows]
        if len({r["result"]["plan_sha256"] for r in records}) != 1:
            raise ValueError("Ranks disagree on plan")
        combined = {}
        for field in ("parameter", "gradient", "exp_avg", "exp_avg_sq", "step"):
            es = [r["errors"][field] for r in records]
            numbers = [v for e in es for v in e.values()]
            if any(not math.isfinite(v) or v < 0 for v in numbers):
                raise ValueError("Nonfinite or negative metric")
            combined[field] = {
                "global_relative_l2": (sum(e["difference_squared"] for e in es) /
                                       max(sum(e["reference_squared"] for e in es), 1e-30))**0.5,
                "worst_rank_relative_l2": max(e["relative_l2"] for e in es),
                "max_abs": max(e["max_abs"] for e in es),
            }
        accepted = (all(e["worst_rank_relative_l2"] <= 0.08 for k, e in combined.items()
                        if k in ("gradient", "exp_avg", "exp_avg_sq"))
                    and combined["parameter"]["max_abs"] <= 3e-6)
        if any(r["accepted"] != accepted for r in records):
            raise ValueError("Acceptance differs from stored numeric errors")
        groups = {}
        for rank in range(len(rows)):
            evidence = json.loads((directory / f"tensors-head{head}-case{case}-rank{rank}.json").read_text())
            for name, e in evidence["groups"].items():
                if name.endswith("/gradient"):
                    a = groups.setdefault(name, [0.0, 0.0])
                    a[0] += e["difference_squared"]
                    a[1] += e["reference_squared"]
        top = sorted([{"name": k, "difference_l2": v[0]**0.5,
                       "reference_l2": v[1]**0.5} for k, v in groups.items()],
                     key=lambda g: g["difference_l2"], reverse=True)[:10]
        comparisons.append({"fp32_head": head, "case": case,
                            "strategy": records[0]["strategy"], "micro_batch": records[0]["micro_batch"],
                            "accepted": accepted, "errors": combined,
                            "probability_difference": records[0]["probability_difference"],
                            "largest_gradient_differences": top})
    profiles = []
    for path in sorted(directory.glob("profile-*/operators.json")):
        profile = json.loads(path.read_text())
        profiles.append({"path": str(path.relative_to(directory)), "rank": profile["rank"],
                         "identity": profile["identity"], "instrumented_seconds": profile["instrumented_seconds"],
                         "largest_device_operators": profile["operators"][:12]})
    complete = len(profiles) == len(rows) * 2
    if not complete and not allow_partial_profiles:
        raise ValueError("Incomplete independent profiler evidence")
    return {"world": len(rows), "batch_sha256": identities.pop(), "comparisons": comparisons,
            "profiles": profiles, "profiles_complete": complete,
            "expected_profile_count": len(rows) * 2, "performance_comparison": False,
            "scope": "one frozen real batch; forward/backward/state diagnostic; not on-policy training",
            "profile_semantics": "operator device times may overlap and cannot be summed as wall time"}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("directory", type=Path)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--allow-partial-profiles", action="store_true")
    args = p.parse_args()
    report = analyze(args.directory, allow_partial_profiles=args.allow_partial_profiles)
    args.output.write_text(json.dumps(report, indent=2))
    print(json.dumps({k: v for k, v in report.items() if k != "profiles"}, indent=2))


if __name__ == "__main__":
    main()
