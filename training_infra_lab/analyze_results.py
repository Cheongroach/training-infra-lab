"""Generate report figures and matched statistics from collected result JSON."""

import gzip
import json
import statistics
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).parent / "results"
FIGURES = ROOT / "figures"
FIGURES.mkdir(exist_ok=True)
rows = json.loads((ROOT / "all_results.json").read_text())
plt.rcParams.update({"font.size": 10, "axes.spines.top": False,
                     "axes.spines.right": False, "savefig.dpi": 160})


def stats(values):
    assert values
    tokens = [v["nonpad_tokens_per_second"] for v in values]
    return {"repeats": len(values), "tokens_per_second": tokens,
            "median": statistics.median(tokens), "min": min(tokens), "max": max(tokens),
            "peak_allocated_gib": max(v["peak_allocated_gib"] for v in values),
            "supervised_tokens_per_second": [v["supervised_tokens_per_second"] for v in values],
            "artifacts": [v["artifact"] for v in values]}


report = {"benchmark": {}, "scale": {}, "sync": {}, "profiles": []}
variants = [("full", "fixed", False), ("full", "dynamic", False),
            ("selected", "dynamic", False), ("chunk", "dynamic", False),
            ("chunk", "dynamic", True)]
labels = ["Fixed / full", "Dynamic / full", "Dynamic / selected",
          "Dynamic / chunk", "Dynamic / chunk + AC"]
colors = ["#94a3b8", "#457b9d", "#2a9d8f", "#e9c46a", "#e76f51"]
fig, axes = plt.subplots(2, 2, figsize=(12, 7.8), layout="constrained")
for col, (hardware, tag) in enumerate((("H20", "h20-benchmark-v1"), ("910B4", "npu-benchmark-v1"))):
    values = []
    for mode, padding, ac in variants:
        v = [r for r in rows if r["tag"] == tag and r["mode"] == mode
             and r["padding"] == padding and r["activation_checkpoint"] == ac]
        summary = stats(v)
        report["benchmark"][f"{hardware}/{mode}/{padding}/ac{int(ac)}"] = summary
        values.append(summary)
    for ax, key, title in ((axes[0, col], "median", "Non-padding tokens/s"),
                           (axes[1, col], "peak_allocated_gib", "Peak allocated memory (GiB / device)")):
        ax.barh(labels, [v[key] for v in values], color=colors)
        if key == "median":
            for i, v in enumerate(values):
                ax.scatter(v["tokens_per_second"], [i] * v["repeats"], color="#172b4d", s=16, zorder=3)
        for i, v in enumerate(values):
            ax.text((v["max"] if key == "median" else v[key]) * 1.02, i, f"{v[key]:,.0f}" if key == "median" else f"{v[key]:.2f}", va="center", fontsize=9)
        ax.invert_yaxis()
        ax.set_xlim(0, max(v[key] for v in values) * 1.2)
        ax.set_title(f"2 x {hardware} | {title}")
        ax.grid(axis="x", alpha=0.15)
fig.suptitle("Qwen2.5-1.5B full SFT: matched ablations\n16 steps; 4 warmup; global batch 8; dots show independent runs", fontsize=13)
fig.savefig(FIGURES / "ablations.png")
plt.close(fig)

fig, axes = plt.subplots(1, 2, figsize=(10, 4.1), layout="constrained")
for n, tag in ((2, "h20-scale2-v1"), (8, "h20-scale8-v1"), (16, "h20-scale16-v2")):
    report["scale"][str(n)] = stats([r for r in rows if r["tag"] == tag])
x = [2, 8, 16]
y = [report["scale"][str(n)]["median"] for n in x]
axes[0].plot(x, y, "o-", color="#2a9d8f", label="Measured median")
axes[0].plot(x, [y[0] * n / 2 for n in x], "--", color="#94a3b8", label="Ideal from 2 GPUs")
for n, value in zip(x, y):
    axes[0].annotate(f"{value:,.0f}", (n, value), xytext=(0, 8), textcoords="offset points", ha="center")
axes[0].set(xlabel="H20 GPUs", ylabel="Non-padding tokens/s", xticks=x)
axes[0].legend(frameon=False)
efficiency = [value / y[0] / (n / 2) * 100 for n, value in zip(x, y)]
axes[1].bar([str(n) for n in x], efficiency, color="#457b9d")
for i, v in enumerate(efficiency):
    axes[1].text(i, v + 2, f"{v:.1f}%", ha="center")
axes[1].set(xlabel="H20 GPUs", ylabel="Strong-scaling efficiency (%)", ylim=(0, 115))
fig.suptitle("Fixed global batch 16 | 20 steps, 4 warmup, 2 runs\n16 GPUs: two physical hosts, RoCE/GDRDMA observed", fontsize=12)
fig.savefig(FIGURES / "strong_scaling.png")
plt.close(fig)
for n, eff in zip(x, efficiency):
    report["scale"][str(n)]["efficiency"] = eff / 100
    report["scale"][str(n)]["speedup_from_2"] = report["scale"][str(n)]["median"] / y[0]

for hardware, tag in (("H20", "h20-sync-v1"), ("910B4", "npu-sync-v1")):
    for deferred in (False, True):
        values = [r for r in rows if r["tag"] == tag and r.get("defer_gradient_sync") == deferred
                  and not any(s.get("profiled", False) for s in r["steps"])]
        if values:
            report["sync"][f"{hardware}/{'last' if deferred else 'each'}"] = stats(values)

# Count actual NCCL kernels only; API/annotation events can duplicate the same
# communication. Durations overlap and are NOT a wall-time decomposition.
for path in sorted(ROOT.glob("**/trace-rank-0.json*")):
    with gzip.open(path, "rt") if path.suffix == ".gz" else path.open() as file:
        trace = json.load(file)
    kernels = [e for e in trace["traceEvents"] if e.get("cat") == "kernel"]
    profile = {"artifact": str(path.relative_to(ROOT))}
    for operation in ("ReduceScatter", "AllGather"):
        selected = [e for e in kernels if "nccl" in e["name"] and operation in e["name"]]
        profile[operation] = {"calls": len(selected), "sum_kernel_ms": sum(e.get("dur", 0) for e in selected) / 1000}
    report["profiles"].append(profile)
(ROOT / "report_metrics.json").write_text(json.dumps(report, indent=2))
print(json.dumps({"figures": str(FIGURES), "matched_groups": len(report["benchmark"]) + len(report["scale"]) + len(report["sync"])}))
