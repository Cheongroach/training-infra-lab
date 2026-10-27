"""Export figures from terminal audits and captured resource points."""

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from summarize_results import summarize


def main(root):
    summary = summarize(root)
    out = root / "figures"
    out.mkdir(exist_ok=True)
    sources = set()
    figures = []
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
                         "axes.spines.top": False, "axes.spines.right": False})

    def save(fig, name):
        fig.tight_layout()
        for suffix in ("png", "svg"):
            target = out / f"{name}.{suffix}"
            fig.savefig(target, dpi=180)
            figures.append(target.name)
        plt.close(fig)

    terminal = [r for r in summary["runs"] if r["terminal"] and r["gpu_hours"] is not None]
    if terminal:
        fig, ax = plt.subplots(figsize=(10, 3.5))
        values = [r["gpu_hours"] for r in terminal]
        ax.bar([r["name"] for r in terminal], values,
               color=["#16866e" if r["training_verified"] else "#64748b" for r in terminal])
        for i, value in enumerate(values):
            ax.text(i, value, f"{value:.3f}", ha="center", va="bottom", fontsize=9)
        ax.set_ylabel("H20 GPU-hours")
        ax.set_title("Terminal Pod residence: preparation, failures and diagnostics included")
        ax.tick_params(axis="x", rotation=20)
        save(fig, "terminal-cost")
        sources.update(root / r["source"] / "verified.json" for r in terminal)

    for run in summary["runs"]:
        folder = root / run["source"]
        if run["training_verified"]:
            updates = run["verified_updates"]
            ids = [u["rollout_id"] for u in updates]
            fig, axes = plt.subplots(1, 2, figsize=(10, 3.5))
            axes[0].plot(ids, [u["grad_norm"][0] for u in updates], "o-", color="#16866e")
            axes[0].set_ylabel("Recorded global grad norm")
            axes[1].plot(ids, [u["completed_samples"] for u in updates], "o-", label="Natural completion")
            axes[1].plot(ids, [u["strict_extracted_samples"] for u in updates], "s-", label="Strict extraction")
            axes[1].set_ylim(0, 66)
            axes[1].set_ylabel("Samples out of 64")
            axes[1].legend(frameon=False)
            for ax in axes:
                ax.set_xlabel("Rollout / optimizer update")
                ax.set_xticks(ids)
            fig.suptitle(f"{run['name']}: audited training signal; no quality improvement claim")
            save(fig, run["name"] + "-training")
            sources.add(folder / "training-audit.json")
        baseline = run["stage_baseline"]
        if baseline.get("passed"):
            fig, ax = plt.subplots(figsize=(10, 3.5))
            rows = baseline["rows"]
            kinds = [("rollout_seconds", "Rollout / 16 GPUs"),
                     ("actor_train_seconds_rank0", "Actor train / rank 0"),
                     ("log_probs_seconds_rank0", "Log probabilities / rank 0"),
                     ("publication_seconds_rank0", "Publication / rank 0")]
            x = np.arange(len(rows))
            for i, (key, label) in enumerate(kinds):
                ax.bar(x + (i - 1.5) * .18, [r[key] for r in rows], .18, label=label)
            ax.set_xticks(x, [r["rollout_id"] for r in rows])
            ax.set_xlabel("Rollout / optimizer update")
            ax.set_ylabel("Recorded stage seconds")
            ax.set_title("Diagnostic stage timers: scopes differ; bars are not additive")
            ax.legend(frameon=False, fontsize=8, loc="upper center",
                      bbox_to_anchor=(.5, -.22), ncol=2)
            save(fig, run["name"] + "-stage-baseline")
            sources.add(folder / "stage-baseline.json")
        benchmark = run["publication_benchmark"]
        if run["training_verified"] and benchmark.get("passed"):
            fig, axes = plt.subplots(1, 2, figsize=(12, 4))
            for i, (name, data) in enumerate(benchmark["comparisons"].items()):
                values = data["wall_seconds"]
                axes[0].scatter(np.full(len(values), i), values, color="#2563eb", zorder=3)
                axes[0].plot([i - .15, i + .15], [data["median_seconds"]] * 2, color="#111827")
                axes[1].bar(i, data["transfer_median_seconds"], color="#16866e")
            for ax in axes:
                ax.set_xticks(range(4), list(benchmark["comparisons"]), rotation=20)
            axes[0].set_ylabel("Critical-rank wall seconds")
            axes[0].set_title("Wall samples: checksum scope changes")
            axes[1].set_ylabel("Critical-rank transfer median seconds")
            axes[1].set_title("Materialize, publish and receive acknowledgement")
            fig.suptitle("Same-content publication only: four balanced repetitions per case")
            save(fig, run["name"] + "-publication")
            sources.add(folder / "publication-benchmark-audit.json")
        # Each captured series keeps the provider's actual Pod aggregation.
        gpu = folder / "grafana/panel-7.summary.json"
        if run["terminal"] and gpu.is_file():
            report = json.loads(gpu.read_text())
            if not report["series"]:
                continue
            fig, ax = plt.subplots(figsize=(10, 3.5))
            for item in report["series"]:
                points = sorted(item["points"])
                pod = next((v for k, v in item["tags"].items() if "task" in k or "pod" in k),
                           json.dumps(item["tags"], sort_keys=True))
                times = [datetime.fromtimestamp(p[0], timezone.utc) for p in points]
                ax.plot(times, [p[1] for p in points], label=str(pod), linewidth=1.2)
            ax.set_ylabel("Source GPU duty-cycle values")
            ax.set_xlabel("UTC capture time")
            ax.set_title(f"{run['name']}: captured Pod series; missing points remain missing")
            ax.legend(frameon=False, fontsize=7)
            fig.autofmt_xdate()
            save(fig, run["name"] + "-gpu-duty")
            sources.add(gpu)
    manifest = {
        "figures": figures, "source_sha256": {
            str(path.relative_to(root)): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in sorted(sources)
        },
        "scope": "Terminal costs, independent audits and captured Pod resource series",
        "quality_improvement_claim": False, "end_to_end_speedup_claim": False,
    }
    (out / "figure-manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({"figures": figures, "sources": len(sources)}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parent)
    main(parser.parse_args().root)
