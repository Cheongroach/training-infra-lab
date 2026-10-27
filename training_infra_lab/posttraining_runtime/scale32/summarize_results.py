"""Recompute expansion costs and summarize only independently collected evidence."""

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path


def read(path):
    return json.loads(path.read_text()) if path.is_file() else {}


def summarize(root):
    runs = []
    for folder in sorted((root / "runs").iterdir()):
        if not any((folder / name).is_file() for name in ("create-response.json", "create-result.json")):
            continue
        terminal = read(folder / "verified.json")
        status = terminal or read(folder / "status.json")
        manifest = read(folder / "manifest.json")
        audit = read(folder / "training-audit.json")
        benchmark = read(folder / "publication-benchmark-audit.json")
        baseline = {}
        if terminal.get("training_verified") and audit.get("passed") and not manifest.get("performance_run"):
            from derive_stage_baseline import derive
            baseline = derive(folder)
        events = [
            json.loads(line)
            for path in sorted((folder / "shared-results/events").rglob("*.jsonl"))
            for line in path.read_text().splitlines() if line.strip()
        ]
        stages = {}
        for event in events:
            if event.get("type") == "metric":
                for key, value in event["metrics"].items():
                    if key.startswith("perf/") and isinstance(value, (int, float)):
                        stages.setdefault(key, []).append({
                            "rollout_id": event.get("rollout_id"), "value": value,
                            "source": event.get("source"),
                        })
        # Train group completion is an observation; semantic verification is separate.
        observed_updates = sorted({
            e["rollout_id"] for e in events
            if e.get("type") == "train_group_step_end" and e.get("role") == "actor"
        })
        runs.append({
            "name": folder.name, "run": status.get("run", status.get("run_sid")),
            "status": status.get("status", "submitted"), "terminal": bool(status.get("terminal")),
            "gpu_hours": terminal.get("gpu_hours"),
            "cost_scope": "Terminal Pod residence including image preparation, assets and diagnostics",
            "training_verified": bool(terminal.get("training_verified") and audit.get("passed")),
            "observed_optimizer_boundaries": observed_updates,
            "verified_updates": audit.get("updates", []) if audit.get("passed") else [],
            "final_evaluation": audit.get("final_evaluation"),
            "publication_benchmark": benchmark,
            "stage_baseline": baseline,
            "stage_metrics": stages,
            "failure": read(folder / "failure-analysis.json"),
            "environment_verified": bool(terminal.get("verified") and folder.name.startswith("preflight")),
            "format_probe_audit_complete": terminal.get("audit_complete", False),
            "format_probe_protocol_passed": terminal.get("protocol_passed", False),
            "allocated_gpus": manifest.get("allocated_gpus"),
            "source": str(folder.relative_to(root)),
        })
    result = {
        "observed_at": datetime.now(timezone.utc).isoformat(), "runs": runs,
        "terminal_gpu_hours": sum(r["gpu_hours"] for r in runs
                                  if r["terminal"] and r["gpu_hours"] is not None),
        "unfinished_runs": [r["name"] for r in runs if not r["terminal"]],
        "verified_training_runs": [r["name"] for r in runs if r["training_verified"]],
        "sota_claim": False, "held_out_quality_improvement_claim": False,
        "end_to_end_speedup_claim": False,
    }
    (root / "results-summary.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    lines = [
        "# 32 卡 GRPO 实测与证据", "",
        f"采集时间：{result['observed_at']}。所有成本从终态 Pod 驻留计算，包含准备、失败和重算；运行中的任务尚未计入。",
        "",
        "配置：4 worker × 8 H20；每 worker 64 CPU / 256 GiB。Qwen3-8B；16 learner，TP2 × DP8、PP1、CP1；16 rollout，8 个 TP2 引擎。"
        "Miles commit `2233d795dcf167be489d1859971c9233393f04c3`。模型使用逐文件 immutable revision 与 SHA256，未验证 HF commit 一致性。",
        "",
        "| 任务 | 状态 | 真实优化边界 | 独立训练验收 | H20·h |",
        "|---|---|---:|---|---:|",
    ]
    for run in runs:
        cost = f"{run['gpu_hours']:.4f}" if run["gpu_hours"] is not None else "待终态"
        lines.append(f"| [{run['name']}]({run['source']}/status.json) | {run['status']} | "
                     f"{len(run['observed_optimizer_boundaries'])} | "
                     f"{'通过' if run['training_verified'] else '未通过/待验收'} | {cost} |")
    lines.extend([
        "", f"已终态扩展任务合计 **{result['terminal_gpu_hours']:.4f} H20·h**。",
        "",
        "![终态任务成本](figures/terminal-cost.png)", "",
        "## 参数与发布验收", "",
        "训练通过要求：完整组与奖励变化、每条样本恰好消费一次、有限行为概率、16 learner rank 参数/Adam m/v 的实际变化、"
        "同 TP 分片的八份 DP 模型副本一致、八个 TP2 引擎内容一致、v0→v3 发布和最终生成 token 消费 v3。"
        "末次评估使用已暴露题目，仅证明协议；不用于能力收益或 SOTA。",
    ])
    for run in runs:
        if not run["training_verified"]:
            continue
        lines.extend(["", f"### {run['name']}", "",
                      "| 更新 | 样本 | 自然完成 | 严格格式 | 奖励不同组 | 非零 advantage token | grad norm |",
                      "|---|---:|---:|---:|---:|---:|---|"])
        for update in run["verified_updates"]:
            lines.append(
                f"| {update['rollout_id']} | {update['admitted_samples']} | {update['completed_samples']} | "
                f"{update['strict_extracted_samples']} | {update['mixed_reward_groups']} | "
                f"{update['nonzero_advantage_tokens']} | {update['grad_norm']} |")
        lines.extend(["", "末次暴露题评估：`" + json.dumps(run["final_evaluation"], ensure_ascii=False) + "`。",
                      "", "![参数信号与格式](figures/" + run["name"] + "-training.png)"])
    lines.extend(["", "## 阶段计时与全部卡时的差别", "",
                  "以下 learner 时间来自已锁定的 actor rank 0 指标，不代表全 rank 最大墙钟；"
                  "不同阶段的计时区间不得随意相加。吞吐分母必须与实际工作量和设备范围对应。"])
    for run in runs:
        baseline = run["stage_baseline"]
        if not baseline.get("passed"):
            continue
        total = baseline["totals"]
        lines.extend(["", f"### {run['name']}", "",
                      "| 更新 | response token | 非零 advantage 比例 | rollout（s） | actor rank0（s） | log-prob rank0（s） | 发布 rank0（s） |",
                      "|---|---:|---:|---:|---:|---:|---:|"])
        for row in baseline["rows"]:
            lines.append(f"| {row['rollout_id']} | {row['response_tokens']} | "
                         f"{row['nonzero_advantage_fraction']:.2%} | {row['rollout_seconds']:.3f} | "
                         f"{row['actor_train_seconds_rank0']:.3f} | {row['log_probs_seconds_rank0']:.3f} | "
                         f"{row['publication_seconds_rank0']:.3f} |")
        lines.extend(["", f"三轮合计 {total['admitted_response_tokens']:,} 个训练回答 token，"
                      f"其中 {total['nonzero_advantage_fraction']:.2%} 为非零 advantage token。"
                      "该比例表示学习信号覆盖，不能当作正确率。",
                      "", f"rollout 阶段：{total['response_tokens_per_rollout_gpu_second_pooled']:.3f} "
                      "response token / rollout GPU·s；分母是 16 个 rollout GPU 的生成计时。"
                      f"全部驻留成本：{total['admitted_response_tokens_per_total_gpu_second']:.3f} "
                      "训练回答 token / 全部 GPU·s；分母包含 32 卡准备、训练、发布及末次评估。"
                      "两者不是同一个吞吐指标，不能互相替代。",
                      "", "![记录的阶段计时](figures/" + run["name"] + "-stage-baseline.png)",
                      "", f"复算入口：`python3 derive_stage_baseline.py --folder {run['source']}`；"
                      f"[原始复算结果]({run['source']}/stage-baseline.json)。"])
    lines.extend(["", "## 等内容权重发布 A/B", "",
                  "固定第三次真实更新后的权重；两次 warmup，普通/packed broadcast × 全量 checksum 开关，各四次平衡顺序测量。"
                  "每轮核验发送名字/dtype/shape/bytes/bucket 顺序、模型、Adam 与八引擎内容。checksum-off 的内容检查在计时外。"])
    for run in runs:
        benchmark = run["publication_benchmark"]
        if not benchmark.get("passed") or not run["training_verified"]:
            continue
        lines.extend(["", f"### {run['name']}", "",
                      "| 方式 | 全部墙钟样本（s） | 墙钟中位数（s） | min/max（s） | 传输中位数（s） |",
                      "|---|---|---:|---|---:|"])
        for name, values in benchmark["comparisons"].items():
            lines.append(f"| {name} | {values['wall_seconds']} | {values['median_seconds']:.6f} | "
                         f"{values['min_seconds']:.6f}/{values['max_seconds']:.6f} | "
                         f"{values['transfer_median_seconds']:.6f} |")
        lines.extend(["", "发布阶段比值：`" + json.dumps(benchmark["speedup_ratios"]) + "`。"
                      "该比值不代表端到端训练加速。",
                      "", f"相同发送布局共 {benchmark['sender_bytes']:,} bytes；"
                      "每轮最大值来自全部 16 learner rank。传输区间包括上游 update_weights 的物化、"
                      "通信和接收确认；不是链路带宽测量。全量 checksum-on 额外包含 learner 模型/Adam "
                      "和八引擎 TP 分片内容 SHA256，以及完成屏障。",
                      "", "checksum 开关的墙钟中位数差（s）：`"
                      + json.dumps(benchmark["checksum_added_median_seconds"]) + "`。"
                      "checksum-off 每轮仍在计时外完成相同内容核验，因此它不是完全不做审计的训练。",
                      "", "![全部发布计时样本](figures/" + run["name"] + "-publication.png)"])
    lines.extend(["", "## 本轮资源时序", "",
                  "只展示 API 返回的实际点，保留 Pod tags、查询窗口和响应 SHA256。"
                  "不同看板的面板编号含义不同，以本次 describe 结果为准；空数据不能当作零利用率。"])
    for run in runs:
        if not run["terminal"]:
            continue
        reports = []
        for path in sorted((root / run["source"] / "grafana").glob("*.summary.json")):
            report = read(path)
            if report.get("readable_points"):
                reports.append((path, report))
        if not reports:
            continue
        lines.extend(["", f"### {run['name']}", "",
                      "| 面板 | 可读 series | 实际点数 | 逐点来源 |",
                      "|---|---:|---:|---|"])
        for path, report in reports:
            lines.append(f"| {report['panel_id']} | {report['readable_series']} | "
                         f"{report['readable_points']} | "
                         f"[JSON]({path.relative_to(root)}) |")
        if any(report["panel_id"] == 7 for _, report in reports):
            lines.extend(["", f"![实际 Pod GPU 时序](figures/{run['name']}-gpu-duty.png)"])
    lines.extend([
        "", "## 失败记录与学习入口", "",
        "train-03 的 Ray worker 端口与平台 GCS 重叠；train-04 的严格格式门禁误处理 decoded EOS，四批共 256 条回答未进入 optimizer。"
        "修复保留全部原始回答、token 和 reward，将格式统计与训练准入分开；格式失败负奖励仍参与完整组训练。",
        "",
        "[参数判断与四道练习](PARAMETER_AUDIT.md) · [实测案例与答案](WORKED_EXAMPLE.md) · [执行规格](SCALE32_PLAN.md) · [机读复算结果](results-summary.json)。"
        "新增 Grafana 资源时序按实际 Pod tags 展示；历史 NSYS 和逐卡图仍属原四卡采集。"
        "本轮没有节点丢失恢复、RDMA 带宽或独立 held-out 质量实验的结论。",
    ])
    (root / "REPORT.md").write_text("\n".join(lines) + "\n")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parent)
    args = parser.parse_args()
    value = summarize(args.root)
    print(json.dumps({key: value[key] for key in
                      ("terminal_gpu_hours", "unfinished_runs", "verified_training_runs")}))
