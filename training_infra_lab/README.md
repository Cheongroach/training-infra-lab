# H20 / Ascend 910B4 Training Infra Lab

第一次阅读请从 [读者版总入口](README-friendly.md) 开始。本文保留原始工程记录和复现命令。

当前主线是 [Qwen3.5-9B 轨迹训练运行时](trajectory_runtime/README.md)：围绕完整 GRPO
更新语义、全 rank 调度、设备数值一致性、参数/Adam 成对回放和一致恢复实现系统组件。
4×H20 已完成 9B 真实 GRPO 执行与独立 profiling，生成权重驻留的三次配对加速中位
1.761×；排序调度没有稳定收益。v9 修复奖励后完成四步：64 条回答全部正常结束，
62 条正确、2 条明确错答；前三步梯度为零，第四步产生非零梯度与 Adam 两阶矩。
尚无数学能力提升或 910B4 新模型训练结论。见 [测量报告](trajectory_runtime/measured-report.md)。

Qwen2.5-1.5B 全参数 SFT 是历史实验底座，见 [实验报告](report.md) 和
[源码学习指南](source-guide.md)。

后续系统方向见 [Qwen3.5 / Qwen3.8 调度与恢复运行时调研](research/2026-10-19-system-direction.md)。
主模型为 Qwen3.5-9B，Qwen3.8-27B 为后续扩展项。
下文只记录已完成的 Qwen2.5 历史实验，结果不能迁移到新模型或 RL。
所有数字来自实际集群任务；本目录独立于 vllm-omni 的推理代码。

## 已完成的主要结果

- 1,543,714,304 个参数全部训练；2×H20 与 2×910B4 均完成 10 项短程消融。
- H20 两机 16 卡完成固定 global batch 的强扩展：中位 12,987 有效 token/s，
  相对 2 卡 4.11×，效率 51.3%；实际跨机 RoCE/GDRDMA 已从日志确认。
- 双卡 910B4 延迟梯度同步：同条件短测中位吞吐 1727→2476 token/s（+43.4%），
  显存 15.09→17.97 GiB/卡；H20 profiler 确认 ReduceScatter 116→29 次/步。
- H20 与 910B4 均通过“连续 6 步 vs 保存/重启续跑”的参数、Adam 状态逐字节一致校验。
- 动态 padding 是短样本的主要收益；分块 CE 未稳定优于监督位置筛选。
  具体负结果、额外同步消融、恢复结论及简历表述均见报告。

这是 1.5B 模型、最多 16 卡的短程实验；未验证长程收敛、千卡规模、弹性恢复或
生产训练吞吐。分块 CE 和延迟梯度同步都是已有方法，本项目不主张算法原创。

## 源码地图

| 文件 | 负责的问题 |
|---|---|
| [prepare_assets.py](https://github.com/Cheongroach/training-infra-lab/blob/main/training_infra_lab/prepare_assets.py) | 固定公开模型/数据版本，assistant labels，分词边界，SHA256 |
| [loss.py](https://github.com/Cheongroach/training-infra-lab/blob/main/training_infra_lab/loss.py) | full / selected / chunk 三种保持同一目标的 loss |
| [correctness.py](https://github.com/Cheongroach/training-infra-lab/blob/main/training_infra_lab/correctness.py) | 梯度、全忽略样本、不等长 mask、FSDP 同步/延迟同步更新校验 |
| [transformer_gate.py](https://github.com/Cheongroach/training-infra-lab/blob/main/training_infra_lab/transformer_gate.py) | 小型 Qwen 与 HF 参考的 padding/重计算/loss 校验 |
| [train.py](https://github.com/Cheongroach/training-infra-lab/blob/main/training_infra_lab/train.py) | FSDP2 owner、BF16、全局 token normalization、计时、训练状态 |
| [checkpointing.py](https://github.com/Cheongroach/training-infra-lab/blob/main/training_infra_lab/checkpointing.py) | DCP、RNG、数据游标、参数/Adam 摘要与加载契约 |
| [verify_fingerprint.py](https://github.com/Cheongroach/training-infra-lab/blob/main/training_infra_lab/verify_fingerprint.py) | 临时参数视图下的只读摘要回归检查 |
| [build_experiment_job.py](https://github.com/Cheongroach/training-infra-lab/blob/main/training_infra_lab/build_experiment_job.py) | 自包含源码快照、已固定镜像、资源与任务执行上限 |
| [submit_job.py](https://github.com/Cheongroach/training-infra-lab/blob/main/training_infra_lab/submit_job.py) / [inspect_job.py](https://github.com/Cheongroach/training-infra-lab/blob/main/training_infra_lab/inspect_job.py) | 预检提交，run → trial → pod → log 证据 |
| [collect_results.py](https://github.com/Cheongroach/training-infra-lab/blob/main/training_infra_lab/collect_results.py) / [export_evidence.py](https://github.com/Cheongroach/training-infra-lab/blob/main/training_infra_lab/export_evidence.py) | HDFS 结果收集、所选环境与验证证据导出 |
| [analyze_results.py](https://github.com/Cheongroach/training-infra-lab/blob/main/training_infra_lab/analyze_results.py) | 按相同条件汇总，生成图表与 profiler kernel 统计 |
| [validate_results.py](https://github.com/Cheongroach/training-infra-lab/blob/main/training_infra_lab/validate_results.py) | 复算吞吐，核对任务终态、数据预算、源码摘要与恢复状态 |

## 本地校验

需要匹配的 PyTorch；Qwen 校验还需 Transformers 4.57.1。

```bash
OMP_NUM_THREADS=1 python3 training_infra_lab/correctness.py
python3 training_infra_lab/verify_fingerprint.py
LAB_DEVICE=cpu python3 training_infra_lab/transformer_gate.py
```

多卡 correctness 使用下面同样的 torchrun 启动方式。小模型仅用于正确性，
不用于报告实际训练性能。

## 直接训练

准备好 `assets/model/`、`train.jsonl`、`test.jsonl`、`manifest.json` 后：

```bash
cd training_infra_lab
LAB_DEVICE=cuda torchrun --standalone --nproc_per_node=2 train.py \
  --assets /path/to/assets --output /path/to/results \
  --mode selected --padding dynamic --seq-length 4096 \
  --global-batch 8 --micro-batch 1 --steps 16 --warmup 4
```

Ascend 使用匹配的 CANN/torch-npu 镜像，source CANN 环境后将 `LAB_DEVICE` 改为 `npu`。
`--defer-gradient-sync` 减少 reduce-scatter 次数，但保留更多未分片梯度；
`--activation-checkpoint` 重算 Transformer block；`--mode chunk` 重算分块 loss。
以报告中的同条件取舍为依据选择开关。

恢复必须保持训练参数和 world size 一致。使用 `--save-checkpoint --checkpoint-dir ...`
保存，以 `--resume ...` 读取；`--deterministic --fingerprint` 用于严格重放检查。
`--stop-after` 表示本进程的最后一步，`--steps` 是整个运行计划的总步数。
主性能实验不启用确定性算法或逐字节摘要。

## Merlin 复现

资源组 `search_douyin_life_nlp`；当前 builder 的路径和镜像针对本工作区配置。
先检查资源可用量，生成请求不会自动提交：

```bash
python3 training_infra_lab/build_experiment_job.py \
  --device cuda --cluster lf --stage sync --devices 2 --seconds 1800 \
  --tag my-sync-run --output-dir /tmp/my-sync-run
python3 training_infra_lab/submit_job.py check /tmp/my-sync-run
# 查看 precheck-result.json，再提交相同的固定请求
python3 training_infra_lab/submit_job.py submit /tmp/my-sync-run
```

阶段：`smoke`、`benchmark`、`scale`、`recovery`、`profile`、`sync`。
16 卡使用 `--stage scale --nodes 2 --devices 8 --cluster lf`。
NPU 使用 `--device npu --cluster hl`。
每个任务自带源码 bundle，worker 不依赖开发机 `/mlx_devbox/` 路径。

预检 `error` 阻止提交；`warn/hint` 展示后处理。SID 保留式合并到
`request_context.precheck_sid`，30 分钟内提交。脚本会检查预检配置未变更，
避免重复提交；入口总超时覆盖安装、资产复制、校验和训练。
每个退出路径将结果归档到个人 HDFS，退出码保留原始训练错误。

```bash
python3 training_infra_lab/inspect_job.py RUN_SID --output /tmp/my-sync-run/evidence
python3 training_infra_lab/collect_results.py \
  --store /mnt/hdfs/life/zhangyang.exp/training_infra_lab/20261019 \
  --output training_infra_lab/results my-sync-run
python3 training_infra_lab/export_evidence.py \
  --jobs /tmp --output training_infra_lab/results my-sync-run
python3 training_infra_lab/analyze_results.py
```

`collect_results.py` 汇总整个本地结果目录，按任务、配置和测量 step 集合分组，
避免把恢复片段、profiler 和普通 benchmark 混算。产物收集须在任务完成后进行。
`analyze_results.py` 的图表针对本次固定的实验标签；自定义标签需调整分组定义。
本次结果完整收集后，运行 `python3 training_infra_lab/validate_results.py`
可重新核对报告使用的训练段和任务证据。

## 数据与证据

- [results/all_results.json](results/all_results.json)：所有已归档训练段的结果索引。
- [results/report_metrics.json](results/report_metrics.json)：报告数字与来源文件映射。
- 每项任务的 `job_evidence.json` 含 run/trial/pod、状态、退出码、镜像和资源配置；
  `validation_events.json` 含实际运行时环境和校验事件。
- 每项任务的 `bundle.tar.gz` 是实际运行源码；`source_manifest.json` 是逐文件 SHA256。
- 大型 Chrome trace 压缩为 `.json.gz`；解压后可用 Perfetto 等工具打开。
- 模型/数据版本、许可与文件校验见 [assets_manifest.json](results/assets_manifest.json)。

测量只计驻留 batch 的训练步，包含前后向/梯度裁剪/AdamW，排除搬运、分词、eval、
checkpoint 和初始化；全体设备吞吐用真实非 padding token 计算。完整口径见报告。
