# 四卡 GRPO 日常监控与 Nsight 诊断实测

2026-10-24，Qwen3-4B / Miles / Ray / 4×H20。诊断作业 `49b6f8781696e68e`、trial `438689893`、Pod `5391254252` 已 done，Pod exit=0。HTML 是实际采集快照，支持离线打开，不是持续刷新的生产监控服务。

## 本次得到什么

- Grafana 历史资源曲线：基线 `70d36e11e929c5c3` / trial `438559238`，18 条序列、1022 点；提交的诊断 run 和历史曲线归属在页面分别标明。
- 本次 GPU 1 秒 CSV：四张卡，共 3656 条记录，涵盖初始化、训练与最终探针。
- NSYS 实际训练 trace：NVTX `grpo/actor_train/rollout=1/rank=0`，**39.50 s / 35,559 kernels / 264,679 CUDA API**。原始 .nsys-rep、SQLite、统计 CSV 和可筛选时间线均已保留。
- NCU 已实际运行隔离 BF16 投影，但 `regex:.*gemm.*` 没有命中实际 `nvjet_sm90...` kernel。返回码 0，日志明确写 `No kernels were profiled`。**没有有效 .ncu-rep 或 SM/DRAM/occupancy/stall counter，不能宣称计数器采集成功。**
- 实际更新三次、最终 version 3 发布通过；回答完整验收 **passed=false**。本次 boxed 子检查 3/4、boxed 与 scorer pred 一致 1/4、正确 0/4，四条均自然结束。不能以进程成功代替质量验收。

## 日常工作人员看什么

| 层次 | 常看的信息 | 能回答什么 | 本次可查看 |
|---|---|---|---|
| 任务与业务 | 状态、进度、有效样本、reward/advantage、梯度、权重版本、评测 | 是否在做有效工作，发布是否可信 | 三轮信号、参数/Adam hash、版本与四条探针 |
| 常驻资源监控 | 逐卡利用率、SM/Tensor/HBM、显存、功耗、CPU/RSS/cache、网络 | 资源饱和、阶段不均衡、进度停滞 | 历史 Grafana + 本次逐卡 CSV |
| 阶段计时 | rollout、过滤重采、learner wait、log-prob、actor train、发布 | 时间消耗在哪个阶段 | 历史三步和本次三步 host timer |
| NSYS 短采集 | NVTX、CPU CUDA API、GPU stream/kernel/memcpy、同步 | 空白位置、提交节奏、复制与计算关系 | learner rank 0 的 rollout 1 |
| NCU 定点采集 | kernel 时长、SM/DRAM、launch、occupancy；加 section 后看 cache/stall | 某个 kernel 的效率与限制 | 本次未命中，失败日志与修正脚本已保留 |
| 在线服务 | TTFT/TPOT、P95/P99、tokens/s、排队、拒绝率 | 是否满足服务 SLO | 本次为训练诊断，未采服务 SLO |

常驻应用/Grafana/DCGM 指标通常为 1–15 秒；异常复现时采 NSYS 的几个步骤，再从 top kernel 中选 NCU。告警同时绑定阶段、持续时间、进度和任务有效性，不能统一要求所有任务 GPU 利用率达到 80%。

## NSYS 实测发现与下一步

**1. 优先计量验收用 CPU 回读。** kernel 时间区间并集为 6.581 s，占 NVTX 16.7%；加入 memcpy/memset 后设备活动并集 14.667 s，占 37.1%。其余区间只是此 rank 没有记录到设备活动，当前 CPU 周期采样和调用栈采集关闭，尚不能完整归因。

Device→Host：973 次、20,114,959,798 bytes = **18.73 GiB**，累计设备复制 8.078 s。host `cudaMemcpyAsync` 累计 8.365 s，`cudaStreamSynchronize` 1.936 s；host API 与 GPU 活动可重叠，不能直接相加为 step 耗时。

固定源码确认：`model.py` 正常 train_step 后调用 `dump_local_weight_checksums`；该函数遍历全部模型参数、buffer 与优化器 state，`hash_tensor_sha256` 执行 `tensor.detach().cpu()...numpy().tobytes()`。本次开启 `--save-local-weight-checksum`，因此这条路径是已有源码证据支持的候选原因。见 [源码摘录与 hash](evidence/checksum-source-excerpts.json)。尚未通过调用栈将每笔复制对应到该路径，不能把全部 18.73 GiB 都归于 checksum。

下一轮可在独立性能运行中关闭全量 checksum/debug dump，并增加 checksum 专属 NVTX；保持本次正确性验收冻结、继续保留其完整审计。按相同 workload 对照有效 tokens/s、step time 和版本正确性，才可报告性能收益。

**2. 先改善 rollout 供给与训练阶段重叠。** 历史基线 learner 等待占比分别 50.3% / 50.8% / 38.3%。本次被 profiler 影响的 step 不能当作新基线。rollout 1 的 56.11 s host 范围中，各卡有 55 个利用率采样；GPU 0/1 平均 15.29% / 15.20%，GPU 2/3 为 0%，该范围包含采集开关和 flush，不能称为稳定训练利用率。数据支持查看 actor/rollout 的轮流工作关系，再验证流水线重叠、过滤重采和分工比例，约束 lag、有效样本和质量。

**3. 算子优化排在阶段归因之后。** 按实际 kernel 名启发式分类：GEMM 6369 次 / 累计 5.127 s，Attention 3564 次 / 0.720 s，NCCL 5633 次 / 0.366 s。GEMM 累计时间占全部 kernel 时间约 76.6%，这不是 wall-clock 临界路径占比。单 learner trace 不足以定多 rank 通信根因。

**4. 采集开销已量化。** host scope 56.114 s，NVTX 39.498 s，范围外差值 **16.616 s**，包含 CUDA Profiler API start/stop、flush 及上下文记账。本次不能据 56 秒和旧 38 秒的差值宣布性能退化，也不能把差值全部精确归因于某一个 API。

**5. 回答输出协议仍应先修。** 完整验收明确记录 `54`→`**54`、`-3`→空串、`3`→`}3` 等 boxed/pred 不一致。当前只保留问题，不改原验收或评分器，不以更多 GPU 掩盖这项质量失败。

## 本次阶段数据

| rollout | learner wait s | log-prob s | actor train s | step total s | wait ratio |
|---|---:|---:|---:|---:|---:|
| 0 | 57.22 | 11.68 | 48.72 | 117.66 | 48.6% |
| 1 | 23.92 | 1.72 | 56.11 | 81.78 | 29.3% |
| 2 | 42.24 | 1.60 | 38.71 | 82.57 | 51.2% |

rollout 1 的 actor train host timer 包含 NSYS start/stop。父子 timer 可能嵌套；发布不直接叠入 step total。其他 rollout 也处于诊断进程，不是独立无 profiler 对照。

## NCU 后续采集入口

已准备 [capture_ncu_projection.py](https://github.com/Cheongroach/training-infra-lab/blob/main/training_infra_lab/posttraining_runtime/observability-20261024/scripts/capture_ncu_projection.py)：按本次实际 `nvjet_sm90` 名称筛选，skip 5 / count 3，set basic，clock-control/cache-control=none。它同时检查 returncode、非空 .ncu-rep、`No kernels were profiled` 和 raw CSV 实际 metric；不会把退出码 0 当作充分成功条件。

```bash
python scripts/capture_ncu_projection.py --output /path/to/new-output \
  --ncu /usr/local/cuda/bin/ncu --python /opt/sglang/bin/python3
```

这条命令需要 GPU 与计数器权限；脚本语法及参数入口已验证，**修正后的 GPU 采集尚未执行**。本轮只启动一个诊断任务，没有自动追加 GPU 重跑。形状 `[256,2560] × [9728,2560]ᵀ`，PyTorch 2.14.1+cu130 / CUDA 13.0 / BF16。隔离算子不能代替真实训练算子的计数器证据；basic 没有的 stall/cache/Tensor pipeline 指标需以后增补对应 section。

## 资源、版本与证据

- Pod 驻留：2026-10-24 13:13:35–13:37:27 UTC，1432 s × 4 GPU = **1.5911 GPU·h**，含启动、训练、探针与 profiling。平台 trial start_at 与 Pod start_at 是不同时间字段，成本按终态 Pod 驻留计量。
- Nsight Systems **2026.4.1.191**；Nsight Compute **2025.3.1.0**，实际版本输出保留。
- Miles commit `2233d795dcf167be489d1859971c9233393f04c3`；Ray 2.58.1 commit `4c7892ef427b19e7585ee28f548a60ab241929e8`；Qwen3-4B revision `1cfa9a7208912126459214e8b04321603b3df60c`；image version `dbdt1t71enl9fha0bn40`。
- 平台 14 项预检成功，SID `fd4b07fb-f90a-43d5-a6fc-05b1a7a5cd5e`；source manifest 16 个文件与实际归档一致、5 个 runtime 文件变更核验、run/trial entry 一致、entry.log 在 Pod stdout 原始字节中唯一对应。
- 独立冻结验收确认参数/Adam 更新与最终发布通过，完整回答验收 false；详情见 [verified.json](evidence/verified.json)。归档审计见 [archive-audit.json](evidence/archive-audit.json)。
- 有效 NSYS 已登记为平台资产 `79c375329e81e2bd79a2f8d435d1ae1adabf3e271b159ca5bc9026e8bc463df7`，关联 arnold_trial `438689893`；见 [上传回执](evidence/profiling-upload-short.json)。本地 .nsys-rep 下载可以直接用桌面 Nsight Systems GUI 打开。平台上传回执未提供 preview_url，不虚构预览链接。

## 口径与交付

历史 GPU Duty / SM / Tensor / HBM / 功耗为四卡平均，显存为四卡总和。显存峰值 213.55 / 382.31 GiB。CPU 面板 limit=16，但提交 32 CPU，口径待核，不称 CPU 打满。RSS/cache/shmem 不直接相加；TCP 面板为 0 不说明没有 NCCL/NVLink/RDMA。

HTML、报告、局部时间线、NSYS 原始报告、SQLite、工具日志、采集代码、规范化历史与独立审计均在离线包中。包中不包含平台全量环境输出、完整 Pod stdout、Ray 环境 dump、训练张量和全量运行归档。完整原始归档保留在本地 evidence，供受控复核。

这次交付是运维视图与一次实际诊断，不包含新的长期任务、线上监控服务部署或优化收益验收。
