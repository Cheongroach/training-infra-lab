# Qwen3.5-9B 训练执行验收

本报告只记录实际完成的验证。4×H20 已完成 Qwen3.5-9B 训练执行与生成驻留优化，生成阶段配对中位加速 1.761×。v9 修复奖励后完成四次更新，其中一次由正常结束的明确错答产生非零梯度和 Adam 矩；尚无调度加速、能力提升或 910B4 新模型训练结论。

## 已验证的工程边界

4×H20，Qwen3.5-9B 文本主干 8,953,803,264 个参数全量训练；FSDP2、FP32 master/梯度归约、BF16 计算、逐 block activation checkpoint、FLA GDN。权重文件和 worker 源码均与提交 manifest 核对 SHA256。原生 PyTorch 卷积与 gated norm，不能把它们称为融合路径。

[后端探针](results/h20-probe-v1/job_evidence.json)已完成 128/512/1024 三个长度桶，每桶 3 步；1024 桶热身后两步约 1.203/1.206 秒，最大 peak allocated 47.94 GiB。这是合成 token 后端测量，不是实际 RL 吞吐。

v7 worker 的 16 项 CPU 回归、每 rank 17 项设备门禁通过。门禁覆盖全词表概率/梯度、GDN 样本隔离、行为概率、全局 loss 归一化、空槽位、延迟同步、两次参数/Adam 更新和重排。tiny 数值门禁通过不能替代 9B 集成验收。

## 真实轨迹：执行到哪一步

任务 [84527e7696acae23](https://ml.bytedance.net/development/instance/jobs/84527e7696acae23)，trial 436726967、pod 5381073942；2026-10-21 20:17:50–20:37:34 UTC，退出 1。任务时间线没有先行人工停止事件；worker 主动抛出回放数值门禁失败，随后退出。详见 [原始归档](results/h20-grpo-v7/job_evidence.json)及[可重算摘要](results/h20-grpo-v7/run-summary.json)。

| 指标 | 第一批 | 第二批 |
|---|---:|---:|
| 问题组/轨迹数 | 4/16 | 4/16 |
| 响应 token | 3,878 | 4,292 |
| 平均奖励 | 1.000 | 0.875 |
| 非零 advantage 组 | 0 | 2 |
| 正常 EOS / 长度截断 | 16/0 | 14/2 |
| 正常结束但答错 | 0 | 0 |
| 完成 live optimizer step | 是，梯度为零 | 否 |

所有零奖励回答都被 384 token 上限截断。不能将这种信号解释为已验证有意义的推理能力学习。第一步 Adam 已初始化，但一阶/二阶矩为零；它不验证非零动量状态下的真实批次重放。

缓存采样与完整重放概率 RMS 误差分别为 0.01557/0.01718，超出 ratio 裁剪区间的 token 比例为 0.0516%/0.0932%，通过预设的行为概率检查。

## 真实回放失败

第二批从同一参数/Adam 暂存状态回放，第一项 slot_sorted、microbatch=1 建立参考；verl、microbatch=2 在四个 rank 的梯度相对 L2 为 17.45%、17.50%、18.40%、17.31%，超过原 8% 门槛。参数最大差仅 1.39e-6，通过原 3e-6 门槛，说明仅检验小学习率下的参数差不足以验证训练一致性。

没有完成 4 配置×3 重复的矩阵，也没有采集 profiler；局部耗时不得整理成加速结论。`analyze_replay.py` 会拒绝这种不完整证据。

对这批数据，verl 与 slot_sorted 的各槽位最大长度以及 padding 总量相同：microbatch=1 共 5,708 个输入 token，microbatch=2 共处理 6,208 个含 padding token。简单排序没有显示减少形状工作量的机会；这是计数分析，不是算子或墙钟性能证明。

## 固定轨迹诊断

任务 [5f217016eaf0f05d](https://ml.bytedance.net/development/instance/jobs/5f217016eaf0f05d) 固定 v7 两批原始轨迹、同一 9B 权重和 warm step，分开比较：同计划重复、只重排、改变 microbatch，以及 BF16/FP32 输出头。记录逐层梯度/Adam 差异和独立 profiler。

该实验不重新采样，属于离策略数值诊断。任务在 3000 秒入口时限退出 124，归档复制退出 0、源码 SHA256 核对通过。四个 rank 各 10 个配置结果齐全，只有 BF16 的四份 profiler 完成，FP32 profiler 缺测。见 [完整数值/部分 profiler 摘要](results/h20-replay-v1/diagnostic-summary.json)及[退出与时间线](results/h20-replay-v1/job_evidence.json)。

| 对照（相对同精度 verl/micro=1） | BF16 梯度全局相对 L2 | FP32 head 梯度全局相对 L2 | 结果 |
|---|---:|---:|---|
| 同计划重复 | 0.973% | 0% | 均通过 |
| slot_sorted / micro=1 | 0.573% | 0.570% | 均通过 |
| verl / micro=2 | 7.677% | 6.508% | 均未通过，Adam 二阶矩超限 |
| slot_sorted / micro=2 | 6.794% | 6.688% | 均未通过，Adam 二阶矩超限 |

FP32 head 的 micro=2 最坏 rank Adam 二阶矩误差 18.68%，超过原 8% 门限。FP32 head 未解决跨 microbatch 一致性；最大梯度差异集中在第 0 层 GDN out_proj、第 3 层 attention v_proj 等模块。该结果只定位差异分布，不能证明某个 GDN kernel 是根因；本次也未精确复现 v7 的 18% 梯度差幅度。

BF16 profiler 中 rank 0 的 `aten::mm` self device time 约 696 ms，FSDP all-gather 132 次、约 241 ms，reduce-scatter 66 次、约 184 ms。算子、NCCL kernel 和 annotation 可能重复归属或重叠，不能相加为墙钟分解，也不能把这一条插桩样本当作性能对照。

## v8：完整执行与生成驻留

任务 [07e722ee19788760](https://ml.bytedance.net/development/instance/jobs/07e722ee19788760)，trial 436794698、pod 5381688702；2026-10-21 21:56:45–22:42:16 UTC，`done`、退出 0、归档复制退出 0，逐文件源码 SHA256 验证通过。见 [任务证据](results/h20-grpo-v8/job_evidence.json)、[时间线](results/h20-grpo-v8/timeline.json)。这 45 分钟包含安装、设备门禁、模型复制、生成比较、全参数/Adam 暂存与逐张量检查、独立 profiling 和四轮真实更新，不是训练步耗时。

固定 microbatch=1，FP32 输出头，Adam epsilon=1e-4。MATH algebra/train 固定 revision `21a5633873b6a120296cce3e2df9d5550074f4a3`，取 Level 4/5、最终 boxed 答案为整数的 64 题，按题目 SHA256 排序、两种难度均衡；样本选择未使用模型回答。数据 SHA256 `689e9289448b34465ff6e43370604f44a8289379330ce36fbedce1acc6fc50d4`。实际使用前 16 题，4 轮×4 题×4 轨迹，上限 768 response token。

18 项 worker CPU 回归、每 rank 18 项设备门禁通过。驻留门禁覆盖：生成与参数/Adam 不变、退出驻留后继续训练、全组显存拒绝、全 rank 注入阶段异常后清理和继续训练。故障注入仅为这个 tiny CUDA 场景，不能推广为任意单 rank 故障恢复。

### 生成阶段

`residency.py` 在无梯度生成期间，通过 FSDP2 公共 API 保留 BF16 聚合权重；全组显存准入，结束后逐模块 reshard 并恢复训练策略。预算含完整 BF16 权重和固定 cache/activation 余量，不能保证任意长度无 OOM。

128 token，同一初始参数、相同随机种子，1 次预热与 3 次随机顺序配对；每次 4 题×4 轨迹，共 2048 response token。四卡的 sampled token、reward 和 log-prob 全部逐值一致。计时包含准入、首次 all-gather、采样与最终 reshard。见 [可重算摘要](results/h20-grpo-v8/residency-summary.json)。

| 配对 | 正常 reshard（秒） | 生成期驻留（秒） | 加速比 |
|---|---:|---:|---:|
| 1 | 12.345 | 7.117 | 1.735× |
| 2 | 12.407 | 7.038 | 1.763× |
| 3 | 12.379 | 7.029 | 1.761× |

中位配对加速 **1.761×**，全卡生成吞吐约 165→291 response token/s；峰值 allocated **16.99→29.44 GiB/卡**。这是优化本运行时的 FSDP 逐 token 采样，不是相对 vLLM/完整 verl 的比较，也不是整个 RL 训练加速。

独立的 16 token profiler 在四卡均观察到 all-gather **528→33 次**，对应 33 个 FSDP group 从每步聚合变为生成阶段聚合一次。rank 0 对应 device annotation 约 875→52.5 ms。CPU/GPU 同名 annotation 不重复累加，算子时间存在重叠，不用它们直接分解墙钟。

### 真实更新与奖励缺陷

[训练审计](results/h20-grpo-v8/training-summary.json)按轨迹 ID 对齐 rollout、batch、mask、每 rank 的完整 history，并复算全组 advantage。

| 更新 | response token | 平均奖励 | 非零 advantage 组 | 正常结束且正确 | 截断：判对 / 判错 | 梯度范数 |
|---|---:|---:|---:|---:|---:|---:|
| 0 | 11,019 | 0.5000 | 2/4 | 7 | 1 / 8 | 1.292 |
| 1 | 10,511 | 0.6250 | 2/4 | 9 | 1 / 6 | 1.376 |
| 2 | 10,353 | 0.5625 | 2/4 | 9 | 0 / 7 | 1.285 |
| 3 | 10,717 | 0.4375 | 2/4 | 5 | 2 / 9 | 1.292 |

64 条轨迹中 30 条正常结束、34 条截断。所有零奖励回答都因长度上限截断，正常结束的错答为 0；四轮虽有非零梯度，仍不能声称验证了有意义的数学能力学习。各轮题目不同，奖励变化不能作为学习曲线。

此外，`last_decimal_exact_v1` 将 4 条截断回答判为正确。逐条检查发现其中 `algebra/train/923/1` 停在含 `sqrt{21}` 的中间算式，尚无最终答案；中间数字碰巧等于参考值。这是奖励判定缺陷。v8 保存原始结果，不重新打分后冒充修复规则下的 on-policy 训练。

第一轮 update 70.76 秒，含首次训练路径开销；后三轮 4.07/4.62/4.64 秒。生成阶段为 74.68/49.12/51.02/50.91 秒。概率重放与诊断阶段未纳入这些分项，不相加冒充完整 RL 比较。训练 update 最大 allocated 47.44 GiB/卡。绝对 Adam 矩范数未保存；回放逐张量对照包含两阶矩。

### Actor 回放：通过门槛，没有稳定加速

第二批从相同参数与 warm-step Adam 状态开始，verl KK 与 slot_sorted 两种 micro=1 计划各 3 次；24 份 rank×配置×重复记录完整，8 份独立 actor profiler 齐全。[回放摘要](results/h20-grpo-v8/replay-summary.json)通过身份、计划、token 分母、完整四轮 history 与数值检查。

最大梯度/一阶矩/二阶矩相对 L2 为 **7.195% / 5.356% / 6.376%**，参数最大差 1.241e-6，通过原 8% / 3e-6 门限；这属于容差验收，不是训练逐值等价，也不覆盖失败的 micro=2。

verl 中位 **4.368 s**，slot_sorted 中位 **4.392 s**。逐对加速为 1.354×、0.979×、0.988×，中位 **0.988×**。首对基线偏慢不足以支持调度收益，应继续保留基线。

## 奖励修复后的验证

新增默认规则 `eos_final_line_decimal_v2`：要求正常 EOS，最后一行必须为 `Final answer: <number>`，完整解析该数字后精确比较。结果分为 correct、incorrect、missing_final_answer、truncated；截断永不获正奖励。旧规则通过显式参数保留用于复现。它仍是整数/十进制子集规则，不是通用数学等价判分。

### v9：修复规则下的真实训练验收

任务 [6f0457157a6955b2](https://ml.bytedance.net/development/instance/jobs/6f0457157a6955b2)，trial 436829699、pod 5382056947；2026-10-21 22:53:50–23:10:09 UTC，`done`、训练退出 0、归档复制退出 0。约 16 分钟包含安装、门禁、权重复制、首次编译、概率回放和诊断，不是纯训练耗时。见 [任务与源码证据](results/h20-grpo-v9/job_evidence.json)、[验收摘要](results/h20-grpo-v9/validation-summary.json)和[逐轮审计](results/h20-grpo-v9/training-summary.json)。

采用与 v8 相同的预先固定数据顺序和前 16 题，4 次 live update、每题 4 条轨迹；response 上限 2048 token、microbatch 输入预算 4096。使用修复后的提示词与奖励规则，保持 FP32 head、epsilon=1e-4、microbatch=1 和生成期驻留。没有按模型结果挑题、重采样或补造 reward。本次没有 actor 回放、生成性能对照或 profiler，不能与 v8 不同长度及提示词的分项耗时比较加速。

28 项 worker CPU 测试、每 rank 18 项设备门禁通过，原数值门限重新核对通过。源码和数据 SHA256 与提交时一致。独立审计从归档文本和固定参考答案重新计算全部奖励，并按轨迹 ID、batch SHA、mask、policy version 对齐四卡更新记录。新增的严格奖励审计测试在 bundle 固化后加入，本地 4 项审计测试退出 0，其中 3 项与 worker 测试重叠；不合并为 32 项独立测试。

| 更新 | response token | 正确 / 明确错答 | 格式失败 / 截断 | 非零 advantage 组 | 梯度范数 | Adam 一阶 / 二阶矩非零元素数 |
|---|---:|---:|---:|---:|---:|---:|
| 0 | 8,024 | 16 / 0 | 0 / 0 | 0/4 | 0 | 0 / 0 |
| 1 | 7,320 | 16 / 0 | 0 / 0 | 0/4 | 0 | 0 / 0 |
| 2 | 6,942 | 16 / 0 | 0 / 0 | 0/4 | 0 | 0 / 0 |
| 3 | 9,929 | 14 / 2 | 0 / 0 | 1/4 | 0.915452 | 7,935,924,616 / 7,935,924,616 |

共 64 条轨迹、32,215 个 response token，全部正常 EOS：62 条正确、2 条明确错答，格式失败和截断均为 0。前三步虽然执行了 AdamW step，GRPO 梯度和两阶矩均为零；只有第四步是已观测到的非零梯度更新。Adam 数量直接扫描实际分片后全组归约，四卡一致，统计位于 update 计时之外；未保存绝对矩范数，也未测量下一步如何使用这些非零动量。

两条错答均来自 `algebra/train/1269`：将 `ab - 3a + 4b = 137` 错误改写成 `(a+4)(b-3)=149`，最终输出 141；正确因式分解为 125，最小差值为 13，同组另外两条输出 13。原始 [四条回答](results/h20-grpo-v9/results/grpo/rollout-3-rank1.json)保留了完整过程。这确认零奖励来自真实数值错答，且与正确回答共同形成组内训练信号。

四轮生成阶段约 182.31/45.46/44.08/93.85 秒，update 约 66.05/4.02/4.30/5.14 秒，均取最慢 rank；首次执行包含冷启动开销，概率回放和 Adam 扫描未计入这些分项。update 峰值 allocated 最大 47.44 GiB/卡。

本次验证了修复后的采样、奖励、全组 advantage、反向与 Adam 更新链路。只有一次非零梯度更新，没有持出集评估、收敛对照或训练后质量测量；62/64 是这批采样的奖励结果，不能解释为训练后的准确率。各步题目不同，也不能把奖励变化画成学习曲线。

复算命令：

```bash
python3 -m training_infra_lab.trajectory_runtime.audit_training \
  training_infra_lab/trajectory_runtime/results/h20-grpo-v9/results/grpo \
  --output /tmp/h20-grpo-v9-training-summary.json
```

## 910B4

两卡 tiny Qwen3.5 FSDP2/HCCL 原生 torch GDN 后端探针 64c1d67f06edd50d 仍为 `started`，trial 436688702、未分配 Pod。[最新资源与任务快照](results/environment-candidates/npu-capacity-v6.json)显示 HL 队列 total/used 均为 16 卡；聚合容量无法确定占用任务或排队 ETA。保留原排队任务，未重复提交。此轮没有 910B4 runtime 成功数据，不与 H20 做性能比较。
