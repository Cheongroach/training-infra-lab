# Qwen3.5 长轨迹训练运行时

2026-10-21：实现与分层验收中。设计见 [调研报告](../research/2026-10-19-system-direction.md)。旧 Qwen2.5 实验保留在上级目录。

已在 v4 worker 通过 15 个 CPU 测试：全词表 log-prob/梯度、padding/GDN 样本隔离、缓存生成概率、GRPO 裁剪、重新分批后的参数/Adam 更新、计划完整性、profile 数据隔离、DCP 参数/Adam/RNG/LR 重放及 epoch fencing、缺失 DCP 分片拒绝提交、EOS/长度终止与真实奖励、非空 Adam 的回放初态恢复、拒绝缺测矩阵/非有限误差/未完成更新。

4×H20 的 Qwen3.5-9B 后端探针完成，8,953,803,264 个文本主干参数全部训练。FP32 参数/归约、BF16 计算，FSDP2 + block 重计算。PyTorch 2.9.1、Transformers 5.2.0、FLA 0.4.2；GDN 使用 FLA，卷积和 gated RMSNorm 使用 PyTorch。

128/512/1024 长度桶各完成 3 步，首步热身。1024 桶后两步约 1.203/1.206 秒，全卡最大 peak allocated 47.94 GiB。桶内含不同长度；输入为合成 token，不能作为实际 RL 吞吐或调度收益。

任务 6051c79b1a2b5502、trial 436459539、pod 5379397302：done、退出码 0。见 [实际源码、环境、每 rank 测量及 SHA256 归档](results/h20-probe-v1/)。

第二阶段入口已实现：`device_gates.py` 先比较 FLA/torch 的概率和梯度，再比较独立全局 batch 与 FSDP 的两步参数、梯度及 Adam 状态，覆盖两种 reduction、重排、空槽位和延迟同步。门限在运行前固定；失败会阻止实际工作负载继续执行。

`grpo.py` 执行同步真实采样 → 规则奖励 → 完整 group advantage → clipped surrogate → Adam 更新。第二次更新之前，以同一批真实轨迹做成对回放；每种配置都热身，每次恢复相同参数及非空 Adam 状态，三次固定种子随机顺序，比较全部分片。回放暂存文件不属于 DCP 持久恢复协议。

任务 `c123923673dca8a4` 在数值门禁失败，证据见 [h20-grpo-v2](results/h20-grpo-v2/job_evidence.json)：FLA/torch BF16 概率最大差 0.00202、梯度相对 L2 0.00408，缓存/完整概率最大差 0.00197，跨样本隔离精确；FP32 sequence-mean 的四种计划通过。token-mean 参数最大差 2.4233e-6 超过 2e-6 门限，实际 GRPO 未执行。

原门禁分别热身 reference/FSDP，导致比较起点存在舍入差异。修订后，任务 `a82e58f0c1f8e1b3` 的 8 项 FP32 计划对照全部通过，参数最大差 2.38e-7，原 2e-6 门限未放宽；FLA/torch 单卡路径也通过。见 [v3 归档与 gate 汇总](results/h20-grpo-v3/gate-summary.json)。

v3 在 BF16 初始化阶段挂起：[调用栈](results/h20-grpo-v3/stacks.txt)显示 rank 0/1 在 Adam `full_tensor().cpu()`，rank 2/3 已进入 `finally` 的进程组清理。原始异常未能输出，尚不能把具体数值差异当作已观测事实。归档后主动终止了本任务的 torchrun launcher，任务最终退出 1，实际 GRPO 未执行。

v4 改为只由 rank 0 生成参考状态并广播；全部初始化 collective 完成后再汇总不一致；异常先打印，再交给 torchrun 终止整组。任务 `e5f1e084165ee7dc` 的 BF16 fixed/sequence-mean 对照未通过：初始参数及 Adam 精确一致，梯度相对 L2 0.04085，但参数最大差 1.739e-4 超过原 3e-5 门限。任务及时退出 1，原始异常可见，实际 GRPO 未执行。见 [v4 证据](results/h20-grpo-v4/gate-summary.json)。

Gloo 双进程故障传播检查已通过：含启动约 3.01 秒退出，原始异常可见，未等待 60 秒 collective 超时。这是 CPU 进程生命周期验证。可选独立 profiler 回放已实现，尚未完成真实设备采集。

`precision_diagnostic.py` 已完成两轮四卡对照：[v1](results/h20-precision-v1/precision-summary.json)、[v2](results/h20-precision-v2/precision-summary.json)，均退出 0、源码 SHA 核对通过、四个 rank 完整结果一致。诊断退出 0 只表示证据采集完成。

发现两个独立问题：普通 autocast 与 FSDP BF16 参数策略的计算路径不同；对齐为可微 BF16 参数视图后，梯度相对 L2 从 4.085% 降到 0.248%，但接近零的梯度仍因舍入改变符号，并被 Adam 放大。BF16 视图下 epsilon=1e-8/1e-6/1e-5 的参数最大误差分别为 9.33e-5/5.00e-5/2.20e-5，只有最后一组通过原 3e-5 门槛。

v5 候选显式采用 `--adam-eps 1e-5`，reference/FSDP/真实 GRPO 保持一致；原默认值 1e-8 仍可复现，FP32 门禁仍使用 1e-8。epsilon 是优化器配置变更，会影响训练轨迹；这里没有收敛或质量结论。完整矩阵未通过：任务 `2e4c7a5e1b878df6` 的 BF16 fixed 通过，但 verl/sequence-mean 的参数最大误差为 5.96405e-5，仍高于原 3e-5，退出 1。见 [v5 归档](results/h20-grpo-v5/gate-summary.json)。真实 GRPO 未执行。

下一候选 `034bfc708a9ff5f1` 显式采用 `epsilon=1e-4`，完整保留各计划结果并在矩阵末尾汇总失败；门限保持不变，任何失败仍阻止真实工作负载。四个 rank 各通过完整 17 项设备门禁，分布式检查结果跨 rank 精确一致；BF16 参数最大误差 8.00192e-6、梯度相对 L2 最大 0.00253929。9B 加载后在采样前因 Transformers 5.2 聊天模板默认返回 BatchEncoding 而退出 1，尚未执行真实更新。见 [v6 归档](results/h20-grpo-v6/gate-summary.json)。

`rollout.py` 现显式请求 `return_dict=False`；新增真实 `PreTrainedTokenizerFast` 回归测试，在本地复现同一异常后验证修复，专项测试退出 0。本地 4 项采样/回放测试断言通过，但进程收尾被沙箱 `/proc` 线程命名限制拦截，不能记为完整进程退出 0；v7 任务 `84527e7696acae23` 将在 worker 重新执行全部 16 项 CPU 测试、设备门禁与真实工作负载。该配置是稳定性试验，尚无质量结论。

v7 worker 的 16 项 CPU 测试与各 rank 的 17 项设备门禁均通过。完成两批真实采样，但仅完成第一步 live update：第一批 4 个组全部正确，advantage/梯度为零；第二批有 2 个信号组，两个零奖励回答均因 384 token 上限截断。第二批回放中，slot_sorted/micro=1 与 verl/micro=2 的梯度相对 L2 差异为 17.31%–18.40%，高于原 8% 门限，尽管参数最大差只有 1.39e-6。任务退出 1，未完成第二步 live update、重复矩阵或 profiler，不能报告调度加速或有效学习。见 [v7 原始归档](results/h20-grpo-v7/job_evidence.json)与[逐项摘要](results/h20-grpo-v7/run-summary.json)。

固定轨迹诊断 `5f217016eaf0f05d` 在 3000 秒时限退出 124，归档与源码 SHA 验证成功。四卡各 10 项数值配置齐全，BF16 四份 profiler 完成，FP32 profiler 缺测。同计划重复与仅重排均通过；改为 microbatch=2 后，两种输出头都因 Adam 二阶矩差异超限而失败（FP32 head 最坏 rank 18.68%）。FP32 head 未解决跨 microbatch 一致性。见 [诊断摘要](results/h20-replay-v1/diagnostic-summary.json)与 [测量报告](measured-report.md)。

v8 任务 `07e722ee19788760` 已完成：4×H20、Qwen3.5-9B、MATH 固定子集、4 次同步非零梯度更新；18 项 worker CPU 测试及每 rank 18 项设备门禁通过，任务与归档复制退出 0，源码 SHA 验证通过。128 token 生成的驻留配对加速中位 **1.761×**，token/reward/log-prob 逐值一致，峰值 allocated 16.99→29.44 GiB/卡；独立 16 token profiler 的每卡 all-gather 528→33 次。该结果只覆盖本运行时生成阶段。

v8 的 micro=1 两计划×3 次回放完整通过原误差门限，梯度/两阶矩仍有可观舍入差异，不能称为逐值等价。排序方案配对加速中位 **0.988×**，没有稳定收益；micro=2 仍未通过。见 [完整测量报告](measured-report.md)与 [v8 归档](results/h20-grpo-v8/job_evidence.json)。

v8 奖励审计发现：64 条轨迹中 34 条截断，所有零奖励均来自截断，正常结束错答为 0；另有 4 条截断被最后数字规则判对，包含中间算式碰巧匹配的误判。不能报告有效数学能力学习。新增 `rewards.py` 的默认 EOS + 明确 final-line 规则，单独记录明确错答、格式失败和截断；旧规则显式保留。

v9 `6f0457157a6955b2` 已完成四步，训练与归档退出 0，源码和数据 SHA 验证通过；28 项 worker CPU 测试、每 rank 18 项设备门禁通过。使用同一固定数据顺序，response 上限 2048 token。独立复算 64 条回答、32,215 response token：62 条正确、2 条明确错答，全部正常 EOS，没有格式失败或截断。前三步梯度和 Adam 矩均为零；第四步有 1 个信号组、梯度范数 0.915452，Adam 两阶矩各有 7,935,924,616 个非零元素。两条错答均将同一道题的因式分解算错，输出 141，另两条正确输出 13。本次只验证了一次由真实错答产生的非零梯度更新，没有能力提升结论，也未重复 actor 回放、驻留性能比较或 profiler。见 [逐轮审计](results/h20-grpo-v9/training-summary.json)与 [验收记录](results/h20-grpo-v9/validation-summary.json)。

910B4 [最新容量与任务快照](results/environment-candidates/npu-capacity-v6.json)仍为 total/used 16/16 卡，原任务 `started`、未分配 Pod。已提交两卡、900 秒的 Qwen3.5 小模型后端探针 `64c1d67f06edd50d`，使用 torch-npu 2.9/CANN 9.0.1 候选镜像，仅测试原生 torch GDN、FSDP2/HCCL 和重计算，不加载 9B 权重。见 [提交记录](results/environment-candidates/npu-backend-submission.json)。保留原排队任务，未重复提交；尚无本轮 NPU runtime 验收结果。

v7 输入为固定 SHA256 的 GSM8K train 前 64 行、384 token 上限；v8 改为固定 MATH train 子集、768 token 上限，仍属于短程工程验收，不覆盖原设计的 16K/32K 长轨迹。比较对象是移植的 verl KK 加合理局部降序和 microbatch 调优，不是完整 verl 框架或 packing 强基线；三次重复位于同一进程组，不是三个独立任务。

## 源码

contracts.py 定义完整 group 和更新身份；rl.py 保留 GRPO 裁剪与 reduction；model.py 提供文本主干和完整词表 log-prob；baselines.py 移植固定版本 verl KK；planner.py 生成全 rank 槽位；cost.py 做环境绑定的拟合与回退；executor.py 负责全局归一化和同步；rollout.py 记录真实行为概率与 reward；state.py/checkpoint.py 实现带 epoch 的同步 DCP 提交；probe.py/build_job.py 构造有时限的自包含探针。

按系统契约阅读代码见 [源码学习指南](source-guide.md)。`analyze_replay.py` 校验全部 rank 的测量、身份与数值结果，再重算成对吞吐；缺测或门禁未通过会拒绝生成成功报告。

```bash
python3 -m unittest training_infra_lab.trajectory_runtime.test_runtime training_infra_lab.trajectory_runtime.test_state_cost training_infra_lab.trajectory_runtime.test_grpo training_infra_lab.trajectory_runtime.test_analysis -v
```

## 尚待验收

9B 真实轨迹上的跨 microbatch 数值一致性（tiny 门禁已通过）；严格奖励下多步非零动量训练与持出集质量评估（v9 已完成四步执行、其中一次非零梯度）；包含 packing 的强基线比较及独立任务复测；910B4 新环境；长序列 packing；verl adapter；多机和 27B；真机故障注入及恢复成本。当前代价模型只有合成数据单测，恢复只有单进程 CPU 验证。

LocalCommitStore 依赖经验证的共享 POSIX flock/rename/fsync，不对 HDFS FUSE/对象存储作持久提交保证。当前仅同 world size，没有异步 checkpoint、8→4→8 或跨 CUDA/NPU 恢复。HDFS 归档是证据导出，不是该恢复协议。
