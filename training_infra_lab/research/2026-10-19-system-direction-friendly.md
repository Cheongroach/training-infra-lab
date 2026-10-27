# Qwen3.5 / Qwen3.8 长轨迹训练运行时：设计读者版

## 研究问题

长短差异很大的 RL 轨迹会造成 rank 间等待、显存压力和恢复重算。项目希望在不改变一次 GRPO 更新所使用的轨迹集合和目标函数的前提下，联合规划：

- 轨迹分配；
- microbatch；
- activation checkpoint；
- 梯度同步；
- checkpoint 与数据进度。

主模型是 Qwen3.5-9B，后续扩展为 Qwen3.8-27B。H20 和 910B4 分别组成同构训练组，不做 CUDA/NPU 混合训练。

本文是设计，不是性能报告。已经完成的结果见 [Qwen3.5 运行时](../trajectory_runtime/README-friendly.md)。

## 为什么不是简单换模型

历史 SFT 项目已经验证 token 归一化、FSDP2、通信和 checkpoint，但没有覆盖：

- prompt group 上的 GRPO advantage；
- 行为概率与训练概率的一致性；
- 变长轨迹的全 rank 执行计划；
- 策略版本；
- RL 数据和 optimizer 的一致恢复。

把 Qwen2.5 换成 Qwen3.5 并不会自动补齐这些语义。

## 模型结构与容量

| 模型 | 层数 | 混合注意力 | 项目角色 |
|---|---:|---|---|
| Qwen3.5-9B | 32 | 24 层 linear attention + 8 层 full attention | 主验证模型 |
| Qwen3.6-27B | 64 | 48 + 16 | 架构备选 |
| Qwen3.8-27B | 64 | 48 + 16 | 扩规模模型 |

如果保留 FP32 参数、梯度和两份 Adam 状态，静态分片状态粗略为 `16 × 参数量 / 卡数` 字节。按标称参数量：

| 模型 | 4 卡 | 8 卡 | 16 卡 |
|---|---:|---:|---:|
| 9B | 33.5 GiB/卡 | 16.8 GiB/卡 | 8.4 GiB/卡 |
| 27B | 100.6 GiB/卡 | 50.3 GiB/卡 | 25.1 GiB/卡 |

这不是显存实测，还不包含 activation、临时聚合权重、通信缓存和输出头。因此 27B 不应直接承诺在 8×64G 上执行长上下文全参数训练。

## 系统边界

| 组件 | 负责什么 |
|---|---|
| profiler | 测真实形状下的时间、峰值显存和 kernel 路径 |
| cost model | 预测候选执行计划的时间与显存 |
| planner | 把固定更新批次映射到 rank 和 slot |
| executor | 保持 collective 顺序、loss 归一化和一次全局更新 |
| training state | 保存模型、Adam、RNG、数据、策略与计划版本 |
| adapter | 把 verl 或原生 FSDP batch 接入统一契约 |

首版固定 FSDP mesh，TP=PP=CP=1。只调整一次 update 内的数据分配和执行配置，避免把动态并行拓扑也混入首轮实验。

## 三条不能破坏的契约

### 1. 算法语义

完整 prompt group 先计算 advantage，之后才允许重新分配轨迹。sequence mean 和 token mean 是不同目标，不能为了负载均衡偷偷切换。

### 2. 分布式执行

所有 rank 使用同样的 slot 数和模块顺序。快 rank 不能少跑 forward，空 slot 也要经过经过验证的零贡献路径。

### 3. 恢复提交

一个恢复点必须同时对应模型、Adam、scheduler、RNG、全局 update ID、轨迹清单、策略版本和执行计划。`optimizer.step()` 返回不等于这些状态已经持久提交。

## 混合注意力的特殊边界

Packing 不只需要隔离 full attention。Qwen3.5 的 Gated DeltaNet recurrent state 和 causal convolution 也必须在样本边界重置。

正确性比较应覆盖同一批样本的：

- 单独执行；
- padding；
- packed；
- logits、loss、梯度和一步更新；
- 改变上一条样本时，下一条样本输出不变。

只重置 position ID 或 attention mask 不能证明 GDN 状态隔离。

## 待验证的三个假设

1. 混合架构实测代价模型是否优于纯 token 或 verl 固定公式；
2. 时间与显存联合规划是否提高有效吞吐、减少 OOM 和重算；
3. H20 与 910B4 是否需要不同的重计算、同步和 GDN 策略。

任何一个假设都允许得到负结果。如果强基线已经足够，应保留强基线。

## 实验顺序

1. 9B 数值与边界测试；
2. H20 和 910B4 后端能力检查；
3. 固定轨迹 actor-update 回放，隔离执行器差异；
4. 同步 GRPO 真实采样；
5. 同 world-size 故障恢复；
6. H20 多机与 27B；
7. 通过前述门禁后，再考虑 8→4→8 或异步。

性能比较必须对已启用合理动态 batch/packing 的强基线。超过全 padding 弱基线不构成系统贡献。

## 成功标准

先要求：

- loss、梯度和 update 语义正确；
- 样本无重复或丢失；
- 策略版本闭合；
- 失败恢复后状态一致；
- 未完成状态不能发布。

再比较：

- update latency 和 p95；
- 有效训练 token/s；
- OOM、拒绝和重算；
- 有效 GPU-hour；
- held-out reward/pass rate。

短程运行最多证明执行链路和质量没有明显异常，不能证明长程收敛等价。

## 来源

原始文档保留完整模型 revision、verl、NeMo-RL、Skrull、Hydraulis、AReaL 和 PyTorch DCP 引用：[完整设计](2026-10-19-system-direction.md)。
