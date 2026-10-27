# 用真实 32 卡记录判断训练是否有问题

数据来自 `runs/train-05`，run `dd561599d5c16cc6`。四个 Pod 终态退出 0，源码/节点归档与实际 stdout 对齐，独立训练审计通过；以下判断只覆盖三次更新。

| 更新 | 自然完成 / 64 | 严格提取 / 64 | 奖励不同组 / 16 | 非零 advantage token | grad norm | 每 rank 变化的真实模型 tensor |
|---|---:|---:|---:|---:|---:|---:|
| 0 | 61 | 26 | 7 | 36,008 | 0.295643 | 202 |
| 1 | 64 | 12 | 5 | 25,736 | 0.205806 | 202 |
| 2 | 63 | 18 | 8 | 44,074 | 0.247101 | 197 |

## loss 接近零，为什么还在学习

第二次更新的 `train/loss = -3.259629e-09`，但梯度范数为 `0.205806`，16 rank 的参数及所拥有的 Adam m/v 均发生实际变化。组内 advantage 的均值接近零，而新旧策略在初始评估点的 ratio 接近 1，平均 surrogate loss 因此可以接近零。

用一个简化的未裁剪例子复算：

`L = -(1/N) Σ Aᵢ exp(log pᵢ − log pᵢ_old)`

当 ratio=1 且 `ΣAᵢ=0`，有 `L=0`，但 `∂L/∂log pᵢ = -Aᵢ/N`。loss 数值为零并不要求各项梯度为零。真实 PPO 还包含 mask、裁剪和参数共享，判断应联查 advantage、grad、optimizer outcome、参数与 Adam 状态。

这里 hash 证明内容变化，不能证明变化大小。Adam m/v 的 hash 也不提供元素的有限性检查；训练 tensor 与梯度的有限性在独立审计中另查。本轮没有全量参数/Adam 元素数值健康分布的结论。

## 三道实测练习与答案

1. **严格提取从 26 降到 12，训练是否已经崩溃？** 先独立写下判断，再查 `training-audit.json`。第二批仍有 5 个奖励不同组、25,736 个非零 advantage token、正有限梯度，所有 rank 更新并完成 v2 发布。系统更新契约通过；格式指标下降是真实问题，不能写成能力提高或忽略负样本。
2. **为什么 TP0/TP1 的 hash 不同，而 DP 副本必须一致？** 用 `pre-update-1-0-rank-*.json` 的 `parallel` 字段分组。TP 保存不同权重切片；固定 TP rank 后的 8 份 DP 模型是同一份副本。Adam 按本 rank 拥有的状态判断，不要求各 DP optimizer 字典相等。
3. **8 个引擎都 ack，是否足够证明下一批用了新权重？** 查 `inference_engine_weight_checksum` 与样本的 `weight_versions.spans`。需要实际内容一致，并且每条响应从 `len(tokens)-response_length` 到 `len(tokens)` 的 token 区间连续覆盖且版本正确。本次最终 16 条回答全部消费 v3；自然完成 10 条、严格提取 4 条、原评分器判对 2 条。题目已经暴露，这些数字只用于协议观察。

可复算入口见 [参数审计方法](PARAMETER_AUDIT.md)，全部训练/eval 样本和事件可在 `site/index.html` 中按“32 卡扩展 · train-05”筛选。先写预期，再看实际事件和答案。

## 吞吐分母练习

同一 train-05 的两个可复算数字：rollout 生成阶段 **241.819 response token / rollout GPU·s**，全部 Pod 驻留成本为 **6.035 admitted response token / 全部 GPU·s**。前者只用 16 个 rollout GPU × 三次生成计时，后者用全部 32 卡的终态驻留（12.5311 H20·h），包含准备、训练、发布及末次评估；两者都只统计三轮训练回答的 272,254 token，排除 prompt 与 eval token。

**练习：能否把 241.819 / 6.035 = 40.07 写成推理优化带来的 40 倍加速？** 不能。两者计时范围和设备范围都不同，缺少同工作量基线。复算：`272254 / (16 × Σ rollout_seconds)` 和 `272254 / (12.5311111111 × 3600)`。数据来自 [stage-baseline.json](runs/train-05/stage-baseline.json)，入口为 `python3 derive_stage_baseline.py --folder runs/train-05`。

三轮非零 advantage token 为 105,818，占回答 token 的 **38.87%**。这衡量学习信号覆盖；不代表 38.87% 回答正确，也不证明增加训练步会改善能力。后两轮 rank0 actor 训练约 49–51 秒、记录的发布约 0.54 秒，提示应先测主导训练路径。rank0 计时还不是所有 rank 的最大墙钟，最终优化决策仍需要全 rank 同工作量对照。

## 发布计时练习

从 `runs/publication-01/publication-benchmark-audit.json` 读取四组样本：
`broadcast/checksum-0`、`broadcast_packed/checksum-0`、`broadcast/checksum-1`、`broadcast_packed/checksum-1`。
先用 `sender_bytes` 与每 rank 的 layout/state 指纹核验相同工作量，再复算各组中位数。
`wall_seconds` 是 16 learner rank 的最大墙钟；`transfer_median_seconds` 仅覆盖上游 update_weights 的物化、传输与接收确认。

**练习：checksum-on 比 checksum-off 慢很多，是否表示 NCCL 很慢？** 不能直接推出。前者还包含完整 learner 模型/Adam 和八引擎 TP 分片的 SHA256 与完成屏障。
先比较 transfer 时间，再看墙钟中位数差。checksum-off 每次仍在计时外核验相同内容，所以这份实验只测发布区间，不能作为完全关闭诊断后的训练吞吐。

**练习：packed 的中位数更小，是否马上替换普通 broadcast？** 先查看四次样本与 min/max 是否重叠，检查收益是否足以影响实际暴露时间。
本轮只有同一任务中的四次平衡重复，没有跨任务、跨节点或多种子性能区间。若收益小于波动，当前结论应保留为局部观察。

实测答案：checksum-off 普通为 0.530577 秒、packed 为 0.522645 秒，中位数比值 1.015177；范围分别为 0.524334–0.537934 和 0.516011–0.538482 秒，重叠明显。
checksum-on 墙钟分别为 30.282540 和 30.262081 秒；传输区间中位数仍约 0.52 秒。普通的 checksum 增量为 29.751963 秒，不是 NCCL 传输额外耗时。
因此本轮支持“全量诊断开销大、packed 差异小且不稳定”的判断，没有端到端加速或 SOTA 证据。

若后续同工作量端到端测量得到发布占比 `f`，局部发布加速 `s` 的理想整体上限为
`1 / ((1-f) + f/s)`。这里的 `f` 必须来自同一端到端计时定义；不能拿 rank0 的阶段时间、NSYS 累计时间或两种 GPU·s 分母拼出它。

## 还需要怎样的证据

三次真实更新验证了分布式系统链路；要判断算法收益，还需要独立未见题、同预算前后配对、多个种子和能力退化检查。要判断端到端性能，还需要关闭重型诊断后的同工作量对照，将 rollout、训练、发布、等待和重算一起计入墙钟与有效 GPU·h。
