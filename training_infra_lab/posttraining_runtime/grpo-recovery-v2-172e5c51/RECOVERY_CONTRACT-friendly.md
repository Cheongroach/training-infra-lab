# 冷进程恢复：需要恢复哪些状态

## 恢复边界

本实验在同一个 Pod、同一个 worker 本地文件系统上完成：

```text
训练 rollout 0 和 1
→ 保存 checkpoint
→ 关闭 Ray 与训练进程
→ 启动新进程
→ 从 rollout 2 继续
→ 发布 version 3
```

它不是节点丢失、断电或跨机器恢复。

## 一个完整恢复点包含什么

| 状态 | 为什么必须恢复 |
|---|---|
| 模型参数 | 决定当前策略 |
| Adam moments | 决定后续优化轨迹 |
| LR scheduler | 防止学习率重复或跳步 |
| Python/NumPy/Torch/CUDA RNG | 保持训练随机状态 |
| datasource cursor | 防止重复或跳过请求 |
| qualification journal | 拒绝批也已经消费样本身份 |
| policy version | 绑定 learner 与 rollout |
| engine tensor 内容 | 版本号相同仍可能加载错权重 |

只恢复模型不足以继续同一训练状态。

## 如何验收

1. 比较两个 TP rank 保存后与加载刚返回时的模型、Adam、scheduler 和 RNG 指纹；
2. 重新计算候选文件 SHA、录取/拒绝决策和全部 sample identity；
3. 要求 datasource cursor 与 journal 覆盖范围一致；
4. 比较两个新引擎的全部 290 个 raw tensor SHA256；
5. 使用新的 publication UUID，避免把 checkpoint 重放的旧事件当成新同步；
6. 恢复后第三轮必须有非零 gradient、参数和 Adam 变化；
7. 最终生成的每个响应 token 必须消费 version 3。

## 为什么不能和原始 HF 权重比较

原始 HF 是 version 0，checkpoint 保存的是训练后的 version 2。恢复后与 version 0 不同是正确现象。

恢复入口只在固定 commit、源码 hash 和 `start_rollout_id=2` 同时匹配时，改为与 checkpoint 提交权重比较；普通入口继续与原始 HF 比较。数值容差没有放宽。

## 仍未覆盖

- 任意指令位置的 crash consistency；
- 磁盘断电和 fsync 持久性；
- HDFS 原始 checkpoint；
- 节点丢失或跨机恢复；
- 与不中断训练逐位相同；
- rollout 引擎采样 RNG 恢复；
- 模型能力提升。

原始精确契约见 [RECOVERY_CONTRACT.md](RECOVERY_CONTRACT.md)。
