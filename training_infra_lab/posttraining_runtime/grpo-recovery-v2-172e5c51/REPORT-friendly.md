# Recovery v2：恢复、继续训练和最终发布均通过

## 研究问题

四卡 Qwen3-4B 在完成两轮训练后，能否关闭全部 Ray/训练进程，从本地 checkpoint 恢复，并继续第三轮更新和权重发布？

## 结果

任务正常结束，独立语义验收通过。

| 检查 | 结果 |
|---|---|
| checkpoint | 20 文件，52.4628 GiB |
| learner 模型与 Adam | 两个 TP rank 保存/加载指纹一致 |
| scheduler 与 RNG | 完整恢复 |
| 数据状态 | 恢复 32 个已消费身份，下一轮为 rollout 2 |
| 新引擎权重 | 两台引擎全部 290 个 tensor 与 version 2 一致 |
| 恢复后训练 | 非零 advantage、gradient、参数与 Adam 变化 |
| 最终发布 | version 3 发布成功 |
| 最终生成 | 4 条回答的 2,048 个响应 token 全部消费 version 3 |

## 三轮训练信号

| 轮次 | 候选批 / 拒绝批 | 非零 advantage token | grad norm |
|---:|---:|---:|---:|
| 0 | 1 / 0 | 9,441 | 0.290 |
| 1 | 1 / 0 | 6,202 | 0.257 |
| 2，恢复后 | 2 / 1 | 15,033 | 0.432 |

三轮中两个 TP rank 的模型和 Adam moments 均发生变化。保存前的 version 2 与恢复后重新发布的 version 2，全部引擎 tensor 变化数为 0；第三轮后的 version 3 改变 202 个 tensor。

## 修复了什么

v1 错误地把训练后的 version 2 与原始 HF version 0 比较。v2 只在限定恢复入口中改为与 checkpoint 提交权重比较，并要求全部 raw tensor SHA256 一致。

历史事件在恢复时会重放。审计器只允许完全一致的旧事件去重，并使用新的 publication UUID 和时间顺序证明这是一次新同步。

## 结论边界

本轮证明：

- 同一 worker 本地 checkpoint 的受控冷进程恢复；
- learner 状态、数据游标和引擎权重的一致恢复；
- 恢复后可以继续训练；
- 最终权重可以发布并被生成消费。

它没有证明：

- 节点丢失或跨机恢复；
- 任意时刻 crash consistency；
- HDFS 持久 checkpoint；
- 与不中断训练逐位等价；
- 能力或吞吐提升。

四条最终 probe 均因 512-token 上限截断、reward=-1，所以只用于版本覆盖验收。

详细证据和任务成本见 [原始报告](REPORT.md)。
