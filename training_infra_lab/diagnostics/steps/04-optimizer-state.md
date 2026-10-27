# 04 分布式参数与优化器状态审计

## 诊断问题

非零奖励差异是否真正传递为梯度、Adam moments 和模型参数变化，并且所有并行副本保持一致？

## 输入参数

| 参数 | 含义 |
|---|---|
| update / rollout ID | 更新边界 |
| learner rank | 训练进程身份 |
| `dp_rank` / `tp_rank` | 数据并行与模型切片身份 |
| gradient norm | 当前更新梯度证据 |
| 参数 SHA256 | 模型内容身份 |
| `exp_avg` / `exp_avg_sq` SHA256 或非零计数 | Adam 状态证据 |
| optimizer outcome | step 是否成功或跳过 |

## 操作步骤

1. 从运行记录重建每个 learner rank 的并行坐标。
2. 在更新前后比较当前 rank 实际拥有的模型 tensor。
3. 分别比较 Adam 一阶矩和二阶矩，不用标量 `step` 代替。
4. 检查 gradient、参数和 optimizer tensor 的有限性。
5. 固定 TP rank，再比较各 DP 模型副本。
6. 排除 companion/witness/version 等非模型元数据。
7. 将结果与该更新的非零相对优势 token 数关联。

## 输出结果

32 卡 train-05 的三次更新：

| 更新 | 非零 advantage token | gradient norm | 每 rank 变化模型 tensor |
|---|---:|---:|---:|
| 0 | 36,008 | 0.295643 | 202 |
| 1 | 25,736 | 0.205806 | 202 |
| 2 | 44,074 | 0.247101 | 197 |

16 个 learner rank 均完成更新；固定 TP rank 后，8 个 DP 副本一致；各 rank 持有的 Adam moments 发生变化。

## 最终诊断结论

train-05 的三次更新均具有非零训练信号、有限梯度、参数变化和 Adam 状态变化，且数据并行副本没有分叉。这比“loss 有值”或“optimizer step 增加”提供更强的更新证据。

## 原理说明

梯度是更新输入，参数变化是更新结果，Adam moments 决定后续优化轨迹。TP rank 持有不同模型切片，不能互相比 hash；DP rank 持有同一切片的副本，成功同步更新后应保持一致。loss 接近零也不意味着梯度为零，因为组内优势可相互抵消标量值，但各 token 梯度仍非零。

## 注意事项

- BF16 参数 hash 不变时，还要检查 FP32 master weight。
- 参数变化可能来自旧 momentum 或 weight decay，必须联查当前 gradient。
- Adam distributed state 不要求不同 rank 的字典结构相同，只检查各自实际拥有的状态。
- SHA256 证明内容相等或变化，不提供变化幅度。
