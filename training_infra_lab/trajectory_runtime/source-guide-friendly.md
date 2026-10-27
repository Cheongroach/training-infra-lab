# 从一条 GRPO 轨迹读懂分布式更新

## 1. 轨迹契约

`contracts.py` 中的每条 `Trajectory` 保存：

- `trajectory_id`：回答身份；
- `prompt_group_id`：同一道题的回答组；
- `policy_version`：由哪个策略生成；
- `input_ids` 和 `response_mask`；
- 生成时保存的 `old_log_probs`；
- reward。

GRPO 的 advantage 必须先在完整 prompt group 上计算，再把轨迹分给不同 rank。若每个 rank 只看局部回答，算法已经改变。

`response_mask` 与 log-prob 都对齐到 `input_ids[1:]`。prompt 最后一个 token 的 hidden state 负责预测第一个 response token，这是最容易出现的一位偏移错误。

## 2. 生成概率为什么还要回放

`rollout.py` 用当前策略采样，并保存每个响应 token 的行为概率。训练前，`grpo.py` 再用训练路径计算一次相同 token 的概率。

两者可能因为缓存生成、dtype、kernel 或模型路径不同而有误差。因此代码检查：

- RMS 误差；
- p99 绝对误差；
- 超出 PPO clip 区间的 token 比例。

概率门禁通过只说明采样和训练路径足够接近，不说明模型学得更好。

## 3. loss 如何跨 rank 保持语义

本项目使用 sequence mean。每条轨迹先产生局部 loss 分子，`executor.py` 再按全局轨迹数归一化。

FSDP 会平均各 rank 的梯度，因此局部 backward 需要乘 `world_size / global_denominator`。这样，不同 rank 分到的轨迹数和长度不同，仍对应同一个全局目标。

不能把每个 microbatch 的 mean 直接相加，也不能为了长度均衡把 sequence mean 偷换成 token mean。

## 4. 为什么所有 rank 要执行同一 slot 结构

FSDP forward/backward 包含 collective。一个 rank 少跑一层或提前结束，其他 rank 可能永远等待。

`planner.py` 生成 `slots[slot][rank]`：

- 每条轨迹恰好出现一次；
- 所有 rank 的 slot 数一致；
- 每个 slot 的 token 预算合法；
- batch SHA 和 world size 固定。

空 slot 不能简单跳过。当前使用最小有效样本并把 loss 置零，以保持相同模块和 collective 顺序。它仍消耗时间，所以规划器应尽量避免空 slot。

## 5. 设备门禁在比较什么

`device_gates.py` 使用一个独立的全局 batch reference，不复用分布式 executor 的分批逻辑。然后比较：

- 参数；
- 梯度；
- Adam `exp_avg`；
- Adam `exp_avg_sq`；
- loss。

tiny 门禁通过不保证 9B 真实批次一定通过。v7 中，microbatch 从 1 改为 2 后，9B 梯度相对差异达到约 18%，而 tiny 测试没有发现。因此 9B 回放是独立验收层。

## 6. 生成期权重驻留

普通 FSDP 在逐 token 生成时重复 all-gather。`residency.py` 在显存允许时：

1. 聚合 BF16 参数；
2. 在整个生成阶段保持聚合状态；
3. 禁止梯度；
4. 生成结束后遍历所有 FSDP module 重新分片；
5. 恢复训练策略。

准入由所有 rank 投票。任一 rank 显存不足，全组都回退普通模式。测试还检查生成前后参数和 Adam 不变、异常后的清理，以及退出驻留后仍能训练。

## 7. 奖励为何要版本化

旧规则 `last_decimal_exact_v1` 只取最后一个数字，可能把截断的中间推导判对。

新规则 `eos_final_line_decimal_v2` 要求：

- `finish_reason == eos`；
- 最后一行匹配 `Final answer: <number>`；
- 数字与固定参考答案完全相等。

输出分为 `correct`、`incorrect`、`missing_final_answer` 和 `truncated`。这样可以区分数学错答、格式失败和没有完成。

## 8. 如何独立复算

`audit_training.py` 不只读取汇总事件。它重新对齐：

- rollout 与 batch 的 trajectory ID；
- response token 数与 mask；
- batch SHA 和 policy version；
- 每个 rank 的更新记录；
- 严格奖励下的原始文本与固定答案；
- 梯度范数和 Adam 非零元素数。

这能证明“保存的结果彼此一致”，但不能代替 held-out 质量评估。

建议实际阅读顺序：

```text
contracts.py
→ rewards.py
→ rollout.py
→ rl.py
→ planner.py
→ executor.py
→ grpo.py
→ audit_training.py
```
