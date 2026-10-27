# 用这套代码理解训练 Infra

这份指南只讲五个问题。读完后，应能解释一次更新优化什么、显存在哪里、FSDP2 如何通信、checkpoint 为什么可能“能加载但不正确”，以及性能数字该如何解读。

## 1. 一次更新到底优化什么

入口是 `prepare_assets.py`、`loss.py` 和 `train.py`。

训练样本包含完整对话，但交叉熵只监督 assistant 回答。位置 `t` 的 hidden state 预测位置 `t+1` 的 token，因此代码使用 `hidden[:, :-1]` 对齐 `labels[:, 1:]`。

全局损失是：

```text
所有 rank、所有 microbatch 的监督 token loss 总和
--------------------------------------------------
所有 rank、所有 microbatch 的监督 token 数
```

这不是“每个 microbatch 的平均 loss 再求平均”。例如两个 microbatch 分别有 10 和 100 个监督 token，正确权重是 10/110 和 100/110，而不是各 1/2。

关键文件：

- `prepare_assets.py`：定位 assistant 的字符区间，再映射到完整文本的 token；
- `loss.py`：返回局部 loss 总和；
- `train.py`：用全局监督 token 数统一归一化；
- `correctness.py`：用不等长样本验证梯度。

## 2. 为什么大词表会占显存

普通输出层产生 `[batch, sequence, vocab]` logits。Qwen 词表很大，这个张量可能成为显存热点。

| 路径 | 计算范围 | 取舍 |
|---|---|---|
| `full` | 所有输入位置 | 最直接，临时 logits 最大 |
| `selected` | 只有监督位置 | 减少无用输出投影 |
| `chunk` | 监督位置分块 | 限制临时 logits，但反向需要重算 |

`selected` 和 `chunk` 解决的问题不同。只写 Python 分块循环也不一定省峰值，因为 autograd 可能保留每一块；这里使用 checkpoint 让反向逐块重算。

动态 padding 又是另一层优化：它减少整个 backbone 的补齐计算，而不是只改变输出层。

## 3. FSDP2 在做什么

一个 layer 的典型流程是：

```text
参数分片 → all-gather 完整参数 → 前向 → 释放
         → 反向前再次 all-gather → 计算梯度
         → reduce-scatter 梯度 → 更新本地参数和 Adam 分片
```

这里每个 Transformer layer 是一个 FSDP 单元，embedding 和共享输出 head 留在同一个 root owner。优化器必须在分片之后创建。

默认每个 microbatch 都 reduce-scatter。`--defer-gradient-sync` 在前几个 microbatch 保留完整梯度，最后一次再同步。因此：

- ReduceScatter 次数下降；
- 参数 AllGather 不会自动下降；
- 未分片梯度驻留更久，显存上升。

是否启用它必须同时看通信次数、step time 和显存。

## 4. checkpoint 为什么可能“能读但不正确”

`checkpointing.py` 保存模型、Adam、scheduler、RNG、下一个 step、world size、训练配置和数据摘要。

恢复实验比较：

```text
连续 6 步
vs.
训练 3 步 → 保存 → 新进程加载 → 再训练 3 步
```

至少要检查两处：

1. 保存点与加载后初始状态是否相同；
2. 连续训练和恢复训练的最终状态是否相同。

曾经出现过一个真实错误：读取 `optimizer.state[param]` 会向 defaultdict 插入不存在的临时参数，随后 checkpoint 保存报错。修复使用 `.get()`，并只遍历优化器真正持有的参数。

所以“文件成功写入”“加载没有异常”和“恢复后的训练轨迹一致”是三个不同结论。

## 5. 如何读性能结果

先确认分母和计时范围：

- 吞吐使用全体设备的非 padding token/s；
- step 时间包含前向、反向、梯度裁剪和 Adam；
- 不包含模型加载、分词、eval 和 checkpoint；
- 多卡取最慢 rank；
- profiler 样本不混入普通吞吐。

扩卡时还要区分：

- 固定 global batch：强扩展；
- 随卡数增加 global batch：弱扩展；
- 单阶段 token/s；
- 整个任务的有效 GPU-hour。

它们不能互相替代。

## 建议亲手做的三件事

1. 把全局 token mean 改成 microbatch mean，观察不等长样本的梯度检查如何失败。
2. 只切换梯度同步时机，先预测 ReduceScatter 次数和显存，再看 profiler。
3. 对照 continuous、split 和 resumed 的状态摘要，分别找出“加载正确”和“最终重放正确”的证据。

完成这些练习后，再阅读 [Qwen3.5 GRPO 源码指南](trajectory_runtime/source-guide-friendly.md)。
