# 从这份代码学习训练 Infra

这套实验把真实 Qwen 全参数 SFT 拆成几个可以独立验证的问题。先读
`README.md` 的测量口径，再按下面的顺序读代码；实测结果与能力边界以
[结果报告](report.md)为准。

## 1. 先确认一次更新究竟优化什么

入口是 `prepare_assets.py`，然后是 `loss.py`。训练样本包含完整对话，
但只有 assistant 回答参与交叉熵。因果预测中，位置 `t` 的 hidden state
用于预测位置 `t+1` 的 label，因此代码对应 `hidden[:, :-1]` 和
`labels[:, 1:]`。`-100` 表示该目标位置不参与损失。

不要用“对话前缀的 token 数”切 assistant 的标签。BPE 分词可能跨越
消息头、换行和正文的边界，前缀单独分词的结果不保证是整段分词的前缀。
这里先对完整模板文本分词，再用字符 offset 判断 token 是否完全落在
assistant 的监督区间；跨边界 token 保守忽略，并统计数量。

一次 optimizer step 的目标是：

```text
L = 所有 rank、所有 microbatch 的有效 token loss 之和
    / 这些 microbatch 的全局有效监督 token 数
```

`loss.py` 返回局部的 **sum**，`train.py` 统一归一化。FSDP 默认平均各
rank 的梯度，所以 backward 前乘 `world_size` 抵消这次平均。
不能把每个 microbatch 的 mean loss 简单平均：短回答会因此获得过高权重。

可手算的例子：两个 microbatch 分别含 10 和 100 个监督 token，每个
token 的 loss 都相同。正确权重是 `10/110` 和 `100/110`，而非各 `1/2`。
`correctness.py` 用不等长 mask 和全忽略样本验证这个契约。

## 2. 大词表为什么会成为显存热点

通常语言模型输出 `logits[B, S, V]`。Qwen 的词表很大，输出层的内存可能
比直觉中“最后一层线性层”的印象大得多。这里可以比较三个分支：

| 分支 | 计算哪些位置 | 反向前保留什么 |
|---|---|---|
| `full` | 所有输入位置 | 全量 logits 与交叉熵所需中间结果 |
| `selected` | 只计算有监督 label 的位置 | 全部监督位置的中间结果 |
| `chunk` | 监督位置分块计算 | checkpoint 输入；反向时重算各块 |

`selected` 减少无监督位置上的输出投影工作量。`chunk` 进一步限制临时
logits 的尺寸，但增加输出层重计算。两者是不同作用，不能只比较首尾
就声称全部收益来自分块。

只写 Python 分块循环通常不够：autograd 可能把所有块的反向激活都留下，
总峰值依旧很大。这里使用非重入 checkpoint，让反向逐块重算。
代码没有使用专用融合交叉熵 kernel；它是一条跨设备可移植的实现路径。

这是已有的优化思想。工程价值在于正确接入训练目标、共享权重、分布式
梯度和设备运行时，并用消融实验确认取舍。

## 3. 动态 padding 与输出层优化有什么不同

在 `train.py:batch_at` 中，固定 padding 把所有样本补齐到设定上限；
动态 padding 按当前 microbatch 最长样本向上对齐到 128。
有效 token 及其标签保持相同，减少的是补齐位置上的计算。

动态 padding 会影响整个 backbone；筛选监督位置只影响输出损失层。
因此需要在相同 padding 下对比 `full`、`selected`、`chunk`，并另外比较
固定与动态 padding。

实验中的 microbatch 为 1，并未实现多样本 packing 或长度均衡调度。
多卡上每个 rank 的序列长度不同，快 rank 可能等待慢 rank；
“省了 padding”不自动等于“各卡工作量均衡”。这是后续可独立研究的问题。

## 4. FSDP2 在一次前向和反向中做了什么

```mermaid
flowchart LR
    A[本 rank 的参数分片] --> B[逐层 all-gather]
    B --> C[该层前向]
    C --> D[释放完整参数]
    D --> E[反向前再次聚合参数]
    E --> F[计算梯度]
    F --> G[reduce-scatter 梯度]
    G --> H[本 rank 更新参数与 Adam 状态分片]
```

`train.py` 先加载模型，把每个 Transformer layer 包成 FSDP 单元，
最后包装根模块；优化器必须在分片之后构造。
输入 embedding 和输出 head 共享参数，因此留在同一个根 FSDP owner 下。
分别包装共享参数的两个使用者，容易破坏参数所有权与生命周期。

模型保存 FP32 参数，前向使用 BF16，梯度归约使用 FP32。观察性能时，
需要同时看计算精度、参数 all-gather、梯度 reduce-scatter、Adam 更新。
“用了 BF16”不代表优化器状态或所有通信也变成 BF16。

梯度裁剪必须基于全局参数梯度范数。不能每个 rank 独立裁剪自己的分片；
这会改变优化算法。当前通过 DTensor 对应的全局 norm 语义执行裁剪。

默认每个 microbatch 都同步。`--defer-gradient-sync` 调用 FSDP2 的
`set_requires_gradient_sync`，只在最后一次 microbatch 打开同步。
已用不等长 mask 对照未分片模型校验梯度和一次更新；同条件性能/显存消融见报告。
读 profiler 时核对 ReduceScatter 次数，不能期待参数 AllGather 自动减少。
它用未分片梯度驻留内存换取较少通信，不能不测内存就默认开启。

## 5. checkpoint 的“能读回来”和“恢复正确”有何区别

`checkpointing.py` 用 DCP 保存模型与 Adam 状态，额外保存：

- 下一个 optimizer step，也对应确定性数据顺序中的位置；
- 每个 rank 的 CPU、设备 RNG；
- world size、训练参数和数据 manifest 摘要。

读取时校验这些条件，防止悄悄换数据、换 loss 模式或改变 batch 定义。
当前恢复契约要求相同 world size，不应称为弹性扩缩容恢复。

恢复实验比较连续运行 6 步，与“运行 3 步 → 新进程加载 → 再运行 3 步”。
检查所有本地持久参数分片和 Adam 状态的字节摘要，外加最终验证 loss。
先比较 split 保存前与 resumed 加载后的摘要，再比较连续与续跑最终摘要。
摘要遍历 optimizer param groups，而不是模型当前可能暴露的临时参数视图；
`optimizer.state` 是 defaultdict，读取不存在的 key 会修改状态。这曾在 NPU
环境中引入孤立条目，导致保存 KeyError；现在使用 `.get()` 和不变量断言。
若设备算子或通信存在非确定性，摘要可能不同；应报告差异并定位，
不能把“任务没报错”当作恢复一致。

这个实验也不等于任意时刻的故障恢复：保存点在 optimizer step 边界，
没有恢复半个 microbatch 或半次参数更新的承诺。

## 6. 如何读性能数字

先看有效非 padding token/s 和监督 token/s，再看峰值 allocated /
reserved 显存。padding 后的 token/s 可用于解释形状，不适合作为唯一收益。
每一步同步设备并取最慢 rank 的耗时，预热步不计入吞吐。

计时包含前向、反向、梯度裁剪、Adam 更新；不包含下载、分词、batch 搬运、
验证和 checkpoint。因此报告的是驻留数据下的训练步吞吐。
Profiler 步另行剔除，避免把采样开销算进普通训练。

跨 H20 与 910B4 比较时，软件栈版本不同。优先回答“各自优化前后怎样”，
再描述本次环境下的绝对数值；不要把一次短测推广成硬件总体优劣。

扩卡时，固定全局 batch 才能讨论强扩展。增加卡数还会改变每卡梯度累积
次数与样本分配，需一并披露。两机实验还必须有不同宿主机的证据；
两个 worker 进程可能实际被放在同一宿主机。

## 7. 适合在面试中讲的工程闭环

从实际瓶颈出发，解释内存来源，给出保持目标不变的实现，再证明 loss、
逐参数梯度和一次更新正确。用相同数据和训练预算做消融，最后说明
吞吐、显存、恢复一致性与扩展效率之间的取舍。

模型规模和卡数只写实测值。短程 SFT 不证明模型质量提升，1.5B 模型
多卡运行也不自动构成“大规模训练系统”。可以讲清楚并复现的系统行为，
比没有证据的规模描述更有价值。


## 8. 建议亲手完成的三次小实验

1. 在 correctness gate 中故意把全局 token mean 改成 microbatch mean，
   观察不等长 mask 的逐参数梯度比较为何失败，再恢复正确公式。
2. 固定 batch/样本，只切换同步时机，预测通信次数与梯度内存的变化，
   再打开已有 profiler 对照预测；不要先追求更高吞吐。
3. 对照恢复实验的 continuous/split/resumed JSON，找出数据从 step 3 开始、
   加载状态一致、最终状态一致分别由什么字段证明。然后解释为何这仍不是
   任意机器故障后的恢复保证。

这些练习能把“AI 生成的实现”转化为你能解释、修改和独立验证的工程能力。


## 9. 如何接回你做过的 RL 项目

FSDP 分片、混合精度、梯度累积同步、显存观测和完整状态恢复，都可以迁移到
RL 的 actor update 阶段。变长 rollout 会把这里的 padding 和 rank 负载问题放大。
但 PPO/GRPO 的 loss 可能有意按序列或其他方式加权；SFT 的全局 token mean
不能机械套用，应先明确算法目标，再验证分布式实现与参考计算一致。
