# 32 卡训练记录：三个容易误判的例子

数据来自成功的 train-05。

## 例一：loss 接近零，为什么仍有学习

第二次更新的 loss 是 `-3.26e-9`，接近零，但：

- gradient norm = 0.205806；
- 25,736 个 response token 的 advantage 非零；
- 16 个 learner rank 的参数变化；
- Adam moments 变化；
- version 2 发布完成。

GRPO 的组内 advantage 均值接近零，初始 ratio 又接近 1，因此平均 surrogate loss 可以接近零。但每个 token 对 log-prob 的梯度仍与其 advantage 有关。

结论：不能只看 loss 标量，必须联查 advantage、gradient、参数和 Adam。

## 例二：严格格式从 26/64 降到 12/64，训练是否崩溃

没有。第二轮仍有：

- 5 个 reward 不同的组；
- 25,736 个非零 advantage token；
- 正且有限的 gradient；
- 全 rank 更新；
- version 2 发布。

系统更新契约通过，但格式指标明显变差。正确表述是“训练执行成功，格式质量仍有问题”，不是“能力提高”或“训练崩溃”。

## 例三：8 个引擎都 ack，是否证明下一批用了新权重

不够。还需要两层证据：

1. 8 个引擎的实际 tensor 内容一致；
2. 下一批每条响应的 `weight_versions.spans` 连续覆盖全部 response token。

train-05 的最终 16 条回答全部消费 version 3，因此发布到消费链路闭合。

## 吞吐分母练习

同一任务有两个数字：

- 241.819 response token / rollout GPU·s；
- 6.035 response token / 全部 GPU·s。

第一个只统计 16 张 rollout GPU 的生成阶段；第二个使用 32 张卡的完整 Pod 驻留。两者相除得到约 40，但不能写成 40× 加速，因为设备和计时范围不同。

## 发布计时练习

checksum-off：

- 普通 broadcast：0.530577 秒；
- packed：0.522645 秒；
- 比值：1.015。

两者范围重叠，所以没有稳定 packed 收益。

checksum-on 约 30.27 秒，但实际传输区间仍约 0.52 秒。多出的约 29.75 秒来自全量 learner 模型、Adam 和引擎分片校验及屏障，不能归因于 NCCL。

## 正确的分析顺序

遇到一次训练记录时，按以下顺序：

1. 样本和 reward 是否形成组内信号；
2. gradient 是否有限且非零；
3. optimizer 是否实际更新参数和 Adam；
4. DP/TP 状态是否一致；
5. 权重是否发布；
6. 下一批是否消费新版本；
7. 独立质量评测是否改善。

前三至六项通过，只证明系统链路；第七项才回答模型效果。

更多练习和复算公式见 [原始案例](WORKED_EXAMPLE.md)。
