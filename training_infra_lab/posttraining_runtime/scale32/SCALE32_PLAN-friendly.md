# 32 卡 GRPO：执行计划与完成状态

## 目标拓扑

```text
4 worker × 8 H20
├── 16 learner：TP2 × DP8
└── 16 rollout：8 个 TP2 引擎
```

模型为 Qwen3-8B，每轮 16 prompt × 4 response，共三次更新。

## 执行顺序

1. **32-rank 通信预验**：验证四节点加入、NCCL、BF16 正反向和身份映射；
2. **格式探针**：验证 tokenizer、EOS、reward 和行为概率；
3. **多机 Ray 启动**：验证端口、依赖、模型路径和 Jobs readiness；
4. **真实训练**：三次更新，审计所有 learner rank 和 rollout 引擎；
5. **最终发布**：version 3 被最终回答实际消费；
6. **发布 A/B**：同内容、同连接下比较普通和 packed broadcast。

## 已完成

- 32-rank NCCL 预验通过，并观察到跨节点 GDRDMA 通道；
- Qwen3-8B 资产逐文件校验；
- train-05 三次真实更新通过；
- publication-01 再次完成三次更新和发布 A/B；
- 16 learner rank、8 个引擎和最终 v3 消费审计通过；
- packed broadcast 没有稳定局部收益；
- 全量 checksum 的额外墙钟约 29.75 秒。

## 关键失败与修复

| 任务 | 失败 | 修复 |
|---|---|---|
| format-01 | shell 参数过长 | 压缩入口并设置大小上限 |
| format-02 | tokenizer 返回 `BatchEncoding` | 显式处理返回类型 |
| train-03 | Ray worker 端口与 GCS 冲突 | 使用不相交端口范围 |
| train-04 | EOS 使严格格式统计全部误报失败 | 格式统计与训练准入分离 |

原始回答、token 和 reward 均保留，修复没有改写历史样本。

## 验收标准

训练执行：

- 完整 group；
- 有限行为概率；
- 非零 advantage 与 gradient；
- 所有 learner rank 的参数和 Adam 变化；
- 固定 TP 后 DP 副本一致；
- 所有引擎内容一致；
- response token 使用正确策略版本。

性能：

- 相同工作量；
- 所有 rank 最大墙钟；
- 有效 token/s 和 GPU-hour；
- 记录准备、过滤、等待和重算；
- profiler 累计时间不当作墙钟。

质量：

- 独立未见题；
- 相同生成预算和评分规则；
- 多个种子；
- 格式通过不能替代能力提升。

## 下一步

不需要为了 packed broadcast 继续扩卡。更有价值的方向是：

1. 关闭重型 checksum 后测完整训练路径；
2. 改善 rollout 与 learner 重叠；
3. 修复输出格式/评分协议；
4. 增加独立 held-out 质量实验；
5. 单独验证节点丢失恢复。

原始计划保留全部任务 SID、资源快照和执行时间线：[SCALE32_PLAN.md](SCALE32_PLAN.md)。
