# 32 卡 Miles 后训练项目：设计读者版

## 项目要解决什么

系统把 32 张 H20 分成：

- 16 张 learner：TP2 × DP8；
- 16 张 rollout：8 个 TP2 SGLang 引擎。

核心问题是：learner 更新后的权重如何可靠地发布给 rollout 引擎，并在真实训练中减少停顿，而不破坏样本、行为概率和策略版本的对应关系。

这份文档最初是执行前方案。后续已经完成的 32 卡结果见 [实测报告友好版](../../posttraining_runtime/scale32/REPORT-friendly.md)。

## 为什么先做观测，而不是直接写优化

固定 Miles 源码已经有：

- 逐 tensor broadcast；
- packed broadcast；
- P2P 权重传输；
- disk-delta；
- 双 buffer 和映射缓存；
- fully async 数据缓冲和版本检查。

因此“打开一个已有开关”不是新的工程贡献。首个可靠工作应当是把一次权重更新拆成可测阶段：

```text
提取权重
→ 格式/分片转换
→ 发送与在途等待
→ 引擎暂停、加载和确认
→ 恢复生成
→ 下一批 token 实际消费新版本
```

只有实测确认某一阶段是瓶颈，才对该阶段做小范围修改。

## 固定实验结构

| 阶段 | 模型与资源 | 目的 |
|---|---|---|
| 四卡预验 | Qwen3-4B；2 learner + 2 rollout | 验证 token、概率、reward 和版本契约 |
| 16 卡跨机预验 | Qwen3-8B；8 learner + 8 rollout | 验证多机栈和网络路径 |
| 32 卡主实验 | Qwen3-8B；16 learner + 16 rollout | 三次真实更新和发布 A/B |

首轮固定同步 GRPO，避免把异步陈旧度与传输协议同时改变。外围可回收 worker 只是后续设计，不属于首轮完成内容。

## 每条轨迹必须保留什么

- prompt group 和 sample identity；
- token IDs 与 response mask；
- 生成时的行为 log-prob；
- reward 与解析状态；
- 实际消费的策略版本；
- 截断和结束原因。

训练前必须验证完整 group、有限概率和版本覆盖。发布版本递增不等于引擎已经收到正确内容；引擎 ack 也不等于下一批 token 实际使用了该版本。

## 正确的比较方法

### 权重发布

固定同一更新后的权重、同一引擎、同一连接和同一 bucket 布局，比较普通 broadcast、packed broadcast 或 P2P。报告：

- 实际字节；
- 初始化和稳态时间；
- 所有 rank 的最大墙钟；
- 引擎不可服务区间；
- 内容一致性。

### 训练吞吐

必须把 rollout、训练、发布、等待、过滤和重算纳入同一个计时定义。发布阶段局部变快不能直接写成 GRPO 端到端加速。

### 学习质量

参数和 Adam 变化只证明发生了更新。能力结论需要独立未见题、相同生成预算和多个种子。

## 后续实际结果

执行后的关键观察是：

- 32 卡 train-05 完成三次真实更新，16 learner rank 和 8 个 rollout 引擎审计通过；
- publication-01 的 packed/普通 broadcast 中位比值为 1.015，范围重叠，没有稳定收益；
- 全量 SHA256 诊断增加约 29.75 秒发布墙钟；
- 输出格式仍有大量失败，训练执行成功不等于回答协议通过。

这些结果说明，下一步优先级应是减少重型诊断的暴露时间、改善训推重叠和修复输出协议，而不是继续只优化 packed broadcast。

## 原始来源

[完整源码调研与执行前方案](README.md)保留固定 Miles commit、源码定位、实验矩阵和参考链接。
