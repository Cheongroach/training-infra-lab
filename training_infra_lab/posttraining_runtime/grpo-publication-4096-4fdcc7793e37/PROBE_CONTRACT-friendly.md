# 修正后的最终回答探针契约

本契约与首次提交相同，但 finish-reason patch 已移到实际的 `generate_hub/single_turn.py`。

## 训练要求

- 4×H20，Qwen3-4B；
- 三次完整 GRPO 更新；
- 每轮有非零 advantage、gradient、参数与 Adam 变化；
- 两个 rollout 引擎权重内容一致；
- 第三轮后实际发布 version 3。

## 回答要求

四个固定 prompt 使用 non-thinking、temperature 0 和 4096-token 上限。每条必须：

- 自然 EOS；
- 有限行为概率；
- 全响应 token 连续消费 version 3；
- 含闭合、非空的 `boxed` 答案；
- 固定 DAPO scorer 可解析；
- 运行时与离线重评分一致。

错答案不导致协议失败；截断、无法解析和错版本会失败。四条训练内题目不用于能力收益结论。

## 执行结果

本次任务在 Ray Jobs 提交阶段返回 500，没有进入训练。因此契约未完成，不能把“正确率”记为 0/4，应记为未测。

原始版本、hash、CPU tests 和提交约束见 [精确契约](PROBE_CONTRACT.md)。
