# Ray-ready 与最终回答探针：验收条件

## 新增的 Ray 门禁

Ray 启动后，先提交一个 1 CPU、0 GPU 的短任务。只有以下条件同时满足，才允许提交训练：

- 唯一 submission ID；
- 固定 Ray build 和 session；
- 日志包含随机校验标记；
- 状态为 `SUCCEEDED`；
- 总等待不超过 120 秒。

只对明确的临时 `No available agent` 或传输失败进行有界等待。训练 Job 本身不重试。未确认提交成功时，不执行 stop。

## 训练与回答条件

沿用最终回答探针：

- 三轮真实更新；
- learner 参数和 Adam moments 变化；
- rollout 引擎内容一致；
- 最终 version 3 发布；
- 四条回答自然 EOS、boxed、可被冻结 DAPO scorer 解析；
- 所有响应 token 消费 version 3。

CPU checks 只验证代码路径，不替代真实 GPU 验收。

## 实际结果

Ray readiness 在 2.794 秒内通过，随后训练完成。训练与权重发布通过，但四条回答中的两条无法被固定 scorer 解析，因此整体协议仍失败。

完整固定版本和异常路径见 [原始契约](PROBE_CONTRACT.md)。
