# Miles 后训练运行时：实验脉络

这个目录记录从四卡功能验证到 32 卡扩展的完整过程。每次失败都保留，因为它们回答不同问题。

## 先看最终状态

| 阶段 | 结论 | 友好版 |
|---|---|---|
| GRPO v2 | 程序完成，但 48 条回答全截断，梯度为零 | [报告](grpo-v2/REPORT-friendly.md) |
| GRPO v3 | 前两步真实更新，第三步无 reward 方差，被门禁停止 | [报告](grpo-v3/REPORT-friendly.md) |
| GRPO v4 | 三步真实更新、参数/Adam 变化和权重回传通过 | [报告](grpo-v4/REPORT-friendly.md) |
| Recovery v1 | checkpoint 已恢复，但错误地与原始 HF 权重比较而失败 | [报告](grpo-recovery-v1/REPORT-friendly.md) |
| Recovery v2 | 同一 worker 冷进程恢复、继续训练和 version 3 发布通过 | [报告](grpo-recovery-v2-172e5c51/REPORT-friendly.md) |
| Publication 首次 | patch 指向错误生成入口，启动前主动取消 | [报告](grpo-publication-4096-eab9f5d659ce/REPORT-friendly.md) |
| Publication 修正 | Ray Jobs 没有可用 agent，训练未启动 | [报告](grpo-publication-4096-4fdcc7793e37/REPORT-friendly.md) |
| Ray-ready | 三步更新和 version 3 发布通过；回答格式协议仍失败 | [报告](grpo-ray-ready-ba214ae231ee/REPORT-friendly.md) |
| Observability | 完成四卡 GPU 指标和 NSYS 诊断；NCU 未采到有效 kernel | [报告](observability-20261024/REPORT-friendly.md) |
| Scale32 | Qwen3-8B 32 卡三步更新通过；packed 发布无稳定收益 | [报告](scale32/REPORT-friendly.md) |

## 一条成功链路需要哪些证据

一次后训练任务成功退出还不够。完整链路至少需要：

1. 轨迹有完整 prompt group、有限行为概率和可解释 reward；
2. learner 的 gradient 非零；
3. 模型参数和 Adam moments 实际变化；
4. 所有 DP/TP rank 状态一致；
5. rollout 引擎收到内容一致的新权重；
6. 下一批响应 token 实际消费新版本；
7. 独立评测才能判断能力是否提高。

本目录大部分工作集中在前六项。第七项仍然有限。

## 四卡得到的主要结论

- `enable_thinking=true`、2048 token 上限会让全部回答在思考阶段截断；
- 关闭 thinking、使用 DAPO scorer 和 4096 token 后可以产生真实组内 reward 差异；
- 有界补采能避免一个无差异候选批让整轮训练失去信号；
- 模型、Adam、scheduler、RNG、数据游标和 rollout journal 可以在同一 worker 冷进程恢复；
- 最终回答自然 EOS 不等于评分协议通过，换行后的 boxed 表达式可能无法被固定 scorer 解析；
- Ray runtime started 不等于 Ray Jobs agent 已经可提交任务。

## 32 卡得到的主要结论

- Qwen3-8B 的 16 learner + 16 rollout 布局完成三次真实更新；
- 16 learner rank、8 个 TP2 引擎和 v0→v3 消费版本闭合；
- 普通/packed broadcast 的局部差异小于本次样本波动；
- 全量模型、Adam 和引擎 SHA256 校验远比实际传输更慢；
- 输出格式失败仍然存在，训练系统通过不等于模型回答协议通过。

## 阅读建议

先读 [v2→v4 的演进](grpo-v2/REPORT-friendly.md)，再读 [恢复契约友好版](grpo-recovery-v2-172e5c51/RECOVERY_CONTRACT-friendly.md)，最后读 [32 卡参数审计](scale32/PARAMETER_AUDIT-friendly.md)。

原始 README 保留全部任务状态、镜像和提交记录：[README.md](README.md)。
