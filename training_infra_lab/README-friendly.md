# Training Infra Lab：从哪里开始读

如果前面的报告仍然难读，请先只读 [一步步认识这个 Training Infra 项目](一步步认识Training%20Infra项目-friendly.md)。它假设你已经理解 embedding、Transformer、梯度和 Adam，从一轮真实 GRPO 开始，每一步只增加一个项目概念。

这个目录记录了四条训练基础设施研究线。它们的模型、框架和结论不同，不能把数字混在一起。

| 主线 | 核心问题 | 已完成的最高层证据 | 建议入口 |
|---|---|---|---|
| 历史 SFT | FSDP2 训练如何兼顾正确性、显存、通信和恢复 | Qwen2.5-1.5B，H20/910B4，最多 16×H20 | [实验报告友好版](report-friendly.md) |
| 原生 GRPO | Qwen3.5-9B 的真实采样、更新和生成期权重驻留能否正确执行 | 4×H20；一次严格奖励下的非零梯度更新；生成阶段 1.761× | [Qwen3.5 运行时入口](trajectory_runtime/README-friendly.md) |
| Miles 后训练 | learner、rollout、权重发布和恢复如何形成可审计闭环 | Qwen3-4B 四卡恢复；Qwen3-8B 32 卡三次更新 | [Miles 运行时入口](posttraining_runtime/README-friendly.md) |
| OPD | 如何把领域 teacher 的能力迁移到 student，并控制保持性 | 36 组补强训练；新留出集三种子验收 | [OPD 友好版](opd/OPD_DESIGN_AND_INTERVIEW-friendly.md) |

## 先看结论

### Qwen3.5-9B 原生 GRPO

- v8 完成四次非零梯度更新，但旧奖励规则会把截断的中间数字误判为正确，因此不能据此判断学习质量。
- v9 修复奖励后生成 64 条回答：62 条正确、2 条明确错答，全部正常结束。前三步因组内奖励相同而梯度为零；第四步产生梯度范数 0.915 和非零 Adam 状态。
- v8 的生成期 FSDP 权重驻留把 128-token 生成阶段中位耗时从约 12.38 秒降至 7.04 秒，即 1.761×。这不是完整 RL 系统加速。
- 910B4 新模型探针尚未获得运行结果。

### Miles 四卡与 32 卡

- 四卡 Qwen3-4B 已证明三次真实更新、参数与 Adam 变化、权重发布和同一 worker 冷进程恢复。
- 四卡最终回答探针虽然自然结束，但固定评分器无法解析其中两条格式，完整回答协议未通过。
- 32 卡 Qwen3-8B 已完成三次真实更新，覆盖 16 learner rank、8 个 TP2 rollout 引擎和最终 v3 消费。
- packed broadcast 相对普通 broadcast 的局部中位比值仅 1.015，样本区间明显重叠，没有稳定加速证据。
- 全量 SHA256 诊断约增加 29.75 秒发布墙钟，说明诊断本身可能成为主要开销。

### OPD

- 补强实验选择 524,288 response-token 的 OPD 配置。
- 新留出 bank 集三种子均值 98.44%，相对 warm student +10.94 个百分点，相对 replay +1.53 个百分点，距领域 expert 0.15 个百分点。
- MBPP 保持集均值相对 warm 为 +0.31 个百分点；每个种子的配对 95% CI 下界均高于预设的 -2 个百分点门槛。
- 结果只覆盖同规模、同 tokenizer 的两个代码任务集；没有验证异步 rollout、奖励 RL、异源词表或线上部署。

## 如何阅读

如果目标是理解系统，按以下顺序：

1. [历史 SFT 源码指南友好版](source-guide-friendly.md)：先理解 token 归一化、FSDP2 和 checkpoint。
2. [Qwen3.5 源码指南友好版](trajectory_runtime/source-guide-friendly.md)：再理解 GRPO group、行为概率和全 rank 执行计划。
3. [32 卡参数审计友好版](posttraining_runtime/scale32/PARAMETER_AUDIT-friendly.md)：最后理解 learner 到 rollout 的权重版本闭环。

如果目标是看结果：

1. [Qwen3.5 测量报告友好版](trajectory_runtime/measured-report-friendly.md)
2. [Miles 四卡实验脉络](posttraining_runtime/README-friendly.md)
3. [32 卡报告友好版](posttraining_runtime/scale32/REPORT-friendly.md)
4. [监控与 Nsight 报告友好版](posttraining_runtime/observability-20261024/REPORT-friendly.md)

## 证据边界

“任务成功退出”只说明程序完成；“参数发生变化”只说明执行了更新；“训练后能力提升”需要独立评测。本文档中的结论始终区分这三层。

原始 Markdown、JSON、日志、归档和源码哈希继续保留。`-friendly.md` 负责解释问题和证据，不替代原始记录。
