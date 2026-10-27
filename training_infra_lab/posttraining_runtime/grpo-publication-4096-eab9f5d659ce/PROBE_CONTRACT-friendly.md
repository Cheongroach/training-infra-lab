# 最终回答探针：验收条件

## 目标

在不重复 52.46 GiB 冷恢复实验的情况下，完成三次真实更新，发布 version 3，并检查四条最终回答是否自然结束且能被固定 scorer 解析。

## 固定设置

- Qwen3-4B；
- 4×H20；
- 2 learner TP2 + 2 rollout TP1；
- DAPO 64 题；
- non-thinking；
- temperature 0；
- 每条最多 4096 response token。

## 训练通过条件

- 三轮完整 4×4 group；
- 样本身份、概率和 learner 输入一致；
- 每轮有非零 advantage 和 gradient；
- 两个 learner rank 的参数与 Adam moments 变化；
- 两个 rollout 引擎内容一致；
- 最终发布 version 3。

## 四条回答通过条件

- 四个固定 prompt 各产生一条回答；
- 原始 finish reason 是自然 stop；
- 行为概率覆盖所有响应 token；
- 每个 token 连续消费 version 3；
- 最后一项 boxed 表达式闭合且非空；
- 固定 DAPO scorer 能解析；
- 离线重评分与运行记录一致。

答案可以错误，但不能截断、解析失败或使用错误版本。正确数单独报告，不用于能力提升结论。

本次提交因 patch 指向错误入口而取消；该契约本身未在 GPU 上执行。精确版本和 hash 见 [原始契约](PROBE_CONTRACT.md)。
