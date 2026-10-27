# OPD：从方法到实测结果

## 一句话定义

On-Policy Distillation（OPD）让 student 自己生成轨迹，再让 teacher 在 student 实际访问的前缀上提供概率监督。

它与离线蒸馏的关键区别不是 teacher 更大，而是监督状态来自当前 student。teacher 可以与 student 同规模，只要它在目标领域更强。

## 什么时候值得用

OPD 适合以下问题：

- 已有强 teacher，希望把能力迁移到较小 student；
- 从共同 checkpoint 训练了多个领域专家，希望融合到一个模型；
- 顺序 RL 导致旧能力回退，希望用历史 checkpoint 恢复；
- 同时使用 teacher 的密集信号和环境 reward。

它不保证补齐模型缺失的领域知识。缺少知识时，继续预训练、SFT 或检索可能更合适。

## teacher 和 student 的资格检查

训练前至少确认：

1. teacher 在同题、同工具和同输出预算下确实优于 student；
2. teacher 在 student 的正确和错误前缀上仍能提供有用分布；
3. tokenizer、模板、response 边界和 EOS 语义一致；
4. 短 pilot 能改善独立 dev，而不是只降低 KL；
5. 非目标领域没有不可接受的回退。

“teacher 更大”不能替代这些检查。不同 tokenizer 下逐 token 相减 log-prob 不是合法的同事件空间 KL。

## 两类常见 loss

### sampled-token policy gradient

student 采样 token `a_t` 后，teacher 和 student 都对它评分：

```text
OPD advantage = log q_teacher(a_t | prefix) - log p_student(a_t | prefix)
loss = - stop_gradient(advantage) × log p_student(a_t | prefix)
```

它传输量小，容易接入 RL 框架，但有 action sampling 方差。

### top-k 或完整分布

top-k 提供更丰富的监督，但必须保留全词表归一化；只对 k 个 logits 重新 softmax 会改变目标。还要明确：

- k 个 token 来自 teacher 还是 student；
- 使用交集还是并集；
- tail probability 如何处理。

完整词表 loss 最精确，但 output head、显存和传输成本更高。

## 异步时要区分三种漂移

1. rollout 来自旧 student；
2. teacher checkpoint 或评分配置改变；
3. 相同权重下，推理 kernel、量化或采样配置造成概率差异。

刷新当前 student log-prob 不能把旧 prefix 变成真正 on-policy。异步方案只能是受控 off-policy 近似，必须记录：

- rollout policy version；
- teacher version；
- 行为 log-prob；
- token 和 action mask；
- tokenizer/template hash；
- 采样配置和结束原因；
- lag、ratio、丢弃原因和有效 token/GPUh。

## 本项目实际做了什么

项目使用同一 Qwen3.5-4B、同 tokenizer 的两个代码任务：

- bank：目标迁移域；
- MBPP：保持域。

student 从 MBPP SFT adapter 开始。bank teacher 来自独立选出的强 adapter。两者不是从共同多域 SFT `S0` 训练出的严格同源专家，因此结果应称为“同规模、同 tokenizer 的两任务后训练”，而不是完整 MOPD 路线。

训练实现：

- student 生成新鲜前缀；
- teacher top-64 修正 reverse divergence；
- 完整词表归一化；
- teacher 冻结；
- 精确 response-token 预算；
- 参数、Adam、scheduler、RNG 和 checkpoint 验收。

没有使用 Miles/verl 执行本轮训练，也没有实现异步 rollout 或环境奖励 RL。

## pilot 结果

pilot 使用 16,384 response token。OPD 三种子在 bank blind 上相对 warm student 平均 +9.27 个百分点，相对 replay +4.53 个百分点，但距 expert 仍有 3.93 个百分点。

MBPP 上，OPD 相对 warm 平均 -0.39 个百分点；置信区间不足以证明非劣。这个结果促使后续扩大预算和冻结保持性门槛。

## 补强实验结果

补强完成 36 组训练，并使用完整 dev 选择 524,288 response-token 配置。最终在新留出集评估三种子：

| 方法 | bank | MBPP 保持集 |
|---|---:|---:|
| warm student | 87.50% | 46.33% |
| bank expert | 98.58% | 未作为保持目标 |
| OPD | 98.44% ± 0.13 pp | 46.64% ± 0.35 pp |
| replay | 96.91% ± 0.24 pp | 45.18% ± 0.23 pp |

因此，所选 OPD：

- 相对 warm student 的 bank 均值为 +10.94 pp；
- 相对 replay 为 +1.53 pp；
- 距 bank expert 为 0.15 pp；
- MBPP 相对 warm 均值为 +0.31 pp。

预先冻结的保持门槛要求每个 seed 相对 warm 的配对 95% CI 下界不低于 -2 pp，三个 seed 均通过。

## 这些结果没有证明什么

- 没有验证异步 rollout；
- 没有验证异源 tokenizer；
- 没有完整词表 loss 消融；
- 没有环境 reward 或 RL + OPD；
- 没有线上 fallback；
- 两个代码任务不代表通用多域融合；
- MBPP 公开数据存在预训练污染边界。

OPD 的进程成本明显高于 replay：所选预算三种子合计 22.74 H20·h，replay 为 3.19 H20·h，约 7.13×。这个比值包含加载、rollout、teacher 评分、更新和保存，不能只归因于 teacher prefill。

## 面试时应能解释

- 为什么 teacher 更强仍可能不适合直接蒸馏；
- top-k 是否保留全词表归一化和 tail mass；
- 为什么刷新 log-prob 不能消除旧 prefix 的 off-policy；
- prompt 比例为什么不等于 token 或梯度比例；
- 单域迁移缺口和多域融合干扰如何区分；
- 为什么参数更新、dev 选择和 final blind 必须分开；
- 为什么 teacher 成本即使被时延隐藏，也仍计入 GPU-hour。

完整数学推导、公开方法比较、每个种子的区间和资源恢复记录见 [原始技术文档](OPD_DESIGN_AND_INTERVIEW.md)。
