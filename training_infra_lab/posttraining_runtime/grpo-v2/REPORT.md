# Miles 四卡 GRPO v2：执行成功，学习验收未通过

2026-10-24 验收。任务 [2c78b85806cd2090](https://ml.bytedance.net/development/instance/jobs/2c78b85806cd2090)，control-plane=`cn`，trial=`438336563`，pod=`5390177759`。LF cluster 22 / group 2079，4 H20、32 CPU、128 GiB；2 learner TP2 + 2 rollout TP1。

**三轮 rollout → learner → 权重发布流程已执行，IPv6 初始 broadcast 修复已在 GPU 链路验证；未证明有效 GRPO 学习。** 独立验收 [verified.json](verified.json) 为 `passed=false`，不作吞吐、收敛或规模化结论。

## 终态与证据完整性

- 平台 `done`，timeline=`job_success`，Pod 退出 0，`GRPO_DRIVER_COMPLETE`、`ORIGINAL_EXIT_CODE=0 ARTIFACT_COPY_RC=0`。原始 [run](observations/terminal-run.json)、[trial](observations/terminal-trial.json)、[pod](observations/terminal-pods.json)、[timeline](observations/timeline.json)、[log-proxy stdout](observations/5390177759.stdout.log) 已收集。
- UTC：created 22:25:59.308 → start_run 22:30:40.643 → job_success 22:43:09.361；启动等待 281.335 s，平台运行 748.718 s。Pod start/stop 为 22:26:28–22:43:00，按四卡估算驻留约 1.102 GPU·h；该值包含启动，不是有效计算卡时或计费账单。
- HDFS `training_infra_lab/miles-grpo/grpo-v2/artifacts.tar.gz` 已下载并安全解包：2,220,101 bytes，SHA256 `df4863cf5b9fe0691910a01881d0c85944ab8c53c091b4e7d73bb40648c4846d`。5 份归档源码均与提交 manifest 一致；[archive-audit.json](archive-audit.json) 核验运行时补丁前后哈希。归档中 `runtime-patch.json.gpu_verified=false` 是训练前记录，保持原样；后续 GPU 结论来自终态事件和校验和。
- v2 与 v1 使用同一 driver、镜像、模型、数据和时限，仅增加 IPv6 补丁。14 项提交预检 success，无 warn/hint；保留合并 `request_context.precheck_sid=82bbc976-4196-4f9a-8e03-b84c67fe0aa4`，30 分钟内提交。门禁为 error 停止、warn/hint 展示后可继续。

## 三轮独立核验

| rollout | 完整组 / 样本 | 截断 | reward 均值 | 混合 reward 组 | 非零 advantage token | grad_norm |
|---|---:|---:|---:|---:|---:|---:|
| 0 | 4 / 16 | 16 | 0 | 0 | 0 | 0 |
| 1 | 4 / 16 | 16 | 0 | 0 | 0 | 0 |
| 2 | 4 / 16 | 16 | 0 | 0 | 0 | 0 |

“完整组”指每题四条样本齐全，回答本身均被截断。48 条样本各有 2048 个 response token；行为 log-prob、learner log-prob 有限。sample index、tokens、response length、行为概率逐条对应，权重版本 spans 完整覆盖生成区间。三轮 learner 各有一个 TP2 cell，两个 rank 都报告 `normal`。

初始同步及前两轮训练后的发布版本为 0、1、2；每次两台引擎各 290 个 tensor 校验和完全一致。但版本间这些 tensor 的校验和全部不变。learner 每 rank 292 个参数记录中，轮间只改变 `model_companion.weight_version`；实际模型参数不变。optimizer 仅两个 param group 的 `step` 从 1→2→3，其他记录字段不变。详见 [weight-diagnosis.json](weight-diagnosis.json)。因此版本与 optimizer 计数递增不能作为学习成功证据。

上游无末轮评测时跳过最后一次 engine 发布，故最后一轮只有 learner step；不将不存在的第 3 版引擎发布当成故障。当前尚未验证“真实改变后的模型权重”回传到 rollout 引擎。

## 已确认的问题

1. **生成长度不足以完成本批回答。** 48/48 响应在 2048 token 截断，全部没有 `</think>` 或 `\boxed`，抽查尾部仍为推导过程。[sample-diagnosis.json](sample-diagnosis.json) 保存计数与短尾部。
2. **评分器要求思考结束标记，纯 `Answer:` 答案也不被接受。** 固定 64 题同时包含 `Answer: \boxed{...}` 和 `Answer:` 指令，标签均为整数。因此不能说提示完全没有 boxed 要求。当前 `--rm-type deepscaler` 对没有思考结束标记的 response 直接给 0，再通过 boxed 提取答案。CPU 调用固定源码重现了全部 48 个零奖励；纯 `Answer: <label>` fixture（含或不含 `</think>`）不被接受，但这不证明每个遵循完整原始提示的答案都不被接受。
3. **零奖励导致无组内学习信号。** 每组 reward 相同，GRPO advantage 全零；KL/entropy 系数均为 0，实际 policy loss 与梯度也全零。模型哈希结果与此一致。没有 crash、OOM 或 NCCL 失败证据需要解释这次终态。

[reward-contract.json](reward-contract.json) 记录了候选 `--rm-type dapo --reward-key score` 的 CPU 验证：64 条正确答案 fixture 被接受，64 条错误答案 fixture 被拒绝；这只是评分协议测试。重评本次真实 48 条 response 仍全部 -1，因此只改评分器也不能使这次数据产生有效 GRPO 梯度。

## 后续实施边界

下一次先对齐 DAPO scorer，再做少量 rollout 资格检查，确认有完整答案、正确的答案提取、组内 reward 差异后才进入三轮训练。短 smoke 可评估 `--apply-chat-template-kwargs '{"enable_thinking":false}'`；若保留 thinking，则应重新核算 response 上限、总 token 数与执行时限。两者都未做 GPU 验证，不能预先声称能解决问题。验收仍要求非零 advantage/梯度、实际参数和 optimizer 更新以及引擎收到改变后的权重。

本轮没有继续提交 GPU 任务或扩至 32 卡。`verify_smoke.py` 已按实测 TP2 事件格式修正，并补上 source 不匹配时禁止反序列化、样本内容与行为概率映射、列长度检查；14 项 CPU fixture 测试通过。本次真实数据再次被拒绝，拒绝原因保留在 `verified.json`。
