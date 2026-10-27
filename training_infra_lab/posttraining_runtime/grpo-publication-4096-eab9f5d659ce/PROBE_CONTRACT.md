# 最终版本完整回答验收

本次仅补齐四条最终版本探针的自然结束与答案解析；此前冷进程恢复的通过结论保留在 `grpo-recovery-v2-172e5c51/`。这次不重复约 52.46 GiB 的 checkpoint 保存/恢复，也不宣称能力或吞吐收益。

固定 Qwen3-4B revision `1cfa9a7208912126459214e8b04321603b3df60c`、64 条 DAPO smoke 数据 SHA256 `67173a6074167cb280f1cf70542ccd1c1f91583defa8698f2428d268858e4fff`、Miles commit `2233d795dcf167be489d1859971c9233393f04c3`、镜像 VID `dbdt1t71enl9fha0bn40`。同一 LF 集群 22、资源组 2079、四张 H20、32 CPU / 128 GiB、2 learner TP2 + 2 rollout TP1。

训练仍要求三次真实更新：有界候选资格检查、完整 4×4 组、样本身份与 learner 映射、有限行为概率、非零 advantage/梯度、TP2 参数与 Adam moments 变化、两台引擎一致。最终 eval 使上游执行第三轮权重发布，要求 version 3。

探针使用冻结数据的前四条提示、每题一个回答，`enable_thinking=false`、temperature 0、响应预算 4096、第三轮后 eval。原有 full/no-eval 与 recovery/resume/512 默认不变。新增 worker patch 只保留原始 eval finish reason，不改变生成、停止或评分行为；patch 以原文件和 commit 哈希拒绝错版本或重复应用。

独立验收要求四个唯一 index 对应原始提示及 prompt token，全部 `completed`，原始停止原因为 `stop` 且 matched token 属于 `[151645, 151643]`，有限行为概率覆盖所有响应 token，version 3 spans 连续无缺口。每条必须含闭合、非空的最终 `\boxed{...}`，冻结 DAPO scorer 的 pred 可解析，重评分与记录一致。错答案可以通过完成/版本验收，正确数单独报告；截断、INVALID、错版本或缺失概率均拒绝。

执行上限 2700 秒，转换 600 秒、训练 1800 秒，另有最多 110 秒终止/归档宽限。仅提交一次，失败后保存证据，不自动重试。平台 `done` 需要 Pod exit 0 和 `job_success` timeline；安全解包、9 份内嵌源码与 runtime patch 哈希先于 PT 反序列化。108 项 CPU tests、64 条真实 tokenizer 渲染、内外层 shell 与内嵌 Python 语法均通过。

任务 `53bfe898ac2210b3` 使用唯一 HDFS 输出 `grpo-publication-4096-eab9f5d659ce`，上传和读回标记一致。14 项平台预检 success，无 warn/hint，保留 `request_context.precheck_sid=c8771796-ca73-43eb-a21a-4b3eabfa52a8` 并于 30 分钟有效期内提交；若预检 error 则停止，warn/hint 展示后可继续。
