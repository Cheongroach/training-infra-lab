# 最终版本完整回答验收

任务 [585440c23b867d70](https://ml.bytedance.net/development/instance/jobs/585440c23b867d70) 补齐四条最终版本探针的自然结束与答案解析。此前冷进程恢复验收保留在 `../grpo-recovery-v2-172e5c51/`，这次不重复约 52.46 GiB checkpoint 保存/恢复。

固定 Qwen3-4B revision `1cfa9a7208912126459214e8b04321603b3df60c`、64 条 DAPO 数据 SHA256 `67173a6074167cb280f1cf70542ccd1c1f91583defa8698f2428d268858e4fff`、Miles `2233d795dcf167be489d1859971c9233393f04c3`、镜像 VID `dbdt1t71enl9fha0bn40`。LF 集群 22、资源组 2079、四张 H20、32 CPU / 128 GiB，2 learner TP2 + 2 rollout TP1。

训练要求三次真实更新：有界候选资格检查、完整 4×4 组、身份与 learner 映射、有限行为概率、非零 advantage/梯度、TP2 参数与 Adam moments 变化、两台引擎一致。第三轮后 eval 触发上游最终权重发布，要求 version 3。

四条探针使用冻结数据的前四条提示，每题一个回答，`enable_thinking=false`、temperature 0、响应预算 4096。原有 full/no-eval 和 recovery/resume/512 默认不变。worker patch 只在实际 `generate_hub/single_turn.py` 的 eval 调用保留原始 finish reason，不改变生成、停止或评分；源码和 commit 哈希不匹配或重复应用时拒绝。

严格验收四个唯一 index 对应原始提示及 prompt token；全部 `completed`；原始停止原因 `stop` 且 matched token 属于 `[151645, 151643]`；行为概率有限且长度等于响应长度；version 3 spans 连续覆盖全部响应 token。每条必须含闭合非空的最终 `\boxed{...}`，DAPO pred 非 INVALID，冻结 scorer 重评分与记录一致。错答案可以通过完成/版本验收，正确数单独报告；截断、解析失败或错版本均拒绝。不宣称四条训练内样本的能力或吞吐收益。

入口上限 2700 秒，转换 600 秒、训练 1800 秒，最多 110 秒终止/归档宽限。平台 done 要核对 Pod exit 0 与 job_success timeline；安全解包、9 份内嵌源码和两份 runtime patch 哈希先于 PT 读取。109 项 CPU tests、64 条真实 tokenizer 渲染、内外层 shell / 内嵌 Python 语法通过，新增调用测试确认实际 single-turn eval 采集 EOS、train 不变。

唯一 HDFS namespace 为 `grpo-publication-4096-4fdcc7793e37`，标记已写入和读回核对。14 项平台预检 success，无 warn/hint，保留 `request_context.precheck_sid=67f80f55-3490-41a2-a5a4-9bf4a7e4de05` 并于 30 分钟内提交。预检 error 停止，warn/hint 展示后可继续。

首次提交 `53bfe898ac2210b3` 在 staging 时因 patch 指向旧入口而主动取消，Pod `start_at=null`，没有训练结论；完整取消记录见 `../grpo-publication-4096-eab9f5d659ce/REPORT.md`。当前修正提交不自动重试，也不根据结果放宽验收。
