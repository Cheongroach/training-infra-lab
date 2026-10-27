# 启动前取消记录

任务 `53bfe898ac2210b3` / trial `438542697` / Pod `5390903111` 已停止，未产生训练或探针结论。原始配置与 14 项成功预检保留，不将其计作完成的 GPU 验证。

提交后、用户入口执行前的静态复查发现：`bounded_rollout.BoundedRolloutFn` 继承的 `InferenceRolloutFn` 默认通过 `generate_hub.single_turn.generate` 生成，而首次停止原因采集 patch 指向旧 `sglang_rollout.py`。它无法提供本次严格验收需要的 raw finish reason。因此在 staging 时主动取消，未等训练结束后放宽检查。

取消 API 成功。终态 `stopped`，Pod status 4 / exit -70000，`start_at=null`、`stop_at=2026-10-24T06:22:25Z`，实时 stdout/stderr 均无 URL。证据见 [终态](cancelled-terminal-run.json)、[退出详情](cancelled-exit-info.json)、[时间线](cancelled-timeline.json) 和 [取消响应](cancel-before-entry.json)。平台没有可用于计算 Pod 驻留 GPU·h 的开始时间，不捏造驻留费用或数值上界。

修正将 patch 放到实际 single-turn 入口，增加直接调用该函数的 eval/train 测试；eval 原始 EOS 记录保留、train 行为不变。修正后的来源、payload、HDFS namespace 和预检重新冻结在 `../grpo-publication-4096-4fdcc7793e37/`。本目录仅保留取消记录。
