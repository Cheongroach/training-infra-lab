# Miles 四卡冷进程恢复与末轮发布验收 v2

**已完成，独立验收 passed=true。** 保留任务 [`172e5c51a339edef`](https://ml.bytedance.net/development/instance/jobs/172e5c51a339edef)，trial `438495582` / Pod `5390798511`，平台 done / job_success，Pod、入口和归档退出码均为 0。详细结果见 [按 SID 隔离的报告](../grpo-recovery-v2-172e5c51/REPORT.md)；本线程独立使用冻结语义验收器所得 [verified.json](verified.json) 与该报告的结果完全一致，归档 SHA256 也一致。

保持 Qwen3-4B、DAPO 64 题、2 learner TP2 + 2 rollout TP1、4 H20、32 CPU/128 GiB、cluster 22/group 2079、原队列和镜像 VID `dbdt1t71enl9fha0bn40`。入口 2700 秒及 110 秒终止宽限，转换最多 600 秒，两段训练与 checkpoint 清单共用 1800 秒上限，转换后要求本地剩余 96 GiB。原 v4 已通过的三轮更新证据保持不变。

完成 rollout 0/1 后保存 tracker=1，关闭旧进程，再由新 Ray/learner 从 rollout 2 继续。checkpoint 共 20 文件、56,331,517,391 bytes（52.4628 GiB）；两个 TP rank 的模型、Adam、scheduler、Python/NumPy/Torch CPU/CUDA/Megatron CUDA RNG tracker 指纹均精确恢复。补采样游标恢复 32 个已消费身份，完整历史及后续候选身份通过审计。指纹见 [restore-fingerprints](restore-fingerprints)、[文件清单](checkpoint-inventory.json)。RNG 验收范围为 learner；rollout 引擎采样随机流未纳入恢复验收。

两台新引擎各 290 个完整 raw tensor SHA256 与保存的训练版本 2 一致，并与 checkpoint 清单绑定。恢复发布使用新 UUID `1b29731cf6d840c68a5233e565ae3daa`；42 行完全一致的历史重放经去重后，新发布、版本、实例与时间顺序独立核验通过。

| rollout | 候选 / 拒绝批 | 非零 advantage token | 梯度范数 |
|---:|---:|---:|---:|
| 0 | 1 / 0 | 9,441 | 0.290185 |
| 1 | 1 / 0 | 6,202 | 0.256558 |
| 2（恢复后） | 2 / 1 | 15,033 | 0.431847 |

共核验 64 条候选，拒绝 16 条、录取 48 条；三轮两个 TP rank 的模型与 Adam moments 均改变。末轮实际发布版本 3，四条固定提示共 2,048 个响应 token 的版本覆盖均为 3。四条 probe 均在 512 token 上限截断、reward=-1，验收结论仅包含真实生成和版本覆盖，不证明答题成功、能力或吞吐收益。

[v1](../grpo-recovery-v1/REPORT.md) 在恢复训练版本 2 后被原始 HF 启动基线比较拦住。本轮 worker 补丁严格绑定 Miles commit 和原文件 SHA256：actor 在两个 TP rank 记录保存后/加载后指纹；train.py 仅对限定恢复入口使用 checkpoint 对应的完整 engine 张量 SHA256 比较，其他入口保留原检查。全部融合参数参与，未放宽数值容差。14 份 worker 源码和三处运行时补丁独立校验通过，见 [archive-audit.json](archive-audit.json)。

最新冻结 CPU 检查为 86 项通过，见 [cpu-checks.json](cpu-checks.json)、[audit-tests.log](audit-tests.log)。原 [audit-manifest.json](audit-manifest.json) 与 [verifier-manifest.json](verifier-manifest.json) 对应快照保留。平台使用 done 成功终态，观测器与 collector 补齐该枚举；collector 额外要求 job_success 和所有 Pod status=3/exit=0。操作适配的前后哈希及完整差异见 [collector-done-adapter.json](collector-done-adapter.json)，语义验收器未改动；本线程实际使用版本见 [collector-manifest.json](collector-manifest.json) / [collector-code](collector-code)。

共享记录出现入口、配置和资源完全相同的重复任务 `a0f976754ea54fc4`。本线程于 04:23:06 UTC 停止该重复任务，保留原任务并恢复单任务四卡范围，见 [清理记录](duplicate-cleanup.json)。两次提交复用 HDFS 输出路径；本线程验证归档内完整入口日志字节序列嵌入保留任务 stdout、所有环境 hostname 匹配，见 [run-provenance-fingerprint.json](run-provenance-fingerprint.json)。仅按相同目录名或源码哈希不能证明归属。

保留任务的提交记录见 [submission-audit.json](submission-audit.json)：预检 SID `3d163f32-25ac-4bd3-9b21-1bbaaaa1e9c8`，04:09:30 UTC 预检、04:09:32 UTC 提交。重复任务的 14 项预检均为 success，无 warn/hint；其预检 SID `b9036769-ef21-44d4-98e3-b55a6220bf49` 和提交身份继续保留在 [precheck-result.json](precheck-result.json) / [state.json](state.json)。门禁为 error 停止、warn/hint 展示后可继续；request_context.precheck_sid 保留式合并并在 30 分钟内提交。

平台终态 Pod 起止字段为 04:10:17–04:41:05 UTC，驻留 30 分 48 秒，保留任务共 2.053333 GPU·h，包含启动；重复任务额外驻留 0.422222 GPU·h，两任务合计 2.475556 GPU·h。该统计不是训练算子活跃时间。终态见 [terminal-evidence.json](terminal-evidence.json)、[execution-state.json](execution-state.json)；原始平台响应与产物保留在 `/tmp/miles-grpo-recovery-172e`，仓库仅保存白名单字段。

范围为同一 worker 本地文件系统上完整轮次边界的受控冷进程恢复。结果归档未包含 52.46 GiB 原始 checkpoint；未验证任意时刻 crash consistency、节点丢失、HDFS 原始 checkpoint 持久性、跨机恢复，或与不中断训练的逐位等价。
