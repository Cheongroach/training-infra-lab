# Miles 四卡冷进程恢复验收：172e5c51

**本轮终态验收通过，passed=true。** 任务 [`172e5c51a339edef`](https://ml.bytedance.net/development/instance/jobs/172e5c51a339edef)，trial `438495582` / Pod `5390798511`，平台 done / job_success，Pod、入口与归档复制均退出 0。[归档归属审计](archive-audit.json) 和 [独立语义验收](verified.json) 均通过。这证明同一 worker 本地 checkpoint 的冷进程恢复及恢复后训练、发布可用；不构成能力或吞吐收益。

配置为 Qwen3-4B、DAPO 前 64 题、2 learner TP2 + 2 rollout TP1、4 H20、32 CPU / 128 GiB、LF cluster 22 / group 2079、镜像 `dbdt1t71enl9fha0bn40`。先完成 rollout 0/1，保存 tracker=1 和完整补采样日志，再关闭 Ray/训练进程；重新加载模型、Adam、scheduler、RNG，从 rollout 2 继续；末轮发布 version 3 并执行四条固定原题生成。

## 恢复状态逐项结果

| 检查 | 本轮证据与结果 |
|---|---|
| learner 模型和 Adam | rank 0/1 的 save 后与 load 刚返回时指纹一致；模型与 optimizer 状态均恢复 |
| scheduler 与 RNG | 完整 scheduler 状态及 Python、NumPy、Torch CPU/CUDA、Megatron CUDA RNG tracker 指纹一致 |
| load 选项 | finetune、no_load_optim、no_load_rng 均关闭，use_checkpoint_opt_param_scheduler 开启 |
| 补采样状态 | 恢复 32 个已消费 sample 身份，completed_rollout_id=1，next_rollout_id=2；cursor 与完整候选 journal 一致 |
| checkpoint | worker 核验 20 文件、56,331,517,391 bytes（52.4628 GiB），包含 iter_0000001 和 rollout/1；恢复前重新核验 |
| 恢复发布 | 新传输 UUID `1b29731cf6d840c68a5233e565ae3daa`，published_version=2；两个新引擎实例收到权重 |
| 引擎完整权重 | 每台引擎全部 290 个 raw tensor SHA256 与保存的 version 2 完全一致，包含融合参数；与 checkpoint 清单绑定 |
| 历史事件 | 119 个归档事件行中 42 行为完全一致的历史重放；去重后 77 行，新恢复发布及 checksum 单独核验 |
| 恢复后继续训练 | 第三轮非零 advantage、正梯度、两个 rank 的模型与 Adam moments 改变 |
| 末轮发布与生成 | version 3 实际发布；四条生成共 2,048 个响应 token，逐 token 版本覆盖均为 3 |

详细提交边界、插件入口和各项检查的原因见 [RECOVERY_CONTRACT.md](RECOVERY_CONTRACT.md)。原始 checkpoint 未包含在 6.35 MiB 结果归档中；离线审计能检查 worker 保存的清单和状态证据，不能重新读取已经结束的 worker 本地 checkpoint。

## 三轮训练与实际权重变化

每批四个 prompt、每个 prompt 四个回答。四个候选批共 64 条样本，拒绝 16 条、录取 48 条；全部候选身份、完整组、有限行为概率、learner 输入映射及 token 版本覆盖通过独立核验。恢复后第一批因没有组内 reward 方差被拒绝，下一批含两个混合奖励组并通过门禁。

| rollout | 候选 / 拒绝批 | 非零 advantage token | 梯度范数 | rank 0/1 各自改变的模型 tensor | 两台引擎各自改变的 tensor |
|---:|---:|---:|---:|---:|---:|
| 0 | 1 / 0 | 9,441 | 0.290185 | 207 / 207 | 207 |
| 1 | 1 / 0 | 6,202 | 0.256558 | 200 / 200 | 200 |
| 2（恢复后） | 2 / 1 | 15,033 | 0.431847 | 202 / 202 | 202 |

模型变化将每轮 pre-update hook 与 post-update local checksum 逐项比较，排除 companion/witness；每个 rank 覆盖 290 个模型 tensor，三轮 Adam moments 均改变。录取批的截断数为 2 / 0 / 1，不影响已验证的组内奖励方差与非零训练信号。见 [diagnostics.json](diagnostics.json)。

保存前 version 2 与冷启动重新发布的 version 2，全部引擎 tensor 的变化数为 **0**；随后 version 3 改变 **202** 个 tensor，两个副本一致。发布日志中的聚合 workers_hash 是 pseudo-hash-1，因此权重一致性结论依赖完整 raw tensor SHA256，不能从那个聚合字符串推出。

## 修复解决了什么

[v1](../grpo-recovery-v1/REPORT.md) 已恢复并发布训练 version 2，但 Miles 的初始化检查仍与原始 HF version 0 比较而失败。本轮一次性 worker 补丁严格绑定 Miles commit `2233d795dcf167be489d1859971c9233393f04c3` 及原文件 SHA256，限定恢复入口和 start_rollout_id=2，改为与最后提交的训练权重比较；其他入口保留原检查。数值容差没有放宽，没有跳过融合参数。

两个 TP rank 在保存后和 load 刚返回时记录状态；新 publication 与 checksum 共享新的 debug_weight_update_id，并核对实例、版本和时间顺序。历史重放只允许完全一致的去重，不能证明新的权重同步。本轮在原 v4 验收基础上补足完整恢复边界和末轮真实发布；原 v4 的既有结论仍按它自己的报告保留。

## 归档归属、源码与预检

另一会话提交了相同入口，复用同名临时目录和 HDFS 输出目录。本报告按 SID 隔离保存，排除被覆盖的另一次提交返回值。归属审计确认全部事件位于本 Pod 起止区间，五次 publication 的 UUID、时间和 candidate/published version 均在本 Pod stdout 匹配；14 份 worker 源码、22 份 auditor 文件、7 项运行时补丁指纹全部一致。

- [源码清单](source-manifest.json)、[auditor 冻结清单](audit-manifest.json)、[归属工具哈希](archive-identity-manifest.json)；冻结前 86 项 CPU 检查通过，另有 16 项归属匹配正反例检查，见 [audit-tests.log](audit-tests.log) / [archive-identity-cpu.json](archive-identity-cpu.json)。
- [提交审计](submission-audit.json)：预检 SID `3d163f32-25ac-4bd3-9b21-1bbaaaa1e9c8`，14 项 success、无 warn/hint；保留式合并到 request_context.precheck_sid 后在 30 分钟有效期内提交。门禁仍为 error 停止、warn/hint 展示后可继续。
- 入口 SHA256 `dff3e3677ecd58246dc49dd9980f1699e5d4cfef7c3fcb83d33f2a7a001e6451`；归档 [artifacts.tar.gz](artifacts.tar.gz) SHA256 `5dea3ac44f46c60d9310c0a2191d2ba5623eeb637d021f1d559f79245cefd2f1`，6,662,318 bytes。
- 使用 cn 控制面和本 Pod 的 log-proxy stdout，关键真实日志保存在 [terminal-log-evidence.log](terminal-log-evidence.log)；完整临时日志路径 /tmp/miles-recovery-followup/v2-stdout.log，SHA256 见 archive-audit.json。终态记录见 [current.json](current.json)、[exit-info.json](exit-info.json)、[timeline.json](timeline.json)。

## 时间与成本

UTC timeline：job_created 04:09:32.445，start_run 04:14:34.464，job_success 04:41:04.768。创建到平台开始运行约 302.019 秒；平台运行约 26 分 30.304 秒。终态 Pod 字段记录 04:10:17–04:41:05，驻留 30 分 48 秒，四卡共 **2.053333 GPU 小时**，包含准备、两次初始化、checkpoint 保存/哈希、训练、生成与归档。

运行态曾观察到 Pod start_at=04:14:32，终态接口修正为 04:10:17；成本使用保存的终态字段，不将两种区间混用。该成本不是训练算子活跃时间，不证明利用率、恢复加速或算法收益。入口 2700 秒加 110 秒终止/归档宽限，转换最多 600 秒，两段训练和 checkpoint 哈希共用 1800 秒上限；任务正常结束，没有追加训练重跑。

## 仍未证明的部分

四条末轮 probe 均在 512 token 截断，reward 全为 -1，答案不可解析。它们证明真实生成使用了 version 3 和完整响应 token 版本覆盖，**没有证明答题成功或模型能力提升**。前三轮有效学习信号也不等于泛化收益。

本轮是完整轮次边界上的受控退出和同一 worker 冷进程恢复，不是任意时刻 crash consistency；没有断电/fsync 持久性、节点丢失、HDFS 原始 checkpoint、跨机恢复，也没有与不中断训练逐位等价的对照。后续若扩展恢复范围，需要按具体故障边界验证这些独立目标；若扩展能力结论，需要冻结独立题集、生成预算和配对评测。
