# 恢复验收的状态边界与证据

本文描述已提交入口的验收设计，实际通过与否以 [REPORT.md](REPORT.md)、归档归属审计和独立语义验收为准。范围是同一 Pod、同一 worker 本地文件系统上，关闭全部 Ray/训练进程后重新启动。

rollout 1 更新完成时的提交状态为模型参数、Adam moments、scheduler、RNG、datasource cursor、全部候选 journal 与已发布 policy version 2。恢复后的下一轮为 rollout 2，采样行为版本为 2；更新完成后发布 version 3。

| 状态 | 检查方式 | 缺失时的后果 |
|---|---|---|
| 两个 TP rank 的模型和 optimizer | save 后与 load 刚返回时的参数、optimizer 指纹比较；Adam moments 单独核对 | 只恢复模型会重置优化轨迹；optimizer metadata 变化也不能代替 moment 更新 |
| scheduler | 完整 state_dict 的递归张量指纹 | 重复或跳过学习率进度 |
| RNG | Python、NumPy、Torch CPU、CUDA 和 Megatron CUDA RNG tracker | 训练随机状态不能在同一边界继续 |
| datasource cursor | epoch_id / sample_offset / sample_index / sample_group_index | 已消费请求可能重复，或跳过未提交请求 |
| qualification journal | 重算每个候选的完整组、概率、奖励、版本覆盖和录取/拒绝决策 | 拒绝批也消耗身份；只保存录取样本不足以恢复游标 |
| rollout engine 权重 | 两台 TP1 引擎各自全部 290 个 raw tensor SHA256，与提交前的 version 2 一致 | version 数字一致仍可能加载错误权重 |
| 新 publication | 新传输 UUID、时间顺序、两台引擎身份及 checksum 对应 | checkpoint 重放历史事件可能被误认作新同步 |
| 恢复后训练和末轮生成 | 第三轮非零 advantage/梯度、参数与 Adam moments 变化；四条生成全响应 token 均为 version 3 | load 成功不能证明可继续训练，也不能证明最后一次更新已可用于推理 |

## checkpoint 提交顺序

冻结的 [Miles driver](research/posttraining_20261022/upstream/miles/train.py) 先执行 rollout_executor.save，再保存 actor model。qualification journal 位于 rollout checkpoint 内；训练 tracker 是后续恢复查找的提交入口。测试先在 rollout 1 保存，再发布 version 2，并捕获真实 engine 权重。

[qualification_checkpoint.py](audit-code/qualification_checkpoint.py) 要求候选文件 SHA256、候选序号、每次决策及重算 assessment 全部相同；录取和拒绝批的所有身份恰好覆盖 range(sample_index)；sample_index 等于 4 倍 sample_group_index，且没有未处理的 datasource buffer；训练配置、输入文件和采样代码指纹与保存时一致。

恢复 journal 前先完整验证，在 staging 目录写入文件后通过 os.replace 切换 active 目录，并保留此前 journal。这里没有磁盘断电后的 fsync 持久性验收，也没有任意中间指令处的故障注入。受控的两轮完成后退出不能推广为任意时刻 crash consistency。

## 为什么恢复比较不能继续使用原始 HF 权重

原始 HF 对应 version 0，训练提交对应 version 2。恢复后将引擎与原 HF 比较，会把正确恢复的训练权重判成不一致。[worker 补丁](audit-code/patch_recovery_probe.py) 只在限定的恢复入口、rollout 起点 2 和冻结源码 SHA256 同时匹配时，切换为 checkpoint 提交比较；其他入口保持原检查。

新的比较不使用容差或跳过融合参数，而是要求每台引擎全部 290 个 raw tensor SHA256 与保存的提交一致。提交 engine 指纹还绑定 checkpoint 文件清单的 canonical SHA256。两个 learner rank 的 save/load 状态、真实引擎指纹、新传输事件三者共同约束恢复状态。

## 归档归属与统计边界

两个会话曾共用同一 HDFS 输出路径，因此先执行 [verify_archive_identity.py](verify_archive_identity.py)：事件必须位于本 Pod 起止区间，各 publication UUID、时间、candidate/published version 必须在本 Pod stdout 出现，worker、auditor 和运行时补丁指纹均须匹配。通过后再对 .pt 产物执行 [verify_recovery.py](audit-code/verify_recovery.py)。相同代码的另一任务不能替代本轮证据。

驻留成本按 4 × (Pod stop − Pod start) / 3600 统计，包含准备、初始化、状态哈希和归档。它不是训练算子 GPU 活跃时间，也不能直接换算为利用率或算法吞吐。

本验收仍不证明：原始 checkpoint 已持久化至 HDFS、节点丢失后可恢复、跨机恢复、与不中断训练逐位等价，或模型能力和吞吐收益。四条末轮原题用于发布与 token 版本覆盖核验，不是独立能力测量。
