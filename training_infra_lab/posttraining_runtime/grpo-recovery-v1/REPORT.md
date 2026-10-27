# Miles 四卡 checkpoint 恢复与末轮发布验收

当前状态：四卡任务 [`115988fdc4b06c6c`](https://ml.bytedance.net/development/instance/jobs/115988fdc4b06c6c) 终态 failed、Pod 退出码 1，独立验收 `passed=false`。两轮更新、checkpoint 保存、冷重启和采样日志恢复已执行；初始化的权重检查阻断了末轮训练，完整恢复与 version 3 发布尚未通过。已完成的三轮有效学习结果仍见 [v4 报告](../grpo-v4/REPORT.md)，其记录保持不变。

本次保持 Qwen3-4B、DAPO 64 题、seed、2 learner TP2 + 2 rollout TP1、LF cluster 22/group 2079 与镜像 VID `dbdt1t71enl9fha0bn40`。单个任务仅使用四张 H20，入口 2700 秒加 110 秒终止/归档宽限，转换最多 600 秒；两段训练与 checkpoint 哈希共用 1800 秒上限。

先执行 rollout 0/1 并保存 checkpoint，再退出、重启 Ray 与训练进程，加载相同 checkpoint 执行 rollout 2。Miles 的 checkpoint iteration 是已完成 rollout ID：两步保存为 `1`，恢复应从 `2` 开始。恢复移除 `finetune/no-load-optim/no-load-rng`，显式选择 step 1 并启用 `use-checkpoint-opt-param-scheduler`，防止学习率进度被额外累加。

补采样插件新增 save/load，保存完整候选批、拒绝理由、已处理身份、游标和源码/训练参数契约。恢复在修改当前记录前验证全部哈希、行为版本及样本身份，并与已恢复 datasource 游标对齐。旧的未提交候选记录移入独立历史目录，保留原字节；恢复安装失败则回滚当前目录。不能把恢复后重新生成的候选当作先前 optimizer 已提交的更新。

两个 TP rank 记录保存后及恢复更新前的实际参数、Adam state、scheduler 和 Python/NumPy/Torch/CUDA tracker RNG 指纹。复用上游 post-save hook，在一次性 worker 中以 commit 和原文件 SHA256 严格限制地开放到两个 rank；只用于本任务的 rank-safe 探针。末轮 eval 使用四条固定原题和 512 token 上限，目的是验证 version 3 的权重发布与真实生成；它不是独立能力评测，不使用分数宣称能力提升。

CPU 检查覆盖实际 v4 的 80 条候选记录保存/恢复、旧身份拒绝、游标错位、缺失候选、契约和版本篡改、未提交记录保留、恢复安装失败回滚，以及参数/Adam/scheduler/RNG 比较和末轮发布要求。54 个独立测试通过，见 [cpu-checks.json](cpu-checks.json)。13 份 worker 源码冻结，见 [source-manifest.json](source-manifest.json)；外层入口、内层 shell 与双阶段训练 shell 均通过语法检查。CPU 结果不等于 Miles/GPU 恢复通过。

本次 checkpoint 仅位于该 worker 的本地文件系统，用于同一任务中的进程重启；不会把数十 GiB 的原始模型/optimizer 分片塞进结果归档。转换后须实测至少 96 GiB 可用本地空间，完整 checkpoint 哈希与恢复前后实际状态纳入证据。任务终止后仅保留小型审计归档，因此不能宣称已验证 HDFS 持久 checkpoint、节点丢失或跨机恢复。

本轮 14 项平台预检全部 success，无 warn/hint。预检 SID `6fdd3c93-7043-4fee-880d-5d360bd1ffb2` 保留式合并到 request_context.precheck_sid，并在有效期内提交。提交门禁仍为 error 停止、warn/hint 展示后可继续。

实测前两轮梯度为 0.36004484 / 0.21188946，五批共 80 条候选中拒绝三批、32 条入训。Checkpoint 有 23 个文件、56,334,419,596 bytes，tracker 为 1；28 个 Ray 进程停止后启动独立恢复进程，采样游标恢复为 sample_index=80 / sample_group_index=20 / sample_offset=20，next rollout=2。恢复后两个 engine 的 version 2 张量 checksum 与第一阶段已发布的 version 2 完全相同。证据见 [diagnosis.json](diagnosis.json)、[archive-audit.json](archive-audit.json)、[verified.json](verified.json)。尚无恢复更新前的两个 rank 指纹，所以不能据此宣称 Adam、scheduler、RNG 已完成独立验收。

根因是 Miles train.py:53 在首次 update_weights 后无条件执行 action=compare，对比 engine 启动时的原始 HF snapshot。恢复的 version 2 已经包含训练更新，与原始 HF 张量相差约 3.81e-6；该检查在 rollout 2 开始前失败。原始归档路径 `/tmp/miles-grpo-recovery-v1/artifacts`，13 份源码与两处运行时补丁哈希均通过核验。Timeline 为系统 actor Merlin 写入 job_failed，没有人工 stop 事件。

另发现审计器需要适配上游事件恢复：resume events 会重放 checkpoint 中的旧事件，恢复后的首次 checksum 继承 rollout_id=1，首次 publication 则为 rollout_id=null。修复应以 debug_weight_update_id 关联新 publication/checksum、严格去重相同历史事件；不能以重复旧事件证明新同步。后续保持同四卡和时间上限，修正恢复时的 HF 初始化比较后做一次验证；保留所有实际状态与末轮发布门禁。
