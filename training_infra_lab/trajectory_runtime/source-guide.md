# 从训练目标到分布式执行

本项目的系统边界是：一次全局更新的轨迹集合和目标函数固定，改变执行计划，并用数值对照证明参数更新没有被悄悄改掉。当前 9B 后端探针通过；真实 GRPO、设备一致性和增量性能的状态以 README 与结果目录为准。

## 1. 先读输入契约

`contracts.py` 中的 `Trajectory` 把 mask 和 log-prob 都定义在 `input_ids[1:]` 上。prompt 最后一个 token 的 hidden state 负责预测第一个 response token，不能把监督位置整体错开一位。

`UpdateBatch` 拒绝旧 policy version、重复轨迹和不完整 prompt group。GRPO advantage 在整个 group 上使用 sample standard deviation 计算，然后才分给各 rank。先分片再在局部计算 advantage，会改变算法。

`rollout.py` 显式请求聊天模板返回 token ID 列表（`return_dict=False`），由真实 fast tokenizer 回归覆盖 Transformers 5.2 默认返回值变化。它保存当前模型实际采样的行为概率，固定 temperature=1、无 top-k/top-p。`grpo.py` 在更新前重新计算响应位置的概率，检查缓存生成和训练重放是否存在过大偏差。规则奖励由 `rewards.py` 显式版本化。v8 的最后数字规则会把截断的中间算式误判为正确；新默认规则要求正常 EOS 和明确的最终答案行，分别记录正确、明确错答、格式失败与截断。只支持整数/十进制答案子集，不提供通用数学等价判断。旧规则仅通过显式配置复现，不能事后重打分后冒充修复规则下的 on-policy 训练。零方差组不补造 reward。

## 2. 看模型和 loss 的内存契约

`model.py` 加载 Qwen3.5 文本主干，实际训练参数数目与整个视觉语言模型不同。`Actor` 同时拥有 embedding、主干和输出头；缓存生成也必须经过 FSDP root，避免绕开分片参数的生命周期。

输出层只处理 response 位置，仍计算完整词表归一化。分块 checkpoint 降低完整 logits 的峰值内存，但它不等价于限制候选词表。CPU 测试直接对照 Hugging Face 的完整 CE 和梯度。

GDN kernel 选择与 gated norm/卷积实现分别记录。看到库已安装、没有报错，不能证明预期快速路径被执行。

## 3. 推导全局归一化

`rl.py` 返回局部分子的和；`executor.py` 将它乘以 `world_size / global_denominator`。FSDP 的梯度平均与这个 world_size 抵消后，才得到真正的全局平均。

对于 sequence mean，denominator 是轨迹数；对于 token mean，它是响应 token 数。每个 microbatch 各自取平均再累积，在不等长和不等样本数时会改变权重。梯度裁剪、Adam step 也必须只在全局更新末尾发生一次。

## 4. 为什么计划必须全局一致

`planner.py` 的 `slots[slot][rank]` 表示每个同步阶段、每个 rank 的轨迹索引。它验证覆盖恰好一次、batch SHA、rank 数和 padding 预算。

`executor.py` 先让所有 rank 汇报验证结果及计划摘要，再进入 FSDP forward。空槽位执行有效的两 token 样本并把 loss 置零，以保持模块遍历和 collective 顺序。空槽位仍消耗资源，必须计入时间。

延迟梯度同步不消除参数 all-gather；它也会增加未分片梯度的驻留量。不能只统计 reduce-scatter 次数便宣称整体更快。

## 5. 基线和设备验收

`baselines.py` 移植固定版本 verl 的 KK 分区；`planner.py` 给基线加了局部长度降序，避免拿未合理对齐的基线比较。当前仍没有 packing，不能将结果称为超越完整 verl 框架。

`device_gates.py` 的独立 reference 直接执行完整全局 batch，避免只让两个使用同一 executor 的方案互相验证。它比较第二次更新时的参数、梯度、一阶和二阶矩，并包含空槽位与 deferred sync。

普通 autocast 不等于 FSDP 的 BF16 `param_dtype`：embedding、残差及非矩阵乘参数的精度也要匹配。独立 reference 使用可微的 BF16 参数视图，保留 FP32 master 和完整全局 batch；不能为了对上结果而复用 executor 的分批逻辑。

精度对齐后也可能出现小梯度抵消、符号翻转和 Adam 更新放大。`precision_diagnostic.py` 同时记录最差参数元素的初值、梯度、更新后值及两阶矩，并显式比较 epsilon。v2 中 1e-5 通过单项门槛，1e-8 和 1e-6 未通过；这是候选优化器配置，既不是浮点下溢的证据，也不是收敛保证。正式验收必须让参考与实际训练使用同一 epsilon，保持原误差门槛，再覆盖全部执行计划。 v5 的完整验收证明单项通过不足：同一 epsilon=1e-5 在 verl 分批下仍失败。现在每个 rank 逐项保存整个矩阵，任一失败都会阻止后续 workload；v6 的 epsilon=1e-4 已通过完整设备门禁，参数最大误差 8.00192e-6；仍不代表训练收敛。

参考模型只在 rank 0 热身和更新，再广播完整参数、Adam 及预期结果。每个 rank 分别热身并不能保证相同的 BF16 初态。初始化检查完成全部张量 collective 后汇总错误，避免一个 rank 提前抛异常、其他 rank 继续进入下一次 collective。

`grpo.py` 的成对回放从同一份参数和非空 Adam 状态开始；全部候选先热身，测量顺序使用固定随机种子。逐个读取所有张量分片比较，内存中只保留单个张量或块。暂存与数值检查时间不属于 actor 更新延迟；规划、collation 和训练步计入 `total_seconds`。数据生成、安装、初始化与回放检查必须在更高层端到端成本中另计。

9B 集成门禁不能由 tiny 测试代替：v7 的 tiny 检查全部通过，真实批次改变 microbatch 后梯度差异仍达 18.40%。仅看参数最大误差会漏掉这类问题，因为 LR、epsilon 和梯度裁剪会缩小单步参数变化。`replay_diagnostic.py` 固定原始轨迹与 warm step，在原 BF16 头和 FP32 头内部分别比较同计划重复、重排、批形状；逐层误差便于区分状态恢复与计算精度问题。FP32 头同时覆盖 generation 和 training，必须重新采样验收后才能用于 on-policy 结论。

零梯度 update 不等于有学习信号。v7 第一批全对，第二批仅长度截断产生零奖励；非空 Adam 状态在这里包含零一阶/二阶矩，不能称为非零动量验证。v9 严格奖励下前三步也全对，直接测得梯度和两阶矩为零；第四步出现正常结束的明确错答，梯度范数 0.915452、两阶矩各有 7,935,924,616 个非零元素。这证明真实错答进入了一次非零更新，未验证下一步使用这些动量或能力提升。

可选 `--profile` 在正式重复结束后恢复同一初态，单独采集两种策略的 CUDA/CPU trace 和算子表。Profiler 内的 update 不算吞吐样本；算子 device time 存在重叠，不能直接相加解释墙钟。

`analyze_replay.py` 从归档批次重算 SHA、token 数与有效梯度组，校验计划内容、完整重复矩阵及数值指标；仅有“accepted=true”不够。 `audit_training.py` 进一步按 ID 对齐各轮 rollout 与 update；严格规则从原始文本和固定答案重新打分，避免把格式失败或截断当作明确数学错答。`optimizer_nonzero_counts` 在 update 正式计时外扫描真实 Adam 分片、全组归约非零元素数，补充只有梯度范数时的状态证据。`collect_evidence.py` 把 worker 源文件逐一与提交时的 SHA 核对，归档同时绑定任务和退出码。

`residency.py` 使用 PyTorch 2.9 的 `set_reshard_after_forward(False)` 在生成期间保留聚合权重。恢复时必须遍历所有 FSDP module，因为 `reshard()` 不递归；之后再设回训练策略。显存准入由全组投票，任何 rank 不足都保持普通模式。这个实现仅适配当前 BF16 参数策略和正常 reshard=True 的运行时，不是任意 FSDP 配置的通用包装器。

驻留验证分三层：tiny 设备门禁比较真实随机采样、参数/Adam 不变与恢复后训练，并检查拒绝准入、注入阶段异常后的清理；9B 对同一随机种子的两模式进行 token/reward/log-prob 精确比较；正式计时与独立 profiler 分开。`analyze_residency.py` 只汇总完整配对，不把生成吞吐换算成整个 RL 系统加速。 v8 已通过全部三次配对，生成阶段中位 1.761×；独立 profiler 确认每卡 16 token 的 all-gather 528→33 次，代价是额外约 12.45 GiB peak allocated。两种 actor 计划的中位配对加速仅 0.988×，因此保留 KK 基线。v8 四轮非零梯度更新已完成，但其奖励审计发现截断误判。v9 修复后 64 条回答均正常结束，62 条正确、2 条明确错答，已完成四步执行；只有第四步梯度非零。v9 没有性能比较，两个版本均未验证数学能力提升。

## 6. 恢复协议与普通保存的区别

`checkpoint.py` 负责 DCP 张量、RNG 和 scheduler；`state.py` 负责提交身份和发布恢复点。所有被 `.metadata` 引用的分片必须存在且长度足够，完整 manifest 包含文件 SHA256。最后才原子发布 `latest.json`。

epoch fencing 防止旧 worker 发布新恢复点，不负责停止它的物理计算。当前协议要求共享 POSIX flock/rename/fsync；HDFS 归档和 GRPO 回放 scratch 都不提供这个持久保证。

CPU 单进程重放通过不代表多机、故障注入、8→4→8 或异步 checkpoint 已通过。这些需要独立故障实验。

worker 异常时也不能无条件 `finally: destroy_process_group()`：已失败的 rank 可能在 NCCL 清理中等待仍进入其他 collective 的同伴，连原始异常都无法输出。`distributed.py` 先打印并刷新异常，再退出本进程，由 torchrun 终止其余 rank。`failure_probe.py` 专门在 Gloo 双进程中验证这个生命周期；它不构成 NCCL/HCCL 故障恢复验收。

## 7. 代价模型什么时候才值得用

`cost.py` 必须绑定模型、设备、kernel 和执行配置，profiling 与 validation 样本隔离；域外输入或误差过大时回退。`choose_profiled_plan` 当前只有合成单测，不能当作已验证优化。

先测真实长度分布、各 slot 的耗时、显存与编译开销。如果简单 KK/排序已达到同样效果，应保留基线。项目价值可以是明确的正确性修复、后端瓶颈与可靠恢复，不需要给每个模块都编造正向加速比。
