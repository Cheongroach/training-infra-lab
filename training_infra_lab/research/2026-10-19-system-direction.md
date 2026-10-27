# Qwen3.5 / Qwen3.8 长轨迹训练：调度与恢复运行时

核实日期：2026-10-19。状态：调研与工程设计；本文没有新的 GPU/NPU 性能结果。

推荐项目是**面向混合注意力模型的长轨迹 RL 训练执行运行时**：在一次算法更新的数据集合不变的条件下，联合规划数据分配、microbatch、重计算与梯度同步，并将执行计划、模型、优化器和数据进度纳入一致恢复。

主验证模型改为 `Qwen/Qwen3.5-9B`，扩规模模型选 `Qwen/Qwen3.8-27B`。H20 和 910B4 分别组成同构训练组，共用规划接口、正确性契约和实验协议。项目重点是训练端；真实 GRPO 用于端到端验收。旧 Qwen2.5 实验只作为历史回归底座，不构成新项目的成果。

## 1. 选题判断与证据边界

先前 `training_infra_lab` 完成的是通用对话数据上的全参数 SFT、性能消融和固定规模恢复。它证明了一些训练数值与工程基础，但没有覆盖 RL 数据语义、复杂执行计划、分布式失败恢复或新模型后端适配。将模型名字替换成 Qwen3.5，不会自动补齐这些能力。

长轨迹负载的工程问题有公开依据：NeMo-RL 已有 packing 与动态 batch，verl 已有工作量均衡，Skrull 和 Hydraulis 进一步联合处理计算、显存和数据分配。因此，“长度不均导致训练慢”和“应联合调度”有多源支持；“本项目方案能在 Qwen3.5/3.8 上进一步获益”仍是待测假设。<a href="#source-4">[cite:4]</a><a href="#source-5">[cite:5]</a><a href="#source-6">[cite:6]</a><a href="#source-7">[cite:7]</a>

| 方向 | 自己需要负责的工程问题 | 与现有工作的边界 | 判断 |
|---|---|---|---|
| 长轨迹训练调度与恢复 | 代价建模、显存准入、全 rank 执行计划、更新语义、状态恢复 | 在现有 packing/FSDP 之上验证新架构与后端差异 | **主选**，系统边界清楚，能逐层验收 |
| Qwen Gated DeltaNet 的 Ascend 算子优化 | varlen 前后向、状态边界、融合、设备内带宽与算力分析 | 官方生态已有 GDN 路径；应针对真实瓶颈修补 | 适合作为主选中的一个深入模块 |
| 完整异步 RL 平台 | rollout 服务、策略版本、背压、权重发布、陈旧度修正 | AReaL 已覆盖大部分机制；与工作区训推项目有重叠 | 深度足够，但首轮范围过大 |
| 独立弹性 checkpoint 系统 | 分片状态、提交协议、失败检测、重建进程组、数据重放 | DCP 已能 reshard；不能只封装保存 API | 主选的可靠性扩展，必要时独立成项目 |

该推荐是面向现有资源和学习目标的工程判断，不是论文新颖性认定。AReaL 的异步训练包含对过期策略样本的算法处理，不能以“生成后做 CE”或“加一个队列”声称复现。<a href="#source-8">[cite:8]</a>

## 2. 新模型与软件栈核实

本次直接读取官方 Hugging Face `config.json` 和模型 API 文件列表。三个模型均有公开 safetensors 文件；只核实了配置、revision 和文件目录，没有下载完整权重，也没有执行新模型训练。

| 模型 | 文本主干 | 官方目录核实 | 项目角色 |
|---|---|---|---|
| Qwen3.5-9B | 32 层，hidden 4096；24 层 linear attention + 8 层 full attention | 4 个 safetensors 文件 | 主模型，完成正确性、训练性能与真实 GRPO |
| Qwen3.6-27B | 64 层，hidden 5120；48 层 linear attention + 16 层 full attention | 15 个 safetensors 文件 | 已确认存在的同架构备选，不增加首轮矩阵 |
| Qwen3.8-27B | 64 层，hidden 5120；48 层 linear attention + 16 层 full attention | 18 个 safetensors 文件 | 较大规模验证 |

三者配置均为 `model_type=qwen3_5`、`Qwen3_5ForConditionalGeneration`；这是配置架构兼容的依据，不能证明任意旧软件栈都能正确训练。上表来自各官方仓库，属于模型发布方的单一权威来源核实。确切 revision 记录在同目录 [model-evidence.json](model-evidence.json)。<a href="#source-1">[cite:1]</a><a href="#source-2">[cite:2]</a><a href="#source-3">[cite:3]</a>

这里的 dense 指 FFN 非 MoE，并不表示所有层都是 full attention。Qwen 官方 Transformers 文档确认 3:1 的 Gated DeltaNet / full attention 结构；文本路径与完整视觉语言路径有不同配置类和加载入口。初版采用文本主干，冻结或排除视觉塔，不把文本路径的参数量称为完整多模态训练规模。<a href="#source-9">[cite:9]</a>

### 2.1 支持范围与版本冲突

| 证据 | 实际能确认什么 | 不能据此推出什么 |
|---|---|---|
| verl 当前 `qwen3_5.py` | 已有 packed convolution、GDN varlen 和 FLA CP 对接源码 | 每个设备、版本和并行组合都通过训练验证 |
| verl Ascend 快速指南，页面标注 2026-03-03 | 参考环境仍固定 Transformers 4.57.6，并写明不支持 5.x | 这份旧组合可以直接运行当前 Qwen3.5 路径 |
| ms-swift 当前 NPU 指南 | 披露 FLA Triton-Ascend GDN 路径和 Qwen3.5 验证范围 | 同样结果已在本地 910B4、9B/27B、长序列全参数 GRPO 上成立 |

这属于文档时间与适用范围的差异，不能挑一页文档便宣布“已完全支持”或“完全不支持”。本地上一轮的 Transformers 4.57.1 环境不能直接作为新方案的已验证环境。<a href="#source-10">[cite:10]</a><a href="#source-11">[cite:11]</a><a href="#source-12">[cite:12]</a>

ms-swift 的 GDN 专项说明列出 torch 2.9.0+cpu、torch-npu 2.9.0.post2、FLA 0.5.2 main、triton-ascend 3.2.1、Transformers 5.12.1；披露的 Qwen3.5-4B 验证是 A2 上 BF16、FSDP LoRA、packing、长度 512、300 步。该页面其他章节还有不同推荐版本，因此应以一套完整匹配组合为候选并实测，不能拼接各节版本。<a href="#source-12">[cite:12]</a>

实现顺序：H20 先锁定上游 verl 的可运行 commit 和依赖；910B4 用独立镜像验证相同模型/目标函数。优先复用同一规划接口；若 NPU 尚不能接入 verl 新模型路径，先以原生 FSDP2/ms-swift 适配器验证训练执行，并明确标注尚未完成 NPU GRPO。不能为填满表格退回 Qwen2.5。

### 2.2 容量约束

以下仅为条件估算：若保留 FP32 参数、FP32 梯度、两份 FP32 Adam 状态，静态分片状态约为 `16 × 参数数 / 分片数` 字节。按标称 9B/27B 估算：

| 模型 | 4 卡 | 8 卡 | 16 卡 |
|---|---:|---:|---:|
| 9B | 33.5 GiB/卡 | 16.8 GiB/卡 | 8.4 GiB/卡 |
| 27B | 100.6 GiB/卡 | 50.3 GiB/卡 | 25.1 GiB/卡 |

这些数值由公式计算，不是显存测量；不含激活、临时 BF16 权重、all-gather、通信缓存、输出头和 rollout/reference 模型。实际以训练参数清单和状态 dtype 复算。27B 在 8×64G NPU 上余量紧，不能直接承诺长上下文全参数训练。首轮 27B 优先放在 16×H20；910B4 先验证 9B。

## 3. 需要复现的机制与不应重复的功能

| 来源 | 已有机制 | 本项目如何使用 | 复现边界 |
|---|---|---|---|
| verl | `calculate_workload`、Karmarkar–Karp 分区、动态 microbatch、梯度累积延迟同步 | 必测强基线和可插拔入口 | 不能宣称首次引入计算量均衡/no-sync |
| NeMo-RL 0.5.0 | MFFD packing、动态 batch、loss wrapper | 移植明确的 batching 机制作为第二基线 | 该版本两条路径互斥；移植不等于完整框架对比 |
| Skrull | profiling、计算/显存约束、GDS、选择性 context parallel | 复现计算均衡与显存均衡的取舍 | 未实现 DACP 时，只称机制复现 |
| Hydraulis | 动态异构并行策略、O-P 分离、两阶段数据分配 | 参考代价建模和计划选择 | “异构策略”不等于 CUDA/NPU 混合训练；不声称复现整个 Hetu |
| AReaL | 异步 rollout、陈旧度控制、PPO 变体 | 另一方向的范围参照 | 首版保持同步 GRPO，不移植异步算法 |
| PyTorch DCP | 分布式保存、异步保存、加载时 reshard | 复用张量状态持久化 | DCP 不是完整训练提交、数据重放或任务恢复服务 |

以上机制分别来自原始论文、官方文档和源码。<a href="#source-4">[cite:4]</a><a href="#source-5">[cite:5]</a><a href="#source-6">[cite:6]</a><a href="#source-7">[cite:7]</a><a href="#source-8">[cite:8]</a><a href="#source-13">[cite:13]</a><a href="#source-14">[cite:14]</a>

最具体的切入点是当前 verl `calculate_workload`：公式为 `24576 * L + L**2`，注释明确按 hidden size 4096 的 7B dense Transformer 标定。它已超出纯 token 计数，但没有在这个函数中区分 Qwen 的线性/全注意力比例、设备 kernel、重计算与同步策略。这个源码观察是单一来源事实；是否造成可观性能损失必须通过 profiling 证明。<a href="#source-4">[cite:4]</a>

## 4. 拟实现的系统与关键契约

### 4.1 模块所有权

以下是计划中的模块接口，不是已生成或已运行的代码。

| 模块 | 输入 → 输出 | 必须自己处理的内容 |
|---|---|---|
| `profile_backend` | 模型、后端、真实 microbatch → 耗时/峰值显存/路径信息 | 分离 GDN、full attention、MLP、通信、输出头；记录编译与 fallback |
| `cost_model` | 长度分布、执行模式 → 时间/显存预测与误差范围 | 按模型 revision、kernel、硬件分别建模；域外样本回退 |
| `batch_planner` | 固定更新批次、资源预算 → 每 rank、每 slot 的索引与配置 | 最慢 rank 时间、显存限制、分组依赖、规划开销 |
| `distributed_executor` | 全局计划 → 梯度与 optimizer update | collective 顺序、microbatch 数、loss 归一化、同步点 |
| `training_state` | 更新日志、模型/Adam/RNG/数据 → 一致 checkpoint | epoch 隔离、提交清单、不完整快照识别、重放 |
| `verl_adapter` / `fsdp_adapter` | 框架 batch 与事件 ↔ 统一接口 | 保留真实 GRPO 语义；CUDA/NPU 差异集中在后端 |

主工程工作集中在组件契约，以及证明它们正确工作的测试与测量。attention、通信库、FSDP、GRPO 和存储原语复用上游实现。

### 4.2 批次规划与有限搜索空间

第一版固定 DP/FSDP mesh、TP=PP=CP=1；在一次 optimizer update 内改变样本到 rank/microbatch 的分配，先不做每步动态切换并行拓扑。这样可以测清调度收益，避免把张量重分片、网络与算法同时混入。

候选配置为“packing/动态 batch 方式、microbatch 预算、重计算档位、梯度累积同步策略”。先以相同配置对比调度器，再逐项放开执行配置选择，否则无法归因。每个配置都要先通过正确性和容量检查。

代价模型可以从可解释的形式开始：

`T = a·ΣL + b·ΣL² + c·序列数 + d·microbatch数 + 已测通信项`

其中 `ΣL²` 针对正确隔离序列后的 full attention；不能把 packed 总长度平方当成每条独立序列平方之和。GDN、padding、对齐、kernel 与重计算的影响由分配置测量吸收；该式是候选模型，不是精确 FLOPs 或性能结论。

目标为最小化预测最慢 rank 的更新时间，加上实际规划开销，并要求各 rank 显存上界不超过预算。拟合与验收使用不同真实 batch；记录预测误差、排序错误、显存低估和拒绝率，不能只报告拟合优度。

未知形状或预测误差过大时，回退到已验证固定预算。若最长单条轨迹已超容量，microbatch=1 也不能解决；只能明确拒绝该配置，或使用已验证重计算/固定 CP 方案。不能悄悄截断轨迹后宣称吞吐提升。

### 4.3 分布式执行契约

FSDP 参数 all-gather 仍然存在，no-sync 只减少梯度通信。不能让快 rank 随意多执行 forward，让慢 rank 自行退出循环。全 rank 必须使用对齐的 slot 数、模块遍历和同步计划；执行前统一确认计划摘要。<a href="#source-13">[cite:13]</a>

空 slot 若确有必要，使用经过验证的最小有效 forward 和零贡献 loss，保持同一 collective 路径；它的时间和内存必须计入开销，不能使用产生 NaN 的全 mask 平均。优先在规划阶段减少空 slot。

同步/重计算配置先按全组一致档位选择。逐 rank 配置只有证明 collective 契约一致后才开放。optimizer step、梯度裁剪和 LR scheduler 都以全局更新为单位，不能随局部 microbatch 数变化。

### 4.4 Qwen 混合注意力的序列边界

Packing 除了隔离 full attention，还必须重置各样本的 GDN recurrent state 和 causal convolution 边界。只重置 position IDs 或 attention mask 不足以证明独立性。

verl 已有 `_prepare_packed_seq_idx`、`_packed_causal_conv1d_fallback`、`_packed_chunk_gated_delta_rule`、`_ConvPrefixExchange` 等路径，应复用并扩展验证。CP 路径检查 FLA `cp_context` 能力；不能因为 full attention 的 Ulysses 可用就宣布整模型 CP 可用。<a href="#source-10">[cite:10]</a>

关键验证：同一批真实短/长样本，单独执行、padding 执行和 packed 执行，在监督/响应位置比较 logits、loss、各模块梯度和一步更新；改变上一条样本内容，不得影响下一条样本的独立输出。CP>1 后续独立验收。

### 4.5 RL 语义

端到端 workload 采用同步 GRPO：当前策略真实 rollout → 规则 reward → 按 prompt group 计算 advantage → policy log-prob/ratio/clipping → 参数更新。初版固定 rollout 布局和同步方式，让变化来自训练执行。

轨迹至少携带 `trajectory_id, prompt_group_id, policy_version, response_mask, old_log_probs, reward`。完成 group 级 advantage 计算后，独立轨迹可按相应 loss 的条件重新分配；组内统计不能因换 microbatch 而变成局部统计。

比较内固定 prompt 集合、每题 rollout 数、更新次数、优化器、loss reduction、截断规则和 KL 配置。序列平均、token 平均等目标各有语义，必须保留选定算法的原始定义，不能为负载均衡自动换成 token 平均。

先固定同一真实 rollout 批次回放训练更新，隔离 scheduler 数值与性能；这只能叫“actor-update 回放”。最终验收必须重新生成轨迹并运行真实 GRPO。现成 DeepSeek-R1 回答没有本策略行为概率，不能直接当作本模型 on-policy GRPO 轨迹。

### 4.6 一致恢复与持久提交

内存中的 `optimizer.step()` 完成不等于持久提交。恢复点由完整 checkpoint 清单决定：模型、Adam、LR scheduler、RNG、全局更新 ID、数据/轨迹清单、策略版本和执行计划版本必须对应同一状态。

DCP 保存张量分片；所有分片成功且清单验证通过后才发布持久恢复点。异步保存先获得一致 staging 快照，再允许源状态变化；限制并发快照数和 CPU staging 内存。存储原子发布方式依据实际 HDFS/文件系统能力实现并验证，不假设任意对象存储的 rename 都是原子的。<a href="#source-14">[cite:14]</a>

未到持久恢复点的更新允许重算，成本计入指标。先终止旧进程组，以运行 epoch 阻止旧 worker 发布结果，再恢复全部训练状态、重放同一数据。保证的是有效恢复链上的更新不重计，不是“任何物理计算都只执行一次”。

单 rank 在 forward/backward 中 OOM 后，其他 rank 可能已进入 collective，不能只在出错 rank 捕获异常、减半 batch 并继续。初版对未预防 OOM/通信异常执行整组重启；确认发生在共同安全边界的容量拒绝，才允许全组重新规划。

第一阶段做同 world size 恢复。第二阶段再做同设备栈 8→4→8：DCP reshard 张量，额外恢复全局数据顺序、更新批次和 LR 进度。改变归约顺序后不承诺逐字节一致；CUDA↔NPU 恢复不是首版范围。DCP 文档也不保证跨 PyTorch 版本 checkpoint 兼容。<a href="#source-14">[cite:14]</a>

## 5. 待证伪的假设与实验协议

### 5.1 三个研究问题

1. **混合架构代价模型是否有增量？** 比较纯 token、verl 固定公式和实测代价模型。若现有公式已足够且规划成本更低，应保留基线。
2. **显存预算是否改善有效吞吐？** 比较固定预算、只预测时间、时间与显存联合规划。观察 OOM、拒绝与重算，不能只看成功 step。
3. **后端差异是否改变执行策略？** 在 H20 和 910B4 分别测重计算、梯度同步和 GDN 路径，检验策略能否迁移，不能提前认定硬件优劣。

这些都是待验证假设。Skrull/Hydraulis 已研究计算与显存冲突，故项目可能贡献混合架构适配、可复现负结果或上游扩展，而非新调度理论。<a href="#source-6">[cite:6]</a><a href="#source-7">[cite:7]</a>

### 5.2 工作负载

用公开真实长短推理样本建立训练形状与边界测试，例如 OpenR1-Math-220k。其生成回答可用于 SFT 正确性/训练内核负载；真实 GRPO 使用问题与参考答案，再由当前 Qwen 策略重新生成。数据卡披露 R1 生成和验证方法，名称中的 220k 不能当作某个 subset 的精确行数。<a href="#source-15">[cite:15]</a>

样本 ID 固定分成 profiling、validation、evaluation，按问题去重，不能让同题不同回答跨集合。实际 tokenization 后公布长度分布；16K/32K 桶无足够真实数据时报告缺测，不能把字符串重复拼长当作真实任务结果。

### 5.3 基线与规模

| 实验层 | 必测对照 | 目标规模与说明 |
|---|---|---|
| 数值/边界 | 独立样本、padding、packing；原始更新与重新规划更新 | 小配置用于单元验证，9B 真实 batch 用于集成验证 |
| 调度隔离 | 固定 batch；token 均衡；verl workload 均衡；实测代价+显存规划 | 9B：H20 4/8/16，910B4 4/8；容量不过的格保留原因 |
| 执行策略 | 相同调度下分别切换重计算/no-sync | 独立消融，基线也允许合理调优 |
| 故障 | 无故障、普通 checkpoint 重启、带一致状态协议重启 | 杀 worker、杀协调器、写 checkpoint 中断、容量压力 |
| 真实 RL | 同步 GRPO 原始训练执行器 vs 新执行器 | H20 先完成；rollout 资源、reward 与算法配置固定 |
| 扩模型 | 原始基线 vs 最终方案 | 3.8-27B、16×H20；通过后才扩大矩阵 |

训练卡数不等于整项 RL 资源预算。trainer 使用 8/16 卡时，rollout/reference 的资源也要单列；不能给两方案不同 rollout 资源却只比较 trainer 卡数。先用 actor-update 回放完成扩展测试，再安排真实 RL。

主比较固定更新数据集合和 token 数，按 trial 成对随机顺序执行，至少三次独立重复；仅在波动影响结论时增加重复。难度与质量差异用真实 RL 曲线验证，不把回放吞吐当成收敛收益。

### 5.4 指标与通过条件

| 指标 | 口径 |
|---|---|
| 更新延迟 | 调度开始到 optimizer update 完成，p50/p95 与分解 |
| rank 等待 | 计算与通信等待分别测量；长 NCCL/HCCL 区间不自动等于网络瓶颈 |
| 执行吞吐 | 非 padding 训练 token / 包含调度与通信的墙钟 |
| 恢复有效吞吐 | 最终有效状态链上的去重训练工作 / 包含失败、重启、重放与最终持久化的墙钟 |
| 显存与主机内存 | 各 rank allocated/reserved 峰值，checkpoint staging 单列 |
| 规划质量 | 预测误差、显存低估率、回退率、规划耗时 |
| RL 质量 | 固定评估集 reward/pass rate、响应长度、KL、clip fraction、无效/截断比例 |
| 资源效率 | 总预留设备小时与角色用量；无价格数据不宣称实际省钱 |

通过条件先要求数值语义、样本计数、故障恢复和无状态污染。性能必须相对**已启用合理 packing/动态 batch 的强基线**有可重复增量；仅超过全 padding 或未调优基线不足以支撑结论。

可预注册继续投入门槛，例如端到端有效吞吐改善至少 10%，或同吞吐附近明显降低失败/恢复损失。10% 是项目选择门槛，不是预测结果。只有 kernel 局部加速而端到端无收益时，结论按实际范围写。

一两晚真实 RL 主要验证系统行为与短程质量未明显异常，不能证明长程收敛等价。质量预算不足时报告“执行效率通过，最终学习效果未确认”。

## 6. 实施阶段与学习顺序

“一到两晚”是代码准备完成后的计算预算，不是同时完成新架构适配、系统开发、论文级实验和长程收敛的保证。

第一阶段实现后端能力检查、9B 正确性、已有强基线和 profiler。只有发现可复现失衡或容量问题，才投入代价模型和规划器；若瓶颈是 NPU GDN fallback，则先修后端路径。

第二阶段实现规划器与执行协议，跑真实 batch 回放、分布变化和同规模失败恢复。第一晚优先完成 9B 两后端实验并收齐失败记录，不铺开所有模型/长度/配置的笛卡尔积。

第三阶段接入真实同步 GRPO，第二晚验证端到端影响，随后做 H20 多机和 3.8-27B 扩规模。8→4→8 恢复是扩展验收项，不能在同规模重启通过后自动标成已完成。

阅读顺序：verl `seqlen_balancing.py` → Qwen `qwen3_5.py` → FSDP collective/梯度归一化 → 规划接口 → DCP 与恢复状态机 → 真实 GRPO adapter。每一步对应可复现错误或通过证据，而不是只阅读一份大脚本。

简历应记录实测完成的模块、模型/卡数、强基线增量、恢复行为和上游提交。当前不能写“实现了该系统”或填入待测数字。旧报告中的 16 卡结果仍属于 Qwen2.5 短程 SFT，不能迁移为新模型或 RL 成果。

## 7. 来源

`main/latest` 页面为本次访问时内容，会继续变化；运行时另行固定代码、镜像和依赖。论文机制、官方支持范围与项目提案不是同一层级的证据。

<a id="source-1"></a>[cite:1] Qwen3.5-9B 官方[配置](https://huggingface.co/Qwen/Qwen3.5-9B/resolve/c202236235762e1c871ad0ccb60c8ee5ba337b9a/config.json)与[模型 API](https://huggingface.co/api/models/Qwen/Qwen3.5-9B)。本次直接读取配置与文件列表。

<a id="source-2"></a>[cite:2] Qwen3.6-27B 官方[配置](https://huggingface.co/Qwen/Qwen3.6-27B/resolve/6a9e13bd6fc8f0983b9b99948120bc37f49c13e9/config.json)与[模型 API](https://huggingface.co/api/models/Qwen/Qwen3.6-27B)。

<a id="source-3"></a>[cite:3] Qwen3.8-27B 官方[配置](https://huggingface.co/Qwen/Qwen3.8-27B/resolve/1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0/config.json)与[模型 API](https://huggingface.co/api/models/Qwen/Qwen3.8-27B)。

<a id="source-4"></a>[cite:4] verl [`seqlen_balancing.py`](https://github.com/verl-project/verl/blob/main/verl/utils/seqlen_balancing.py)。实际读取 raw 源码：`calculate_workload`、`karmarkar_karp`、`rearrange_micro_batches`。

<a id="source-5"></a>[cite:5] NeMo-RL 0.5.0：[Sequence Packing and Dynamic Batching](https://docs.nvidia.com/nemo/rl/0.5.0/design-docs/sequence-packing-and-dynamic-batching.html)。该版本的约束不外推至后续版本。

<a id="source-6"></a>[cite:6] [Skrull: Towards Efficient Long Context Fine-tuning through Dynamic Data Scheduling](https://arxiv.org/html/2505.19609v1)，2025，第 3–4 节与性能建模附录。

<a id="source-7"></a>[cite:7] [Hydraulis: Balancing Large Transformer Model Training via Co-designing Parallel Strategies and Data Assignment](https://arxiv.org/html/2412.07894v2)，2025 修订版，第 5–7 节与代价模型附录。

<a id="source-8"></a>[cite:8] [AReaL: A Large-Scale Asynchronous Reinforcement Learning System for Language Reasoning](https://arxiv.org/abs/2505.24298)，本次摘要页显示最新 v5，2026-03-02。引用异步解耦与陈旧度处理，不移用论文加速比。

<a id="source-9"></a>[cite:9] [Transformers Qwen3.5 文档](https://huggingface.co/docs/transformers/main/en/model_doc/qwen3_5)，混合结构、文本/多模态类与 GDN 路径。

<a id="source-10"></a>[cite:10] verl [`qwen3_5.py`](https://github.com/verl-project/verl/blob/main/verl/models/transformers/qwen3_5.py)，本次读取 packed convolution、GDN、CP 源码。

<a id="source-11"></a>[cite:11] [verl Ascend Quickstart](https://verl.readthedocs.io/en/latest/ascend_tutorial/ascend_quick_start.html)，页面标注 2026-03-03，旧 Transformers 限制与支持表。

<a id="source-12"></a>[cite:12] [ms-swift NPU Support](https://swift.readthedocs.io/en/latest/BestPractices/NPU-support.html)，重点 Verified Combinations 和 Qwen3.5 FLA Patch Notes。

<a id="source-13"></a>[cite:13] [verl Performance Tuning Guide](https://verl.readthedocs.io/en/latest/perf/perf_tuning.html)，动态 batch、remove padding、重计算、梯度同步与显存取舍。正文与页头日期可能不同步。

<a id="source-14"></a>[cite:14] [PyTorch Distributed Checkpoint](https://docs.pytorch.org/docs/stable/distributed.checkpoint.html)，保存/加载、resharding、异步 staging 与兼容边界。

<a id="source-15"></a>[cite:15] [OpenR1-Math-220k 数据卡](https://huggingface.co/datasets/open-r1/OpenR1-Math-220k)，真实问题、生成轨迹、验证来源与 subset 范围。
