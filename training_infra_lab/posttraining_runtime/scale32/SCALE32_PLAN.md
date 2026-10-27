# GRPO 扩展至 32 卡

用户授权：2026-10-25 明确允许扩展卡数，“32 卡是一个相对舒服的数量。更多卡我也不阻拦。”32 卡为首轮目标；新增资源应有对应实验，未指定累计卡时预算。

## 已执行

- cn 控制面、group 2079 当前查询：LF cluster 22 可见 104 H20，HL cluster 20 可见 32 H20。聚合余量不保证节点调度或互联。原始查询 request ID：`20261025121409C8F36BF2000DE22CB941`。
- 已检查 11 个 `miles-grpo` 历史任务，均为终态。
- 跨节点环境预验已完成：run `0487f5d136d7b5b9`，trial `439159056`，终态 `done / finished`，四个 Pod 均 status=3、exit=0；全部 32 rank 核验通过。保留初次 `started / queued` 查询证据。
- 资源：LF 同组保证队列，4 worker × 8 H20，每 worker 64 CPU / 256 GiB。沿用 image VID `dbdt1t71enl9fha0bn40`。
- 全部 14 项平台预检 success；SID `a552a2fb-eab2-4368-9f12-5ec7b3f82030` 已保留式合并进最终请求，在有效期内提交。
- 入口上限 720 秒，超时额外 110 秒终止及归档；torchrun 上限 480 秒，额外 15 秒清理。每节点独立保存归档。此任务仅验证环境、32-rank NCCL 和 BF16 正反向，不加载模型。
- [提交与证据](runs/preflight-01/)、[构建入口](https://github.com/Cheongroach/training-infra-lab/blob/main/training_infra_lab/posttraining_runtime/scale32/build_preflight.py)。原始 create 回执 SID 位于顶层；不得因缺少 `data.sid` 重复提交。
- 四节点归档的源文件 hash、32-rank NCCL 求和 528、BF16 loss=128 和预期梯度全部匹配。32 份 NCCL 日志出现跨节点 `NET/soft_bonding_v8/.../GDRDMA` 通道；无带宽、延迟或性能增益结论。
- 入口日志与四条实际 Pod stdout 精确匹配，但日志下载标签的顺序与 `ARNOLD_ID` 不一致；以归档 bootstrap 和 rank 身份闭合，避免从下载顺序推断 rank。独立核验见 [verified.json](runs/preflight-01/verified.json)。
- 终态 Pod 驻留合计 3.7644 H20·h，包含镜像准备与初始化；没有模型训练。
- 独立格式任务 `64bf39daf0b766d2` / trial `439161981` 在生成前失败，退出码 126。stderr 明确为 shell 单参数 `Argument list too long`，入口约 150,682 bytes，未生成产物；成本 0.3956 H20·h。新构建入口压缩至约 54,968 bytes，另设 96 KiB 参数上限，保留原失败请求与日志，不覆写历史证据。
- [format-02](runs/format-02/) 在 SGLang 就绪后、请求生成前因 Transformers 返回 `BatchEncoding` 而失败，退出 1，未生成回答；归档源码与入口已核验，成本 0.4656 H20·h。
- [format-03](runs/format-03/) run `fc234509cf0d43b9` / trial `439173098` 终态 done，exit=0，48 条回答的评分、EOS 与概率逐条独立复算通过；成本 0.7133 H20·h。显式提示文件 SHA256 `498d4e3342c97fd7e35223093d0211c921c92fd54ec9b19bcd69fd67e883259f`，保留原 64 题顺序与标签，固定原评分器，不强制解码。原提示确定性格式通过 2/16，新提示 10/16，新提示随机 8/16。`audit_complete=true`、`protocol_passed=false`；格式改善不能写成格式已解决或 held-out 能力提升。
- Qwen3-8B 的 14 文件共 16,397,443,109 bytes（15.2713 GiB）已逐文件校验并上传 HDFS，`verified=true`、`uploaded=true`。来源为 ModelScope 的逐文件 immutable revision + SHA256，尚未验证 Hugging Face revision 一致性；不得混写为某个 HF commit。
- format-03 后曾要求至少半数完成且严格提取；train-04 的实测回放证明这会阻止所有候选进入训练。当前 train-05 将严格格式统计与训练准入分开：保留整个候选批的原始回答、原始评分和解析失败负奖励，要求至少半数自然完成、完整组与奖励变化；补采仍最多四批。`strict_extraction_passed`、`format_contract_passed` 独立报告，训练通过不代表格式通过。真实候选的 CPU 回放通过，非有限概率/重复 ID 被拒，全部响应/评分保持不变。
- [train-03](runs/train-03/) 的 18 文件压缩包源码回读、AST 和 bash 检查通过，shell 参数 71,514 bytes；平台 14 项预检 success，无 warn/hint，保留式合并 precheck_sid 后于有效期内提交。run `03c584556b4a96a9` / trial `439180304`，2026-10-25T05:17:23Z 为 started、0 Pod。尚无真实多节点 Ray/8B 训练结论；后台 [collect_train32.py](https://github.com/Cheongroach/training-infra-lab/blob/main/training_infra_lab/posttraining_runtime/scale32/collect_train32.py) 采集终态证据。
- 已实现 [train32_checks.py](https://github.com/Cheongroach/training-infra-lab/blob/main/training_infra_lab/posttraining_runtime/scale32/train32_checks.py) 与 [audit_train32.py](https://github.com/Cheongroach/training-infra-lab/blob/main/training_infra_lab/posttraining_runtime/scale32/audit_train32.py)：实际 TP/DP/物理节点记录，16 rank 参数/Adam 矩变化，8 个 DP shard 的 token/概率/身份归属，8 个 TP2 引擎的 checksum 与最终 v3 发布、原评分器复算。
- 本轮已终态的环境/格式任务合计 5.3389 H20·h；32 卡训练仍未计入终态成本。train-01 / train-02 是被替代的本地未提交草稿，禁止报告为 GPU 任务。
- train-03 已终态 failed，模型训练未开始：实际 Ray 日志 `worker_ports ... port number 11516 ... used by other components`，平台 GCS 端口落在 Ray 默认 10002–19999 worker 范围。退出信息聚合接口返回顶层 `trials/pods`，采集器现已适配。两个节点归档与实际 stdout 完整匹配，其余两个被集群联动终止，归档不可用；保留这一证据缺口。驻留成本 1.8244 H20·h，累计已终态 7.1633 H20·h，无 8B 训练结论。
- train-04 使用与 GCS 不相交的 1000 个显式 worker 端口，并在 bootstrap 记录范围；CPU 对全部 65535 个可能 GCS 端口验证不相交。18 文件冻结包 71,978 bytes，尚以真实 Ray 加入、训练及终态审计作为验收。
- train-04 在 14 项预检 success、无 warn/hint 后保留式合并 SID 并提交：run `428f9ac06ea21524` / trial `439186603`。2026-10-25T05:37:29Z 四个 Pod 均运行；镜像准备导致最早与最晚 Pod 启动相差 263 秒，尚未获得真实训练终态。
- 已准备 [publication_benchmark.py](https://github.com/Cheongroach/training-infra-lab/blob/main/training_infra_lab/posttraining_runtime/scale32/publication_benchmark.py) 与[独立审计器](https://github.com/Cheongroach/training-infra-lab/blob/main/training_infra_lab/posttraining_runtime/scale32/audit_publication_benchmark.py)：训练后 v3 固定权重，在同一 NCCL 连接、同一 8 引擎、同一 bucket 布局上做两次 warmup 和 16 次平衡顺序测量（普通/packed broadcast × 全量 checksum 开关，各四次）。保存每 rank 计时、发送 bytes/布局及模型/Adam/引擎内容指纹。checksum-off 的正确性核验在计时区间外；仅比较发布阶段，不声称端到端训练加速。代码已通过 CPU AST/配置检查，尚未提交 GPU 性能实验。
- train-04 四节点在 05:41:14–05:41:18Z 加入 Ray；逐物理节点 GPU 探针通过，权重转换于 05:42:58Z 保存。转换产物为两份约 8.19 GB 的 distcp 加元数据；05:51Z 共享目录校验完毕并发布 manifest，随后三个非主节点复制和校验。日志中的 `Stack: <no Python frame>` 后进程已推进至复制阶段，不能单独解释成训练崩溃。
- SSH 默认 `python3` 与训练虚拟环境不同：第一次 packed 支持探针报 `No module named sglang`；改用实际 `/opt/sglang/bin/python3` 后返回 `supports_multi_dtypes=true`。该结果只确认 API 能力，尚无 packed GPU 计时结论。
- 已新增[分布式参数审计方法与四道练习](PARAMETER_AUDIT.md)、有界[运行阶段采集器](https://github.com/Cheongroach/training-infra-lab/blob/main/training_infra_lab/posttraining_runtime/scale32/inspect_live.py)和[资源面板采集器](https://github.com/Cheongroach/training-infra-lab/blob/main/training_infra_lab/posttraining_runtime/scale32/capture_panel.py)。当前 Grafana 已取得 CPU、内存、GPU 利用率与显存的可读数据；HDFS panel 42/48 返回无可读点，不能据此报告带宽。
- 下一实验可通过 [freeze_converted_cache.py](https://github.com/Cheongroach/training-infra-lab/blob/main/training_infra_lab/posttraining_runtime/scale32/freeze_converted_cache.py) 绑定终态 source 的四节点源码/身份、模型校验和全部节点转换就绪观测，再复用转换产物；每个新节点仍重新复制并校验全部文件。准备产物与训练成功分开记录，允许准备已完成但后续 admission 失败的缓存复用，不声称 optimizer 更新。
- train-04 已终态 failed；四份源码/身份/entry 与实际 stdout 均核验通过。四批共 256 条回答因 `GRPO_QUALIFICATION_EXHAUSTED` 被拒，零 optimizer 更新。原 gate 将 decoded EOS `<|im_end|>` 留在正则输入中，严格提取全部误报 0；修复后四批为 25/14/19/20，仍未达到原 32 条阈值。完成数 61/63/63/63，奖励不同组 7/5/6/6，身份/概率 fatal errors 为空。保留原始响应、token 和 reward。
- 新 gate 仅在 completed 回答的末尾去除 `<|im_end|>` 供格式统计，格式与训练 admission 分开报告；完整组、自然完成、有限概率、奖励变化及逐 rank/engine 内容审计继续作为训练门禁。真实四批 CPU 回放通过，无样本修改；格式仍失败。train-04 Pod 驻留 15.5822 H20·h，平台终态 start_at 与早期 running 查询有差异，成本统一以终态记录计。累计完成扩展任务 22.7456 H20·h。
- train-05 冻结 20 文件、81,062 bytes；14 项预检 success，无 warn/hint，保留式合并 precheck_sid 并在有效期内提交 run `dd561599d5c16cc6`。仍为 4×8 H20、16 learner +16 rollout，复用 train-04 的六文件转换缓存（`source_conversion_verified=true`、`source_training_verified=false`）。待新任务终态验收。

- train-05 已终态 done/finished，四 Pod exit=0；四份冻结源码/节点归档与实际 stdout 一一匹配。独立 `training-audit.json` 通过：3 次真实更新、192 条完整组训练样本、16 learner rank 的真实模型参数与本地 Adam m/v 变化、TP 固定后的 DP 模型一致、八个 TP2 引擎各 582 个 checksum 一致、v0→v3 发布及最终 16 条回答全 token 消费 v3。每轮每 rank 变化模型 tensor 为 202/202/197，gradient norm 为 0.295643/0.205806/0.247101。
- train-05 自然完成 61/64/63，严格提取 26/12/18，奖励不同组 7/5/8；格式契约仍失败。末次暴露题 16 条回答中自然完成 10、严格提取 4、原评分器判对 2，仅作协议观察。驻留成本 12.5311 H20·h，累计已终态扩展任务 35.2767 H20·h，无 held-out 能力或 SOTA 结论。
- publication-01 的同内容发布 A/B 已完成 20 文件冻结、AST/bash/压缩回读验证。14 项平台预检 success、无 warn/hint，SID 保留式合并后提交 run `2e633b4e08807737` / trial `439212279`；仍 4×8 H20，等待终态训练与发布审计。该任务的训练阶段上限 3000 秒，入口总上限 5400 秒；不得把排队/运行写成性能证据。
- publication-01 已终态 done，四 Pod status=3、exit=0，四节点源码/身份/entry 与实际 stdout 一一匹配，独立训练与发布审计均通过。再次完成 3 次真实更新、192 条训练回答；全部 16 rank 模型/Adam 状态、8 引擎一致性与末次 16 条回答全 token 消费 v3 通过。末次暴露题自然完成 12、严格提取 4、原评分器判对 0，无能力收益结论。
- 18 次同内容发布含两次 warmup，四种配置各四次平衡重复；每次发送 16,381,470,720 bytes，16 rank 计时与布局复算通过。checksum-off 普通/packed 墙钟中位数 0.530577/0.522645 秒，比值 1.015177；样本范围重叠，不足以证明稳定性能优势。checksum-on 中位数 30.282540/30.262081 秒，校验增加约 29.75 秒；不代表端到端训练加速。
- 发布审计修正了算法边界：Miles 末次事件的 `checksum` 使用 16 位 `tensor_hash`，基准的 `raw_checksum` 使用完整 SHA256，不比较不同算法的 digest。严格检查末次 learner 完整状态、八个发布 cell/update 身份、tensor 名称覆盖，以及基准每轮 raw SHA256/布局与参考完全一致。五项真实证据篡改回归通过，原始失败审计快照保留于本地。
- publication-01 驻留成本 19.4844 H20·h，包含准备、训练与重型内容校验；全部扩展终态合计 54.7611 H20·h。GPU/显存/CPU/内存面板重新按实际 Pod 时间与四物理主机采集，分别取得 517/1034/524/1310 个可读点。原看板 `hostnum` 自动查询错误主机时返回空值，显式绑定 `ray-cluster-verified.json` 的主机后恢复；不能将首次空数据解释成零利用率。

## 后续执行顺序

1. train-05 与 publication-01 均通过终态独立验收；四节点归档及共享 results 已收集。按 ARNOLD_ID 与 entry 内容匹配 stdout，不从下载顺序猜 rank。此前失败与成功诊断均保留，无需重跑。
2. 已完成格式探针，尚未修复全部格式失败。对 8B 训练候选批独立复算原评分器、自然 EOS、严格提取占比及失败明细；保留负样本及拒绝候选的全量轨迹。
3. 跨节点 Miles/Ray 启动：按平台实际 `ARNOLD_ID`、`ARNOLD_WORKER_NUM`、`METIS_WORKER_0_HOST/PORT` 建集群；在提交训练前核验所有节点加入、GPU 资源、依赖、数据、模型路径和 Jobs 就绪。
4. 32 卡诊断：Qwen3-8B，16 learner（2 节点，TP2 × DP8、PP1、CP1）＋16 rollout（2 节点，8 个 TP2 引擎）；3 次更新，每次 16 prompt × 4 response。末次 eval 为已暴露的前 16 题，仅核验最终发布与格式；尚没有独立质量评测。
5. 同内容发布 A/B 已完成，没有稳定 packed 收益证据。后续若继续性能研究，优先测关闭重型诊断后的完整训练路径与训推重叠，使用相同有效工作量；当前没有必要只为发布传输增加卡数。

## 验收

- 正确性：完整组、身份与 token 映射、有限行为概率、非零学习信号、所有 learner rank 参数/Adam 实际变化、所有引擎内容一致、生成与消费时策略版本。
- 恢复：按独立故障边界验模型、优化器、scheduler、RNG、采样状态与重新发布，不从同 worker 冷恢复推出节点丢失可恢复。
- 效果：固定独立题集、相同生成预算与评分规则的前后配对结果。格式通过与参数变化不能代替质量收益。
- 性能：有效 tokens/s、成功组/GPU·h、阶段暴露时间与引擎暂停区间；记录启动、准备、过滤和重算成本。Nsight 的累计时间不能直接作墙钟加速比。
- 平台提交：预检 error 停止，warn/hint 先展示；保留 request_context 其他字段并合并 precheck_sid，在 30 分钟内提交。发生不明确的写请求超时，先查询已有任务，不直接重发。

## 已复现的输出协议问题

固定 `math_dapo_utils.py` 在 CPU 上对正确整数的结果：

| 生成文本 | pred / score |
|---|---|
| `Answer: \boxed{54}` | `54 / +1` |
| `Final Answer:` 后换行再输出 `$$\boxed{54}$$` | 空串 / -1 |
| `**Final Answer: \boxed{54}**` | `54** / -1` |

`-3` 和 `3` 重现同一模式。这证明该固定评分器的解析行为；新的提示能否使模型遵守格式仍需真实运行验证。
