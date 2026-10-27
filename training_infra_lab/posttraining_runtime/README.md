# Miles 四卡运行验收

四张 H20 的依赖、单机 NCCL 和 BF16 正反向已通过独立归档验收；真实 GRPO 三轮有效学习、参数变化与前两轮权重回传也已通过独立验收。跨机 RDMA 未验证。

## 最新执行状态（2026-10-24）

**Ray 提交修复验证（70d36e11）终态：done；训练与最终发布通过，完整回答协议未通过，整体 passed=false。** [报告](grpo-ray-ready-ba214ae231ee/REPORT.md) / [独立验收](grpo-ray-ready-ba214ae231ee/verified.json) / [归属审计](grpo-ray-ready-ba214ae231ee/run-provenance.json) / [任务](https://ml.bytedance.net/development/instance/jobs/70d36e11e929c5c3)。CPU Jobs 检查 2.794 秒 SUCCEEDED，唯一训练 Job 正常提交；三轮梯度 0.466318/0.446185/0.198871，两个 TP rank 参数和 Adam moments 每轮改变，双引擎校验和一致，最终 version 3 发布。80 条候选中 48 条入训；四条最终回答全部自然 EOS、version 3 完整覆盖，共 8,528 响应 token，但两条 Final Answer 后换行令冻结 DAPO 规则捕获 $$、pred 为空，回答协议验收拒绝；正确数 0/4，不宣称能力或吞吐收益。Pod/入口/归档均退出 0；11 份源码、三份 runtime patch、15 份 auditor 和三个 Ray 日志快照已核验，125 项 CPU 检查及 14 项预检通过，终态驻留 1.453333 GPU·h。只执行一次修正训练，未自动重跑。现有四张 H20 足够；旧任务 agent 不可用的底层原因仍未证明，旧失败证据独立保留。

**最终版本完整回答验收（585440c2）终态：failed，passed=false。** [报告](grpo-publication-4096-4fdcc7793e37/REPORT.md) / [严格验收](grpo-publication-4096-4fdcc7793e37/verified.json) / [归属审计](grpo-publication-4096-4fdcc7793e37/run-provenance.json) / [任务](https://ml.bytedance.net/development/instance/jobs/585440c23b867d70)。10 文件资产校验和双进程转换完成；Ray 训练 Job 提交返回 `500: No available agent to submit job`，随后清理返回 404。没有 rollout、梯度、权重发布或四条 probe 响应，完整回答目标尚未完成；agent 不可用的底层原因未确认。Pod/入口退出 1、归档复制退出 0；9 份内嵌源码、两份 runtime patch、13 份冻结 auditor 和日志归属通过核验，109 项 CPU tests 通过。终态 Pod 驻留 0.671111 GPU·h。本次未自动重试；首次 staging 取消任务 `53bfe898ac2210b3` 单独归档且成本未知。此前冷恢复及 v4 的通过结论按各自证据保留。

**GRPO 冷进程恢复 v2（172e5c51）终态：passed=true。** [报告](grpo-recovery-v2-172e5c51/REPORT.md) / [独立验收](grpo-recovery-v2-172e5c51/verified.json) / [归档归属审计](grpo-recovery-v2-172e5c51/archive-audit.json) / [任务](https://ml.bytedance.net/development/instance/jobs/172e5c51a339edef)。两个 learner TP rank 的模型、Adam、scheduler、RNG 恢复指纹一致，采样游标恢复 32 个身份；20 文件 checkpoint 约 52.46 GiB。两台新引擎各 290 个 raw tensor SHA256 与提交 version 2 一致，新传输 UUID 已核对；42 行历史重放单独去重。64 条候选中 48 条入训，三轮梯度 0.290185/0.256558/0.431847；两个 rank 和两台引擎逐轮各改变 207/200/202 个模型 tensor，Adam moments 均变化。末轮实际发布 version 3，四条生成的 2,048 个响应 token 版本覆盖通过，但四条均在 512 token 截断、reward=-1，不构成能力收益。平台 done，Pod/入口/归档均退出 0；终态 Pod 驻留 2.053333 GPU·h。86 项冻结 CPU 检查及归属正反例通过。范围仅同一 worker 本地文件系统上的受控冷进程恢复，未验证节点丢失、HDFS 原始 checkpoint、跨机或不中断训练逐位等价。该 SID 独立归档；同路径重复任务 a0f976754ea54fc4 已由本线程停止，额外驻留约 0.422222 GPU·h，见 [清理记录](grpo-recovery-v2/duplicate-cleanup.json)；本线程另行核验了 [归档入口日志的逐字节归属](grpo-recovery-v2/run-provenance-fingerprint.json)。

**GRPO v4 终态：三轮真实更新验收通过，passed=true。** [报告](grpo-v4/REPORT.md) / [独立验收](grpo-v4/verified.json) / [任务](https://ml.bytedance.net/development/instance/jobs/daef6d92f62242e4)。每轮候选批数为 2/3/1，共 96 条，拒绝 48 条、录取 48 条，完整身份、版本、概率及 learner 映射核验通过。三轮非零 advantage token 为 18224/9895/6676，梯度为 0.3125/0.1729/0.3868；两个 TP rank 参数与 Adam moments 改变，前两轮引擎分别改变 207/202 个模型 tensor，两个副本每次一致。平台 done，Pod/入口/归档均退出 0。33 项 CPU tests、8 份源码与 IPv6 patch 哈希通过；四卡 Pod 驻留约 0.911 GPU·h，包含启动。末轮无 eval，依照上游跳过最终引擎发布；不宣称能力或吞吐收益。本次限定四卡验证已完成。

**GRPO v3 终态：两轮真实更新与改变后的权重回传已核验，末轮因无 reward 方差被拦截。** [报告](grpo-v3/REPORT.md) / [验收](grpo-v3/verified.json)。梯度 0.6337、0.2140，非零 advantage token 2782、11678；两台引擎各次校验和一致，两次实际改变 206、205 个 tensor。第三批全部负分，未进入 learner，因此完整三轮 passed=false。21 项 CPU tests 通过。后续 v4 已实现并验证有界补采样，见上方最新结果。

**GRPO v2 已完成独立验收：进程成功，学习失败。** [完整报告](grpo-v2/REPORT.md) / [验收 JSON](grpo-v2/verified.json)。任务 `2c78b85806cd2090`，trial `438336563`，pod `5390177759`：平台 done / job_success，Pod、入口、归档均退出 0；三轮共 48 条样本，全部在 2048 token 截断，reward、advantage、梯度均为 0。两台 rollout 引擎在版本 0/1/2 的每次发布中校验和一致，但模型权重没有变化；learner 仅 companion 版本和 optimizer step 计数推进。`grpo_verified=false`。

IPv6 初始 broadcast 修复已在此次 GPU 链路验证，5 份归档源码及 patch 前后哈希一致。归档 SHA256=`df4863cf5b9fe0691910a01881d0c85944ab8c53c091b4e7d73bb40648c4846d`。这证明同步流程执行成功，尚未证明有真实学习更新可供同步。

完整 DAPO 提示同时写了 `Answer: \boxed{...}` 和 `Answer:`，不能说没有 boxed 要求。v2 的 `deepscaler` 要求思考结束标记及 boxed 答案；48 条响应都在结束思考之前截断。候选 `--rm-type dapo --reward-key score` 的 64 对正确/错误 fixture 已通过 CPU 评分协议检查；对 v2 真实 48 条回答重评分仍全部 -1。仅更换评分器不足以产生组内 reward 方差。

**GRPO v1 终态与修复**：任务 `9a2f2d0a2c88b986` / trial `438326241` / pod `5390124078` 已 failed，Pod 与入口退出码均为 1、归档退出码为 0。UTC timeline：created 22:09:53.870、start_run 22:10:23.245、job_failed 22:19:57.054。模型校验和 checkpoint 转换完成、两台 SGLang 引擎 ready；首轮权重同步在 `broadcast.py::connect_rollout_engines_from_distributed` 失败：裸 IPv6 拼进 `tcp://{master_address}:{master_port}`，`urllib.parse` 报 `Port could not be cast to integer value`。后续 Gloo connection closed、Ray worker died 是退出链条中的现象，不能据此称为 OOM。归档 SHA256=`a8ecd87d933576b8db8eb94f5a6336a9bb7ee9fcc1eaad4bd5a329f837bd725a`，3 份内嵌源码哈希核验一致，仅 16 env_report、2 engine_env_report，无 rollout/训练事件。[v1 验收](grpo-v1/verified.json) passed=false。

[IPv6 补丁](https://github.com/Cheongroach/training-infra-lab/blob/main/training_infra_lab/posttraining_runtime/broadcast-ipv6.patch) 复用 Miles 的 `wrap_ipv6`，TCP rendezvous URL 加方括号、选择地址对应 socket family，传给 SGLang 的地址保留裸格式。已核对 SGLang 固定 commit 的 `WeightUpdater.init_weights_update_group` 使用 `NetworkAddress.to_tcp()`，接收端无需相同修复。7 项测试覆盖 IPv4、裸/括号/带 scope 的 IPv6、原始故障、重复 patch 拒绝和真实 CPU IPv4/IPv6 TCPStore；这些不等于 GPU broadcast 通过。[patch_broadcast_ipv6.py](https://github.com/Cheongroach/training-infra-lab/blob/main/training_infra_lab/posttraining_runtime/patch_broadcast_ipv6.py) 只在一次性任务容器中修改镜像内源码，要求 base commit 和原文件 SHA256 完全匹配，并记录修改后哈希。

v2 保持 v1 的同一训练入口哈希、镜像、模型、数据、卡数和时限，仅增加 IPv6 补丁。14 项预检 success，无 warn/hint；保留式合并 `request_context.precheck_sid=82bbc976-4196-4f9a-8e03-b84c67fe0aa4` 后于 30 分钟内提交。Qwen3-4B 的 10 文件共 8,060,896,856 bytes，逐一校验上游 LFS SHA256 / Git blob SHA1，Worker 再查 SHA256；清单见 [model-manifest.json](model-manifest.json)。入口上限 2700 秒 + 110 秒终止/归档宽限，转换 600 秒、训练 1800 秒；排队启动另计。所有提交均以 error 阻止，warn/hint 展示后可继续。

[verify_smoke.py](https://github.com/Cheongroach/training-infra-lab/blob/main/training_infra_lab/posttraining_runtime/verify_smoke.py) 核验完整组、有限概率、样本 index/tokens/length/行为概率映射、非零 advantage/梯度、TP2 两个 rank 的完成状态、optimizer/参数哈希及两台引擎版本/权重。14 项 CPU fixture 测试通过；实际 v2 被拒绝，原因是无有效学习信号和无模型权重变化。上游无末轮评测时跳过最后一次 engine 发布，验收预期为三轮 learner step、初始同步与前两轮更新同步。

v3 四卡任务 `73c87741b20db5a8` / trial `438282983` / pod `5389805624` 已 **failed，退出码 255**。UTC 时间：创建 21:01:45.810、平台 start_run 21:06:46.109、job_failed 21:07:22.354；Pod 运行区间 21:06:24–21:07:30。约 300.3 秒启动等待与约 36.2 秒平台运行时间需要分开统计。没有真人 stop 事件。stdout/stderr URL、TLS 的 std/sandbox/pre_check_log/hdfs_fuse、Pod 配置、HDFS 输出均未取得有效运行证据；Kelemetry 仅根 span 无事件。因此 **不能从 255 推断 GRPO/CUDA/NCCL 错误**，也没有探针通过结果。原始记录在 [job-v3/observations](job-v3/observations)。

进一步查询平台镜像要求并通过 v4 构建检查原 v3，确认原镜像没有 tiger 用户、sudo 和 `/opt/tiger`、`/opt/log/tiger`、`/var/log/tiger`，`/root` 为 700。它不满足 Arnold 启动前提；缺少 v3 的运行日志，不能将它直接写成该次退出的已证实唯一原因。依据：[镜像准入规则](https://bytedance.larkoffice.com/wiki/Bn1HweJqKiUrjokAWH9cpnchnMf)与 [image-build-v4.log](image-build-v4.log)。

[Dockerfile.platform](https://github.com/Cheongroach/training-infra-lab/blob/main/training_infra_lab/posttraining_runtime/Dockerfile.platform) 在 v3 上补齐 tiger UID 1000、sudo 和运行目录，只对镜像内公开源码路径开放遍历；`runuser -u tiger` 已通过 sudo、可写目录、源码读取和 Miles commit 检查。v4 VID `dbdt1t71enl9fha0bn40`、build `170530`、Bytebuild `182818386` 已发布，LF 状态 OK；CPU 构建检查不代替 GPU 验证。Torch/SGLang/Ray/TE 元数据未变，Miles 发行版元数据在 tiger 下显示 0.1.2，而 v3 root 构建输出为 0.1.3，运行探针已确认实际导入 `/root/miles/miles/__init__.py` 且源码为目标 commit，发行包元数据并非源码身份依据。

新的 [job-v4](job-v4) 为同配置四卡预检，已提交 [0b220fc10762ca42](https://ml.bytedance.net/development/instance/jobs/0b220fc10762ca42)，trial `438304391`、pod `5389958077`；平台 14 项预检通过，预检 SID `8fcf11a6-0107-415d-93ed-2a4ceebf2677` 已保留式合并并于有效期内提交。任务已成功：timeline 为 job_success、Pod 退出 0、入口和归档退出 0。下载归档后由 [verify_preflight.py](https://github.com/Cheongroach/training-infra-lab/blob/main/training_infra_lab/posttraining_runtime/verify_preflight.py) 独立核验所有源码哈希、依赖报告、四 rank 和数值契约，[verified.json](job-v4/verified.json) 返回 passed=true。增加每个模块导入前的进度快照、入口 tee 日志、HDFS 挂载标记核对和启动标记；归档的 mkdir/tar 一并纳入时间限制。HDFS 输入标记已从本机上传并读回验证，避免把容器本地同名目录误当远程挂载。本地源码/shell 语法、payload hash、缺依赖拒绝验证通过；归档核验器的缺失 rank/源码篡改拒绝路径也已用合成 fixture 验证。

本次实际运行栈：Python 3.12.15、PyTorch 2.14.1+cu130、Transformers 5.19.0、Ray 2.58.1、TE 2.17.0+miles、FlashAttention 2.7.4.post1（FA3 interface 亦可导入）。SGLang commit `3a938213c1b349e1a8df0b5fdc57ef36a41c743b`，Megatron commit `2fbe25a442a539b2796cdf40e77dfb2c90782529`。四 rank 均为 NVIDIA H20、NCCL 2.30.7、all-reduce=10、BF16 loss=128、梯度逐元素等于 2/128。平台 byted-wandb/pip 初始化报错没有阻止用户入口；训练阶段是否依赖这类平台集成仍需实际检查。`ibv_devinfo` 缺失，跨机/RDMA 未验证。整个归档 SHA256=`13186f85a38e5ba0164b0cec541ac3bf8499157ae31b39b6b436d3230eff2854`。

[run_smoke.py](https://github.com/Cheongroach/training-infra-lab/blob/main/training_infra_lab/posttraining_runtime/run_smoke.py) 是 v1/v2 已执行的三轮同步 GRPO prepare/train 入口，保留失败学习验收的原始配置以便复现。Qwen3-4B revision=`1cfa9a7208912126459214e8b04321603b3df60c`，固定 DAPO 前 64 题数据 revision=`2e65612930298bde4c5d58fd97b3f23a483aaff9`。配置 2 learner TP2 + 2 rollout TP1、4 prompts×4 answers/批、2048 response 上限。已追加 train dump、audit events 和 per-rank 权重/optimizer SHA256 输出。进程退出零仍需审计完整组、reward、行为概率、非零学习信号、optimizer 和权重版本。

以下为之前准备过程的历史记录，以本节最新状态为准。

## 当前环境与资源证据

本轮 `control-plane=cn` 查询配额组 `2079`，资源接口返回：

| 集群 | 队列 | 当时 H20 余量 |
|---|---|---:|
| cloudnative-hl，20 | compute-124-hl-cloudnative-ai-search.douyin.life.nlp-guarantee | 16 |
| cloudnative-lf，22 | compute-124-lf-cloudnative-ai-search.douyin.life.nlp-guarantee | 4 |

查询条件为 `{"page_size":30,"filter":{"group_sid":"2079"}}`，响应 request ID 为 `202610221050024628ECB8D4279B8D51FE`。这些是聚合容量快照，不保证调度成功、排队时长或节点互联。资源查询返回的型号 key 为 `NVIDIA-H20`；早先 `NVIDIA_H20` 的过滤查询为空，不能据此判断没有卡。

既有镜像 VID `d7iu6hbc77u4ebjdshsg` 的实际描述为 Python 3.11、torch 2.9.1、CUDA 12.9。响应 request ID 为 `20261022104955FB001DB180D0BDF74F7C`。它不是所选 Miles 固定版本的已验证镜像。

镜像名称搜索命中 `miles` 和 `sunabako-native-miles-runner-huochen` 两个非公开镜像对象，分别为 `d611a03c77u3t7qp09sg`、`db73tl71enlausgh8un0`。名称命中不证明依赖匹配或拉取权限。前者版本查询在 35 秒后超时；后者返回 HTTP 403、`21605 ImagePermissionDenied`，request ID 为 `20261022105219DA83031609FA4FD6C7D1`。原始查询结果见 [环境查询证据](../research/posttraining_20261022/environment-discovery.json)。权限拒绝不作为 token 失效处理。本轮未创建训练任务，也未以不匹配镜像替代。

## 运行方式

将 `environment_probe.py` 与 `run_environment_probe.sh` 一起放入已经安装匹配 Miles/SGLang/Megatron 的镜像，分配四张 H20：

```bash
bash run_environment_probe.sh /path/to/results
```

第一阶段只检查依赖和源码 revision，最多 120 秒；失败时退出，禁止进入 GPU 阶段。第二阶段四 rank NCCL 求和、BF16 matmul/backward，最多 180 秒。每阶段超时后最多额外 15 秒清理，总执行上限约 330 秒，不含平台启动和结果归档。

依赖源码默认要求 Miles `2233d795dcf167be489d1859971c9233393f04c3`。SGLang、Megatron 的可识别 Git revision 与包路径会被记录，但尚未建立它们的匹配 revision 锁文件；环境探针通过也不能代替依赖版本组合验收。

每个 rank 保存独立 JSON，启动器检查四份报告后才发布成功摘要。错误报告保留缺失模块、CUDA/collective 错误。探针不安装包、不下载模型、不读取或输出认证环境变量。

NCCL 日志、GPU 驱动、拓扑和 RDMA device 列表用于后续核对。单节点求和成功只能证明该环境下 collective 正确，不能证明跨机 RDMA 或 GPU Direct 正常。

## 后续门禁

确认可用的匹配镜像后先运行本探针；再固定模型/数据 revision，并执行 2 learner + 2 rollout 的三次真实 GRPO 更新，验收完整 group、reward、行为概率、非零学习信号、optimizer 状态与权重版本回传。环境成功和训练成功必须分开报告。

真实作业提交继续使用 Merlin 预检：最终 payload 直接包装为 `job_run_context`；`data.status=error` 时停止，`warn/hint` 展示后继续；保留合并 `request_context.precheck_sid`，在 30 分钟有效期内提交。目前没有最终镜像 VID、没有创建 payload、没有预检 SID，也没有实际作业 SID。

## 专用镜像构建与任务准备（2026-10-23）

当前已创建私有镜像 `zhangyang-exp-miles-h20-preflight`，SID `dbd5h7f1enl62v3c94hg`，构建版本 VID `dbd5hi71enlcujgf99j0`、build ID `170427`。最新本轮查询显示 China-North-LF 为 BUILDING，尚未可运行。[构建日志](https://cloud.bytedance.net/bytebuild/log/182393037)。

[Dockerfile](https://github.com/Cheongroach/training-infra-lab/blob/main/training_infra_lab/posttraining_runtime/Dockerfile)固定公开 Miles amd64 基础镜像 digest `sha256:cb2a5f35a7de0915cb05fbdb0d2e7a7be0b495eb35c71ef3f10192401bcf18b0`，保留编译依赖，再固定 Miles 源码。该组合是兼容性候选，没有完整上游 release-lock，不能称为已验证配套版本；构建时执行基础 import，真机探针记录实际依赖。

两个旧私有 Miles 镜像均已返回 ImagePermissionDenied；当前账号可用 SGLang 名称搜索为空，见 [最新查询](image-query-20261023.json)。因此转为构建自己拥有权限的镜像，不修改他人镜像权限。

[job-v1/create.json](job-v1/create.json)是自包含四卡环境任务，LF cluster=22、group=2079，入口执行上限 420 秒加最多 20 秒清理。无模型下载、无训练；模型闭环仍是下一阶段。镜像 READY 和资源重新核对后，按同一 payload 执行平台预检，无 error 再提交。目前只完成任务准备，未获得任务 SID。
