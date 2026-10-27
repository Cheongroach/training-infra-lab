# Miles 四卡 GRPO v4：三轮真实更新验收通过

任务 [daef6d92f62242e4](https://ml.bytedance.net/development/instance/jobs/daef6d92f62242e4)，control-plane=cn，trial=438459211，pod=5390706384。**平台 done / job_success；Pod、入口、归档均退出 0；独立验收 [verified.json](verified.json) passed=true，errors=[]。** 三轮非零学习信号、真实参数更新、改变后的权重回传及全部候选批身份核验通过。

保持 v3 镜像 VID `dbdt1t71enl9fha0bn40`、Qwen3-4B 模型 revision、DAPO 64 题及 seed，LF cluster 22/group 2079，4 H20/32 CPU/128 GiB，2 learner TP2 + 2 rollout TP1。关闭 thinking、DAPO score、4096 response 上限与三轮训练不变。

| rollout | 候选批 / 拒绝批 | 录取批完成 / 提取 | 正分 / 混合组 | 非零 advantage token | grad_norm |
|---|---:|---:|---:|---:|---:|
| 0 | 2 / 1 | 16 / 14 | 2 / 2 | 18224 | 0.3125005 |
| 1 | 3 / 2 | 14 / 14 | 1 / 1 | 9895 | 0.1728782 |
| 2 | 1 / 0 | 13 / 12 | 3 / 1 | 6676 | 0.3867532 |

每个录取批均为完整 4 组×4 回答。共生成 6 批/96 条，拒绝 3 批/48 条，录取 3 批/48 条；三个拒绝批均仅因无组内 reward 方差。sample index 0–95 无重复，录取 index 为 16–31、64–79、80–95。全部发样/返回身份、单输出 lineage、行为版本及跨度、有限概率和批次决定重新计算通过；三个录取批与训练前 rollout 全字段一致，并逐样本核对 learner 的 tokens、长度、行为概率，以及 loss mask 尺寸与有效 advantage。全部 response 共 193198 tokens，拒绝批占 95306；这些是补采消耗，不是吞吐指标。

两台 rollout 引擎每次 290 个 tensor 校验和一致，版本 0→1→2；前两次更新分别改变 207、202 个实际模型 tensor。两个 learner TP rank 各有 292 个参数哈希，step 0→1 各改变 202 个实际模型参数，step 1→2 各改变 203 个，Adam moments 同步改变；首轮非空 moments 及引擎 version 0→1 的真实权重变化补足首次更新证据。三个 optimizer step 的两个 rank 均正常完成。无末轮 eval 时，上游跳过最终引擎发布，故 `final_update_published=false`；已证明三轮 learner 更新、初始同步及前两轮改变后的权重回传，未声称末轮也回传。[diagnosis.json](diagnosis.json) 保留完整计数。

每步最多四个候选批，每批 4 题×4 回答。每批先完整归档再判断，完成与答案提取均至少 8/16、至少一个组内 reward 方差非零；通过批原样进入 learner。无学习信号时保持 rollout ID 与策略版本、推进数据源补采；概率、样本身份、单输出 lineage 或权重版本覆盖错误立即停止。单个候选批也限制数据源只发一批，防止上游吞掉生成异常后无限补发。最多 12 批/192 个回答，另受 1800 秒训练与 2700 秒入口上限约束。

[bounded_rollout.py](https://github.com/Cheongroach/training-infra-lab/blob/main/training_infra_lab/posttraining_runtime/bounded_rollout.py) 接入 Miles class-based rollout；[qualification.py](https://github.com/Cheongroach/training-infra-lab/blob/main/training_infra_lab/posttraining_runtime/qualification.py) 保存完整候选批及决定。[artifacts.tar.gz](artifacts.tar.gz) 内 `results/qualification/candidate-{step}-{attempt}.json` 保留全部样本、发样身份与版本跨度；`step-{step}.json` 保存哈希、决定与理由。上游 sample ownership checker 沿用关闭状态；自有归档承担补采筛选审计，其他显式丢样事件仍拒绝，本轮没有此类事件。审计源码与 CPU tests 已冻结在 [verifier](verifier/verify_smoke.py)，哈希见 [audit-tools-manifest.json](audit-tools-manifest.json)。

33 项 CPU 测试通过：覆盖四批耗尽、非有限概率、旧策略、身份复用、来源及单输出 lineage、跨度缺失、候选归档缺失、拒绝理由伪造和录取内容篡改等路径。本次 GPU 验证了实际拒绝后补采、完整批次录取与三轮更新；四批耗尽和格式/概率错误中止路径仍属于 CPU fixture 验证。复查 v3 历史三批分别通过、通过、仅因无 reward 方差可重采。

归档 4205201 bytes，SHA256=`255e8d43a5ab60ab32e83833ea9c042ace2fa07141c85af3f9f999bb204f17fb`。8 份提交源码和 IPv6 patch 前后哈希一致，见 [archive-audit.json](archive-audit.json) / [source-manifest.json](source-manifest.json)。HDFS 标记上传并读回一致。[14 项预检](precheck-result.json) success，无 warn/hint；保留合并 `request_context.precheck_sid=0e552bdf-3c0c-4d9e-ba89-819401ed35f5` 后在 30 分钟内提交，[最终请求](checked-request.json) 已归档。error 停止，warn/hint 展示后可继续。

[run](terminal-run.json)、[trial](terminal-trial.json)、[pod](terminal-pods.json)、[exit-info](exit-info.json)、[timeline](timeline.json) 已收集。UTC created=02:25:45.306、start_run=02:29:08.763、job_success=02:41:37.924；启动等待 203.457 秒、平台运行 749.161 秒。Pod 区间 02:26:43–02:40:23，四卡驻留估计 0.911 GPU·h，包含启动，非有效计算卡时或计费结果。原始展开日志与观测位于 `/tmp/miles-grpo-v4-final`。本任务已结束，无继续提交。

此次通过的是限定四卡的同步训练闭环。筛选后分数不能代表无条件准确率；同一配置与 seed 也不保证并发采样逐条复现，不能据此宣称能力、吞吐或单变量收益。跨机 RDMA、规模扩展、恢复/异步路径及末轮权重发布未验证。
