# Miles 四卡 GRPO v3：两轮更新通过，末轮被资格检查拦截

任务 [6bd6e2df40142fef](https://ml.bytedance.net/development/instance/jobs/6bd6e2df40142fef)，control-plane=cn，trial=438444081，pod=5390665130。同一 v4 镜像、Qwen3-4B revision、DAPO 64 题、LF cluster 22/group 2079，4 H20/32 CPU/128 GiB，2 learner TP2 + 2 rollout TP1。

**已证明两次有效 learner 更新及改变后的权重回传；完整三轮目标尚未通过。** 第 2 轮全部负分、无组内 reward 方差，训练前检查主动抛出 `GRPO_ROLLOUT_GATE_REJECTED: no within-group reward variation`。平台 failed、Pod/入口退出 1、归档退出 0。[verified.json](verified.json) passed=false，仅有 driver 未完成、末轮 gate 拒绝、末轮 train dump 缺失三项拒绝。

| rollout | 完成 / 提取 / 总样本 | 正分 | 混合 reward 组 | 非零 advantage token | grad_norm | learner |
|---|---|---:|---:|---:|---:|---|
| 0 | 15 / 15 / 16 | 1 | 1 | 2782 | 0.6337099 | 已更新 |
| 1 | 16 / 16 / 16 | 1 | 1 | 11678 | 0.2139573 | 已更新 |
| 2 | 12 / 11 / 16 | 0 | 0 | — | — | 未进入 |

两台 rollout 引擎每次 290 个 tensor 校验和一致，版本 0→1→2；两次更新分别改变 206、205 个 tensor。learner 两个 TP rank 从第 0 到第 1 轮各有 205 个实际模型参数哈希改变，Adam moment 哈希也改变。前两轮的概率、样本映射、advantage、梯度、模型参数、optimizer moment 和引擎校验和核验通过。[diagnosis.json](diagnosis.json) 保留完整计数。

本轮使用 DAPO score、关闭 thinking、4096 response 上限及每批训练前资格检查。原提示同时包含 `Answer: \boxed{...}` 和 `Answer:`，不是完全没有 boxed 要求；deepscaler 要求思考结束标记，而 v2 回答全部在结束思考之前截断。实际数据的关闭 thinking 前缀已核验。不能将结果归因于单个参数，不能据此宣称能力或吞吐提升。

[run](terminal-run.json)、[trial](terminal-trial.json)、[pod](terminal-pods.json)、[timeline](timeline.json) 已收集。UTC created=01:45:16.526、start_run=01:50:12.794、job_failed=02:02:38.864；启动等待 296.268 秒、平台运行 746.070 秒。Pod 区间 01:46:02–02:02:17，四卡驻留估计 1.083 GPU·h，包含启动，非有效计算卡时或计费结果。启动期 Kelemetry 仅取得调度审计，TLS 为空；后来入口正常运行，不能将早期空日志当成已证实启动故障。

[artifacts.tar.gz](artifacts.tar.gz) 为 1,877,188 bytes，SHA256=c8ca176eda0e61dd5c0e6f399b674c3619c183574bd636e548bbb7c027ac2f15；6 份源码及 IPv6 patch 前后哈希核验一致，见 [archive-audit.json](archive-audit.json)。原始展开日志与观测位于 /tmp/miles-grpo-v3-final。14 项预检 success，无 warn/hint；保留合并 precheck_sid=f8dcdb8b-554e-4ee0-bfc5-d069116665c4 后在 30 分钟内提交。error 停止、warn/hint 展示后可继续。21 项 CPU fixture 测试通过，仅证明验收器与拒绝路径。

下一步在同一策略版本下补采候选批，每轮最多 4 批，每批 4 题×4 回答。每批先归档再决定是否进入 learner；被淘汰批次保留理由、sample identity 和版本信息。达到上限仍无有效批则停止。筛选后分数不能代表无条件模型准确率。继续限定四卡，不扩至 32 卡。
