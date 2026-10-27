# 四卡最终版本完整回答验收：585440c2

**本次执行失败，严格验收 `passed=false`。** 任务 [`585440c23b867d70`](https://ml.bytedance.net/development/instance/jobs/585440c23b867d70)，trial `438544641` / Pod `5390906090`，平台 failed，Pod 与入口退出 1、归档复制退出 0。失败发生在 Ray 提交训练 Job 时，尚未创建 learner / rollout；没有三轮更新、version 3 发布或四条回答。本次没有自动重试，也没有放宽验收。

此前[冷进程恢复验收](../grpo-recovery-v2-172e5c51/REPORT.md)的 `passed=true` 仍按其独立证据保留。这次要补齐的是四条最终版本回答的自然 EOS 与 boxed 答案，目标尚未完成。

## 失败位置与证据

[原始异常摘录](failure-log-evidence.log)记录：

```text
ray_job.py:81 client.submit_job(...)
RuntimeError: Request failed with status code 500:
No available agent to submit job, please try again later..

During handling of the above exception, another exception occurred:
ray_job.py:86 subprocess.run(... ray job stop ..., check=True)
CalledProcessError: ... returned non-zero exit status 1.
ORIGINAL_EXIT_CODE=1 ARTIFACT_COPY_RC=0
```

清理命令先打印 `404: Job miles-efae1d1dbfab49a5a1b0f98eb21a0ffc does not exist`，随后异常链保留了原始提交 500。404 属于提交失败后的清理错误；不能把它当作首先导致任务失败的原因。固定 Miles 源码 `miles/utils/external_utils/ray_job.py::_run_and_stop_job` 在 `submit_job` 的 finally 中无条件停止 Job，解释了异常覆盖路径。

已确认的故障层是 **Ray Jobs 提交服务当时没有可用 agent**。agent 未就绪、注册延迟、进程退出或连接异常等底层原因尚未确认：查询本 Pod 在 UTC 06:34–06:36 的 [DashboardAgent](tls-dashboard-agent.json)、[Dashboard](tls-dashboard.json)、[Raylet](tls-raylet.json) TLS 流均为 0 条。没有据此归因为 CUDA、NCCL、OOM、评分器或 probe patch。平台初始化中的 pip 权限错误也未阻止用户入口与 checkpoint 转换。

| 阶段 | 实际结果 |
|---|---|
| 调度 / 启动 | K8s Bind、Pulling 已记录；随后入口实际执行 |
| 模型资产 | 10 文件逐一校验通过，约 8.06 GB |
| checkpoint 转换 | 双进程保存成功，随后记录 `GRPO_CHECKPOINT_PREPARED` |
| runtime patch | IPv6 与实际 `generate_hub/single_turn.py` EOS 采集补丁均应用并记录哈希 |
| tokenizer / probe 配置 | 64 条闭合 non-thinking prefix；4 条冻结 eval prompt、4096 预算、temperature 0、预期 version 3 已落盘 |
| Ray runtime | 06:34:26.048 记录 runtime started |
| 训练 Job 提交 | 500，无可用 agent；没有成功提交记录 |
| 学习 / 发布 / 探针 | 无事件和 PT 文件，真实更新数与 probe 响应数均为 0 |

上表只证明初始化和配置执行过，不能证明 GPU 上的 EOS 采集、权重广播或 GRPO 学习成功。[独立验收](verified.json)明确拒绝缺失 driver completion、audit events、三轮训练数据及最终 publication/checksum 的结果；collector 的非零退出来自该严格拒绝，下载和安全解包本身成功。

## 配置与冻结边界

Qwen3-4B revision `1cfa9a7208912126459214e8b04321603b3df60c`；64 条 DAPO 数据 SHA256 `67173a6074167cb280f1cf70542ccd1c1f91583defa8698f2428d268858e4fff`；Miles `2233d795dcf167be489d1859971c9233393f04c3`；镜像 VID `dbdt1t71enl9fha0bn40`。LF cluster 22 / group 2079，4 H20、32 CPU / 128 GiB，计划 2 learner TP2 + 2 rollout TP1。完整验收条件见 [PROBE_CONTRACT.md](PROBE_CONTRACT.md)。

本次不重复已有冷恢复约 52.46 GiB checkpoint 保存 / 恢复。训练仍要求三次真实更新；四条 probe 要求原始 EOS、闭合非空 boxed 答案、冻结 DAPO 重评分一致、version 3 spans 连续覆盖全部响应。答案正确数与完成 / 版本验收分开；四条训练内题目不支持能力收益结论。本次没有生成答案，正确率记为未测，不能记作 0/4。

## 归属、源码与 CPU 检查

[归档审计](archive-audit.json)核验安全解包 22 个条目、9 份内嵌源码与两份 runtime patch 前后哈希，先于任何 PT 读取。[运行归属](run-provenance.json)核验归档 `entry.log` 的全部 1,219,576 bytes 连续出现在本 Pod stdout 的真实运行部分，排除启动时回显的代码包；13 份冻结 auditor 文件无漂移。

- 提交、终态 run、trial 的入口脚本完全一致，SHA256 `c2621e26174d2ed9390c9816533c521e031d2b34f33fe2b5612b4f8cd512fccc`。
- [结果归档](artifacts.tar.gz)共 122,112 bytes，SHA256 `467349afcb156bf95dfda1eba5f7bed2be7c9246ae3266c98b38458773689bd9`。
- [86 项既有 CPU tests](cpu-tests.log)通过；[23 项 probe tests](probe-tests.log)通过，包含实际 single-turn eval EOS 采集、train 路径不变与严格拒绝条件。合计 109 项；CPU 检查不代表本次 GPU 验收通过。
- [提交审计](submission-audit.json)：14 项预检 success，无 warn/hint；保留式合并 `request_context.precheck_sid=67f80f55-3490-41a2-a5a4-9bf4a7e4de05`，预检后 26.424 秒创建任务，位于 30 分钟有效期内。规则为 error 停止、warn/hint 展示后可继续。
- 证据来自 cn 控制面、run → trial → Pod 的 log-proxy stdout/stderr，保存在 `observations/`；终态见 [terminal-evidence.json](terminal-evidence.json)、[exit-info.json](exit-info.json)、[timeline.json](timeline.json)。

## 时间与成本

UTC：job_created 06:23:55.738 → start_run 06:29:08.321，创建到平台开始运行 **312.583 秒**；job_failed 06:34:55.623，平台运行 **347.302 秒**。timeline 没有真人 stop、抢占或超时事件。

终态 Pod 起止字段为 06:24:52–06:34:56，共 **604 秒**，四卡驻留 **0.671111 GPU·h**，包含镜像启动、资产校验、转换、失败和清理。[cost.json](cost.json)保留计算依据。这不是 GPU 算子活跃时间，也不是训练吞吐或利用率。

首次提交 [`53bfe898ac2210b3`](../grpo-publication-4096-eab9f5d659ce/REPORT.md)在 staging 因补丁指向旧入口被主动取消，Pod `start_at=null`；其成本未知，单独保留，不能按 0 计。修正后的本次失败成本不与此前成功恢复混用。

## 后续边界

本次限定执行与复盘已收尾，完整回答目标仍待验证。下一次执行前应补齐 Ray dashboard agent 的实际就绪证据及本地日志归档；只看到 `Ray runtime started` 不足以证明 Jobs 可提交。启动等待应有明确期限，提交失败要保留原始 500，清理仅针对确认已提交的 Job。不能以无控制重试替代这项检查，也不能修改本次冻结验收器把失败变成通过。
