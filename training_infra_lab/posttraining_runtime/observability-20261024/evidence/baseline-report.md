# Ray 提交修复验证：70d36e11

**Ray 提交、三轮真实更新和最终 version 3 发布通过；完整回答协议验收未通过，整体 `passed=false`。** [任务](https://ml.bytedance.net/development/instance/jobs/70d36e11e929c5c3) 已 `done`，trial `438559238`、Pod `5390937143`，Pod/入口/归档均退出 0。[独立验收](verified.json) 的 `semantic_audit.passed=true`、`publication_probe.passed=false`。本次只创建一个平台任务、只提交一个训练 Job，没有自动重跑。现有四张 H20 足够，无需协调额外资源。

修复包含最长 120 秒的 1 CPU / 0 GPU Ray Jobs 完成检查，固定 Ray build/session 和随机日志标记匹配后才放行训练。实际检查在 **2.794 秒**内 `SUCCEEDED`，随后唯一训练 Job `miles-5a16f9827ae5453389ff23ec71f08ffa` 提交并完成。Miles launcher 只清理确认提交成功的 Job，清理失败保留原始异常，在 Ray 停止前保存日志；本轮没有 `No available agent` 或清理错误。`dashboard_agent.log` 记录 07:20:32.144 注册路由、07:20:33.198 启动 CPU 检查、07:20:35.368 启动训练。旧失败任务缺少可用 agent 日志，因此**不能把本次成功写成旧任务底层原因已被证明**；CPU fixtures 验证了精确 500 等待和清理失败路径，本轮证明门禁后的真实提交与训练可执行。详细冻结条件见 [PROBE_CONTRACT.md](PROBE_CONTRACT.md)。

原模型、数据、镜像和四卡配置保持：Qwen3-4B revision `1cfa9a7208912126459214e8b04321603b3df60c`，Miles `2233d795dcf167be489d1859971c9233393f04c3`，Ray 2.58.1 / `4c7892ef427b19e7585ee28f548a60ab241929e8`，镜像 VID `dbdt1t71enl9fha0bn40`；LF cluster 22 / group 2079，4 H20、32 CPU / 128 GiB，2 learner TP2 + 2 rollout TP1。

三轮共生成 5 个候选批次、80 条训练候选，拒绝其中 32 条、48 条入训。两个 learner TP rank 的参数与 Adam moments 每轮实际变化，两台 rollout 引擎每次 raw tensor 校验和一致，最终 version 3 发布核验通过。训练候选允许原冻结规则下的截断样本；最终回答另按全部自然 EOS 的严格条件检查。

| rollout | 候选批数 / 拒绝批数 | 非零 advantage token | gradient norm | 发布版本 |
|---|---:|---:|---:|---:|
| 0 | 2 / 1 | 18,525 | 0.466318 | 1 |
| 1 | 2 / 1 | 15,488 | 0.446185 | 2 |
| 2 | 1 / 0 | 6,367 | 0.198871 | 3 |

四条冻结提示均使用 temperature 0、4096 token 预算，全部自然 EOS `151645`，行为概率有限，version 3 spans 连续覆盖全部响应，共 8,528 个响应 token。四条都有闭合非空 boxed 表达式，但第 2、3 条未通过冻结 DAPO 提取一致性；官方评分正确数 **0/4**。

| probe index | 响应 token | boxed 值 | DAPO pred | 协议检查 |
|---|---:|---|---|---|
| 0 | 2,212 | 54 | 54 | 通过 |
| 1 | 1,480 | 13 | 13 | 通过 |
| 2 | 4,062 | -1 | 空 | 未通过 |
| 3 | 774 | 5 | 空 | 未通过 |

失败原因已用原始响应及冻结 scorer 重评分确认：[格式分析](answer-format-analysis.json)。probe index 2、3 结尾为 `Final Answer:` 后换行，再写显示公式 `$$` 与 boxed 数值；默认 DAPO 模式使用 `(?i)Answer\s*:\s*([^\n]+)`，捕获 `$$` 后归一化为空。这使两个 `pred` 不可解析，且与 boxed 值不一致。现有提示本已要求末行 `Answer: \boxed{...}`，模型没有遵循；增加 GPU 不解决这一输出协议问题。保留原评分器、提示和严格失败结果，未切换 strict-box 模式或改写响应。下一阶段应单独验证输出格式约束或评分协议调整，不能覆盖本次冻结验收。

[归属审计](run-provenance.json) 通过：归档 `entry.log` 原始字节完整、唯一地存在于实际 Pod stdout，已排除入口源码回显；提交、终态 run、trial 的入口 SHA256 一致。安全解包、11 份源码、三份 runtime patch 和诊断文件哈希均在 PT 读取前核验，15 份 auditor 冻结哈希未变。[归档审计](archive-audit.json) 共 186 个成员、16,965,002 字节，SHA256 `6fc2d0e69cd0dd76ecf1e5ce5da86f6fabe64b2224b518c3436ef65edaea9eb7`；readiness/launcher/archive 三份快照各 40 个限定 Ray 日志，前两份 session 与就绪证明一致。

125 项 CPU 检查通过：86 项既有检查、23 项完整回答检查、16 项 Ray HTTP / 异常保留 / 日志检查；64 条真实 tokenizer prefix、内外 shell 和内嵌 Python 语法通过，见 [cpu-validation.json](cpu-validation.json)。14 项平台预检全部 success，无 warn/hint；SID `8933396b-0823-40d3-b4de-eef06d92a897` 保留式合并到 `request_context.precheck_sid` 并在 30 分钟内提交。预检 error 停止，warn/hint 展示后可继续。新 HDFS namespace 标记已写入并读回核对。

[终态成本](cost.json)：UTC 创建 07:06:52.918、平台 start_run 07:12:03.995、job_success 07:29:38.944；创建到平台运行 311.077 秒，平台运行 1,054.949 秒。按终态 Pod `start_at=07:07:51`、`stop_at=07:29:39`，驻留 1,308 秒，**1.453333 GPU·h**，包含启动且不等于 GPU 算子活跃时间。旧失败任务 `585440c23b867d70` 的 `passed=false` 与 0.671111 GPU·h 保留；更早 staging 取消任务成本仍未知。此前冷恢复成功证据独立保留。本轮不宣称模型能力、吞吐或冷恢复收益。
