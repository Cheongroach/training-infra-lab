# Ray 提交修复与四卡完整回答验收

本次为用户授权的一次修正验证，不自动重试训练。旧失败任务 `585440c23b867d70` 的证据与 `passed=false` 保留。

固定 Miles `2233d795dcf167be489d1859971c9233393f04c3`、Ray 2.58.1 / `4c7892ef427b19e7585ee28f548a60ab241929e8`、Qwen3-4B revision `1cfa9a7208912126459214e8b04321603b3df60c`、64 条 DAPO 数据 SHA256 `67173a6074167cb280f1cf70542ccd1c1f91583defa8698f2428d268858e4fff`、镜像 VID `dbdt1t71enl9fha0bn40`。LF cluster 22 / group 2079，4 H20、32 CPU、128 GiB，2 learner TP2 + 2 rollout TP1。

Ray 启动后先执行最长 120 秒的 CPU Jobs 检查，进程上限 135 秒。唯一 submission ID 始终先 GET 核对，再提交打印随机校验标记的任务，申请 1 CPU / 0 GPU。只允许这个无训练任务在精确 `500: No available agent` 或临时传输失败时等待和核对；SUCCEEDED、固定 Ray build/session 与日志标记全部匹配才放行。训练提交没有重试。未确认提交成功时不清理不存在的训练 Job；训练与清理同时报错时保留训练原始异常。Runtime patch 必须匹配固定 commit 和原始文件 SHA256。只保存限定 Ray 日志文件尾部，每个快照最多 64 文件 × 256 KiB，不覆盖旧快照。

训练与回答标准沿用[原验收条件](../grpo-publication-4096-4fdcc7793e37/PROBE_CONTRACT.md)：三轮完整 4×4 GRPO 分组，有限行为概率、非零 advantage / 梯度、TP2 参数与 Adam moments 实际变化、两个 rollout 引擎权重一致；最终 version 3 发布。四条冻结提示各生成一个回答，temperature 0、non-thinking prefix、4096 token 预算，全部自然 EOS、闭合非空 boxed 答案、冻结 DAPO scorer 重评分一致、version 3 spans 连续覆盖响应。错答案单独记正确数，截断或错版本均拒绝；不宣称能力、吞吐或冷恢复收益。

新入口内嵌 11 份源码，15 份 auditor 文件冻结。125 项 CPU 检查包括 86 项既有检查、23 项完整回答检查、16 项 Ray HTTP / 清理 / 日志检查；64 条真实 tokenizer prefix、内外层 shell 和内嵌 Python 语法通过。CPU 检查不能替代本次 GPU 验收。

入口上限 2700 秒、转换 600 秒、训练及就绪检查合计上限 1800 秒。安全解包、来源哈希、三份 runtime patch 和诊断文件哈希核对必须先于任何 PT 读取；同时核对平台 run → trial → Pod 日志归属、Pod exit、timeline 和驻留 GPU·h。预检 error 停止，warn/hint 展示后可继续；保留式合并 `request_context.precheck_sid` 并于 30 分钟内提交。
