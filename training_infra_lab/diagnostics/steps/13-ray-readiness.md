# 13 Ray Jobs Readiness 诊断

## 诊断问题

Ray runtime 启动以后，Ray Jobs agent 是否已经能够可靠接收训练 Job？

## 输入参数

| 参数 | 设置 |
|---|---|
| timeout | 120 s |
| process bound | 135 s |
| readiness resources | 1 CPU / 0 GPU |
| submission ID | 唯一随机 ID |
| marker | 随机日志校验标记 |
| required identity | 固定 Ray build / session |
| training retries | 0 |

## 操作步骤

1. Ray runtime 启动后，先 GET 唯一 readiness submission ID。
2. 提交一个只打印随机 marker 的 CPU 小任务。
3. 仅对精确 `500: No available agent` 或临时传输失败做有界等待。
4. 要求状态 `SUCCEEDED`、build/session 匹配且日志包含 marker。
5. readiness 通过后只提交一次训练 Job。
6. 未确认训练 Job 提交成功时，不执行 stop。
7. 同时保存 dashboard agent、dashboard 和 raylet 限定日志。

## 输出结果

旧任务在训练 Job 提交时返回 500，随后无条件 cleanup 又返回 404，次生错误覆盖了异常链。

修正任务的 readiness 在 2.794 s 内成功，随后唯一训练 Job 提交并完成三次真实更新，没有再出现 `No available agent`。

## 最终诊断结论

readiness 门禁后的真实 Ray Jobs 提交链路可用。该结果不能反推旧任务的底层原因一定是 agent 注册延迟，因为旧任务缺少对应日志。

## 原理说明

“Ray runtime started”只说明核心进程启动，不保证 Jobs agent 已注册路由并能够接收请求。CPU 小任务验证控制面，不消耗训练 GPU；随机 marker 和 session identity 防止读取到历史 Job。

## 注意事项

- 训练 Job 不做无控制重试。
- 清理只针对确认提交成功的 Job。
- cleanup 失败不能覆盖原始训练异常。
- readiness 成功不替代 learner/rollout 的 GPU 训练验收。
