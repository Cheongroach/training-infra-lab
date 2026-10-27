# 01 任务生命周期取证

## 诊断问题

任务是否真正开始运行、在哪个 trial 和 Pod 上执行、怎样结束，以及日志来自哪个实体？

## 输入参数

| 参数 | 含义 | 示例 |
|---|---|---|
| `control_plane` | Merlin 控制面 | `cn` |
| `job_run_sid` | JobRun 唯一标识 | `49b6f8781696e68e` |
| `trial_sid` | 一次实际 trial | 由 JobRun 返回 |
| `trial_pod_sid` | 日志和退出码对应的 Pod | 由 trial 查询返回 |
| `stream` | 实时日志流 | `stdout` / `stderr` |

## 操作步骤

1. 用 JobRun SID 查询终态和 `runtime_context.trial_id`。
2. 用 trial SID 枚举 Pod，保存 Pod SID、名称、开始时间、停止时间和退出码。
3. 用 Pod SID 获取实时 log-proxy 日志；URL 为空时改查 TLS 归档。
4. 查询 timeline，区分 `job_created`、`start_run`、`job_success`、`job_failed`、人工 stop、超时和抢占事件。
5. 用 exit-info 交叉核对 trial 状态与 Pod 退出码。

## 输出结果

标准输出是一份结构化记录：

```text
JobRun SID
trial SID
Pod SID / Pod name
JobRun status
Pod exit status
job_created / start_run / terminal timestamps
stdout / stderr evidence source
```

项目中的成功运行同时要求平台终态成功、Pod exit 0、入口退出 0、归档复制退出 0。失败运行保留最先出现的原始异常，不能被后续清理错误覆盖。

## 最终诊断结论

该步骤只能判断任务的执行边界和终止方式。它可以区分：

- 排队尚未获得 Pod；
- 已启动但用户入口未执行；
- 用户程序失败；
- 平台超时或抢占；
- 正常完成。

它不能单独判断梯度是否非零、参数是否更新或模型质量是否提高。

## 原理说明

平台资源是 `run → trial → Pod` 三层结构。JobRun 表示用户提交，trial 表示一次实际运行尝试，Pod 才是日志和退出码的归属单位。只查询 run 状态会丢失实际执行实体；只读取日志又可能把其他 trial 或历史 Pod 的记录误认为本轮证据。

## 注意事项

- UTC 和本地时区必须统一后再计算时间差。
- 实时日志 URL 是临时签名资源，不归档 URL 本身，只归档下载后的脱敏内容。
- `done`、`success`、Pod status 和 trial status 的枚举不同，应保存原始字段并显式映射。
- 日志缺失不能自动解释为任务没有运行。
