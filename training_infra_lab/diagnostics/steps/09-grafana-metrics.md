# 09 Grafana 常驻指标诊断

## 诊断问题

训练不同阶段的 GPU、显存、CPU、内存和网络资源表现是什么，是否存在持续资源异常？

## 输入参数

| 参数 | 含义 |
|---|---|
| JobRun / trial | 目标任务 |
| Pod names / hosts | 过滤真实物理节点 |
| time window | 使用 Pod 起止时间 |
| dashboard / panel ID | 指标来源 |
| sampling interval | 常见 1–15 秒 |

## 操作步骤

1. 从任务取得 Grafana dashboard。
2. 描述 dashboard，确认 panel ID 的实际含义。
3. 用真实 Pod host 和终态时间窗口查询 panel data。
4. 保存原始响应 SHA256、series 标签、点数和单位。
5. 将指标时间与 rollout、actor train、publication 等阶段对齐。
6. 空响应先检查 host/template 变量，不把空数据解释为零。
7. 多卡指标同时看平均值和逐卡差异。

## 输出结果

publication-01 实际取得：

| Panel | 可读 series | 点数 |
|---|---:|---:|
| 112 | 8 | 1,034 |
| 119 | 20 | 1,310 |
| 7 | 4 | 517 |
| 90 | 8 | 524 |

首次使用 dashboard 自动 `hostnum` 得到空值；显式绑定实际四个物理主机后恢复数据。

## 最终诊断结论

Grafana 适合判断持续的资源饱和、逐卡不均衡和阶段性等待。首次空数据来自查询变量错误，不能解释为 GPU 利用率为零。面板数字必须结合本次 dashboard describe 结果，不能跨看板复用编号含义。

## 原理说明

Grafana 指标由周期采样构成，适合观察秒级趋势，但看不到单个 kernel 和短同步。指标标签决定查询对象；错误 host 过滤会返回空序列，而不是目标资源的零值。

## 注意事项

- GPU Duty、SM、Tensor、HBM 是不同指标。
- 显存总和与每卡峰值不能混写。
- CPU limit 与任务申请 CPU 可能口径不同。
- TCP 为零不能证明没有 NCCL、NVLink 或 RDMA。
