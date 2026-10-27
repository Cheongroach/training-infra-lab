# 10 Nsight Systems 时间线诊断

## 诊断问题

一次 learner 训练阶段中，CPU CUDA API、GPU kernel、内存复制、同步和空白区间如何分布？

## 输入参数

| 参数 | 含义 | 本次设置 |
|---|---|---|
| target process | learner rank | rank 0 |
| NVTX range | 采集阶段 | `grpo/actor_train/rollout=1/rank=0` |
| profiler tool | Nsight Systems | 2026.4.1.191 |
| output | `.nsys-rep` + SQLite | 两种格式均保留 |
| CPU sampling | 周期/调用栈 | 本次关闭 |

## 操作步骤

1. 在目标 actor train 阶段加入唯一 NVTX range。
2. 只对一个代表 rank、一个更新阶段采集，控制 profiler 开销。
3. 导出 `.nsys-rep` 和 SQLite。
4. 从 StringIds 解析 kernel 和 API 名称。
5. 计算 kernel 活动区间并集，不把跨 stream 累计时间当墙钟。
6. 分别统计 memcpy 方向、次数、字节和设备时间。
7. 用源码和 NVTX 对候选瓶颈做交叉验证。

## 输出结果

实际 trace：

- NVTX 范围 39.50 s；
- 35,559 个 kernel；
- 264,679 个 CUDA API；
- kernel 活动区间并集 6.581 s；
- 加 memcpy/memset 后设备活动并集 14.667 s；
- Device→Host 973 次，18.73 GiB，设备复制累计 8.078 s；
- GEMM 累计 5.127 s，Attention 0.720 s，NCCL 0.366 s。

源码显示训练后全量 checksum 路径会把参数和 optimizer state 搬到 CPU。

## 最终诊断结论

全量 checksum 是 Device→Host 大流量的强候选来源，应先在独立性能任务关闭该路径并增加专属 NVTX，再做同 workload 对照。当前没有调用栈证据，不能把全部 18.73 GiB 精确归因于 checksum。GEMM 占累计 kernel 时间较高，但不等于占墙钟 76.6%。

## 原理说明

Nsight Systems 记录时间关系，适合识别 CPU 提交、GPU 活动、复制和同步是否重叠。区间并集回答“至少有设备活动的墙钟”，累计 kernel 时间回答“所有 kernel 持续时间总和”，两者在并行 stream 下不同。

## 注意事项

- profiler start/stop 和 flush 会增加时间，本次范围外差值为 16.616 s。
- 单 rank trace不能代表所有 rank。
- 无 CPU sampling 时，设备空白区间不能完整归因。
- NSYS 定位阶段瓶颈，不提供单 kernel 的完整硬件计数器。
