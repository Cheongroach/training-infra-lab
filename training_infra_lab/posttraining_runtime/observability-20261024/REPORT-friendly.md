# 四卡 GRPO 监控与 Nsight：诊断读者版

## 研究问题

一次成功退出的 GRPO 作业，时间到底消耗在 rollout、等待、训练、权重校验还是 GPU kernel？常驻指标、NSYS 和 NCU 各自能回答什么？

## 采集内容

- 4×H20，Qwen3-4B；
- 四卡 1 秒 GPU 资源数据，共 3,656 条；
- 历史 Grafana：18 条序列、1,022 点；
- learner rank 0 的 NSYS trace；
- NCU 隔离投影尝试；
- 训练事件、参数更新和最终回答协议。

训练完成三次更新并发布 version 3；最终回答协议仍失败。

## 阶段时间

| 轮次 | learner wait | log-prob | actor train | step total |
|---|---:|---:|---:|---:|
| 0 | 57.22 s | 11.68 s | 48.72 s | 117.66 s |
| 1 | 23.92 s | 1.72 s | 56.11 s | 81.78 s |
| 2 | 42.24 s | 1.60 s | 38.71 s | 82.57 s |

第二轮 actor train 包含 NSYS start/stop，不能与普通运行直接比较。父子 timer 也可能重叠，不能简单相加。

## NSYS 的主要发现

采集区间：

- NVTX actor train：39.50 秒；
- 35,559 个 GPU kernel；
- 264,679 个 CUDA API 调用。

GPU kernel 活动区间并集为 6.581 秒；加入 memcpy/memset 后设备活动并集为 14.667 秒。其余区间只表示此 rank 没记录到设备活动，不能自动称为 CPU 瓶颈。

最值得继续验证的现象是 Device→Host 复制：

- 973 次；
- 18.73 GiB；
- 设备复制累计 8.078 秒。

源码在每次 train step 后遍历模型参数、buffer 和 optimizer state，执行 `.cpu().numpy().tobytes()` 计算 SHA256。这个路径是合理候选原因，但当前 trace 没有调用栈，不能把全部 18.73 GiB 都归因于它。

## kernel 分类

| 类别 | 次数 | 累计 device time |
|---|---:|---:|
| GEMM | 6,369 | 5.127 s |
| Attention | 3,564 | 0.720 s |
| NCCL | 5,633 | 0.366 s |

GEMM 约占累计 kernel time 的 76.6%，但累计 device time 可能跨 stream 重叠，不等于墙钟临界路径占比。

## NCU 为什么没有结果

NCU 命令返回 0，但过滤器 `.*gemm.*` 没有命中实际 `nvjet_sm90...` kernel。日志明确记录 `No kernels were profiled`，因此：

- 没有有效 `.ncu-rep`；
- 没有 SM、DRAM、occupancy 或 stall counter；
- 退出码 0 不能算采集成功。

修正脚本已改用真实 kernel 名，并同时检查报告文件和实际 metric；修正后的 GPU 采集尚未执行。

## 下一步优先级

1. 在独立性能任务中关闭全量 checksum/debug dump；
2. 给 checksum 路径增加专属 NVTX；
3. 保持 workload 相同，比较 step time、有效 token/s 和版本正确性；
4. 再研究 rollout 与 learner 重叠；
5. 阶段归因完成后，才做 kernel 级优化。

输出格式协议也应先修。当前 boxed 与 scorer pred 不一致，更多 GPU 不能解决这个问题。

## 结论边界

本次交付是一套监控视图和一次实际诊断。它没有部署持续在线监控，也没有验证优化收益。

Pod 驻留为 1.5911 GPU·h，包含启动、训练、探针和 profiling，不是 GPU 算子活跃时间。

原始数据、版本和完整口径见 [原始报告](REPORT.md)。
