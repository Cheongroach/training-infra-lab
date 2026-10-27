# 11 Nsight Compute 定点诊断

## 诊断问题

选定 GPU kernel 的 SM、DRAM、occupancy、cache 和 stall 指标是否真正被采集？

## 输入参数

| 参数 | 含义 | 初次设置 |
|---|---|---|
| target command | 隔离 BF16 投影 | 固定矩阵形状 |
| kernel filter | kernel 名匹配 | `regex:.*gemm.*` |
| launch skip/count | 跳过预热、限制采样 | 修正脚本为 skip 5 / count 3 |
| metric set | 计数器集合 | `basic` |
| output | `.ncu-rep` 与 raw CSV | 必须非空 |

## 操作步骤

1. 从 NSYS 找到真实热点 kernel 名称。
2. 用 NCU kernel filter 限定少量代表 launch。
3. 检查命令 return code。
4. 检查日志是否包含 `No kernels were profiled`。
5. 检查 `.ncu-rep` 存在且非空。
6. 导出 raw CSV，确认请求的 metric 实际存在。
7. 保存工具版本、矩阵形状、dtype 和控制选项。

## 输出结果

初次采集 return code 为 0，但日志明确写：

```text
No kernels were profiled
```

原因是实际 kernel 名为 `nvjet_sm90...`，不匹配 `.*gemm.*`。因此没有有效 `.ncu-rep`，也没有 SM、DRAM、occupancy 或 stall 计数器。

修正脚本改用实际 `nvjet_sm90` 名称，并增加报告、错误文本和 raw metric 三重检查；修正后的 GPU 采集尚未执行。

## 最终诊断结论

本次 NCU 计数器采集失败，不能报告任何 kernel 硬件效率结论。工具退出码 0 不是采集成功的充分条件。

## 原理说明

NCU 只对过滤器命中的 kernel 注入硬件计数器。命令本身成功启动但没有命中 kernel 时，进程可正常退出。必须同时验证命中数量、报告文件和实际 metric。

## 注意事项

- 隔离 GEMM 不能替代真实训练 kernel。
- `basic` set 不一定包含完整 cache/stall 指标。
- NCU 开销很高，只采少量 launch。
- kernel 名可能随 PyTorch、CUDA 和编译策略改变。
