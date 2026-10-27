# Recovery v1：状态恢复成功，启动比较错误

## 目标

完成两轮训练后保存 checkpoint，关闭 Ray 和训练进程，再从新进程恢复第三轮，并发布最终 version 3。

## 已经做到的部分

- 前两轮有非零梯度；
- checkpoint 包含 23 个文件，约 56.33 GB；
- 新进程恢复到 `next rollout = 2`；
- 数据游标恢复到已消费 80 条候选；
- 两个 rollout 引擎恢复为训练后的 version 2；
- 引擎完整 tensor checksum 与第一阶段发布的 version 2 一致。

## 为什么仍然失败

Miles 启动检查把恢复后的 version 2 与原始 Hugging Face version 0 比较。

```text
正确恢复的训练权重
≠ 原始未训练权重
→ 初始化比较报错
→ 第三轮训练未开始
```

差异约 `3.81e-6`，这是训练后应当存在的变化，不是恢复损坏。

## 结论

本轮证明 checkpoint 和数据游标确实被加载，也证明 rollout 引擎拿到了保存前的训练权重。但由于缺少恢复更新前两个 learner rank 的完整指纹，不能宣称模型、Adam、scheduler 和 RNG 已完成独立验收。

修复方向不是放宽容差，而是让恢复路径与 checkpoint 中最后提交的训练权重比较，同时保留普通启动路径与原始 HF 权重比较。

完整任务和错误证据见 [原始报告](REPORT.md)。
