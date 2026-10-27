# Recovery v2：四卡冷进程恢复通过

保留任务 `172e5c51a339edef` 完成并通过独立验收。详细、按任务 SID 隔离的读者版见：

- [恢复结果](../grpo-recovery-v2-172e5c51/REPORT-friendly.md)
- [恢复契约](../grpo-recovery-v2-172e5c51/RECOVERY_CONTRACT-friendly.md)

本目录用于保留当时线程的提交、去重和 collector 适配证据。

核心结论：

- checkpoint 约 52.46 GiB；
- 两个 learner TP rank 的模型、Adam、scheduler 和 RNG 指纹恢复一致；
- 数据游标和全部候选 journal 恢复一致；
- 两个新 rollout 引擎加载完整 version 2；
- 恢复后第三轮产生非零梯度并改变参数与 Adam；
- version 3 发布并被四条最终生成消费。

四条最终生成均在 512 token 截断，所以这次只证明恢复、发布和版本覆盖，不证明答题成功。

原始线程报告见 [REPORT.md](REPORT.md)。
