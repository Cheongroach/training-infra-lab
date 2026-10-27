# 06 Checkpoint 恢复审计

## 诊断问题

关闭原进程后，新进程是否恢复了同一训练状态，并能继续产生有效更新和新策略版本？

## 输入参数

| 参数 | 含义 |
|---|---|
| checkpoint inventory | 文件、字节数和摘要 |
| model / Adam fingerprints | 两个 learner TP rank |
| scheduler state | 学习率进度 |
| RNG fingerprints | Python、NumPy、Torch、CUDA |
| datasource cursor | 已消费数据位置 |
| qualification journal | 录取和拒绝候选身份 |
| committed policy version | 保存时 rollout 权重 |

## 操作步骤

1. 在完整更新边界保存模型、Adam、scheduler、RNG、数据游标和候选 journal。
2. 生成 checkpoint 文件清单并核验完整性。
3. 关闭 Ray、learner 和 rollout 进程。
4. 新进程加载 checkpoint。
5. 在任何新更新前比较 save-after 与 load-return 指纹。
6. 检查游标、已消费身份和候选 journal 一致。
7. 把新 rollout 引擎与 checkpoint 提交版本比较，而不是与原始 HF 权重比较。
8. 执行下一次更新，检查非零梯度、参数和 Adam 变化。
9. 发布下一策略版本并检查 token 消费。

## 输出结果

Recovery v2：

- checkpoint 20 文件，56,331,517,391 bytes（52.4628 GiB）；
- 两个 learner TP rank 的模型、Adam、scheduler 和 RNG 指纹一致；
- 恢复 32 个已消费身份，下一轮为 rollout 2；
- 两个新引擎的 290 个 tensor 与保存的版本 2 一致；
- 恢复后 gradient norm 0.431847；
- 版本 3 发布并被最终回答消费。

## 最终诊断结论

同一 worker、本地文件系统、完整 rollout 边界上的冷进程恢复通过。v1 失败不是 checkpoint 损坏，而是错误地将训练后的版本 2 与原始版本 0 比较。

## 原理说明

恢复正确性由状态身份决定，不由“load 没报错”决定。模型、optimizer、scheduler、RNG 和数据位置共同定义后续训练轨迹。参考状态必须与 checkpoint 提交版本一致。恢复后继续训练和发布用于证明加载状态可实际使用。

## 注意事项

该实验没有验证断电、反向中途崩溃、节点丢失、HDFS 原始 checkpoint 或跨机恢复。结果归档不包含 52.46 GiB 原始 checkpoint，只包含 worker 生成的清单与指纹证据。
