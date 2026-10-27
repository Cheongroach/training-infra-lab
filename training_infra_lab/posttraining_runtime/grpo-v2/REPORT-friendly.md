# GRPO v2：程序成功，但没有学习

## 问题

四卡 Miles 链路是否能完成三轮 rollout、learner 更新和权重发布？

## 观察

任务正常退出，IPv6 broadcast 修复在 GPU 链路生效。三轮各生成 16 条回答，但 **48/48 全部达到 2048-token 上限**，没有结束思考或产生可解析答案。

| 轮次 | reward 均值 | 有 reward 差异的组 | 非零 advantage token | grad norm |
|---|---:|---:|---:|---:|
| 0 | 0 | 0 | 0 | 0 |
| 1 | 0 | 0 | 0 | 0 |
| 2 | 0 | 0 | 0 | 0 |

版本号和 optimizer step 都递增，但实际模型参数、Adam moments 和引擎权重内容没有变化。

## 结论

本轮只证明控制流和通信能够执行。它没有发生有效 GRPO 学习。

根因链是：

```text
回答全部截断
→ scorer 全部给相同 reward
→ 组内 advantage 全零
→ policy gradient 全零
→ 参数和 Adam 不变
```

“任务成功”“版本递增”和“optimizer step 增加”都不能替代参数与 Adam 内容变化。

## 下一步

v3 改用 DAPO scorer、关闭 thinking、提高 response 上限，并在训练前检查组内 reward 是否有差异。

完整任务、成本和归档证据见 [原始报告](REPORT.md)。
