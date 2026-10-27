# 12 权重发布 A/B 实验

## 诊断问题

在发送内容完全相同的条件下，packed broadcast 是否稳定快于普通 broadcast，checksum 又带来多少额外开销？

## 输入参数

| 参数 | 设置 |
|---|---|
| fixed model state | 第三次真实更新后的同一权重 |
| learner ranks | 16 |
| rollout engines | 8 个 TP2 |
| sender bytes | 16,381,470,720 |
| methods | `broadcast` / `broadcast_packed` |
| checksum | on / off |
| repetitions | 2 warmup；四种配置各 4 次 |
| order | 平衡顺序 |

## 操作步骤

1. 固定模型和 optimizer 状态。
2. 两种方法使用同一 NCCL 连接和同一引擎集合。
3. 比较 tensor 名称、dtype、shape、bytes 和 bucket 顺序。
4. 每种配置完成 warmup 后进行四次正式测量。
5. 墙钟取 16 个 learner rank 最大值。
6. checksum-off 在计时区间外执行完整内容核验。
7. 汇报全部样本、中位数和 min/max，不只报最佳值。

## 输出结果

| 配置 | 中位墙钟 |
|---|---:|
| broadcast / checksum off | 0.530577 s |
| packed / checksum off | 0.522645 s |
| broadcast / checksum on | 30.282540 s |
| packed / checksum on | 30.262081 s |

checksum-off 的局部比值为 1.015177，两组范围明显重叠。checksum 增量约 29.75 s；传输区间中位仍约 0.52 s。

## 最终诊断结论

当前证据不支持 packed broadcast 有稳定收益。全量内容诊断是发布墙钟的主要开销，优先级高于继续优化约 0.52 s 的传输阶段。

## 原理说明

A/B 实验只有在工作量、连接、引擎、布局和初始状态相同时才能归因。四次同任务重复可显示局部波动，但不足以建立跨任务稳定收益。checksum-on 还包含模型、Adam、引擎分片 hash 和屏障，不是网络传输。

## 注意事项

- 1.015× 不能写成端到端训练加速。
- min/max 重叠时不应只报告中位优势。
- checksum-off 仍需在计时外验证内容，避免性能与正确性失联。
- 发布占完整 step 的比例未测时，不能计算整体加速上限。
