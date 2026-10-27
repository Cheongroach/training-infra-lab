# 05 权重发布与策略版本消费审计

## 诊断问题

learner 更新后的参数是否完整到达所有 rollout 引擎，并被下一批生成 token 实际使用？

## 输入参数

| 参数 | 含义 |
|---|---|
| update ID | learner 更新身份 |
| publication UUID | 一次权重传输身份 |
| policy version | 发布前后版本 |
| learner tensor list | 名称、dtype、shape、bytes、digest |
| engine identity | rollout 引擎身份 |
| engine tensor digest | 接收内容 |
| `weight_versions.spans` | 每条响应 token 的实际版本区间 |

## 操作步骤

1. 把 learner 更新结果绑定到唯一 publication UUID。
2. 检查发送 tensor 的名称、dtype、shape 和字节覆盖完整。
3. 检查每个 rollout 引擎的发布确认。
4. 比较相同 TP 分片的引擎 tensor digest。
5. 拒绝只凭 `published_version` 整数判断成功。
6. 对下一批每条回答计算响应 token 的绝对区间。
7. 检查版本 span 无空洞、无重叠，并覆盖全部响应 token。

## 输出结果

train-05 完成 v0→v1→v2→v3 发布。8 个 TP2 引擎每次内容一致。最终 16 条回答的全部响应 token 均消费版本 3。

恢复实验中，保存前版本 2 与恢复后重新发布的版本 2，全部 290 个引擎 tensor 变化数为 0；第三次更新后的版本 3 改变 202 个 tensor。

## 最终诊断结论

训练更新、权重发布和后续策略消费形成闭环。发布 RPC 成功、版本数字增加、单个引擎 ack 都不是充分条件；完整内容和 token 级版本覆盖共同通过才可下结论。

## 原理说明

策略版本是元数据，参数 digest 是内容身份，token span 是实际使用证据。三者分别约束“应该发布什么”“实际收到什么”“后续用了什么”。缺少任一层都可能出现版本正确但内容错误，或内容已到达但旧请求仍在消费旧版本。

## 注意事项

- TP0 与 TP1 是不同切片，只在对应切片间比较。
- 历史事件重放必须按 UUID 和时间去重，不能当作新发布。
- pseudo-hash 不能替代完整 raw tensor SHA256。
- 最后一轮没有 eval 时，上游可能不发布最终权重，必须按真实事件报告。
