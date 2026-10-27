# 07 阶段计时与吞吐口径

## 诊断问题

时间消耗发生在哪个阶段，吞吐数字的设备范围和计时范围是否一致？

## 输入参数

| 参数 | 含义 |
|---|---|
| response token count | 实际训练回答 token |
| rollout seconds | 生成阶段最慢或定义好的墙钟 |
| actor train seconds | learner 训练阶段 |
| log-prob seconds | 概率重算阶段 |
| publication seconds | 权重发布阶段 |
| GPU count by role | learner / rollout 数量 |
| Pod residency | 全任务时间 |

## 操作步骤

1. 为 rollout、learner wait、log-prob、actor train、publication 建立不重叠或明确可嵌套的计时范围。
2. 记录每个阶段的 token 工作量。
3. 对多 rank 阶段使用定义好的最大墙钟，不用单 rank 累计时间冒充全局墙钟。
4. 分别计算阶段吞吐和全任务资源吞吐。
5. 标注 profiler start/stop、warmup、初始化和归档是否包含在分母中。
6. 禁止把不同设备范围或时间范围的吞吐相除为加速比。

## 输出结果

train-05 三轮共 272,254 个训练回答 token：

- rollout 阶段：241.819 response token / rollout GPU·s；
- 全部 Pod 驻留：6.035 admitted response token / 全部 GPU·s。

前者只使用 16 张 rollout GPU 的生成时间；后者使用 32 张 GPU 的完整驻留时间，包含准备、训练、发布和最终评估。

## 最终诊断结论

两个数字都可复算，但回答不同问题。`241.819 / 6.035` 不能解释为 40 倍优化。性能结论必须固定相同 workload、设备集合和计时边界。

## 原理说明

吞吐是“工作量 / 资源时间”。改变工作量定义、GPU 数量或计时范围，就改变了指标。NSYS kernel 累计时间还可能跨 stream 重叠，也不能直接相加成墙钟。

## 注意事项

- rank0 host timer 不等于全 rank 最大墙钟。
- response token、非零 advantage token、prompt token 是不同工作量。
- profiler 运行不能作为无 profiler 基线。
- 阶段时间可嵌套时，必须说明，不能相加。
