# 32 卡 Qwen3-8B GRPO：结果读者版

## 研究问题

1. 16 learner + 16 rollout 的多机布局能否完成真实 GRPO 更新？
2. learner、8 个 rollout 引擎和每条生成 token 的策略版本能否闭合？
3. packed broadcast 是否比普通 broadcast 更快？
4. 重型参数审计本身有多大开销？

## 配置

- 4 个 worker，每个 8×H20；
- Qwen3-8B；
- learner：TP2 × DP8，共 16 卡；
- rollout：8 个 TP2 引擎，共 16 卡；
- 每轮 16 个 prompt，每个 4 条回答；
- 三次 optimizer update。

所有结论来自终态任务和独立审计。暴露题评估只验证最终版本消费，不用于能力提升。

## 训练结果

train-05 完成三次真实更新：

| 轮次 | 回答 | 自然结束 | 严格格式 | 有 reward 差异的组 | 非零 advantage token | grad norm |
|---|---:|---:|---:|---:|---:|---:|
| 0 | 64 | 61 | 26 | 7 | 36,008 | 0.296 |
| 1 | 64 | 64 | 12 | 5 | 25,736 | 0.206 |
| 2 | 64 | 63 | 18 | 8 | 44,074 | 0.247 |

16 learner rank 的模型参数和本地 Adam moments 每轮变化；固定 TP rank 后，8 个 DP 副本一致。8 个 TP2 rollout 引擎内容一致，v0→v3 发布通过，最终 16 条回答的全部 response token 消费 version 3。

这证明分布式训练和发布链路完成。严格格式通过率仍低，因此不能写成回答协议已经解决。

## 吞吐必须看分母

三轮共有 272,254 个训练 response token。

- rollout 阶段：241.819 response token / rollout GPU·s；
- 全部 Pod 驻留：6.035 admitted response token / 全部 GPU·s。

前者只计算 16 张 rollout GPU 的生成计时；后者使用 32 张卡的完整驻留，包含准备、训练、发布和评估。两者相除不能解释为 40× 加速。

## 权重发布 A/B

publication-01 固定第三次更新后的同一权重，比较普通和 packed broadcast。每种配置四次平衡重复。

| 配置 | 墙钟中位数 |
|---|---:|
| broadcast，checksum off | 0.530577 s |
| packed，checksum off | 0.522645 s |
| broadcast，checksum on | 30.282540 s |
| packed，checksum on | 30.262081 s |

checksum-off 的 packed/普通比值为 1.015。两组 min/max 区间明显重叠，没有稳定性能优势。

实际传输区间约 0.52 秒。打开全量 checksum 后墙钟增加约 29.75 秒，主要增加的是模型、Adam 和 8 个引擎分片的内容扫描与屏障，不是 NCCL 传输。

## 失败实验提供了什么信息

- `format-01`：入口过大，shell `Argument list too long`；
- `format-02`：Transformers 返回 `BatchEncoding`，生成前失败；
- `train-03`：Ray worker 端口与平台 GCS 冲突；
- `train-04`：严格格式门禁错误地保留 decoded EOS，256 条回答均未进入训练；
- `train-05`：把格式统计与训练准入分开后成功。

训练准入要求完整 group、自然结束、有限概率和 reward 差异；格式失败仍以原负 reward 留在组中。否则过滤掉所有负样本会改变 GRPO 信号。

## 资源成本

所有终态扩展任务合计 54.7611 H20·h，包括准备、失败和重算。成功任务：

- train-05：12.5311 H20·h；
- publication-01：19.4844 H20·h。

这些是 Pod 驻留成本，不是 GPU kernel 活跃时间或平台账单。

## 可以得出的结论

- 32 卡同步 GRPO 执行和版本链路通过；
- 参数、Adam、引擎内容和 token 消费版本均可独立审计；
- packed broadcast 在本次局部实验中没有稳定收益；
- 全量 checksum 是本次发布墙钟中的主要诊断开销；
- 输出格式仍是独立未解决问题。

没有节点丢失恢复、独立 held-out 质量、RDMA 带宽或端到端优化收益结论。

## 证据入口

- [参数审计友好版](PARAMETER_AUDIT-friendly.md)
- [实测练习友好版](WORKED_EXAMPLE-friendly.md)
- [执行计划友好版](SCALE32_PLAN-friendly.md)
- [机读汇总](results-summary.json)
- [原始完整报告](REPORT.md)
