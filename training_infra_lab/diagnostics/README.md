# 性能检测与系统诊断方法

本目录把项目中已经实际使用的方法拆成独立步骤。每个步骤只回答一个诊断问题，并固定记录输入、输出、结论、原理和边界。

建议顺序：

1. [任务生命周期取证](steps/01-job-lifecycle.md)
2. [源码与产物归属审计](steps/02-provenance-integrity.md)
3. [回答、奖励与轨迹审计](steps/03-trajectory-reward.md)
4. [分布式参数与优化器审计](steps/04-optimizer-state.md)
5. [权重发布与版本消费审计](steps/05-policy-publication.md)
6. [checkpoint 恢复审计](steps/06-checkpoint-recovery.md)
7. [阶段计时与吞吐口径](steps/07-stage-timing.md)
8. [GPU-hour 成本核算](steps/08-resource-cost.md)
9. [Grafana 常驻指标诊断](steps/09-grafana-metrics.md)
10. [Nsight Systems 时间线诊断](steps/10-nsys-trace.md)
11. [Nsight Compute 定点诊断](steps/11-ncu-kernel.md)
12. [权重发布 A/B 实验](steps/12-publication-ab.md)
13. [Ray Jobs readiness 诊断](steps/13-ray-readiness.md)
14. [输出格式与评分协议诊断](steps/14-output-protocol.md)

这些步骤遵循同一原则：程序成功、数据完整、数值正确、性能改善和模型质量是不同结论，不能互相替代。
