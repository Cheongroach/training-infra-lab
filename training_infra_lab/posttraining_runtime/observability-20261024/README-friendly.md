# 四卡 GRPO 监控包：如何使用

打开 `observability-20261024.html` 即可离线查看，不需要联网或启动服务。解压后保持 `assets/`、`_shared/` 和 `evidence/` 的相对路径。

## 建议查看顺序

1. 先看训练是否产生有效 reward、gradient、参数和版本变化；
2. 再看 rollout、等待、log-prob、actor train 和发布的阶段时间；
3. 对齐四卡利用率、显存、功耗与阶段；
4. 异常阶段再打开 NSYS；
5. 只对少量已确认的热点 kernel 使用 NCU。

## 主要文件

- [诊断报告友好版](REPORT-friendly.md)
- `evidence/artifacts/results/gpu-1s.csv`：四卡 1 秒资源数据
- `evidence/artifacts/results/nsys/learner-rank0-6626.nsys-rep`：NSYS 原始报告
- `evidence/artifacts/results/nsys/learner-rank0-6626.sqlite`：可查询的 NSYS 数据
- `evidence/profile-summary.json`：结构化诊断摘要
- `evidence/verified.json`：独立训练与回答协议验收
- `evidence/artifacts/results/ncu-capture.log`：本次 NCU 未命中的原始证据

## 使用边界

页面是一次已采集任务的快照，不会自动轮询。Profiler 会改变运行时间，因此不能把诊断任务直接与普通任务比较加速。

本次训练参数更新和 version 3 发布通过，但最终回答协议失败。进程成功、GPU 利用率和模型质量是不同结论。

原始操作说明见 [README.md](README.md)。
