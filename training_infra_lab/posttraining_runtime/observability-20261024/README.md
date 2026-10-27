# 四卡 GRPO 监控与 Nsight 诊断

直接打开 [observability-20261024.html](observability-20261024.html)。解压交付包时保留 assets、_shared 和 evidence 的相对路径；无需安装依赖或联网，图表可缩放、切换逐卡指标、筛选 kernel，底部可下载原始报告。

- [REPORT.md](REPORT.md)：本次数据、诊断结论、失败边界和下一步。
- [NSYS .nsys-rep](evidence/artifacts/results/nsys/learner-rank0-6626.nsys-rep)：用 Nsight Systems GUI 打开完整时间线。
- [NSYS SQLite](evidence/artifacts/results/nsys/learner-rank0-6626.sqlite)：只读分析，数据表结构在 evidence/nsys-schema.json。
- [逐卡 1 秒 CSV](evidence/artifacts/results/gpu-1s.csv)：四卡利用率、显存、功耗、温度与时钟。
- [诊断汇总](evidence/profile-summary.json) 与 [独立验收](evidence/verified.json)：数据均来自实际作业，回答完整验收 false。
- [NCU 原始失败日志](evidence/artifacts/results/ncu-capture.log)：实际 nvjet kernel 未命中 gemm 字符串，没有有效 NCU 计数器。
- [NCU 修正脚本](https://github.com/Cheongroach/training-infra-lab/blob/main/training_infra_lab/posttraining_runtime/observability-20261024/scripts/capture_ncu_projection.py)：下次在 GPU 环境执行；本次未追加重跑。
- [浏览器验证](evidence/browser-validation.json)：桌面、手机、打印与交互检查结果。

日常用法：先看有效训练信号与评测，再对齐阶段和逐卡资源曲线；异常时打开 NSYS 查 CUDA API / memcpy / kernel / stream，选少量代表 kernel 做 NCU。任何性能收益都需要相同 workload、未开启 profiler 的独立对照。

这个页面是已采集数据的快照，不会自动轮询任务。scripts/observe_job.py 的平台查询依赖本机 merlin-cli 认证；build_data.py 需要工作区保留的历史基线。离线包展示不依赖这些脚本或完整张量归档。
