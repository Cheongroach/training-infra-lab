# 全量指标与数据离线页面

解压后用 Chrome / Edge / Firefox 打开 `index.html`，保持 `assets`、`_shared`、`evidence` 与页面的相对路径。页面无需联网、启动服务或安装依赖。

当前汇总两次四卡 H20 任务的已归档数据：

| 数据 | 覆盖 |
|---|---:|
| 指标序列 | 230 条 |
| 指标原始采样点 | 27,064 个 |
| NSYS 数据表 | 63 张 |
| NSYS 原始行 | 827,115 行 |
| Rollout / eval 样本 | 104 条 |
| 结构化运行事件 | 108 条 |
| 可浏览数据集 | 71 个 |

历史基线 run：`70d36e11e929c5c3`；诊断 run：`49b6f8781696e68e`。数据采集于 2026-10-24。

1. **全部指标**：按任务、类别、名称和 rank 搜索；最多同时选择 4 条曲线，单位不同时分图。时间范围同时作用于图表、统计与逐点表，所选数据可导出 CSV。完整 230 条指标的 JSON 在来源与下载页。
2. **NSYS 全部表**：选择任意表，搜索某个字段或全表；分页、页码跳转、显示全部列。点击“详情”查看该行全部字段；导出保留全部字段和当前筛选结果。无需先加载全部 82 万行。
3. **训练与原始数据**：查看两次任务的全部事件、prompt / response / reward / token / log-prob、训练张量字段、配置与验收。单元格是摘要，记录详情和 JSON 导出保留完整数组。
4. **来源与下载**：查看全部数据集的原始列数、行数和 SHA256；原始 NSYS 报告、SQLite、四卡 CSV 一并保留。原始 `.nsys-rep` 可以用 Nsight Systems GUI 打开。

所有 SQLite 原始列都保留；`resolved_name` 通过 StringIds 映射，`relative_start_s` 相对选定 NVTX 起点，`duration_us` 由原始 start/end 计算。超出 JavaScript 安全范围的整数保留为十进制字符串。页面中的环境与认证记录隐藏了敏感认证值；所有数值指标保留。

NCU 本轮筛选未命中实际 `nvjet_sm90` kernel，没有有效计数器。页面明确展示缺失状态和原始错误。CPU 周期与上下文切换采样关闭，但实际 OS runtime / callchain 数据仍可浏览。未采集线上服务 SLO 或多 rank NSYS 对照。

完整回答验收仍为 `false`；实际参数更新和 version 3 发布通过。这些 profiling 数据用于诊断，性能收益需要相同 workload 的独立对照。

页面支持 UTC / UTC+8；CSV / JSON 导出全部在浏览器本地完成。打印布局可使用浏览器的打印功能。
