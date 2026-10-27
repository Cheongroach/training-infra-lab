# 02 源码与产物归属审计

## 诊断问题

归档结果是否来自目标任务提交时的源码，而不是其他任务、后续修改或同名 HDFS 目录？

## 输入参数

| 参数 | 含义 |
|---|---|
| 提交源码 manifest | 文件相对路径与 SHA256 |
| worker 归档 | 任务生成的 `artifacts.tar.gz` |
| JobRun / trial / Pod 身份 | 平台执行实体 |
| 入口脚本 SHA256 | 提交、终态 run、trial 三方比较 |
| stdout 原始字节 | 运行归属匹配 |
| HDFS marker | 输出 namespace 与任务身份 |

## 操作步骤

1. 在提交前生成逐文件 SHA256 manifest。
2. worker 启动后把实际执行源码和同一 manifest 写入归档。
3. 收集时先检查 tar member：拒绝绝对路径、`..`、符号链接和未知顶层目录。
4. 安全解包后重新计算每个源码文件 SHA256。
5. 比较提交 manifest、归档 manifest 和解包文件。
6. 比较提交、终态 run 和 trial 中的入口脚本。
7. 将归档 `entry.log` 的字节序列与目标 Pod stdout 匹配，排除启动时源码回显。
8. 对复用 HDFS 路径的任务，再用时间、hostname、publication UUID 和策略版本绑定归属。

## 输出结果

输出包括：

- `submitted_source_verified: true/false`
- bundle 与 artifact SHA256
- 每个源码文件的核验结果
- 安全解包成员数和总字节数
- 入口脚本三方一致性
- Pod stdout 归属匹配结果
- HDFS marker 读回结果

## 最终诊断结论

全部核验通过时，可以说明“报告基于本次任务实际执行的固定源码和归档”。核验失败时，只能说明证据归属未闭合，不能继续读取 `.pt` 或用同名目录中的结果下结论。

## 原理说明

文件名、目录名和 Job 名称都不是内容身份。SHA256 约束内容，run/trial/Pod 和时间区间约束执行实体，两者共同建立“这份代码由这个任务执行并产生这份结果”的链路。

## 注意事项

- 在来源核验前不要反序列化 `.pt`。
- 相同源码的两个任务仍是两个不同实验。
- manifest 只证明列入清单的文件；清单本身也必须冻结。
- 大模型权重不进入 Git 仓库，只保存不可变 revision 和逐文件摘要。
