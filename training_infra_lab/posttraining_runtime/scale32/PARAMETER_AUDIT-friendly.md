# 32 卡训练：如何证明一次更新真的完成

## 拓扑

Qwen3-8B 使用 32×H20：

- 16 learner rank：TP2 × DP8；
- 16 rollout rank：8 个 TP2 引擎。

一次完整更新的证据链是：

```text
完整 prompt group 和 reward
→ 8 个 DP shard 每条样本恰好消费一次
→ 16 learner rank 产生 gradient
→ 模型参数与 Adam m/v 变化
→ 发布新版本
→ 8 个引擎内容一致
→ 下一批 response token 实际消费新版本
```

## 不同并行维度该怎样比较

TP0 和 TP1 保存不同参数切片，hash 不同是正常的。验证 DP 副本一致性时，应固定 TP rank，再比较 8 个 DP 副本。

Adam 可能由 distributed optimizer 分片。每个 rank 只需证明自己拥有的 `exp_avg` 和 `exp_avg_sq` 存在、有限并发生合理变化；不能要求所有 DP rank 的 optimizer 字典逐字相同。

## 常见现象如何诊断

| 现象 | 合理解释或下一步 |
|---|---|
| reward 全相同，advantage 全零 | 当前 batch 没有 GRPO 学习信号 |
| gradient 为零，参数仍变化 | 可能来自旧 momentum 或 weight decay |
| gradient 非零，参数和 Adam 不变 | 检查 LR=0、AMP skip、冻结或 optimizer 未执行 |
| Adam 变化，BF16 参数 hash 不变 | 更新可能小于 BF16 舍入分辨率 |
| 固定 TP 后 DP hash 不一致 | 副本可能分叉，不能只看 rank 0 |
| learner 更新，引擎 hash 不变 | 权重发布断链 |
| 引擎内容一致，token 仍用旧版本 | 路由或暂停/恢复断链 |

版本号、optimizer step 和 ack 都只是元数据，不能替代内容检查。

## loss 接近零不等于没有梯度

PPO/GRPO 的组内 advantage 均值常接近零。初始 ratio 接近 1 时，平均 surrogate loss 可以接近零，但每项对 log-prob 的梯度仍与 advantage 成正比。

因此判断一次更新要联查：

- 非零 advantage token；
- gradient norm；
- optimizer outcome；
- 模型参数；
- Adam m/v。

## 如何审计真实运行

使用 `audit_train32.py` 对归档重新计算：

```bash
python training_infra_lab/posttraining_runtime/scale32/audit_train32.py \
  --results training_infra_lab/posttraining_runtime/scale32/runs/train-05/shared-results \
  --code training_infra_lab/posttraining_runtime/scale32/runs/train-05/node-0/code \
  --source-manifest training_infra_lab/posttraining_runtime/scale32/runs/train-05/source-manifest.json \
  --output /tmp/train32-recomputed.json
```

先验证源码 hash，再读取本实验自有的 `.pt` 归档。

## 如何读发布性能

普通和 packed broadcast 必须发送完全相同的：

- tensor 名称；
- dtype 和 shape；
- 字节数；
- bucket 顺序；
- 最终模型、Adam 和引擎内容。

局部发布比值定义为：

```text
median(普通 broadcast 墙钟) / median(packed broadcast 墙钟)
```

它只描述权重发布阶段，不能推导训练 token/s 或收敛速度。全量 checksum 还会扫描 learner 模型、Adam 和引擎分片，必须与纯传输时间分开。

完整练习和公式见 [原始参数审计](PARAMETER_AUDIT.md)。
