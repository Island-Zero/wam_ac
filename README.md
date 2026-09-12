# WAM-AC

基于预测动作一致性的 training-free 自适应去噪。本仓库提供 Fast-WAM 在 LIBERO 上的 `adaptive`、`10-step` 与 `2-step` 评测。

## 快速开始

先跑通 Fast-WAM 官方 LIBERO 测评，再复用同一 Python 环境、checkpoint 和数据资源。

```bash
git clone https://github.com/Island-Zero/wam_ac.git
cd wam_ac
cp libero_adaptive_2_10/configs/default.json libero_adaptive_2_10/configs/local.json
```

按 [LIBERO README](libero_adaptive_2_10/README.md) 修改 `local.json` 中的资源路径，然后执行：

```bash
# 三种方法各 1 episode 的 smoke
bash libero_adaptive_2_10/scripts/smoke.sh

# 三种方法各 2,000 episode，默认单路后台运行
bash libero_adaptive_2_10/scripts/evaluate_2000.sh --detach
```

详细方法、配置、结果查看和续跑说明见 [LIBERO 使用说明](libero_adaptive_2_10/README.md)。模型权重、外部依赖、本机配置和运行输出不提交到仓库。

## LIBERO 结果

| 方法 | Spatial | Object | Goal | Long | 成功率 | 平均 NFE | 加速 |
|---|---:|---:|---:|---:|---:|---:|---:|
| 2-step | 96.60% | 99.00% | 96.40% | 92.40% | 96.10% | 2.000 | — |
| adaptive | 97.20% | 99.40% | 95.40% | 94.40% | 96.60% | 3.238 | 2.24× |
| 10-step | 96.00% | 99.40% | 96.60% | 94.80% | 96.70% | 10.000 | 1.00× |

每种方法评测 40 tasks × 50 episodes，共 2,000 次；每个子集 500 次。加速为单路动作块 wall-clock 平均耗时之比：10-step 529.09 ms，adaptive 236.20 ms。计时包含观测预处理、video DiT 编码、action DiT 去噪及动作后处理，不包含环境步进。2-step 尚未测量单路加速，故留空。NFE 按动作块加权，不能直接换算成实际加速倍数。
