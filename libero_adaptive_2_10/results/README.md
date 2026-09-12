# LIBERO 评测结果

| 方法 | Spatial | Object | Goal | Long | 成功率 | 平均 NFE | 加速 |
|---|---:|---:|---:|---:|---:|---:|---:|
| 2-step | 96.60% | 99.00% | 96.40% | 92.40% | 96.10% | 2.000 | — |
| adaptive | 97.20% | 99.40% | 95.40% | 94.40% | 96.60% | 3.238 | 2.24× |
| 10-step | 96.00% | 99.40% | 96.60% | 94.80% | 96.70% | 10.000 | 1.00× |

每种方法评测 40 tasks × 50 episodes，共 2,000 次；每个子集 500 次。加速为单路动作块 wall-clock 平均耗时之比：10-step 529.09 ms，adaptive 236.20 ms。计时包含观测预处理、video DiT 编码、action DiT 去噪及动作后处理，不包含环境步进。2-step 尚未测量单路加速，故留空。NFE 按动作块加权，不能直接换算成实际加速倍数。

`existing_2000_episode_results.csv` 的成功率字段为 0–1 比例；`existing_task_outcomes.json` 保存 120 个任务结果；`provenance.json` 保存来源标识和原始文件 SHA-256。
