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
