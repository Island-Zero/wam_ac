# 目录结构

```text
wam_ac/
├── code/                         # WAM-AC 自有代码
│   └── base_adaptive_2_10/       # 当前基础 Adaptive 2/10 实现
├── repos/                        # 第三方源码，不提交到本仓库
│   └── FastWAM/                  # 从 Fast-WAM 官方仓库自行克隆
├── models/                       # DiffSynth/Fast-WAM 依赖的本地模型文件
└── outputs/                      # 实验输出与日志
```

Fast-WAM 的 RoboTwin checkpoint 推荐放在：

```text
repos/FastWAM/checkpoints/fastwam_release/
├── robotwin_uncond_3cam_384.pt
└── robotwin_uncond_3cam_384_dataset_stats.json
```

RoboTwin 仿真代码位于 Fast-WAM 的 `third_party/RoboTwin/`，仿真 assets 应按 Fast-WAM/RoboTwin 官方说明放入其 `assets/` 目录。仅运行 inference 不需要训练轨迹数据集，但需要 checkpoint、dataset statistics、RoboTwin 代码和仿真 assets。

当前可运行方法请查看：

```text
code/base_adaptive_2_10/README.md
```

该方法的脚本默认按上述相对结构寻找 `repos/FastWAM`，也可以通过环境变量 `FASTWAM_ROOT` 和 `DIFFSYNTH_MODEL_BASE_PATH` 覆盖路径。
