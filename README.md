# WAM-AC

WAM-AC 是面向 World Action Model 的 training-free 自适应计算项目。

RoboTwin 可运行代码放在 [`wam_ac/code`](wam_ac/code)，LIBERO 模块放在 [`libero_adaptive_2_10`](libero_adaptive_2_10)；外部依赖、模型和运行输出只保留约定目录，不提交第三方源码或大文件。

推荐从仓库根目录开始：

```bash
git clone https://github.com/Island-Zero/wam_ac.git
cd wam_ac
```

详细目录约定与当前方法入口见 [`wam_ac/README.md`](wam_ac/README.md)。

单卡运行 RoboTwin clean 50 tasks × 20 seeds 的 Full-10/Adaptive 配对评测：

```bash
bash wam_ac/code/paired_clean_50x20/run.sh
```

完整层级如下，注意仓库内部还有一层同名的 `wam_ac/` 项目目录：

```text
wam_ac/                            # Git 仓库克隆目录
├── README.md
└── wam_ac/                       # 项目目录
    ├── code/
    ├── repos/
    ├── models/
    └── outputs/
```

## LIBERO

已跑通 Fast-WAM 官方 LIBERO 评测后，按 [LIBERO README](libero_adaptive_2_10/README.md) 设置资源路径。支持 `adaptive`、`10-step` 和 `2-step`：

```bash
bash libero_adaptive_2_10/scripts/smoke.sh
bash libero_adaptive_2_10/scripts/evaluate_2000.sh --detach
```

正式评测默认每种方法 2,000 episode。上述路径均相对于 Git 仓库根目录。
