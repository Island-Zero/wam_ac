# WAM-AC

WAM-AC 是面向 World Action Model 的 training-free 自适应计算项目。

仓库中的可运行代码统一放在 [`wam_ac/code`](wam_ac/code)；外部依赖、模型和运行输出只保留约定目录，不提交第三方源码或大文件。

推荐从仓库根目录开始：

```bash
git clone https://github.com/Island-Zero/wam_ac.git
cd wam_ac
```

详细目录约定与当前方法入口见 [`wam_ac/README.md`](wam_ac/README.md)。

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
