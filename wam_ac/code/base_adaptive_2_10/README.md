# Fast-WAM + RoboTwin 基础版 Adaptive 2/10

本目录提供 WAM control-redundancy 项目中使用的、可独立运行的
**基础版 Adaptive 2/10 router**。

代码不依赖 `wam_control_redundancy_sampling`、`c3ache` 或已有实验输出，唯一的
外部源码依赖是原版 Fast-WAM 仓库及其中的 RoboTwin 环境。

## 1. 推荐目录结构

默认按照下面的相对路径寻找 Fast-WAM：

```text
wam_ac/
├── code/
│   └── base_adaptive_2_10/       # 本目录
├── repos/
│   └── FastWAM/                  # 原版 Fast-WAM 仓库
└── models/                       # Fast-WAM 基础模型组件
```

在该结构下，本目录相对原版 Fast-WAM 的路径是 `../../repos/FastWAM`。如果本地
目录结构不同，可以显式设置：

```bash
export FASTWAM_ROOT=/absolute/path/to/FastWAM
export DIFFSYNTH_MODEL_BASE_PATH=/absolute/path/to/models
```

## 2. 方法说明

对于每个 action chunk，策略首先运行 Fast-WAM 原生的 Official-2 sampler。
第一次 ActionDiT evaluation 得到起点 velocity，并构造一步 terminal estimate：

```text
A_coarse = z0 - v(z0, sigma=1)
```

`A_coarse` 和完成后的 Official-2 action 都会经过 Fast-WAM 原生 action decoding
和反归一化，转换为 RoboTwin 的双臂绝对关节位置命令。

随后取两个候选实际将要执行的前 `E=24` 个动作，通过 RoboTwin
Aloha-AgileX URDF 分别计算左右臂末端位置。路由分数为：

```text
eta = 1/(2E) * sum_{b in {L,R}} sum_{h=0}^{E-1}
      ||p_b,h^(2) - p_b,h^(coarse)||_2
```

当前冻结阈值为 `0.009702832328772866 m`，即约 `9.702832 mm`。该阈值来自
独立336-chunk calibration artifact的chunk-pooled Q75。

路由规则为：

- `eta <= 9.702832 mm`：直接执行Official-2，真实ActionDiT NFE为2；
- `eta > 9.702832 mm`：从相同observation、condition和初始噪声进入Exact Official-10；
- fallback复用Official-2 probe与Official-10完全相同的第一次velocity；
- Official-2第二次evaluation位于 `sigma=5/6`，不能接入Official-10；
- fallback真实成本为2次probe加9次新增evaluation，共11 NFE。

这里衡量的是action command诱导的运动学控制差异，不是机器人经过接触、摩擦和物体
动力学后真实到达状态的差异，也不应解释为安全证书。

## 3. 运行依赖

运行前需要：

1. 按原版Fast-WAM README安装Python环境；
2. 在 `$FASTWAM_ROOT/third_party/RoboTwin` 下安装RoboTwin；
3. 将RoboTwin仿真assets放在 `$FASTWAM_ROOT/third_party/RoboTwin/assets`；
4. 下载Fast-WAM RoboTwin checkpoint及对应的dataset statistics JSON；
5. Python环境中安装NumPy和SciPy，SciPy用于CPU URDF正向运动学。

推荐checkpoint目录：

```text
$FASTWAM_ROOT/checkpoints/fastwam_release/
├── robotwin_uncond_3cam_384.pt
└── robotwin_uncond_3cam_384_dataset_stats.json
```

仅运行inference或RoboTwin闭环evaluation时，不需要完整的RoboTwin训练轨迹数据集，
但仍然需要RoboTwin仿真代码和assets。

## 4. 安装RoboTwin policy入口

```bash
cd /path/to/base_adaptive_2_10
export FASTWAM_ROOT=/absolute/path/to/FastWAM
bash install_policy.sh
```

脚本只会创建下面的软链接：

```text
$FASTWAM_ROOT/third_party/RoboTwin/policy/base_adaptive_2_10
  -> <本目录>/policy/base_adaptive_2_10
```

如果目标位置存在其他文件或指向其他目录的软链接，脚本会报错，不会覆盖现有内容。

## 5. 运行RoboTwin闭环测试

使用默认checkpoint路径：

```bash
export FASTWAM_ROOT=/absolute/path/to/FastWAM
export DIFFSYNTH_MODEL_BASE_PATH=/absolute/path/to/models

bash run_robotwin.sh \
  --task blocks_ranking_rgb \
  --episodes 2 \
  --gpu 0 \
  --output ./outputs/blocks_ranking_rgb
```

显式指定checkpoint和统计文件：

```bash
bash run_robotwin.sh \
  --task open_laptop \
  --checkpoint /path/to/robotwin_uncond_3cam_384.pt \
  --stats /path/to/robotwin_uncond_3cam_384_dataset_stats.json \
  --episodes 20 \
  --gpu 0 \
  --output ./outputs/open_laptop
```

切换RoboTwin setting：

```bash
bash run_robotwin.sh \
  --task open_laptop \
  --setting demo_randomized \
  --episodes 2 \
  --gpu 0 \
  --output ./outputs/open_laptop_randomized
```

RoboTwin成功率结果保存在指定输出目录中。路由诊断追加写入：

```text
<output>/adaptive_episode_records.jsonl
```

每条episode记录包含task、setting、seed、成功结果、环境步数、chunk数量，以及每个
chunk的position score、路由决策和真实ActionDiT NFE。运行时终端也会打印：

```text
[base-adaptive-2/10] chunk=3 eta_mm=7.4210 decision=official2 nfe=2
[base-adaptive-2/10] chunk=4 eta_mm=13.0852 decision=full10_fallback nfe=11
```

## 6. CPU测试

无需加载模型即可运行FK和输入检查：

```bash
export FASTWAM_ROOT=/absolute/path/to/FastWAM
python -m unittest discover -s tests -v
```

## 7. 方法边界

本目录固定以下设置：

- action horizon：`H=32`；
- execution horizon：`E=24`；
- Official-2 probe；
- Exact Official-10 fallback；
- `9.702832 mm` position threshold；
- 双臂EEF position trajectory mean readout。

本实现不包含historical overlap、gripper guard、progressive target、动态execution
horizon、warm start、ActionDiT cache或learned router。

## 8. 完整复现实验的注意事项

本目录用于复现方法实现。若要逐数值复现实验结果，还必须冻结：

- Fast-WAM代码commit；
- RoboTwin代码与assets版本；
- checkpoint与dataset statistics；
- task configuration；
- 实际被evaluator消费的seed manifest；
- GPU、CUDA和主要依赖版本。

不同RoboTwin版本或初始化流程可能导致相同task和seed无法到达完全一致的simulator
state。跨方法实验应保存实际episode记录，并在相同有效 `(task, seed)` 交集上配对比较。
