# LIBERO Adaptive Action Denoising

在已经跑通的 Fast-WAM LIBERO 环境中，比较 `adaptive`、`10-step` 和 `2-step`。本目录提供动作一致性路由和统一评测入口，不重复安装 Fast-WAM，也不包含 checkpoint、LIBERO 数据和基础模型权重。

以下命令均从 **`wam_ac` 仓库根目录**执行。先进入你克隆的仓库：

```bash
cd wam_ac
```

## 1. 复用已跑通的 Fast-WAM 环境

先激活运行 Fast-WAM 官方 LIBERO 测评时使用的 Python 环境，确认 `python` 指向该环境。脚本默认使用当前 `python`；也可通过 `EVAL_PYTHON` 指定解释器。

本方法针对原始 Fast-WAM LIBERO release checkpoint：`libero_uncond_2cam224.pt`，使用其匹配的 dataset stats 和 `sigma_shift=5.0`，不默认兼容其他 checkpoint 的调度参数。

需要已有：

- Fast-WAM 源码及其可运行的 LIBERO 评测依赖。
- LIBERO 源码、任务资源、初始状态文件及有效的 LIBERO 配置。
- Fast-WAM checkpoint、dataset stats，以及基础模型缓存。

## 2. 修改资源路径

复制配置模板；已经配置过时无需重复覆盖：

```bash
cp libero_adaptive_2_10/configs/default.json libero_adaptive_2_10/configs/local.json
```

编辑 `libero_adaptive_2_10/configs/local.json`，将以下字段替换为**你自己的资源位置**：

| 字段 | 应填写什么 |
|---|---|
| `fastwam_root` | Fast-WAM 仓库根目录，里面应有 `configs/`、`src/` 和 `experiments/libero/eval_libero_single.py` |
| `libero_root` | LIBERO 仓库根目录，里面应有 `libero/` Python 包 |
| `checkpoint` | 已用于官方评测的 `libero_uncond_2cam224.pt` 文件完整路径 |
| `dataset_stats` | 与该 checkpoint 配套的 `libero_uncond_2cam224_dataset_stats.json` 文件完整路径 |
| `libero_config_path` | 官方评测所使用的 `LIBERO_CONFIG_PATH` 目录，通常含 `config.yaml`；其中的数据、任务和初始状态路径也必须有效 |
| `model_base_path` | 官方评测使用的 `DIFFSYNTH_MODEL_BASE_PATH`，即基础模型缓存根目录 |

例如，若你的 Fast-WAM 在 `/home/user/projects/FastWAM`，就把 `fastwam_root` 改为该路径，并将 `checkpoint` 指向实际 checkpoint 文件，而不是只填其所在文件夹。所有资源路径支持绝对路径、`~` 和环境变量；相对路径按 **配置文件所在目录**解析。

其他字段默认保持不变：

```json
{
  "seed": 42,
  "sigma_shift": 5.0,
  "adaptive_threshold": 0.07091929394170104
}
```

`local.json` 已加入 `.gitignore`，不提交个人机器路径。如果使用其他配置文件，在任一脚本后加 `--config /path/to/your_config.json`。

先检查资源路径和任务计划：

```bash
bash libero_adaptive_2_10/scripts/smoke.sh --dry-run
```

`--dry-run` 不加载模型、不启动 GPU，只检查路径和评测计划。它不等于渲染、依赖或 checkpoint 验证。

## 3. 跑 smoke

```bash
bash libero_adaptive_2_10/scripts/smoke.sh
```

默认单 worker，Spatial task 0，三种方法各 1 episode，共 3 episode。前台运行可以直接看到启动错误；详细 worker 日志保存在输出目录。

默认输出：`libero_adaptive_2_10/outputs/smoke/`。检查：

```bash
cat libero_adaptive_2_10/outputs/smoke/completion.json
cat libero_adaptive_2_10/outputs/smoke/summary.csv
```

正常结束应有 `complete: true`、`tasks: 3`、`episodes: 3`，summary 中出现三种方法。机器人 episode 失败与程序异常不同：前者记录为失败试验，后者写入 `ERROR.json`。

同一输出目录中的完整任务会被跳过。需要真正重新测一轮时换输出目录：

```bash
bash libero_adaptive_2_10/scripts/smoke.sh --output libero_adaptive_2_10/outputs/smoke_retry
```

## 4. 正式评测：每种方法 2,000 episode

```bash
bash libero_adaptive_2_10/scripts/evaluate_2000.sh --detach
```

默认对三种方法各评测 40 task × 50 episode，**总计 6,000 episode**，单路执行。使用当前已经跑通 smoke 的配置。

只跑一种方法，例如 adaptive：

```bash
bash libero_adaptive_2_10/scripts/evaluate_2000.sh \
  --methods adaptive --output libero_adaptive_2_10/outputs/adaptive_2000 --detach
```

将 `adaptive` 换为 `10-step` 或 `2-step` 可分别运行基线。独立实验使用不同输出目录。

如果关注成功率吞吐、GPU 显存足够，可以使用三路并行：

```bash
bash libero_adaptive_2_10/scripts/evaluate_2000.sh \
  --workers 3 --gpu 0 --output libero_adaptive_2_10/outputs/comparison_2000 --detach
```

三个模型依次加载，已加载的 worker 立即领取任务；全部就绪后并行评测。单张 80GB GPU 上三个 worker 共约 75–76GB。不要同时再启动其他占显存的评测。**比较 wall-clock 加速时使用单路，并确保 GPU 没有其他负载。此入口主要输出成功率与 NFE，并非完整动作块 wall-time 专用入口。**

`--detach` 让运行脱离终端，SSH 断开也可继续；去掉它则前台运行。全部任务完成后自动退出、释放显存，不自动调参或继续跑其他实验。

## 5. 查看进度和结果

默认正式评测输出为 `libero_adaptive_2_10/outputs/evaluation_2000/`：

```bash
cat libero_adaptive_2_10/outputs/evaluation_2000/status.json
cat libero_adaptive_2_10/outputs/evaluation_2000/summary.csv
cat libero_adaptive_2_10/outputs/evaluation_2000/suite_summary.csv
tail -f libero_adaptive_2_10/outputs/evaluation_2000/worker_0.log
```

| 文件 | 内容 |
|---|---|
| `status.json` | 阶段、已完成任务数、已落盘 episode 数、总量 |
| `summary.csv` / `.json` | 每种方法的成功数、成功率、平均 NFE |
| `suite_summary.csv` / `.json` | 分套件成功率和平均 NFE |
| `completion.json` | 全部评测完成标志 |
| `ERROR.json` | 程序异常；结合 worker 日志定位 |
| `worker_0.log` 等 | 每个 worker 的评测过程和异常 |
| `tasks/<method>__<suite>_<id>/result.json` | 完整任务结果及成功/失败 trial 编号 |
| `tasks/.../episode_XX.json` | 每次 episode 的动作块分数、决策和 NFE |
| `tasks/.../job.json` | 任务配置及初始状态哈希 |
| `run_config.json`、`provenance.json` | 已解析配置和代码来源哈希 |

成功率按 episode 计算，平均 NFE 按动作块加权。summary 只汇总完整完成的 task，运行中可能落后于 status 中的 episode 数。`10-step` 的 NFE 应为 10，`2-step` 应为 2；`adaptive` 接受时为 2、回退时为 11，平均取决于实际轨迹。

## 6. 停止、续跑与排错

优雅停止默认正式评测：

```bash
touch libero_adaptive_2_10/outputs/evaluation_2000/STOP
```

worker 会跑完当前已领取的整个 task 后退出，不会立即中断其环境历史。等待 worker 退出后，重新执行**相同命令、相同配置和输出目录**即可续跑。完整任务跳过，中断任务从 trial 0 重跑，不完整数据归档。不同配置不能混用同一输出目录。

出现程序失败时先检查 `ERROR.json`、`failed/` 和 worker 日志。修复原因并确认旧 worker 已退出后，用原命令加 `--retry-failed` 显式重试。普通停止后的续跑不需要该参数。

## 方法与协议

`adaptive` 比较初始速度给出的粗预测 `a_coarse=z0-v0` 与原生两步结果 `a2`，使用前 10×7 个归一化连续动作元素：

```text
relative_l2 = sqrt(mean((a_coarse-a2)^2)) /
              max(sqrt((mean(a_coarse^2)+mean(a2^2))/2), 1e-8)
```

分数 ≤ 0.07091929394170104 时接受两步结果，否则从相同初始噪声按原生十步 schedule 重放，复用第一次预测，因此回退总 NFE=11。不做物理空间映射、额外夹爪规则或维度加权；环境执行动作时仍沿用官方后处理。

阈值来自全 40 task 校准与候选筛选，正式评测固定，不重新校准。预测长度 32、执行长度 10、seed=42、sigma shift=5、CFG=1，使用原生 task / episode 循环。每个 task 复用环境，正式评测使用初始状态 0–49。模型常驻并在每个 task 前恢复初始化后的 RNG 状态。不保存视频。相同 seed 不保证跨设备或环境版本逐位一致。

NFE 仅统计 action expert 前向次数，不等于端到端加速倍数。观测编码、预处理和环境执行还存在固定开销。

## 参考结果与开发测试

`results/` 保存已有评测摘要及原始来源：adaptive 96.55% / NFE 3.262，2-step 96.10% / NFE 2；历史 10-step 96.70% / NFE 10。

已配置 `local.json` 并使用 Fast-WAM 环境时，可执行：

```bash
(cd libero_adaptive_2_10 && python -m unittest discover -s tests -v)
```

两类用户脚本共同放在 `scripts/` 下：`smoke.sh` 和 `evaluate_2000.sh`。根目录的 `run_2000.sh` 仅保留为兼容入口。
