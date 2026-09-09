# RoboTwin Clean 50 Tasks × 20 Seeds 配对评测

该 campaign 在一张 GPU 上比较：

- Fast-WAM Exact Full-10 baseline；
- WAM-AC Adaptive 2/10。

两种方法使用同一份冻结 manifest，共 50 个 RoboTwin clean tasks、每个任务 20 个 seeds，即每种方法 1,000 episodes，总计 2,000 episodes。

## 一键启动

从 Git 仓库根目录执行：

```bash
export WAM_AC_ROOT="$(pwd)/wam_ac"
export FASTWAM_ROOT="$WAM_AC_ROOT/repos/FastWAM"
export DIFFSYNTH_MODEL_BASE_PATH="$WAM_AC_ROOT/models"

bash wam_ac/code/paired_clean_50x20/run.sh
```

默认使用 GPU 0。指定其他单张 GPU：

```bash
GPU_ID=1 bash wam_ac/code/paired_clean_50x20/run.sh
```

只跑其中一种方法：

```bash
bash wam_ac/code/paired_clean_50x20/run.sh --method full10
bash wam_ac/code/paired_clean_50x20/run.sh --method adaptive_2_10
```

默认结果写入 `wam_ac/outputs/paired_clean_50x20_v1/`。脚本以 method/task 为断点单位，已经存在且 seeds 完整一致的任务会自动跳过。

## 配对和报告口径

脚本逐任务运行 20 个 episodes，并从 evaluator 日志核验实际消费的 seed 序列。任何方法出现 seed drift 时立即停止，避免产生伪配对结果。

50 个任务中，`adjust_bottle`、`beat_block_hammer`、`handover_block`、`hanging_mug`、`open_microwave` 曾用于 9.702832 mm 阈值 calibration。因此最终应分别报告：

- 全部 50 tasks 的 RoboTwin clean benchmark；
- 排除上述 5 个 calibration tasks 后的 held-out 45-task 主结果。

不得把全 50-task 数字称为完全独立的 held-out confirmation。

## 注意

- 这是单卡串行 campaign，不将运行时 latency 直接当作严格的单进程 latency benchmark；
- 默认关闭评测视频，避免 2,000 episodes 产生大量视频文件；
- Adaptive 的终端日志包含每个 chunk 的路由分数和 2/11 forwards；
- 干净官方 RoboTwin evaluator 未包含 episode instrumentation hook 时，不影响标准成功率与脚本日志解析，但不会额外生成 `adaptive_episode_records.jsonl`。
