# Code review and smoke validation

本次审查重点：采样分支、实际 NFE、任务队列、后台启动、停止与续跑。

已修复：

- 控制器异常路径现在发送 STOP 后等待已启动的 worker 退出，避免控制器退出却有 worker 继续占用 GPU。STOP 仍按整个任务边界生效，不截断评测历史。
- 已完成实验续跑时，completion.json 保留 tasks / episodes 字段。
- 所有任务提前结束时，不再加载剩余 worker；对小规模 smoke 尤其有用。
- 拒绝 NaN / Infinity 等非有限 adaptive 阈值。
- 控制器锁改用上下文管理，退出时关闭。

验证：5 项 unittest 通过，包括原生 scheduler 的 adaptive 两分支与 2-step / 10-step 输出、NFE、shadow 审计、指标边界、完成结果跳过以及非有限阈值拒绝。

真实模型复测输出：outputs/review_smoke。Spatial task 0，三种方法各 1 episode。该 smoke 只验证最小链路，不替代全任务回归或三路并发压力测试。

已知限制：任务内 SIGKILL 需重新执行该 task；优雅停止要等当前完整 task 结束。错误不会自动按成功率重试。历史 10-step 结果仍来自另一计时版入口，不能称作当前代码的严格回归基线。相同 seed 不保证环境与 GPU 跨运行逐位复现。

复测完成：三种方法各 1/1 成功；10-step 为 7 chunks / 70 forwards，2-step 为 7 / 14，adaptive 为 7 / 23（NFE 3.286）。completion.json 确认 3 tasks / 3 episodes，GPU 显存释放至 0 MiB。再次执行相同 smoke 命令，跳过已完成任务、未加载模型，完成记录保留计数。
