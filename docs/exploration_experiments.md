# 三条独立探索与服务器排队

三个配置从同一 early 基础出发，均为 CIFAR10 alpha=0.1、LR=0.15、300 轮，保持种子、划分、骨干和原日程。第一阶段不合并方法：

- `trusted_lr015_risk`：A 组风险权重，详见 trusted_risk.md。
- `trusted_lr015_bc`：B 教师软目标＋B/C 特征学习，不启用 risk 或 proximal。
- `trusted_lr015_prox`：中期参数约束，不启用 risk 或 B/C 教师。

## 您有两块空闲卡、另一块卡仍在跑旧实验

以下 0、1、2 和 12345 均为示例，请替换成实际 GPU 编号和旧实验的训练 Python PID。先在服务器项目根目录激活原训练环境。

**方案 A：只使用两块空闲卡，自动排队三项。**

```bash
python scripts/run_explorations.py --gpus 0 1 --dry-run
nohup python -u scripts/run_explorations.py --gpus 0 1 > exploration_queue.log 2>&1 &
```

risk 使用第一块卡，BC 使用第二块卡；其中任何一个结束后，prox 接着使用那块卡。最多同时跑这三个新实验中的两个，完全不使用旧实验的 GPU。

**方案 B：第三项等待旧实验结束后，在第三块卡上运行。**

```bash
python scripts/run_explorations.py --gpus 0 1 --after-gpu 2 --after-pid 12345 --dry-run
nohup python -u scripts/run_explorations.py --gpus 0 1 --after-gpu 2 --after-pid 12345 > exploration_queue.log 2>&1 &
```

risk 和 BC 立即分别运行。prox 等指定的旧训练进程退出，再使用第三块卡；因此旧实验结束后，新实验最多三项并行。如果旧任务有多个占用 GPU 的进程，请等待它们都完成后手动启动第三项，不使用单 PID 模式。

PID 应是实际训练 Python 进程，不是启动后就退出的 shell 或 nohup 包装进程。可用 `nvidia-smi` 查看，再用 `ps -p 12345 -o pid,args` 核对。脚本会校验进程存在并记录 Linux 进程启动时间，避免 PID 被复用后继续等待；不会杀死旧任务，不会自动选择 GPU，也不根据显存空闲猜测任务结束。它不验证旧任务是否训练成功，只等待退出。

不要同时执行 A、B 两种方案，以免重复启动。每项实验有独立时间戳结果目录；队列启动日志在 `results/exploration_launches/<时间_PID>/`，总调度日志为 `exploration_queue.log`。训练非零退出也会释放其队列位置，队列继续其余实验，最后用非零退出码报告失败。中断队列不会终止已独立启动的训练，但不会再启动尚在等待的任务。

如果喜欢手动控制：

```bash
bash scripts/run_experiment.sh trusted_lr015_risk 0 7
bash scripts/run_experiment.sh trusted_lr015_bc 1 7
# 旧实验完成、确认 GPU 2 空闲后：
bash scripts/run_experiment.sh trusted_lr015_prox 2 7
```

这三条是独立前台命令，需要分别在终端/tmux 会话执行。新方法均从头训练，不使用旧 early 检查点续训。

## B/C 版的具体边界

A 路由、A 硬目标及其权重始终由原学生分支计算，不使用 EMA 教师筛 A，不引入多原型或几何软标签融合。额外教师只负责 B/C。前 30 轮与 early 的学生训练相同，第 31—90 轮余弦渐进启用。

B 的损失从原学生弱增强目标逐步过渡到教师弱增强软目标，沿用原 B mask、权重、温度和全无标签批分母。B 的原启动日程不变。B/C 同时学习学生强增强特征预测教师弱增强特征，系数从 0 升至 0.10；不把其他图片作为负例，不给 C 硬类别，不对 A 增加特征损失。

投影头为 dim→dim→128，预测头为 128→256→128，使用 LayerNorm，不增加 BN 统计。辅助头按客户端持久保存，随常规检查点恢复；不上传聚合，也不参加测试推理。每轮本地教师骨干从该轮全局模型初始化，教师投影头从该客户端学生投影头初始化，随后每步用 0.99 EMA 更新。教师分支停止梯度；BN buffers 复制当前学生值，整数计数不做 EMA。原学生 BN 行为不变，因此这不是 BN 修复实验，也不是跨通信轮持久服务器 EMA。

辅助头初始化使用隔离的 CPU 随机状态，避免改变数据增强随机流。只报告原学生全局测试准确率，未增加教师测试或集成。

原 `metrics.csv` 的 L_B 是实际混合后的 B 损失。`dynamics.csv` 新增 `L_feature_effective` 并计入总 loss 重构，同时记录 `teacher_b_visits`、`teacher_b_precision`、`bc_feature_std`。前 30 轮未使用额外教师目标，teacher 统计的零是无访问，不能理解为 0% 正确率。原 `a_prec/b_prec` 仍描述学生路由预测，B 的 wrong_mass 也沿用该口径，不能直接解释为教师软分布的错误量。特征标准差为强增强归一化骨干特征的批内标准差逐维平均，再按本地步数汇总，仅供观察退化，不是充分的坍塌判据。

核心风险：教师可稳定错误，辅助头可吸收特征损失而骨干受益有限，也可能与分类目标竞争。最终 acc、A 保留量、教师 B 精度、特征标准差和有效损失需要一起判断。0.10 和 0.99 是探索起点，不是已证实最优设置。

## 中期 proximal 版

每个客户端在本轮开始固定服务器参数 theta0，增加：

```text
L_prox = 0.5 * mu(round) * sum_over_trainable_parameters((theta - theta0)^2)
```

原始求和，不按参数量平均。BN affine 权重包括在内，running_mean/running_var 等 buffers 不包括。前 30 轮 mu=0；31—60 轮余弦升至 0.01；61—200 轮保持；201—250 轮余弦降至 0；此后关闭。保留 LR、本地步数、FedAvg、BN 与全部原 A/B/C 目标。`dynamics.csv` 新增 `L_prox_effective` 并计入总 loss 重构。

它测试本地参数偏离是否影响整体表现；不能因为曲线更平就认为成功，也不承诺解决第 83 轮。两条新路线均保存第 82、83、84 轮更新快照，格式见 trusted_risk.md。

## 上传与验证

建议同步整个代码仓库，保留服务器数据与旧运行目录。如单独复制，除上一版 risk 相关文件外，本次还需要 `exploration.py`、`fl_runner.py`、`options.py`、`training_dynamics.py`、两个新 YAML、`scripts/run_experiment.sh` 和 `scripts/run_explorations.py`。未新增第三方运行依赖。

CPU 测试覆盖三条路线前 30 轮一致、相同输入下 A/原型目标和路由不变、三个训练阶段真实优化器、教师停止梯度、隐藏标签隔离、总损失重构、prox 梯度方向与日程，以及两种队列的启动顺序。完整 GPU 效果需服务器实验验证。
