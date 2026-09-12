# REV13 后续三组实验

三组均从头训练。已有 `reference`、`high_lr`、`trusted` 配置与命令保留，
新实验以已完成的 **legacy / LR=0.15 / 300轮** 为共同参照。

1. `trusted_lr015`：trusted，初始 LR=0.15，300轮；相对原有 trusted 只提高 LR。
2. `legacy_lr015_500`：legacy，初始 LR=0.15，500轮；相对 high_lr 只改变总轮数。
3. `legacy_lr020`：legacy，初始 LR=0.20，300轮；相对 high_lr 只提高 LR。

配置均是完整快照。其余设置保持一致：CIFAR-10、alpha=0.1、每类500个标签、
20个客户端每轮选8个、local_epochs=5、num_workers=16、partition_seed=0、
lr_min=0.0001、lr_mid_factor=1、teacher=0、A/B cap=0、b_conf_rescue=0。
默认模型与客户端抽样种子均为7。未修改FedAvg、原型数量或C组损失。

## 服务器启动

进入仓库并激活已有 PyTorch 环境，在三个终端分别执行（GPU编号按空闲卡修改）：

```bash
# 实验1：可信几何 + 0.15，300轮
bash scripts/run_experiment.sh trusted_lr015 0 7

# 实验2：老几何 + 0.15，500轮完整计划
bash scripts/run_experiment.sh legacy_lr015_500 1 7

# 实验3：老几何 + 0.20，300轮
bash scripts/run_experiment.sh legacy_lr020 2 7
```

每条命令只启动一个前台训练，不自动后台运行。需要断开SSH时请在tmux/screen会话中运行。
三个任务各用16个DataLoader worker，请确保服务器CPU和内存可容纳并行训练。

可先检查启动命令，不导入PyTorch、不开始训练：

```bash
bash scripts/run_experiment.sh trusted_lr015 0 7 --dry-run
bash scripts/run_experiment.sh legacy_lr015_500 1 7 --dry-run
bash scripts/run_experiment.sh legacy_lr020 2 7 --dry-run
```

SEED可省略，默认为7；换成17或27会同时改变模型/客户端抽样种子，划分种子仍为0。
启动器生成包含实验名、种子、时间戳和进程ID的新run_id，结果保存到：
`results/CIFAR10/runs/rev13_<experiment>_s<seed>_<timestamp>_<pid>_a0.1/`。
保留现有config、acc、metrics、dynamics、updates、client_updates、geometry_audit等日志；
trusted另有trust_reference。新的命令不会选择旧实验检查点。

## 500轮的含义

500轮实验按500轮从头计算余弦学习率和阶段时长，**不是先复现300轮再追加200轮**：

- Phase1为第1–150轮，辅助损失约从第106轮开始接入。
- Phase2从第151轮开始，几何gate用75轮达到1；满足覆盖率等退出条件时，
  最早第226轮进入Phase3，实际切换以日志为准。
- 初始LR=0.15时，第300轮LR约0.051824，第500轮降至0.0001。
- 300轮版本Phase1到90轮结束、gate过渡45轮，第300轮LR已为0.0001。

不要给这组传入旧300轮实验的 `--resume`。检查点LR控制包含总轮数，
不兼容的调度会被拒绝。中断恢复只能使用同一实验、同一总轮数和同一配置。

## 检查与比较

```bash
# 轻量配置与学习率检查，不需要PyTorch/pytest
python -m unittest discover -s tests -p test_experiment_configs.py -v
python -m unittest discover -s tests -p test_training_dynamics.py -v

# 服务器已有PyTorch/pytest时，训练前运行核心检查
python -m pytest tests/test_ppfpsl_routing.py tests/test_trusted_geometry.py tests/test_geometry_audit.py tests/test_training_dynamics.py tests/test_experiment_configs.py tests/test_update_diagnostics.py tests/test_run_guards.py -q
```

主要终点：最后30轮平均准确率；同时记录最高/最终准确率、中期相邻轮变化和大跌次数、
A/B正确率及有效系数、实际耗时。300轮与500轮应同时按通信轮数与训练阶段观察，
不能直接将二者第136–200轮都解释为同一阶段；500轮峰值有更多挑选机会。
新组合不保证收益相加；先用seed7探索，再做配对种子重复验证。
