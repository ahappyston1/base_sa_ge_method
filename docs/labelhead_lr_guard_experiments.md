# Labelhead：两组学习率实验与一组保护解耦实验（附可选收尾版）

三组独立对照，均以现有 labelhead 为基础，从头训练 300 轮。保持数据划分、种子、标签预算、骨干、FedAvg、本地训练步数、辅助头拟合/校准和 BC 特征系数。所有阶段及目标渐增时点不因学习率日程改变。没有更换骨干或加入外部预训练。

## 实验 A：labelhead_constant

配置：`configs/experiment_bc_targets_labelhead_constant.yaml`。第 1–300 轮 LR 始终为 0.15，原 labelhead 方法保持不变。直接检验全程不衰减是否有益；后期仍可能震荡，不能预设一定优于余弦。

## 实验 B：labelhead_step

配置：`configs/experiment_bc_targets_labelhead_step.yaml`。

- 第 1–200 轮 LR=0.15。
- 第 201–300 轮 LR=0.05。
- 原 labelhead 方法保持不变，检验持续保持较强学习能力能否改善准确率。

第 201 轮是明确的三倍降档；不再执行原余弦下降。训练 loss 可能降低，但错误监督也可能被加强，因此同时检查 best、末 30 轮均值、波动、纠错/误改和原型写入质量。

## 可选实验：labelhead_step_tail（不在本次默认运行组合内）

配置：`configs/experiment_bc_targets_labelhead_step_tail.yaml`。

- 第 1–200 轮 LR=0.15。
- 第 201–270 轮 LR=0.05。
- 第 271–300 轮按余弦由 0.05 平滑降至 0.001：
  `0.001 + 0.5*(0.05-0.001)*(1+cos(pi*(round-270)/30))`。
- 第 270 轮仍为 0.05，第 271 轮开始下降，第 300 轮精确为 0.001。

方法仍是原 labelhead。与阶梯实验 B 的前 270 轮日程完全相同，用于判断强化学习后是否需要低 LR 收尾。两组均从头运行；没有新增跨配置 checkpoint 分叉功能，不要直接跨配置使用 checkpoint 恢复。

这些新日程只改变客户端优化器 LR，不改变局部 momentum 重置规则、BN、目标 ramp 或阶段门槛。旧 `lr_min=0.0001` YAML 字段保留用于原配置兼容，新日程不使用该字段；输出 config/checkpoint 的 `lr_controls` 记录实际生效日程及最小值。

## 实验 C：labelhead_guard

配置：`configs/experiment_bc_targets_labelhead_guard.yaml`。

继续使用原 LR=0.15 的 300 轮余弦日程，最小 0.0001，只修改辅助头保护动作及其耦合。

1. **替代标签纠错规则不变。** 辅助头可用性、历史一致性、原类别存在条件、教师双视图置信度 0.60，以及 `0.75*辅助分布+0.25*教师平均分布` 均保留。相同输入下，纠错集合和目标不变；实际训练轨迹变化后，纠错数量仍可能变化。
2. **部分恢复监督。** 使用原 labelhead 保护集合，不增加 0.85/0.80 门槛。仅对原 targets 非硬目标、随后被辅助头保护的 A，使用 `L_protected=0.5*L_hard+0.5*L_before`。`L_before` 是保护前 targets 的分类目标（候选集损失或零分类损失等）。外层仍为 `(1-gate)*L_hard+gate*L_protected`，再乘原样本权重；不把系数重复应用。0.5 是实验参数，不是正确率估计。
3. **分类保护与原型写入分离。** 对原 targets 已判为非硬目标的 A，即使辅助头允许恢复硬 CE，也不因这个动作恢复旧标签原型写入。原型系数按既有 targets 强度逐步衰减，r120 起为 0。纠错产生的新类别仍不写入硬原型。
4. **特征任务保持原 labelhead 规则。** 相同输入下，A 特征 mask 与原 labelhead 完全相同，不因部分保护或原型阻断扩大特征学习集合。B/C 目标和系数不变。

第 61–120 轮渐进启用；没有针对猫/狗/青蛙硬编码不同规则。这是“部分保护 + 原型写入解耦”的组合实验，不能只凭准确率把收益归因于单项。它同时减少正确和错误保护的硬监督，仍需实验验证。阻断的是伪标签对 p_loc 等原型更新的贡献；trusted_ref 仍由真标签建立。以后样本重新满足原 targets 的硬目标条件，仍可正常写入。

guard 控制版本为 2，checkpoint 会拒绝恢复旧版阈值 guard 的训练状态。两个学习率实验不受此次修改影响。

## 新诊断与恢复保护

保留原 targets、auxiliary 和混淆矩阵日志。C 在 dynamics.csv 额外记录：

- `guard_protected/guard_protected_correct`：部分保护的访问数及其中原答案正确数。
- `guard_blocked_protected`：部分恢复分类监督但继续限制原型写入的访问数。
- `guard_blocked_correct_mass/guard_blocked_wrong_mass`：这部分相对恢复写入的名义正确/错误权重，乘目标渐增系数，不含后续可选原型写入过滤器。
- `guard_ce_reduced_correct_mass/guard_ce_reduced_wrong_mass`：相对原 labelhead 保护动作减少的硬 CE 系数质量，包含 gate 和 0.5 系数；不是梯度大小或净损失变化。

原 `experiment_c*_protected` 在本实验表示保护资格集合，不代表完全恢复硬标签；部分恢复量由 guard 字段报告。`target_c*_a_hard_released_*_mass` 已包括部分保护减少的硬 CE 系数。路由计数沿用原 labelhead 状态，不能用状态 0 的计数推断全部使用完整硬 CE。

隐藏真标签只用于日志，不能用于选样、权重、学习率或目标。计数为训练访问，不是独立样本数。沿用现有 checkpoint 保存全局模型、两套样本历史、原型、随机状态和阶段状态；记录新学习率/保护参数并拒绝跨实验误恢复。旧配置默认行为及 checkpoint 控制字典保持兼容。

额外保存完整的 r30、60、90、120、180、200、225、250、270、280、300 checkpoint。新增快照不增加训练轮数。

## 服务器运行

使用原服务器训练 Python 环境。只填当前空闲 GPU，编号遵循 CUDA_VISIBLE_DEVICES。显式指定本次三个实验；不带 `--experiments` 时，脚本仍默认旧 evidence/labelhead/separation 三组。

```bash
python scripts/run_target_experiments.py --gpus 0 1 --experiments labelhead_constant labelhead_step labelhead_guard --dry-run
nohup python -u scripts/run_target_experiments.py --gpus 0 1 --experiments labelhead_constant labelhead_step labelhead_guard > labelhead_followups_queue.log 2>&1 &
```

上述命令先并行运行 A/B；有卡释放后自动运行 C。每卡最多一个新任务；一个任务失败后不再领取待运行任务，已启动任务继续。不会停止服务器上其他实验。每组自动使用不同带时间戳的 run_id，日志在 `results/target_experiment_launches/`。

如果只有一张空闲卡，使用 `--gpus 0`。也可以先跑 C，把 experiments 顺序改为 `labelhead_guard labelhead_constant labelhead_step`。

单独启动示例：

```bash
bash scripts/train.sh --config configs/experiment_bc_targets_labelhead_guard.yaml --gpu_id 0 --run_id "labelhead_guard_$(date +%Y%m%d_%H%M%S)"
```

中断后使用相同配置恢复自己的 checkpoint，不要使用新队列命令代替恢复：

```bash
bash scripts/train.sh --config configs/experiment_bc_targets_labelhead_step_tail.yaml --gpu_id 0 --resume results/CIFAR10/runs/实际运行目录/checkpoint.pt
```

本地检查包含学习率边界、原阶段不变、纠错目标不变、保护与原型/特征解耦、隐藏标签隔离、实际优化器训练、r270→271 checkpoint 重放和项目实际 ResNet CPU 训练路径。完整 CUDA 实验及最终提升仍需服务器验证。
