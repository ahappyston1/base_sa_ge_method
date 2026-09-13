# 三组后续实验启动说明

所有命令在仓库根目录执行，使用现有 PyTorch 环境。GPU 编号可改。
三个实验输出到各自含时间戳的新目录，不会覆盖之前完成的结果。

```bash
# 1. 新方法：REV14，300轮主训练 + 75轮低学习率收尾
bash scripts/run_experiment.sh trusted_multi_tail 0 7

# 2. 原 trusted + LR0.15，保留原300轮日程，再以0.0001训练200轮
bash scripts/run_experiment.sh trusted_lr015_tail500 1 7

# 3. 原 trusted + LR0.15，300轮，只提前可信软降权
bash scripts/run_experiment.sh trusted_lr015_early 2 7
```

这是三个独立进程；不要把多个命令指向同一张显存不足的GPU。只有一张GPU时顺序执行即可。
在任意命令末尾加 `--dry-run` 可只查看命令。

## 实验2：延长收尾，不拉长日程

配置：`configs/experiment_trusted_lr015_tail500.yaml`。
相对原 `trusted_lr015`，只增加 `baseline_schedule_rounds: 300` 并把 `max_rounds` 改为500。

- 默认从头跑500轮；前300轮使用原300轮学习率、阶段、B/原型损失开启安排。
- Phase1仍是前90轮；通常Phase2为91–135轮。Phase3退出/进入规则沿用原实现，不由测试准确率决定。
- 第300轮LR到达0.0001；第301–500轮继续保持0.0001，不重启、不回升。
- 第300轮另存 `main_end.pt`，第500轮结果位于 `checkpoint.pt`。
- 保留原几何、原loss、原模型和客户端步数；没有启用REV14教师、多原型、裁剪或特征损失。

它回答的是“原来300轮模型继续低学习率训练是否受益”，不是“500轮缓慢余弦日程是否更好”。
学习率很小，可能提升有限；不能预先把额外200轮视为有收益。

恢复本实验中断的运行时：

```bash
python ppfpsl.py --config configs/experiment_trusted_lr015_tail500.yaml \
  --gpu_id 1 --seed 7 --partition_seed 0 --sample_seed 7 \
  --run_id 该实验原run_id \
  --resume results/CIFAR10/runs/该实验目录/checkpoint.pt
```

建议按上面的新命令从头运行。已有原300轮checkpoint的LR/几何控制相同时也可恢复延长，但旧checkpoint未保存Python和全局NumPy随机状态，不能宣称与从头500轮逐位一致。新的checkpoint补存这些状态。

## 实验3：提前可信权重，不提前整个阶段

配置：`configs/experiment_trusted_lr015_early.yaml`。
只新增 `trusted_weight_start: 30`、`trusted_weight_end: 60`，原300轮日程不动。

- 第1–30轮不刷新额外参考，也不改变loss权重。
- 第31–60轮，可信软权重的gate从0按余弦平滑升至1，第45轮为0.5。
- 第61轮以后保持1。原Phase2开始时，不重新将权重gate置零。
- 在Phase1，额外参考只用于A/B损失的软降权；A/B/C路由公式、B损失启用时点、有标原型loss目标及原型写入规则不变。
- 在Phase2/3使用原trusted逻辑，但软权重gate已提前完成。
- 每个类别仍受其trust、参考有效性和参考年龄限制。gate=1不意味着无条件相信几何；也没有硬删除“未知参考”样本。
- 权重导致模型轨迹变化后，实际桶占比/伪标签质量当然可能随之变化；“路由不变”指公式与日程，不指训练后每张图的分组永远相同。

新增 `dynamics.csv` 的 `trusted_weight_gate` 字段，与原 `gate` 区分：后者仍是原阶段gate。
已有 `trust_authority_mean`、`trust_valid_fraction`、`trust_refreshes` 会从31轮起记录实际参考作用。
`L_A_effective`、`L_B_effective`及错误权重系数统计包含提前降权效果；无标签真值只用于只读诊断。

观察31–90轮与136–200轮的大幅回落是否减少，以及末30轮准确率是否提高；不是仅看曲线是否更平滑。

## 验证

```bash
python -m pytest tests -q
```

新增测试检查：前300轮学习率/阶段快照一致；301–500轮无LR反弹；早期开关不提前B/原型日程；第91轮权重不回退；实际CPU本地训练在开关生效前参数一致、生效后软权重确实进入loss；有效loss重构一致。
完整CUDA训练由服务器运行，本地测试不能替代收敛验证。
