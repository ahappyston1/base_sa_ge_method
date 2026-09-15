# BC 第 225 轮分叉：检查与运行

## 全自动运行（推荐）

在原来的训练环境中、仓库根目录执行。GPU编号必须是您已预留的空闲卡，脚本不会侦测或终止其他人的进程：

```bash
python -u scripts/run_bc_tail_auto.py --gpus 0 1
```

顺序：原BC前225轮 → 完整checkpoint校验 → 与旧BC逐轮严格比较1–225轮 → 原BC恢复226–230轮 → 严格比较这5轮 → 启动三个尾段实验。两张卡先跑breliability/aguard，任一成功完成后空闲卡接featurehalf；一张卡则顺序执行，三张卡则三个同时运行。前缀和恢复检查使用指定的第一张卡。

默认参考是仓库内已完成的 `rev13_trusted_lr015_bc_s7_20260914_181549_541595_a0.1`，也可以指定：

```bash
python -u scripts/run_bc_tail_auto.py --gpus 0 1 --reference-run /绝对路径/原BC运行目录
```

如果已经生成完整225轮前缀，可跳过重跑，但仍执行两道对比检查：

```bash
python -u scripts/run_bc_tail_auto.py --gpus 0 1 --prefix-checkpoint /绝对路径/round_0225.pt
```

严格比较metrics中的acc/phase/lr，以及dynamics中的phase/lr/gate/aux_scale/trusted_weight_gate。比较使用CSV原始数值，不四舍五入、不设置容差；缺轮、重复轮、非有限数值或任意不一致都不能通过。不只比较第225轮。CSV一致并不证明参数完全一致，真实GPU复现仍以这一恢复检查作为可观测校验。

所有作业采用独立run_id。总控状态在 `results/bc_tail_launches/<时间_PID>/status.json`，比较结果在prefix_comparison.json/replay_comparison.json，子进程输出在同目录各任务.log。任何前置检查失败都会停止后续启动并以非零码退出。尾段作业失败则取消尚未启动的任务，已经运行的其他任务继续完成。中断总控不会主动杀掉已启动作业，重启前先查看日志中的PID，避免重复占卡。

需要断开SSH后继续运行，可使用：

```bash
nohup python -u scripts/run_bc_tail_auto.py --gpus 0 1 >> bc_tail_auto.log 2>&1 &
tail -f bc_tail_auto.log
```

本地已测试完整控制流程与失败分支（模拟训练进程），没有在本机运行GPU训练。总控不会自动把严格检查失败改为“差不多通过”。

## 当前检查结论

2026-09-15：本地 results 中没有 .pt 文件，无法验证服务器真实 checkpoint。原程序每轮覆盖 checkpoint.pt，不保留第225轮；只有跨原日程额外延长时才保存 main_end.pt。已结束的300轮 BC 若只有 checkpoint.pt，通常就是300轮，不能回退到225轮。

原格式已保存 global_model（包括BN）、p_ref/p_ref_valid、client_states（包括每个客户端 bc_heads、rho、p_loc）、历史准确率、独立客户端抽样 NumPy 状态、Python/全局 NumPy/Torch CPU/CUDA RNG、schedule、几何及LR配置。

通信轮结束后没有待延续的优化器动量：代码每次客户端训练会清空 optimizer.state；教师从全局模型重置，target projector 从客户端 projector 重置，所以这两个临时教师不需要跨轮保存。客户端 projector/predictor 则必须恢复。

## 服务器先检查

在装有 torch 的环境中，进入新的代码目录执行：

```bash
find results/CIFAR10/runs -type f -name '*.pt'
python scripts/bc_tail_experiment.py inspect --checkpoint /绝对路径/BC运行目录/round_0225.pt
```

可把检查路径换成现有 checkpoint.pt，程序会打印实际 round。仅允许完整、原版 BC 的第225轮状态用于这组分叉。第300轮、更新诊断快照、缺少客户端头/原型、缺少随机状态的文件会被拒绝。旧格式需要同目录的原 config.json 检查记录过的实验设置；旧格式没有完整训练参数快照，不能据此证明未记录的参数完全相同，运行时应使用仓库原BC配方。

## 若没有225轮快照

用当前代码重跑同一个原BC前缀。不要把 max_rounds 改成225，它会改变余弦学习率和阶段日程。只停止执行，不改变300轮日程：

```bash
bash scripts/train.sh --config configs/experiment_trusted_lr015_bc.yaml --gpu_id 0 --run_id bc_prefix225_20260915 --stop_after_round 225
```

新程序默认额外保存 round_0225.pt，checkpoint.pt 也会停在225轮。保存采用临时文件加原子替换。先确认 run_id 没有被之前运行使用；换新名称可避免进入旧运行。若已有早于225轮的完整原BC checkpoint，也可用原BC配置及 --resume 路径、独立新 run_id、--stop_after_round 225 接着训练，保留原300轮日程。

## 从共同225轮分出四条后段

下面都是新目录运行，不覆盖源checkpoint。把路径和GPU改成实际值，不要在同一张满载GPU上同时启动。

```bash
python scripts/bc_tail_experiment.py control --checkpoint /绝对路径/round_0225.pt --gpu 0 --stop-after-round 230
python scripts/bc_tail_experiment.py breliability --checkpoint /绝对路径/round_0225.pt --gpu 0
python scripts/bc_tail_experiment.py aguard --checkpoint /绝对路径/round_0225.pt --gpu 1
python scripts/bc_tail_experiment.py featurehalf --checkpoint /绝对路径/round_0225.pt --gpu 2
```

加 --dry-run 可在完成checkpoint检查后只打印命令。上面的control只跑226–230轮用于恢复检查，保持原300轮日程。对比新旧前缀及这5轮，吻合后使用旧完整BC作为主要对照；明显不吻合时再补跑新control到300轮。不传--stop-after-round则control执行到300轮。其余三组分别为B权重重分配、A教师分歧保护、特征系数减半。脚本一次启动一个实验，不自动抢占GPU、不提交或推送Git。

程序显式允许原BC225轮到后段配方的转换，其他几何/LR差异继续严格拒绝。检查已记录的数据集、标签数、种子、客户端数、局部epoch、batch及workers等设置。恢复后从226轮执行，始终使用原300轮日程。

## 三个改动

- breliability：226–260轮余弦爬升；教师两个弱增强的置信度/JS一致性打分。对B做中心化重分配，保持整批B权重总量，单样本最多0.5–1.5倍原权重。目标仍为第一个教师视图的软分布。
- aguard：教师两视图同意替代类别、都至少0.90置信且与A标签不同，才追加最多20%减权。226–260轮爬升。不改A标签、不抬高权重、不重新分配扣下的权重。仅改变A分类loss，不改变原来的原型写入规则，以隔离该实验。
- featurehalf：226–260轮将特征系数0.10降至0.05，之后保持。A、B目标与教师更新不变。

额外弱增强在DataLoader worker中生成，并恢复其Python/NumPy/Torch CPU随机状态，避免改变既有弱/强增强的后续随机流。前225轮不产生额外视图。教师额外前向为eval，不修改BN。无标签真值只参与诊断，不进入目标、权重或门控。

## 日志口径

继承的前225轮准确率保留在acc.csv与checkpoint中；metrics/dynamics等逐轮诊断只包含本次实际执行的226–300轮，原目录保留前缀日志。不要把继承的历史当作重新运行的数据。

dynamics.csv新增tail_gate及只读计数：

- tail_b_old_mass/new_mass、old_wrong/new_wrong：同一教师argmax下重分配前后的权重，都是原始访问权重总和。
- tail_score_0/1/2_visits/correct：可靠性小于0.6、0.6–0.9、至少0.9的B访问与教师正确计数。
- tail_a_guard_visits/correct：A保护触发数及其中原学生伪标签正确数；removed_correct/removed_wrong是扣下的正确/错误权重。
- tail_b_true_prob/true_nll：教师给真实类别的概率总和与负对数概率总和，除以teacher_b_visits得到均值。真值只读。

原有b_wrong_mass仍按学生路由标签计错，不能当作教师软监督的真实错误量。新增teacher argmax权重指标也不是完整软分布质量，需结合true_prob/true_nll。

## 验证范围

CPU测试验证225轮原BC更新一致、260轮真实反向传播、隐藏标签隔离、B权重守恒及边界、A减权触发、损失重建、额外视图随机流隔离和分叉拒绝条件。完整GPU复现仍需服务器control后段与原BC对应轮次比较，硬件/软件变化也可能影响数值复现。本地没有GPU或真实225轮权重，不声称已完成服务器续跑验证。

如果新后段中断，使用其自己的checkpoint和相同配方正常 --resume（不要再传 --bc_fork 1）。--bc_fork专门用于从原BC225轮创建新实验，不用于后段断点恢复。
