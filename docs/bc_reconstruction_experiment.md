# BC 特征重构与原型实验：第一阶段

分支：`codex/bc-reconstruction-prototypes`。以 `experiment_trusted_lr015_bc.yaml`
为基线，新增独立配置和辅助模块；不是 REV14 trusted_multi 配方。
本地 CPU 功能验证已完成，完整 CIFAR10 GPU 收敛与收益尚未验证。

## 研究判断与前面方案的修正

要检验的假设是：**让 B/C 数据学习在部分语义信息缺失时恢复教师表示，能否让
BC 已使用的有标原型具有更好的类内紧凑性、类间区分性和查询分类能力。**

有标少会带来估计方差；仅接纳高置信伪标签还会带来选择偏差。重构不能凭空
补齐类别覆盖，也不能自动修复错误的教师语义。第一阶段只检验表示这一环。

此前讨论不能直接照抄成实现，原因如下：

1. BC 的 trusted 几何和有标原型损失使用 `refresh_reference` 每个本地 epoch
   重建的有标中心。历史 `z_a_sum/p_new/p_ref` 仍存在，但不控制 trusted
   Phase2/3 的主要几何判断。只改 `_proto_write_weight` 很可能得到一项不影响
   当前主要目标的实验。
2. `AE(mask(stopgrad(z_teacher))) -> stopgrad(z_teacher)` 只训练 AE。
   要改善学生，输入端必须来自可求导的学生特征，且教师目标停止梯度。
3. 原 BC 已有 projector/predictor 特征自蒸馏。再叠加一个类似目标会同时改变
   辅助损失强度，难以解释收益。本版替换该项，不额外叠加同类损失。
4. 低重构误差不等于伪标签正确。背景相似、常见模式和模型偏见都可能使错误
   样本易于重构；稀有但正确样本则可能难于重构。筛选会进一步缩窄类内覆盖。
5. 冻结/统计得到的原型若没有梯度，直接给其增加 separation loss 不会训练
   骨干。本版不添加此类无梯度损失，不同时引入多原型、A 重路由或全局参考池。

这借鉴了测试时自监督的思想，但执行于联邦训练期，不使用测试图片更新参数。
真正的 TTT 是在测试样本上优化后再预测，不能将本实验称为已实现 TTT。
参考：[TTT, ICML 2020](https://proceedings.mlr.press/v119/sun20b.html)；
[TTT-MAE](https://arxiv.org/abs/2209.07522)。这些工作提供动机，
不证明本方案在 FSSL/BC 上有效。

## 实际计算图

同一张本地无标签训练图像已有弱、强两种视图，不新增数据增强调用。

```text
强视图 -> 学生 ResNet -> 归一化 256 维特征 zs -> mask -> 瓶颈解码头 -> z_hat
                          ↑                                  |
                          +--------- 重构梯度 ----------------+

弱视图 -> 原 BC EMA 教师 -> 归一化 256 维特征 zt -> stop-gradient 目标
                     \-> 类别软分布 -> 原 BC B 组 KL

学生表示随本地学习更新
    -> 下个本地 epoch 对唯一有标样本重新抽取特征
    -> support 拟合可信原型、query 估计半径/margin/trust
    -> 原 BC 几何软降权和有标原型损失
```

解码头是 256→256→64→256→256，隐藏部分使用 LayerNorm/GELU，没有新 BN。
64 维是信息瓶颈；最后输出 256 维，再归一化。目标直接取教师骨干特征，
不经过可学习的目标 projector。教师仍为原 BC 的本地 EMA，每轮从服务器
学生初始化、轮内 decay=0.99；没有新增跨轮服务器 EMA。

训练项：

`L_rec = mean_batch[1_(B or C) * (2 - 2*cos(D(mask(zs)), stopgrad(zt)))]`

正式实验以 `L_feature_effective = bc_gate * 0.10 * L_rec` 替换旧特征项。
BC gate 仍在 31–90 轮余弦渐增；前 30 轮学生保持原 BC 行为。
损失按完整无标签 batch 平均，只有 B/C 有梯度，A 没有新增类别或特征目标。
原 A CE、B 教师 KL、分类门槛、可信几何、有标原型损失系数均沿用 BC。
表示改变后，后续轮次的预测和路由自然可能改变；不声称全程样本分组恒定。

遮挡每行恰好 floor(0.25*d) 个特征维度，至少保留一个维度。不做 dropout
反向缩放。25% 和 64 维仅是研究起点，不是已证明的最优超参数。

decoder 按客户端保存在 checkpoint，不上传、不参加测试推理。其独立 SGD
使用原 BC 的 LR、momentum=0.9、weight_decay=1e-4，和 BC 主优化器一样每个
本地通信轮重置 momentum。正式实验两个优化器从同一次反传获得各自梯度。
初始化使用隔离 RNG；mask 使用独立 CPU Generator，种子由 round/epoch/step
决定，不改变模型、增强和客户端抽样的随机流。

## 三个配套实验及各自能回答的问题

- `bc_rec_audit`：学生完整保留原 BC。独立头以 `zs.detach()` 重建
  `zt.detach()`，执行独立反向；探针损失不加入学生总损失。
  它是带同类诊断的 BC 对照，同时检验误差的排序信息。它不证明筛选已有效。
- `bc_rec_unmasked`：用相同的 64 维瓶颈头和同样的原始教师目标替换 BC
  特征项，训练时 mask=0。用来区分目标/头结构变化与遮挡的作用。
- `bc_rec_masked`：正式候选，在 unmasked 基础上训练时 mask=25%。
  masked 对 unmasked 的比较才主要识别遮挡作用。

三份完整 YAML 复制自原 BC，只增加 reconstruction 控制；主优化器日程、
通信轮数、分区、客户端抽样、LR、每轮本地步数都保持相同。
头结构和原始教师目标相对原 BC 已经改变，因此 masked 对 BC 的收益不能
单独归因于 mask。相同 loss 系数也不保证梯度范数一致。

## 新增诊断与解释边界

`reconstruction_audit.csv`：每个 round/client 按学生伪标签类别、A/B/C、
置信度带和误差分箱累计正确/错误访问数。置信度带依次为
`<0.60`、`[0.60,0.95)`、`[0.95,0.99)`、`>=0.99`。误差范围 0–4，40 箱。
两个独立分数为：

- `cross_view`：mask 后学生强视图重建教师弱视图的误差，混合了增强难度、
  师生差异和重构能力，不能纯粹解释成异常分数。
- `teacher_self`：mask 后教师特征重建自身的误差，更接近自重构。但 head
  的训练输入是学生强视图，此分数也存在输入分布差异，仍是待验证探针。

所有配置评分使用相同的四组固定 mask，均取 25% 遮挡，先平均误差再分箱。
评分发生在当前 batch 更新前；隐藏无标真值只在 optimizer.step 后累计诊断。
已在此前访问训练过的图片不因此成为独立验证样本。A 不训练 decoder，
其误差分布也不能直接套用 B/C 的阈值。此版本没有误差到权重的转换函数。

`reconstruction_reference.csv`：每个本地 epoch、每类的有标 query 访问数、
最近原型预测正确数、到真类中心的余弦距离总和、真类对最近他类 margin 总和。
复用 BC 实际 support/query 划分和同次 eval 特征，不新增骨干 forward，
不足两类有效参考时无记录。query 只留出于中心拟合，仍参与有监督骨干训练；
这些指标不是独立验证准确率，也不是“真实总体原型误差”。

`dynamics.csv`：`L_feature_effective` 仍是实际进入学生损失的特征项；
`rec_objective` 是未乘 gate/0.10 的重构目标（已按整批平均），
`rec_probe_only` 区分 audit 的独立训练头。原有 loss_reconstructed 必须与
loss_observed 对齐；audit 头的额外优化不计入学生 loss。

首先比较 masked/unmasked/audit 的 query accuracy、own distance、margin、
原 `trust_reference.csv` 和最终分类准确率。单独降低重构 loss 不代表成功，
缩小类内距离但同时丢失类间 margin 也不代表成功。原 `bc_feature_std` 是
辅助检查，不能独立证明没有表示坍塌。

然后看 reconstruction AUROC 是否在**同一预测类别、置信度带和桶内**仍有
排序信息。只看总体 AUROC，容易把分类置信度/常见类别的差异误当成新证据。
不要用无标签真值或测试标签据此在线调阈值；任何控制只用训练有标证据。

## 运行和汇总

在服务器项目根目录使用原 PyTorch 环境，先验证：

```bash
python -m pytest tests/test_bc_reconstruction.py tests/test_explorations.py -q
bash scripts/run_experiment.sh bc_rec_masked 0 7 --dry-run
```

再在明确空闲的 GPU 上分别运行；下面使用同一编号，表示顺序执行：

```bash
bash scripts/run_experiment.sh bc_rec_audit 0 7
bash scripts/run_experiment.sh bc_rec_unmasked 0 7
bash scripts/run_experiment.sh bc_rec_masked 0 7
```

先固定 seed=7 做机制检查，有证据后再配对其他种子，不能把连续轮次当成独立
重复。主指标保留第300轮和271–300轮均值，best 作为辅助。额外头的训练和
每批四 mask×两个分数的前向会增加计算/同步/CSV开销；没有新增评分用骨干
前向，但不能声称相同300轮具有相同计算预算。服务器运行时须记录实际耗时。

```bash
python scripts/summarize_bc_reconstruction.py results/CIFAR10/runs/运行目录 \
  --start 91 --end 300 --output reconstruction_summary.json
```

汇总脚本输出按箱近似的错误伪标签 AUROC：误差越大判为越可能错误，
同箱作平局；缺少正确/错误任一组返回 null。输出还包含类内 query 几何。
所有计数是累计访问数，同一图片/客户端/轮次之间相关。

从头启动三个新配置；不把旧 BC、其他模式的 checkpoint 直接接入。
恢复时必须使用原配置，geometry_controls 包含模式、mask、瓶颈、版本等，
已参与客户端缺少 decoder 状态会报错。恢复支持通信轮边界；不支持本地
batch 中断。新诊断 CSV 沿用基线追加行为，如从较早快照回退，分析前应去除
快照之后的旧日志，不能将重跑行作为额外样本累计。

## 下一阶段：满足证据条件后才扩充原型

若表示有所改善，且误差对错误 A 的判别具有额外信息，再单独实现“伪标签
扩充实际 trusted reference”。这必须进入 `refresh_reference` 的有效中心，
不能只更新历史 p_ref：

1. 每个 epoch 用确定性特征和去重 ID 构建候选，排除有标样本重复写入。
2. 以有标中心为锚，另设伪标签总质量相对有标质量的上限；每个样本权重小
   不足以阻止海量无标样本淹没锚点。缺失有标类别保持无效，不自造可信原型。
3. 固定原始有标参考筛选候选，然后一次性重建扩充参考；不在同批中反复用
   扩充中心自证其伪标签。构建、校准和查询使用一致编码器及预处理。
4. 保留真实有标 query，不把高置信伪标签当成校准真值；证据少时不给重构
   信号决策权。伪标签可能曾受这些 query 训练出的模型影响，仍不能称独立校准。
5. 对照必须包括“相同候选、不用重构误差的质量受限扩充”，以分离增加样本
   与重构加权的收益。检查正确样本被丢弃的比例、罕见模式覆盖和每类改善。

本阶段不预先乘五种可靠性分数、不改变 A/B/C 路由，也不增加服务器原型聚合
策略。这样第一轮结果能明确回答表示假设，失败时也能定位，不必解释多个
同时改变的机制。

## 本地验证结果

CPU PyTorch 2.14：56 项测试、3 个子测试通过。覆盖实际 ResNet8 梯度，教师
停止梯度，audit 学生/BC头逐位不变，前30轮一致，三个阶段隐藏标签隔离，
全batch损失分母，总loss核对，decoder 保存恢复与缺失状态拒绝，分箱AUROC，
有标query统计和BC相关回归。CUDA完整训练、最终准确率及真实通信轮耗时未验证。
