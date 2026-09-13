# REV14：可信多原型与教师引导实验

本版本对应“EMA 教师、多原型 trusted、B/C 分层学习、独立收尾”的组合方案。
实现是待验证的研究配方，不承诺达到 88%。不改 REV13 的训练循环或六个旧配置。

## 服务器启动

使用现有装有 torch/torchvision/PyYAML 的环境，在仓库根目录执行。
代码兼容现有 PyTorch 2.1 API；本地实际验证环境为 CPU PyTorch 2.14，未在 CUDA 上跑完 CIFAR10。

```bash
# 推荐：一次得到300轮主训练结果和375轮收尾结果
bash scripts/run_experiment.sh trusted_multi_tail 0 7

# 如果只跑同预算300轮，使用这一条（无需两条都跑）
bash scripts/run_experiment.sh trusted_multi 0 7

# 只打印命令，不启动训练
bash scripts/run_experiment.sh trusted_multi_tail 0 7 --dry-run
```

可通过 `PYTHON=/path/to/python` 选择解释器。GPU 编号为命令第二个参数。
输出目录为 `results/CIFAR10/runs/rev14_trusted_multi[_tail]_s7_<时间戳>_<进程号>_a0.1/`。
第300轮另存 `main_end.pt`，每轮原子更新 `checkpoint.pt`，不自动启动额外任务。

断点恢复使用原配置和原 checkpoint，不需要手填 run_id：

```bash
python ppfpsl.py --config configs/experiment_trusted_multi_tail.yaml \
  --gpu_id 0 --seed 7 --partition_seed 0 --sample_seed 7 \
  --resume results/CIFAR10/runs/你的运行目录/checkpoint.pt
```

不可直接恢复 REV13 checkpoint，也不允许把300轮配置的checkpoint直接改成375轮恢复。
应从一开始选择375轮配置；它与300轮配置的前300轮日程一致。恢复会验证方法、配置与划分哈希，并修剪超过checkpoint轮次的CSV尾部。

## 固定条件与额外成本

- ResNet8、scaling=4，从头训练；没有外部预训练或额外数据。
- CIFAR10 每类500个标签，总5000个；原有 Dirichlet 分配实现、划分种子0、20个客户端每轮抽8个。
- 原有有标弱增强、无标弱/强增强，以及有标并入无标池的约定不变。
- 本地SGD、momentum=0.9、weight_decay=1e-4；每客户端 `local_epochs * floor(N_u / batch_labeled)` 步，默认仍95步。
- 新增投影头/预测头仅用于训练，最终评估仍是学生ResNet8的分类器；不选择教师或集成结果替代主指标。
- 额外成本：EMA模型与冻结参考编码器、弱视图和参考前向、辅助头参数、原型上传、参考下发。相同300通信轮不代表相同FLOPs或总通信字节。
- 375轮属于扩大训练预算；论文主比较使用第300轮及271–300轮均值，收尾结果单独报告。

## 实际算法

1. 每轮开始冻结服务器EMA教师作为几何编码器。所有参与客户端在本地用该编码器提取有标签参考；不会把原始图片传到服务器。
2. 每类按稳定样本ID一半support、一半query，query不参与聚类。样本不足4个时没有本地原型。类别原型上限3，每个聚类至少8个support；不足时退回更少中心。query只留出于原型拟合，并未留出于有监督训练，不视为独立校准集。
3. 用query预测精度的Wilson下界和样本量得到类别可信度启发值。服务器合并当前轮上传的支持中心，优先合并相近形态，按支持数加权；每类最多3个。全局可信度是本地可信度的支持加权汇总，不是全局校准概率。
4. 为避免特征空间漂移，本版只使用当前轮参考，不保留陈旧原型。缺少某类参考时退回教师分类分布。不同类原型相似度通过支持质量归一化的log-sum-exp汇总，防止原型多的类别获得数量优势。
5. 本地EMA教师以服务器教师初始化，每步以0.99跟随学生；服务器教师在聚合后以0.8跟随全局学生。分类软目标使用本地EMA；几何查询仍由本轮冻结编码器提取，与参考保持一致。
6. 第1–10轮仅监督损失；第11–60轮把无标、原型、特征损失和几何作用平滑开启。保持confidence与geometry的意见融合，最大几何混合系数0.5，再乘当前类别可信度和ramp。
7. A要求融合置信度≥0.95且分类/可信几何一致，使用软硬混合目标；B为其余置信度≥0.60样本，学习完整融合软分布；C为其余样本。本版B尚未增加单独的候选集合负样本对比项。
8. 特征目标为学生预测头对EMA教师投影的停止梯度余弦损失；A/B/C权重为0.1/0.5/1，全局系数0.2。有标原型损失系数0.1。教师与几何目标全部detach。
9. 强增强使用批统计，但不写BN running统计；有标/弱增强写running统计。教师BN缓冲从学生复制；服务器浮点BN统计按样本权重平均，整数计数器取max。没有宣称实现严格的全局BN重校准。
10. 客户端参数更新按历史更新范数中位数的EMA限制异常幅度，默认上限为该尺度的2倍；再用原样本权重聚合。第一个回合使用当前客户端范数中位数，不使用测试准确率作任何控制。
11. 300轮余弦LR从0.15下降至0.0001；可选75轮独立收尾，默认保持0.0001，没有学习率重启、没有降低接纳门槛；目标仍按当前证据更新，并非冻结样本名单。

有效总loss（每项已包含系数）：

`L_sup + ramp * (lambda_A*L_A + lambda_B*L_B + lambda_proto*L_proto + tm_feature_weight*L_feature)`

无标各项按整个无标batch平均，避免某个小组样本少时被均值归一化放大。
监督原型项仅对有有效真类参考的有标样本计算。

## 类别纠偏

`tm_distribution_align: 0.0` 默认关闭。启用时用本地有标频率（加一平滑）与本地教师预测EMA做有界修正，比例限制在0.5–2；不强迫客户端十类均匀。它没有引入额外全局类别频率聚合，属于可选的有限本地版本。

## 输出与观察重点

- `acc.csv`：每轮全局学生测试准确率。
- `metrics.csv`：实际有效五项loss、ramp、LR、A/B/C占比与诊断正确率、错误A有效系数、融合目标熵、原型覆盖与上传字节、特征标准差、更新裁剪比例、耗时。
- `teacher_correct` / `fused_correct`：教师与融合目标argmax正确访问数/无标访问数；`correction_good`为改对，`correction_bad`为改错，两者都应看。
- `geometry_valid` 实际表示混合系数>0的访问比例，受ramp与可信度影响；不是单纯的“有中心比例”。
- `feature_std` 为归一化投影在batch维的标准差均值，持续接近0提示特征塌缩；不是准确率。
- `client_updates.csv`：每客户端样本权重、训练步数及原始累计量，可按访问数汇总。
- `partition.json` / `config.json`：实际划分、完整参数、版本和划分哈希。
- `checkpoint.pt`：学生、教师、各客户端预测先验、裁剪历史、所有随机状态及准确率历史。

隐藏无标真值只在optimizer.step之后参与诊断，测试标签只用于evaluate。
不要混用REV13分析脚本：REV14 metrics的loss已含系数，字段口径不同；尤其A/B/C路由发生变化，桶正确率不是同一样本集合的直接比较。

## 测试

```bash
python -m unittest discover -s tests -p 'test_trusted_multi.py' -v
```

包含实际ResNet8反向传播、五项loss梯度、EMA与BN缓冲、多原型归一化、缺失类别、异常更新裁剪、隐藏真值不影响训练，以及两轮模拟联邦训练中断恢复与连续训练参数完全一致。
这是CPU功能验证，不是完整收敛验证。本地独立`.venv-tm`仅用于测试，不上传服务器。
