# bc_targets 三个独立改进实验

三个配置分别从 bc_targets 出发，互不叠加；完整训练 300 轮。保持骨干、标签数量、数据划分、种子、LR=0.15、FedAvg、BC 特征系数 0.10 和原阶段日程。测试仍只用原主模型，不做辅助头集成，不使用外部预训练或额外标签。

## 1. bc_targets_evidence：保护被过度分流的 A

配置：`configs/experiment_bc_targets_evidence.yaml`。

保留 targets 原路由和 B/C 目标，只修改部分 A 的最终目标类型。原型参考不足是中性信息，不构成竞争类别的正面证据。教师双视图都支持原类别且概率至少 0.65，成熟历史支持原类别且至少 0.60，并且没有可靠几何竞争者时，保留原硬目标。几何竞争者必须有可信参考、落在自身半径内，且相对原类别的相似度优势超过原类别 margin_scale 的 0.5 倍。

这不是恢复所有不受几何支持的样本：缺少参考、但也缺少支持原类别的正面证据，仍沿用 targets 的候选/特征目标。保留原几何权重，不额外提升权重。目标变化沿用 61–120 轮渐增；原型写入和额外 A 特征学习跟随最终状态，避免一边纠错一边继续写原标签。

本版复用原真标签 support/query 参考的距离统计，没有新增模型级交叉验证校准器，也不把 0.65 当成真实正确率。query 从中心拟合中留出，但没有从既往 backbone 监督训练中留出。

## 2. bc_targets_labelhead：真标签辅助判别来源

配置：`configs/experiment_bc_targets_labelhead.yaml`。

为避免本地特征空间漂移，本版不是直接 FedAvg 本地辅助头。第 31 轮起，每轮只使用本轮在线客户端已有的真标签数据：

1. 以本轮初始全局编码器、eval 模式、确定性归一化提取单位特征。按稳定样本 ID 去重；不访问无标签池用于拟合。
2. 客户端内按类别交替分为两折，上传各折 `XᵀX / XᵀY / 类别计数`。X 含一个偏置维度。服务器汇总后解带正则的线性最小二乘分类头（ridge=1，偏置正则=0.01）。原始图片和逐样本特征不作为服务器消息。
3. 用另一折拟合的头预测留出折，客户端上传按预测类别与分数间隔分箱的计数。每类至少 8 个验证预测、Wilson 下界达到 0.65，才允许从固定间隔网格选可用阈值；最小间隔 0.10。验证不足时阈值无穷大，直接弃权。此下界只是启发式可靠性门槛，经过网格选择，不能声称为严格的泛化保证。
4. 汇总全部在线真标签统计拟合当轮最终辅助头。每类至少 8 个真标签才能参与辅助预测；原伪标签对应类别缺失时也禁止该头覆盖原目标，避免凭空否定缺失类别。
5. 客户端保留一份本轮冻结的全局编码器，以同一特征空间计算辅助头的双弱增强判断；两个视图都通过间隔门槛且同类才算可用。
6. 独立保存每个无标签样本的辅助预测历史：至少两次过去客户端参与、20 轮时效、旧/新 EMA 为 0.7/0.3。同一次参与的重复访问只算一次。不能将当前 batch 写入后立即当作历史证据。
7. 辅助历史与当前判断一致且历史概率至少 0.60，当前 BC 教师双视图也支持该类别且概率均至少 0.60，才改动 A。若支持原标签则保留原硬目标；若支持替代类别，则采用 `0.75 × 辅助分布 + 0.25 × 当前教师平均分布`。辅助分布由 ridge 分数经温度 0.15 的 softmax 得到，**不是校准的正确率**。

替代软目标直接替换原 A 目标，由原 targets 渐增系数控制；渐增完成后没有强制保留 50% 旧标签。被改目标的 A 不向替代类写硬原型，也不继续写旧硬标签。B/C 保留原 targets 逻辑。

实现模拟两次统计通信及头广播；相比原 BC 有额外通信、真标签特征提取和矩阵求解成本。没有增加标签，辅助头不反向训练 backbone。它仍共享既有 backbone，且 backbone 见过这些真标签，不能称为完全独立或无偏教师。没有实现差分隐私，不应将“不上传图片”等同于隐私保证。

## 3. bc_targets_separation：真标签样本关系学习

配置：`configs/experiment_bc_targets_separation.yaml`。

目标分流保持 targets。第 61 轮起增加有标签第二弱视图，对当前批次按真实图片 ID 去重，用真标签定义同类正例、异类负例。单类别 batch 返回零损失，避免无负例时推动塌缩。

使用单位 backbone 特征，不引入新的投影头或跨客户端特征库。温度 0.20，有效系数为 `0.03 × targets_gate`。负例根据当前真标签样本的分类混淆概率给予 1–2 倍有界权重，该权重停止梯度，且绝不把伪标签用于正负例关系。没有硬编码猫/狗/青蛙。现有原型损失仍保留；新目标作用于实际样本对而非再增加一个类别中心目标。

第二视图使用 batch 统计但不写 BN running buffers，也不增加教师更新次数。原型首次初始化仍用原单视图。额外增强会改变第 61 轮后的随机流，不能声称与原 targets 在后续逐位相同。

## 日志与检查

新实验的 `dynamics.csv` 保留原 targets 日志，并增加：

- `experiment_c*_protected*`：被保留原硬目标的访问量及正确/错误名义权重。权重是相对于原 targets 目标状态的名义量，未乘 targets 渐增，不是梯度或最终准确率。
- `experiment_c*_aux_eligible/aux_correct/aux_correct_when_student_wrong`：辅助头是否提供不同且有用的判断。
- `aux_stable/aux_teacher_agrees/aux_override/aux_fix/aux_harm`：区分头不可靠、历史未稳定、教师不支持和最终覆盖目标。fix/harm 指目标 argmax，不表示模型已经学会。
- `experiment_a_pred{c}_true{t}`：按原预测类别与诊断真类统计的 A 混淆矩阵，不能用于反向决定训练。
- `L_separation_effective/separation_active_batches/separation_unique_labeled`：实际新增损失、有多类参与的 batch 数及独立真标签样本访问数。
- labelhead 单独写 `labelhead_calibration.csv`，记录每轮各类支持数、OOF 预测数/正确数、间隔阈值。

隐藏真值只供 audit。所有计数默认是重复训练访问，不是独立图片数量。必须同时看 best、最后 30 轮均值、正确监督保留量和错误目标变化。不能因切换比例下降或伪标签精度上升，就直接判定方法成功。

## 启动与恢复

安装/激活原服务器训练环境。三组都从头跑，不能拿旧 BC/targets checkpoint 直接切换实验身份。配置及方法版本会写进 config/checkpoint，跨配置恢复被拒绝。每轮保存完整 checkpoint，并额外保留 30、60、90、120、225 轮。

两张空闲 GPU（示例 0、1）先跑 evidence、labelhead，任意一个结束后自动启动 separation；单张卡则依次跑。GPU 编号使用当前 CUDA_VISIBLE_DEVICES 映射后的编号。

```bash
python scripts/run_target_experiments.py --gpus 0 1 --dry-run
nohup python -u scripts/run_target_experiments.py --gpus 0 1 > target_experiments_queue.log 2>&1 &
```

队列不会检测其他占卡任务，也不会中断它们，请只填空闲 GPU。每张指定卡最多启动一个新任务；某任务失败后停止领取后续任务，已经启动的任务继续。启动日志在 `results/target_experiment_launches/`，结果在各自独立带时间戳的 runs 目录。

也可单独启动：

```bash
bash scripts/train.sh --config configs/experiment_bc_targets_evidence.yaml --gpu_id 0 --run_id "bc_targets_evidence_$(date +%Y%m%d_%H%M%S)"
bash scripts/train.sh --config configs/experiment_bc_targets_labelhead.yaml --gpu_id 0 --run_id "bc_targets_labelhead_$(date +%Y%m%d_%H%M%S)"
bash scripts/train.sh --config configs/experiment_bc_targets_separation.yaml --gpu_id 0 --run_id "bc_targets_separation_$(date +%Y%m%d_%H%M%S)"
```

中断后，用对应 YAML 和原运行目录的 checkpoint 恢复，不要使用队列命令重新开始来代替恢复：

```bash
bash scripts/train.sh --config configs/experiment_bc_targets_labelhead.yaml --gpu_id 0 --resume results/CIFAR10/runs/实际运行目录/checkpoint.pt
```

两个样本历史均在客户端 checkpoint 中。辅助头另存当轮快照用于审计，下一轮按新全局编码器重新拟合，不复用旧空间参数；不需要持久化辅助优化器。separation 没有新增持久化模型状态。

本地验证覆盖 CPU 小模型多阶段真实优化、隐藏真值扰动、损失重构、轮边界保存/恢复对照，以及项目实际 ResNet-8/scaling=4 的三个实验训练路径。还覆盖辅助头统计聚合、弃权、日志重放和两卡队列限制。实际 ResNet 的 train/eval 返回值与标准 PyTorch 模块不同，新增辅助编码器已按该实现适配。完整服务器 CUDA 训练和最终 acc 尚未验证，不能承诺无任何运行环境问题或必然提分。
