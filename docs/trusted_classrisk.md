# trusted_lr015_classrisk v2：有向类别冲突条件风险

2026-09-14。替代尚未正式运行的 classrisk v1，保持 risk、bc、prox 配置。config.json 中 class_risk_version=2，不能从旧版本或其他方法的 checkpoint 续训。

## 为什么重做

原 risk 已按预测类别×置信带×几何状态校准。v1 仅将冲突上限从 0.60 降至 0.45—0.60，不能增加错误抑制。v2 删除该倍率，检验具体的“预测类别 c → 竞争原型类别 d”是否提供额外错误证据。所有类别对遵循同一规则，不预设猫/狗/青蛙彼此混淆。

## 查询与分组

每个本地 epoch 复用原参考提取。查询不参与中心拟合，但参加过监督训练，仍可能偏容易。只用有标签查询校准，无标签真值仅用于日志。

查询按预测类别、竞争类别、eval 置信度是否达到 0.99、竞争优势/本类 margin_scale 是否达到 2 分组；须满足当前 A 门槛及原 risk 的 conflict 条件。原 conflict 仍要求优势超过 0.5×margin_scale、双方参考有效、图片落入竞争类别自己的半径、竞争 trust>0。

每次刷新重新计数，不把相同查询跨 epoch 重复出现当成独立样本。没有引入额外 EMA 教师。

## 条件风险比较

每格子与“同预测类别、同置信带的其他查询”比较。比较组排除当前格子，包含几何支持、距离异常和其他冲突。这检验置信度之外的条件信息，不是隔离竞争类别本身的因果作用。

格子至少 4 个查询，比较组至少 8 个查询。各计算 z=1.96 的 Wilson 区间：

    delta = max(pair_lower - rest_upper, 0)
            - max(rest_lower - pair_upper, 0)
    delta = clip(delta, -0.20, +0.20)

数量不足、区间重叠、参考无效、eval 与原训练预测类别不一致或 eval 置信度不达门槛时，delta=0，精确回退原 risk。区间是操作门槛，不是经过多重检验校正的显著性声明。

## 权重修正

先完整计算原 risk 的 penalty。仅证据支持时：

    new_penalty = clip(risk_penalty + delta, 0, 0.85)
    new_A_weight = 1 - original_authority * new_penalty

authority 沿用原启动门控、双方参考可信度和年龄衰减。只修正 conflict；distance、neutral、unknown 不变。没有二次乘原权重，不改 A 标签、路由、B、原型损失、FedAvg 或 BN。权重总体下界 0.15，实际受 authority 进一步限制。前 30 轮与原 risk 一致。

这些数值是预先固定的研究选择，不是已经证明最优的参数。

## 诊断

- pair_risk_audit.csv：按轮次、客户端、预测类、竞争类、置信带、强度记录 A 访问数/正确数、原 risk 和新权重总量、两者错误权重、满足数量门槛的访问量、实际加强/放松量。原 risk 是同一新模型上旧规则的即时对照，不是旧实验训练轨迹。
- pair_calibration.csv：逐 epoch 非空冲突格子的查询数/错误数。未出现表示没有查询，不代表零错误率。跨 epoch 累加不是独立样本数。
- risk_calibration.csv：原类别/置信带/几何状态统计，提供比较组计数。
- risk_audit.csv：继续记录实际权重，old_weight 仍是原 trusted，不是 risk。
- dynamics.csv 使用新实际 A 权重，总 loss 口径不变；保留第 82—84 轮快照。

先看真实介入比例，再看正确/错误权重是否分离，最后看 300 轮最高、最终和末 30 轮平均准确率。很少触发说明细粒度查询证据稀少，不能据此判定类别对方向无效。放松权重也可能放过错误答案；查询偏乐观的风险没有被消除。

## 服务器复制

建议另建 SAGE_classrisk 代码目录，复用原环境和数据，保持旧目录运行旧实验及队列。服务器已有前三个实验完整代码时，更新以下七个文件：

    options.py
    fl_runner.py
    trusted_geometry.py
    trusted_risk.py
    pair_risk.py
    scripts/run_experiment.sh
    configs/experiment_trusted_lr015_classrisk.yaml

新目录也必须具备原项目其余模块，不能只放这七个文件。不覆盖旧 results、checkpoint、配置和排队脚本。用空闲 GPU 从头训练：

    bash scripts/run_experiment.sh trusted_lr015_classrisk 0 7 --dry-run
    bash scripts/run_experiment.sh trusted_lr015_classrisk 0 7

dry-run 仅验证启动命令；实际训练分支另有 CPU 测试。最终准确率需完整 GPU 实验确认。
