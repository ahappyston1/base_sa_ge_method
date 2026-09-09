CIFAR-10 结果目录
================

布局：一次实验 = 一个目录，不再有散落在根目录的文件。

  runs/<run_id>_a<alpha>/
      acc.csv        每轮测试精度：round, acc
      metrics.csv    每轮完整指标（51 列，列序见下）
      config.json    该次实验的关键配置与三个随机种子
      train.log      训练日志（每轮一行汇总）
      launch.out     启动脚本的原始 stdout（含 tqdm、异常栈）
      checkpoint.pt  最新一轮的断点，可用 --resume 续训
      tensorboard/   仅在 --use_tensorboard 时生成

  archive/           历史实验（v1/v2/SAGE，以及没有 run_id 的早期 run）

run_id 是启动时间戳。续训：
  bash scripts/train.sh --dataset CIFAR10 --alpha 0.1 \
    --resume results/CIFAR10/runs/<run_id>_a0.1/checkpoint.pt
run_id 从目录名反推，续训会写回同一个目录。


metrics.csv 的列顺序
--------------------
按「先看什么」从左到右排，同类相邻：

  1. 主进度  phase, round, acc, loss
  2. 损失分解 L_sup, L_A, L_B, L_proto
  3. 次进度  best_acc, best_round, gate
  4. 路由    a_ratio, b_ratio, c_ratio, m_a, geom_drop_rate
  5. 质量    a_prec, b_prec, hce_rate
  6. 原型    proto_a_share, proto_a_dirty, proto_a_unique,
             proto_lab_mass, proto_a_mass, lab_mult
  7. 可靠性  rel_r, rel_m, rel_alpha
  8. 门槛    tau0, delta0, eta_B, tau_warmup, cap_a, cap_b,
             lambda_A_scale, lambda_B_scale, use_a_for_proto, lr
  9. 原始计数（供核对上面的比例）
             cnt_u, cnt_a, cnt_b, cnt_c, pass_s, geom_drop,
             a_total, a_correct, b_total, b_correct,
             hce_hc, hce_den_hc, n_batches

几个容易看错的列：
  m_a            = pass_s / cnt_u，通过置信门槛的比例（几何筛选之前）
  geom_drop_rate = geom_drop / pass_s，过了置信门槛又被几何刷掉的比例
  a_prec/b_prec/hce_rate 用了无标真标签，只作诊断，不参与训练与阶段判定
  proto_a_unique 按样本 ID 去重后的 A 质量；同一样本多次进 A 只算一次
  lab_mult       有标特征累计量 / 独立有标样本数，反映一轮内的重复访问倍数

列定义集中在 fl_runner._metrics_row()，顺序由 tests/test_metrics_csv.py 钉住。
