# -*-coding:utf-8-*-
"""
命令行超参数集中定义。

说明：部分参数（如每类标注数 num_labeled、通信轮数 num_rounds）在 fl_runner.fixmatch()
里按数据集硬编码覆盖。本项目使用 PPFPSL 压力感知原型 + A/B/C 路由方法。

YAML：可用 `--config path/to.yaml` 批量设参；合并顺序为「先读 YAML 作为默认值，再解析命令行」，
故命令行显式传入的项会覆盖 YAML。需安装 PyYAML（见 requirements.txt）。
"""
import argparse
import math
import os
import sys


def _parser_destinations(parser):
    return {
        a.dest
        for a in parser._actions
        if getattr(a, 'dest', None) not in (None, 'help', 'config')
    }


def _yaml_path_from_argv(argv):
    if '--config' not in argv:
        return ''
    i = argv.index('--config')
    if i + 1 >= len(argv):
        return ''
    cand = argv[i + 1]
    if cand.startswith('-'):
        return ''
    return cand


def _apply_yaml_defaults(parser, yaml_path: str) -> None:
    try:
        import yaml
    except ImportError as e:
        raise ImportError('请安装 PyYAML: pip install PyYAML') from e
    with open(yaml_path, 'r', encoding='utf-8') as f:
        raw = yaml.safe_load(f)
    if not raw:
        return
    dests = _parser_destinations(parser)
    merged = {}
    for k, v in raw.items():
        if k == 'config' or not isinstance(k, str):
            continue
        if k not in dests:
            print(f'[options] YAML 忽略未知键（非 argparse 参数）: {k}')
            continue
        if v is None:
            continue
        merged[k] = v
    if merged:
        parser.set_defaults(**merged)


def args_parser():
    """解析命令行，返回 argparse.Namespace。"""
    parser = argparse.ArgumentParser()
    path_dir = os.path.dirname(__file__)

    parser.add_argument(
        '--config',
        type=str,
        default='',
        help='YAML 超参文件路径；与 argparse 同名字段会作为默认值，命令行可逐项覆盖。留空则不读 YAML。',
    )

    parser.add_argument('--gpu_id', type=int, default=0, help='单卡训练时使用的 CUDA 设备编号')
    parser.add_argument('--dataset', type=str, default='CIFAR100',
                        help='数据集：CIFAR10 / CIFAR100 / SVHN / CINIC10')
    parser.add_argument('--num_clients', type=int, default=20,
                        help='联邦学习中客户端总数')
    parser.add_argument('--num_online_clients', type=int, default=8,
                        help='每一轮随机参与训练的客户端数量')
    parser.add_argument('--max_rounds', type=int, default=0,
                        help='若大于 0 则覆盖各数据集默认的 num_rounds，便于快速调试')
    parser.add_argument('--mu', default=2, type=int,
                        help='无标注分支：弱视图 1 份 + 强视图 mu 份中的 mu（与 FixMatch 一致）')
    parser.add_argument('--alpha', type=float, default=1,
                        help='Dirichlet Non-IID 浓度；0 表示 IID，越小通常异质性越强')
    parser.add_argument('--threshold', default=0.95, type=float,
                        help='伪标签置信度阈值，用于 mask')
    parser.add_argument('--lambda_u', default=1, type=float,
                        help='无监督 KL 项相对有监督 CE 的权重')
    parser.add_argument('--batch_size_local_labeled_fixmatch', type=int, default=128)
    parser.add_argument('--kappa', default=0.5, type=float,
                        help='历史 SAGE/CDSC 预留，当前主流程未使用')
    parser.add_argument('--local_epochs', type=int, default=5,
                        help='每轮通信中每个客户端本地训练 epoch 数')
    parser.add_argument('--batch_size_local_labeled', type=int, default=128)
    parser.add_argument('--batch_size_local_unlabeled', type=int, default=128)
    parser.add_argument('--batch_size_test', type=int, default=512)
    parser.add_argument('--num_workers', type=int, default=8,
                        help='本地 DataLoader 的进程数。0 为单进程（最慢）；增广在 CPU 上是主瓶颈，'
                             '8 约提速 5 倍。worker 由 torch 按 base_seed+worker_id 播种，仍可复现，'
                             '但与不同 num_workers 的历史 run 不再逐位一致')
    parser.add_argument('--lr_local_training', type=float, default=0.1)
    parser.add_argument('--lr_mid_factor', type=float, default=1.0, help='LR multiplier ramp; 1=baseline, 0.5=contrast')
    parser.add_argument('--lr_mid_start', type=int, default=60)
    parser.add_argument('--lr_mid_end', type=int, default=90)
    parser.add_argument('--lr_min', type=float, default=1e-4,
                        help='PPFPSL cosine LR floor; 1e-4 preserves REV11, try 0.001 for a tail-only contrast')
    parser.add_argument('--lr_distillation_training', type=float, default=0.1)

    # 各数据集根目录（需含 torchvision 或 CINIC 标准文件夹结构）
    parser.add_argument('--path_cifar10', type=str, default=os.path.join(path_dir, 'data/CIFAR10/'))
    parser.add_argument('--path_cifar100', type=str, default=os.path.join(path_dir, 'data/CIFAR100/'))
    parser.add_argument('--path_svhn', type=str, default=os.path.join(path_dir, 'data/SVHN/'))
    parser.add_argument('--path_cinic10', type=str, default=os.path.join(path_dir, 'data/CINIC10/'))


    #------------- 消融实验 / 其它方法预留超参 -------------#
    parser.add_argument('--SAGE_fixed_p', default=0.5, type=float,
                        help='Fixed parameter used in SAGE ablation study instead of dynamic adjustment')
    parser.add_argument('--ablation_kappa', default=2, type=float,
                        help='Hyperparameter for controlling sensitivity of exponential decay in SAGE ablation study')

    parser.add_argument('--seed', type=int, default=7,
                        help='模型 RNG：torch 初始化、cudnn、DataLoader shuffle。不控制数据划分。')
    parser.add_argument('--partition_seed', type=int, default=0,
                        help='数据划分 RNG：每类有标/无标切分与 Dirichlet 客户端划分。与 --seed 独立。')
    parser.add_argument('--sample_seed', type=int, default=None,
                        help='每轮在线客户端抽样 RNG。默认与 --seed 相同。')

    parser.add_argument('--resume', type=str, default='',
                        help='指定某个 .pt 续训；不填则新开实验（带时间戳，不覆盖正在跑的文件）')
    parser.add_argument('--run_id', type=str, default='',
                        help='实验目录名后缀；留空则用启动时间 YYYYMMDD_HHMMSS。续训同一实验时传入该 id')

    # ------------- 基线方法（FedProx、FedLabel、FreeMatch、FedMatch 等）预留 -------------#
    # FedProx
    parser.add_argument('--lambda_prox', default=0.001, type=float,
                        help='coefficient of FedProx')
    # FedLabel
    parser.add_argument('--fedlabel_lambda', default=1, type=float,
                        help='FedLabel Hyperparameter')

    # FreeMatch
    parser.add_argument('--freematch_sat_ema', default=0.999, type=float,
                        help='ema weight for FreeMatch SAT')
    parser.add_argument('--sat_loss_ratio', default=1, type=float,
                        help='Weight for FreeMatch SAT loss')
    parser.add_argument('--saf_loss_ratio', default=0.05, type=float,
                        help='Weight for FreeMatch SAF loss')

    # FedMatch
    parser.add_argument('--fedmatch_confidence_threshold', type=float, default=0.75)
    parser.add_argument('--fedmatch_lambda_s', default=10, type=float)
    parser.add_argument('--fedmatch_lambda_iccs', default=0.01, type=float)
    parser.add_argument('--fedmatch_lambda_l1', default=0.0001, type=float)
    parser.add_argument('--fedmatch_lambda_l2', default=10, type=float)  # 10
    parser.add_argument('--fedmatch_H', default=2, type=int)
    parser.add_argument('--T', default=1, type=float,
                        help='Temperature of pseudo-labeling')

    parser.add_argument('--num_epochs_label_distillation', type=int, default=50)
    parser.add_argument('--num_epochs_unlabel_distillation', type=int, default=50)
    parser.add_argument('--batch_size_label_distillation', type=int, default=128)
    parser.add_argument('--batch_size_unlabel_distillation', type=int, default=128)

    # ------------- PPFPSL -------------#
    parser.add_argument('--pp_warmup_ratio', type=float, default=0.3, help='warm-up 轮数占比 × num_rounds')
    parser.add_argument('--pp_geom_ratio', type=float, default=0.4, help='phase2 geometry 轮数占比')
    parser.add_argument('--tau_warmup', type=float, default=0.95, help='phase1 无标伪标签置信阈值')
    parser.add_argument('--tau0', type=float, default=0.85, help='类自适应置信阈值基线')
    parser.add_argument('--delta0', type=float, default=0.10, help='类自适应 margin 阈值基线')
    parser.add_argument('--pp_a', type=float, default=0.08, help='tau 随 Norm(rho) 线性项系数')
    parser.add_argument('--pp_b', type=float, default=0.12, help='delta 随 Norm(rho) 线性项系数')
    parser.add_argument('--lambda_p', type=float, default=0.5, help='聚合参考原型时 A 桶权重系数')
    parser.add_argument('--pp_lambda_mix', type=float, default=0.6, help='混合原型 lambda_mix')
    parser.add_argument('--mu_rho', type=float, default=0.9, help='pressure EMA 系数')
    parser.add_argument('--pp_eps', type=float, default=1e-6, help='数值稳定用 eps')
    parser.add_argument('--pp_min_class_count', type=int, default=5, help='参与 pressure 的每类最少样本数')
    parser.add_argument('--eta_B', type=float, default=0.60, help='B 桶可靠性阈值')
    parser.add_argument('--w_min', type=float, default=0.05, help='A 桶样本权重下界')
    parser.add_argument('--w_gamma', type=float, default=1.0, help='A 桶权重中置信度的幂 gamma_w')
    # 解耦「进 CE」与「写原型」：A 全量进硬 CE 不变，仅在写原型时更严。默认 0 = 与旧行为逐位一致。
    parser.add_argument('--proto_conf_floor', type=float, default=0.0,
                        help='写原型的置信度硬门槛：低于此的 A 样本不写原型（不影响 CE）；0=关闭')
    parser.add_argument('--proto_w_extra', type=float, default=0.0,
                        help='写原型的额外置信度幂：w_proto=w*s^extra，越大越压低低置信样本（不影响 CE）；0=关闭')
    # 几何/ C 桶诊断：GT 仅离线统计，只写 diag.csv，绝不参与路由/损失/原型。默认开启，可关。
    parser.add_argument('--diag_geom', type=int, default=1,
                        help='1=记录几何判别与 C 桶原因诊断到 diag.csv（不改训练）；0=关闭')
    parser.add_argument('--diag_s_lo', type=float, default=0.95,
                        help='几何判别诊断的置信下界（含）')
    parser.add_argument('--diag_s_hi', type=float, default=0.99,
                        help='几何判别诊断的置信上界（不含）')
    parser.add_argument('--w_Tp', type=float, default=1.0, help='A 桶权重 sigmoid(m/T_p) 的温度')
    parser.add_argument('--lambda_A', type=float, default=1.0)
    parser.add_argument('--lambda_B', type=float, default=1.0)
    parser.add_argument('--lambda_proto', type=float, default=0.1)
    parser.add_argument('--pp_teacher', type=int, default=0,
                        help='1=弱视图用冻结全局教师；0=学生弱视图（与 81%% 实验一致）')
    parser.add_argument('--pp_b_ratio_cap', type=float, default=0.0,
                        help='B 桶占无标签上限；0 表示不截断')
    parser.add_argument('--mu_p', type=float, default=0.99, help='本地原型 EMA')
    parser.add_argument('--alpha0', type=float, default=0.80,
                        help='gate=0 时 r_i 中 softmax 的权重')
    parser.add_argument('--alpha_min', type=float, default=0.30,
                        help='gate=1 时 r_i 中 softmax 的权重（其余为原型间隔）')
    parser.add_argument('--alpha_max', type=float, default=0.95)
    parser.add_argument('--pp_k0', type=float, default=0.0, help='保留项；α 现在随 gate 过渡，默认 0')
    parser.add_argument('--pp_gamma', type=float, default=1.0, help='高压类进一步降低 α，更看间隔')
    parser.add_argument('--pp_m0', type=float, default=0.05, help='间隔可靠性中心：m=m0 时映射为 0.5')
    parser.add_argument('--pp_mT', type=float, default=0.08, help='间隔可靠性温度，越小对负间隔越严')
    parser.add_argument('--pp_b_min_margin', type=float, default=0.0,
                        help='B 要求原型间隔不低于此值；默认 0 即原型反对 ŷ 的样本进 C')
    parser.add_argument('--pp_proto_T', type=float, default=0.1, help='prototype contrastive 温度 T')
    parser.add_argument('--pp_phase3_geom_boost', type=float, default=0.35,
                        help='phase3 对 tau/delta 几何项的额外放大（随 Norm）')
    parser.add_argument('--pp_tau_ceiling', type=float, default=0.99,
                        help='Phase2/3 置信门槛上限；0 恢复 REV10 不设上限，建议对照 0.99')
    parser.add_argument('--pp_complete_gate', type=int, choices=(0, 1), default=1,
                        help='1=Phase2 完成 gate 退火后才允许退出（包括超时）；0=REV10 gate>=0.95')
    parser.add_argument('--pp_b_conf_rescue', type=int, choices=(0, 1), default=0,
                        help='1=置信度达eta_B也可进B；仍检查B间隔、保留原可靠性KL权重；默认0旧路由')
    parser.add_argument('--pp_geom_mode', choices=('legacy', 'trusted'), default='legacy',
                        help='legacy=current geometry; trusted=fresh labeled references with measured authority')
    parser.add_argument('--pp_trust_min_count', type=int, default=4,
                        help='Minimum unique labeled examples per local class for trusted geometry')
    parser.add_argument('--pp_gate_anneal_ratio', type=float, default=0.15,
                        help='warmup 结束后，将 tau/delta 从 Phase1 余弦退火到目标值的轮数占比')
    parser.add_argument('--pp_warmup_aux_ratio', type=float, default=0.3,
                        help='warmup 最后这段比例内逐步加入 L_proto 与 B 桶，避免 Phase1 过弱')
    parser.add_argument('--pp_adaptive', type=int, default=1,
                        help='1=按停留轮数/gate/A覆盖率推进阶段，不按GT调门槛；0=固定比例')
    parser.add_argument('--pp_adapt_ema', type=float, default=0.8, help='调度指标 EMA 系数')
    parser.add_argument('--pp_adapt_step', type=float, default=0.008, help='每轮门槛调整步长')
    parser.add_argument('--pp_target_a_prec', type=float, default=0.90, help='A 桶目标精度')
    parser.add_argument('--pp_target_a_ratio', type=float, default=0.22, help='A 桶目标占比')
    parser.add_argument('--pp_target_b_prec', type=float, default=0.75, help='B 桶目标精度')
    parser.add_argument('--pp_warmup_min_ratio', type=float, default=0.30, help='Phase1 最少轮数占比')
    parser.add_argument('--pp_warmup_max_ratio', type=float, default=0.40, help='Phase1 最多轮数占比')
    parser.add_argument('--pp_geom_min_ratio', type=float, default=0.12, help='Phase2 最少轮数占比')
    parser.add_argument('--pp_geom_max_ratio', type=float, default=0.35, help='Phase2 最多轮数占比')
    parser.add_argument('--pp_phase1_exit_a_prec', type=float, default=0.70, help='仅日志；切阶段不再使用')
    parser.add_argument('--pp_phase1_exit_acc', type=float, default=0.58, help='仅日志；切阶段不再使用')
    parser.add_argument('--pp_phase2_exit_a_ratio', type=float, default=0.08, help='离开 Phase2 所需 A 占比 EMA（路由计数，无 GT）')
    parser.add_argument('--pp_aux_ready_a_prec', type=float, default=0.68,
                        help='历史项；当前 aux 按轮数打开')
    parser.add_argument('--pp_aux_ready_acc', type=float, default=0.50,
                        help='历史项；当前 aux 按轮数打开')
    parser.add_argument('--pp_a_ratio_cap', type=float, default=0.0, help='A 桶最大占比；0 表示不截断')
    parser.add_argument('--pp_a_ratio_cap_p1', type=float, default=0.0, help='Phase1 A 桶最大占比；0 表示不截断')
    parser.add_argument('--pp_proto_a_prec', type=float, default=0.85, help='A 精度 EMA 低于此则不用 A 更新原型')
    parser.add_argument('--pp_dirty_a_step_mult', type=float, default=3.0, help='A 又大又脏时加快收紧门槛')
    parser.add_argument('--pp_min_c_ratio', type=float, default=0.15, help='C 占比低于此则提高 eta_B，强制拒识')
    parser.add_argument('--hce_tau', type=float, default=0.95, help='HCE 高置信统计阈值')
    parser.add_argument('--use_tensorboard', action='store_true', help='写入 TensorBoard 标量')

    parser.add_argument('--experiment_engine', choices=('baseline', 'trusted_multi'), default='baseline')
    parser.add_argument('--baseline_schedule_rounds', type=int, default=0,
                        help='Baseline LR and phase horizon; 0 uses max_rounds, explicit 300 preserves its schedule during a longer run')
    parser.add_argument('--trusted_weight_start', type=int, default=0,
                        help='0 disables early weighting; 30 starts positive soft weighting at round31')
    parser.add_argument('--trusted_weight_end', type=int, default=60)
    parser.add_argument('--trusted_a_risk', type=int, choices=(0, 1), default=0)
    parser.add_argument('--bc_teacher', type=int, choices=(0,1), default=0)
    parser.add_argument('--bc_ema', type=float, default=.99)
    parser.add_argument('--bc_feature_weight', type=float, default=.10)
    parser.add_argument('--bc_a_repair', choices=('none','recover','correct','joint'), default='none')
    parser.add_argument('--bc_tail', choices=('none','breliability','aguard','featurehalf'), default='none')
    parser.add_argument('--bc_fork', type=int, choices=(0,1), default=0,
                        help='Explicitly fork complete baseline BC round225 state into a new run')
    parser.add_argument('--save_checkpoint_rounds', type=str, default='225')
    parser.add_argument('--stop_after_round', type=int, default=0,
                        help='Stop execution early without changing the LR/phase horizon')
    parser.add_argument('--mid_prox_mu', type=float, default=0.)
    parser.add_argument('--risk_distance_cap', type=float, default=.15)
    parser.add_argument('--risk_conflict_cap', type=float, default=.60)
    parser.add_argument('--risk_prior_count', type=float, default=8.)
    parser.add_argument('--diagnostic_update_rounds', type=str, default='',
                        help='Comma-separated rounds to save pre/local/fused model states for offline diagnosis')
    parser.add_argument('--tm_main_rounds', type=int, default=300)
    parser.add_argument('--tm_tail_rounds', type=int, default=0)
    parser.add_argument('--tm_tail_lr', type=float, default=0.0001)
    parser.add_argument('--tm_start', type=int, default=10)
    parser.add_argument('--tm_ramp', type=int, default=50)
    parser.add_argument('--tm_local_ema', type=float, default=0.99)
    parser.add_argument('--tm_server_ema', type=float, default=0.8)
    parser.add_argument('--tm_prototypes', type=int, default=3)
    parser.add_argument('--tm_min_support', type=int, default=8)
    parser.add_argument('--tm_geometry_mix', type=float, default=0.5)
    parser.add_argument('--tm_feature_weight', type=float, default=0.2)
    parser.add_argument('--tm_clip_factor', type=float, default=2.0)
    parser.add_argument('--tm_distribution_align', type=float, default=0.0)

    yp = _yaml_path_from_argv(sys.argv).strip()
    if yp and os.path.isfile(yp):
        _apply_yaml_defaults(parser, yp)
    elif yp:
        raise FileNotFoundError(f'--config 指定的 YAML 不存在: {yp}')

    args = parser.parse_args()
    if args.bc_a_repair != 'none':
        if not args.bc_teacher or args.bc_tail != 'none' or args.bc_fork:
            parser.error('A repair requires BC, no tail variant, and no BC checkpoint fork')
        if args.dataset != 'CIFAR10' or args.T != 1.0 or args.max_rounds != 300:
            parser.error('A repair recipes require CIFAR10, T=1 and 300 rounds')
        if args.trusted_weight_start != 30 or args.trusted_weight_end != 60:
            parser.error('A repair requires original BC trusted ramp 30-60')
    if (args.bc_tail != 'none' or args.bc_fork) and not args.bc_teacher:
        parser.error('BC tail/fork requires bc_teacher=1')
    if args.bc_tail != 'none' and (args.max_rounds != 300 or args.baseline_schedule_rounds not in (0,300)
                                   or args.bc_feature_weight != .10):
        parser.error('BC tail recipes require max_rounds=300, original 300-round horizon and feature weight=0.10')
    if args.bc_fork and (not args.resume or not args.run_id):
        parser.error('BC fork requires --resume and a new explicit --run_id')
    if args.stop_after_round < 0 or (args.max_rounds and args.stop_after_round > args.max_rounds):
        parser.error('stop_after_round must be within max_rounds')
    try:
        rounds = sorted(set(int(x) for x in args.save_checkpoint_rounds.split(',') if x.strip()))
        if any(r <= 0 for r in rounds):
            raise ValueError()
    except ValueError:
        parser.error('save_checkpoint_rounds must contain positive integers')
    args.save_checkpoint_rounds = ','.join(map(str, rounds))
    if args.bc_teacher or args.mid_prox_mu:
        if args.experiment_engine != 'baseline' or args.pp_geom_mode != 'trusted' or args.pp_teacher:
            parser.error('Exploration requires baseline trusted with pp_teacher=0 (A remains student-routed)')
        if args.trusted_a_risk or (args.bc_teacher and args.mid_prox_mu):
            parser.error('Run risk, BC teacher, and proximal directions independently first')
    if not (0 <= args.bc_ema < 1 and math.isfinite(args.bc_feature_weight) and args.bc_feature_weight >= 0
            and math.isfinite(args.mid_prox_mu) and args.mid_prox_mu >= 0):
        parser.error('Invalid teacher or proximal controls')
    try:
        diagnostic_rounds = [int(x.strip()) for x in args.diagnostic_update_rounds.split(',') if x.strip()]
    except ValueError:
        parser.error('diagnostic_update_rounds must contain comma-separated positive integers')
    if any(r <= 0 or (args.max_rounds and r > args.max_rounds) for r in diagnostic_rounds):
        parser.error('diagnostic_update_rounds must lie within the experiment')
    args.diagnostic_update_rounds = ','.join(str(r) for r in sorted(set(diagnostic_rounds)))
    if args.trusted_a_risk:
        if args.experiment_engine != 'baseline' or args.pp_geom_mode != 'trusted':
            parser.error('trusted_a_risk requires baseline engine with trusted geometry')
        if not (0 <= args.risk_distance_cap <= args.risk_conflict_cap <= 1
                and math.isfinite(args.risk_prior_count) and args.risk_prior_count > 0):
            parser.error('Require 0 <= distance cap <= conflict cap <= 1 and positive risk_prior_count')
    if args.baseline_schedule_rounds < 0 or (args.max_rounds and args.baseline_schedule_rounds > args.max_rounds):
        parser.error('baseline_schedule_rounds must be 0 or at most max_rounds')
    if args.trusted_weight_start < 0 or (args.trusted_weight_start and not args.trusted_weight_start < args.trusted_weight_end):
        parser.error('Require 0 <= trusted_weight_start < trusted_weight_end')
    if args.trusted_weight_start and (args.experiment_engine != 'baseline' or args.pp_geom_mode != 'trusted'):
        parser.error('Early trusted weighting requires the baseline engine and trusted geometry')
    if args.baseline_schedule_rounds and args.experiment_engine != 'baseline':
        parser.error('baseline_schedule_rounds is only for the baseline engine')
    if args.experiment_engine == 'trusted_multi':
        if args.dataset != 'CIFAR10' or args.alpha <= 0:
            parser.error('trusted_multi currently supports CIFAR10 with alpha > 0 only')
        if not (args.tm_main_rounds > args.tm_start >= 0 and args.tm_ramp > 0 and args.tm_tail_rounds >= 0):
            parser.error('Invalid trusted_multi schedule')
        if args.max_rounds not in (0, args.tm_main_rounds + args.tm_tail_rounds):
            parser.error('max_rounds must equal tm_main_rounds + tm_tail_rounds; use the separate schedule fields')
        if not (0 <= args.tm_local_ema < 1 and 0 <= args.tm_server_ema < 1):
            parser.error('EMA rates must be in [0,1)')
        if not (1 <= args.tm_prototypes <= 8 and args.tm_min_support >= 2):
            parser.error('Require 1..8 prototypes and >=2 support examples per prototype')
        if not (0 <= args.tm_geometry_mix <= 1 and 0 <= args.tm_distribution_align <= 1):
            parser.error('Mixture coefficients must be in [0,1]')
        if not (0 < args.tm_tail_lr <= args.lr_local_training and 0 <= args.tm_feature_weight < float('inf') and 0 < args.tm_clip_factor < float('inf')):
            parser.error('Invalid trusted_multi loss, clipping or tail LR')
        if not (0 < args.eta_B < args.tau_warmup < 1 and 0 < args.pp_proto_T < float('inf')):
            parser.error('Require 0 < eta_B < tau_warmup < 1 and positive finite prototype temperature')
        if not all(0 <= v < float('inf') for v in (args.lambda_A,args.lambda_B,args.lambda_proto)):
            parser.error('Loss weights must be finite and nonnegative')
        if not (2 <= args.num_clients and 1 <= args.num_online_clients <= args.num_clients and args.local_epochs > 0 and args.mu > 0 and args.batch_size_local_labeled_fixmatch > 0 and args.num_workers >= 0):
            parser.error('Invalid client, batch, epoch or worker configuration')
    if args.pp_trust_min_count < 4:
        parser.error('pp_trust_min_count must be >= 4 for disjoint support/query splits')
    if args.pp_geom_mode == 'trusted' and (args.pp_teacher or args.pp_b_conf_rescue):
        parser.error('trusted geometry requires student pseudo-labels and pp_b_conf_rescue=0')
    if not (0 <= args.lr_mid_start < args.lr_mid_end and 0 < args.lr_mid_factor <= 1):
        parser.error("Require 0 <= lr_mid_start < lr_mid_end and 0 < lr_mid_factor <= 1")
    if not (0 < args.lr_min <= args.lr_local_training < float('inf')):
        parser.error('Require finite 0 < lr_min <= lr_local_training')
    if not (args.pp_tau_ceiling == 0 or 0 < args.pp_tau_ceiling < 1):
        parser.error('--pp_tau_ceiling 必须为 0（旧行为）或严格介于 0 与 1 之间')
    if getattr(args, 'sample_seed', None) is None:
        args.sample_seed = int(args.seed)
    return args

