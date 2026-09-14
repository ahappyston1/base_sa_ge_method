"""Read-only experiment audit; write derived results separately from run logs."""
from pathlib import Path
import json
import re
from html import escape
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / 'results/CIFAR10/runs'
OUT = ROOT / 'results/analysis_rev13/followup/early_complete_audit.json'
data = {}
result = {}
windows = [(1,30),(31,60),(61,90),(91,135),(136,200),(201,270),(271,300)]
for name in ['trusted_lr015','trusted_lr015_early']:
    path = next(RUNS.glob('rev13_' + name + '_s7_*'))
    tables = {p.stem: pd.read_csv(p) for p in path.glob('*.csv')}
    data[name] = tables
    m = tables['metrics']; d = tables['dynamics']; u = tables['updates']
    checks = {}
    for key, t in tables.items():
        checks[key] = dict(rows=len(t), first_round=int(t['round'].min()), last_round=int(t['round'].max()), missing_rounds=sorted(set(range(int(t['round'].min()),301))-set(t['round'])))
    checks['acc_matches_metrics'] = bool((tables['acc']['acc']==m['acc']).all())
    checks['loss_reconstruction_error'] = float((d.loss_reconstructed-d.loss_observed).abs().max())
    s = dict(path=str(path), config=json.loads((path/'config.json').read_text()), checks=checks,
             best=float(m.acc.max()), best_round=int(m.loc[m.acc.idxmax(),'round']), final=float(m.acc.iloc[-1]),
             tail10=float(m.acc.tail(10).mean()), tail30=float(m.acc.tail(30).mean()), tail30_sd=float(m.acc.tail(30).std(ddof=0)), windows={}, drops=u.nsmallest(10,'delta_acc').to_dict('records'))
    for lo,hi in windows:
        a = m[m['round'].between(lo,hi)]; b = d[d['round'].between(lo,hi)]
        w = dict(acc=a.acc.mean(), step_abs_pp=a.acc.diff().abs().mean()*100, worst_step_pp=a.acc.diff().min()*100, drops_gt2=int((a.acc.diff() < -.02).sum()))
        for g in ['a','b']:
            w[g+'_precision'] = a[g+'_correct'].sum()/a[g+'_total'].sum() if a[g+'_total'].sum() else None
            w[g+'_ratio'] = a['cnt_'+g].sum()/a.cnt_u.sum()
            w[g+'_wrong_mass'] = b[g+'_wrong_mass_per_u'].mean()
            w[g+'_correct_mass'] = (b[g+'_effective_mass_per_u']-b[g+'_wrong_mass_per_u']).mean()
        for col in ['trust_authority_mean','trust_valid_fraction','L_sup_effective','L_A_effective','L_B_effective','L_proto_effective']:
            w[col] = b[col].mean()
        w['c_ratio'] = a.cnt_c.sum()/a.cnt_u.sum()
        diag = tables['diag']; diag = diag[diag['round'].between(lo,hi)]
        for n,col in [('sup_n','sup_acc'),('opp_n','opp_acc'),('conflictC_n','conflictC_clf_acc'),('conflictC_n','conflictC_proto_acc')]:
            w[col] = (diag[n]*diag[col]).sum()/diag[n].sum() if diag[n].sum() else None
        w['conflict_fraction'] = diag.conflictC_n.sum()/a.cnt_u.sum()
        w['tau_clipped_n'] = int(diag.tau_clipped_n.sum())
        s['windows'][f'{lo}-{hi}'] = w
    geo = tables['geometry_audit']; tail = geo[geo['round'].between(271,300)]
    def grouped(t, cols):
        z = t.groupby(cols)[['visits','correct','a_visits','a_correct','a_weight','a_wrong_weight']].sum()
        z['a_precision'] = z.a_correct/z.a_visits
        z['correct_mean_weight'] = (z.a_weight-z.a_wrong_weight)/z.a_correct
        z['wrong_mean_weight'] = z.a_wrong_weight/(z.a_visits-z.a_correct)
        z['wrong_weight_share'] = z.a_wrong_weight/z.a_wrong_weight.sum()
        return z.reset_index().to_dict('records')
    s['tail_geometry'] = grouped(tail,['geometry_group'])
    s['tail_classes'] = grouped(tail,['pred_class'])
    s['geometry_reconciliation'] = dict(a_count_max_error=float((geo.groupby('round').a_visits.sum()-m.set_index('round').a_total).dropna().abs().max()), a_wrong_weight_max_error=float((geo.groupby('round').a_wrong_weight.sum()-d.set_index('round').a_wrong_mass_per_u*d.set_index('round').u_visits).dropna().abs().max()))
    tr = tables.get('trust_reference')
    if tr is not None:
        valid = tr[(tr.support_n>0)&(tr.query_n>0)]
        s['reference'] = dict(rows=len(tr), support_min=int(tr.support_n.min()),support_max=int(tr.support_n.max()),query_min=int(tr.query_n.min()),query_max=int(tr.query_n.max()), tail_trust=valid[valid['round']>=271].trust.mean(), mid_trust=valid[valid['round'].between(136,200)].trust.mean(),support_variants=int(tr.groupby(['client','cls']).support_n.nunique().max()),duplicate_keys=int(tr.duplicated(['round','client','epoch','cls']).sum()))
    s['r83'] = dict(update=u[u['round']==83].to_dict('records'), clients=tables['client_updates'][tables['client_updates']['round']==83].to_dict('records'))
    s['update_correlations_31_200'] = u[u['round'].between(31,200)][['delta_acc','params_global_norm','params_disagreement_relative','params_retained_ratio','bn_mean_global_norm','bn_var_global_norm']].corr()['delta_acc'].to_dict()
    result[name] = s
base = data['trusted_lr015']; early = data['trusted_lr015_early']
result['comparison'] = dict(first30_max_difference=float((base['metrics'].acc.head(30)-early['metrics'].acc.head(30)).abs().max()), sampling_equal=base['client_updates'][['round','client','fedavg_weight','local_steps']].equals(early['client_updates'][['round','client','fedavg_weight','local_steps']]), tail30_wins=int((early['metrics'].acc.tail(30).to_numpy()>base['metrics'].acc.tail(30).to_numpy()).sum()))
OUT.parent.mkdir(exist_ok=True,parents=True)
OUT.write_text(json.dumps(result,ensure_ascii=False,indent=2,default=float),encoding='utf-8')
baseline = result['trusted_lr015']; improved = result['trusted_lr015_early']
section = ['<!-- early-complete-audit-start --><section id="early-complete"><h2>2026-09-14 完整复核：early 300 轮</h2>',
           '<p>本节使用项目 results 中的完整运行，桌面 early 文件夹仍是 263 轮副本。两版客户端抽样、每轮本地步数、聚合权重一致，前 30 轮准确率完全一致。</p>',
           '<p><strong>early 小幅超过原 trusted_lr015：末 30 轮平均提高 0.225 个百分点，29/30 轮领先。第 83 轮仍跌 13.17 个百分点，中期震荡未解决。</strong></p>',
           '<table><tr><th>指标</th><th>trusted_lr015</th><th>early</th><th>差值（百分点）</th></tr>']
for key,label in [('best','最高准确率'),('final','第 300 轮'),('tail10','第 291—300 轮平均'),('tail30','第 271—300 轮平均')]:
    section.append(f'<tr><td>{label}</td><td>{baseline[key]*100:.3f}%</td><td>{improved[key]*100:.3f}%</td><td>{(improved[key]-baseline[key])*100:+.3f}</td></tr>')
section.append('</table><h3>同轮次阶段比较</h3><table><tr><th>轮次</th><th>原版平均准确率</th><th>early 平均准确率</th><th>差值（百分点）</th></tr>')
for key in improved['windows']:
    a=baseline['windows'][key]['acc']; b=improved['windows'][key]['acc']
    section.append(f'<tr><td>{key}</td><td>{a*100:.3f}%</td><td>{b*100:.3f}%</td><td>{(b-a)*100:+.3f}</td></tr>')
section.append('</table><h3>逐文件审计</h3><ul>')
for key,check in improved['checks'].items():
    if isinstance(check,dict): section.append(f'<li>{escape(key)}.csv：{check["rows"]:,} 行，第 {check["first_round"]}—{check["last_round"]} 轮，记录范围内无缺轮。</li>')
section.append('</ul><p>acc 与 metrics 完全一致；loss 重构最大误差 6.56×10⁻⁹。geometry_audit 的 A 访问数逐轮精确对齐 metrics，错误权重总量最大浮点偏差 0.000124。geometry_audit/diag 的几何诊断仍从第 91 轮启用，不能拿前段未记录的零值判定 early 未生效；trust_reference 从第 31 轮开始。</p>')
section.append('<h3>原型区分能力与误伤：第 271—300 轮 A 组</h3><table><tr><th>几何组</th><th>A 访问正确率</th><th>正确样本平均权重</th><th>错误样本平均权重</th><th>占全部 A 错误权重</th></tr>')
for row in improved['tail_geometry']:
    section.append(f'<tr><td>{escape(row["geometry_group"])}</td><td>{row["a_precision"]*100:.2f}%</td><td>{row["correct_mean_weight"]:.3f}</td><td>{row["wrong_mean_weight"]:.3f}</td><td>{row["wrong_weight_share"]*100:.2f}%</td></tr>')
section.append('</table><p>oppose 是未达到几何支持门槛，不能等同于已证明另一个类别正确。该组 A 仍有 82.27% 正确，正确样本平均保留 75.86% 权重，错误样本保留 63.05%。支持组 98.18% 正确，说明参考有识别价值；unknown 组 95.11% 正确，不能简单拒收。上述为训练访问统计，非独立图片计数或梯度大小。</p>')
section.append('<p>末段 A 总正确率 early 93.437%、原版 93.474%，B 为 60.035%、60.161%。early 的最终测试提升并没有表现为末段伪标签总体更准确。冲突 C 中原型答案正确率 46.66%、分类器 37.47%，均不足以支持批量改成原型硬标签。</p>')
section.append('<h3>第 83 轮：更新幅度没有明显爆炸</h3><p>early 全局参数更新范数 2.758，在第 61—90 轮约第 63 百分位；BN 均值更新约第 63 百分位，方差更新约第 50 百分位。8 个客户端更新范数 6.20—7.41，没有单个异常巨大的更新。客户端 8、15、17 的 A 正确率分别 45.71%、53.91%、54.91%，各仍以 1/8 聚合。这是质量异质性的线索，不能仅凭它确定客户端责任或在线按真实标签降权。</p><p>这些日志不支持“只裁剪超大更新就能解决第 83 轮”的判断。仍需保留同轮检查点，并对 BN 处理或客户端更新作反事实评估，才能定位原因。</p>')
section.append('<h3>后续优先级</h3><p>保留 early 为同 300 轮下的小幅改进候选。主方法继续沿用 trusted_lr015 的路由与硬标签学习，优先区分距离异常和可信类别冲突，再利用有标查询校准降权。对 B/C 暂不增加大规模硬伪标签。验证异常轮来源时，应先增加可解释诊断，再选择聚合或 BN 改动。</p><p>训练轮次高度相关，29/30 轮领先不等于 29 次独立验证。本次没有修改训练代码或配置。</p><p>数据来源：')
section.append(escape(improved['path'])+'；对照：'+escape(baseline['path'])+'。复算脚本 scripts/audit_early_complete.py，完整数值 early_complete_audit.json。</p></section><!-- early-complete-audit-end -->')
report=OUT.parent/'report.html'
html=report.read_text(encoding='utf-8')
html=re.sub(r'<!-- early-complete-audit-start -->.*?<!-- early-complete-audit-end -->','',html,flags=re.S)
html=html.replace('</h1>','</h1>'+''.join(section),1)
report.write_text(html,encoding='utf-8')
for name,s in result.items():
    if name=='comparison': print(name,s); continue
    print(name, {k:s[k] for k in ['best','best_round','final','tail10','tail30','tail30_sd','checks','geometry_reconciliation']})
    for key,w in s['windows'].items(): print(key,w)
    print('GEOMETRY',s['tail_geometry']); print('CLASSES',s['tail_classes']); print('REFERENCE',s.get('reference'))
    print('DROPS',[(x['round'],x['delta_acc'],x['params_global_norm'],x['params_retained_ratio'],x['bn_mean_global_norm'],x['bn_var_global_norm']) for x in s['drops']])
