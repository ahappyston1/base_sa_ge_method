"""Render self-contained, reproducible REV13 report and SVG plots."""
import json, math, html
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]; OUT=ROOT/'results/analysis_rev13'
S=json.loads((OUT/'summary.json').read_text(encoding='utf-8'))
D=json.loads((OUT/'series.json').read_text(encoding='utf-8'))
A=json.loads((OUT/'audit_summary.json').read_text(encoding='utf-8'))
names=['reference','high_lr','trusted']
labels={'reference':'Legacy · LR 0.10','high_lr':'Legacy · LR 0.15','trusted':'Trusted · LR 0.10'}
colors={'reference':'#67727c','high_lr':'#1267b1','trusted':'#bd6421'}
def plot(title,table,key,lo=1,hi=300,scale=1,ylabel='',smooth=1):
    width,height=960,330; left,right,top,bottom=72,24,42,50
    series={}
    for n in names:
        rows=D[n][table]; points=[]
        for i,r in enumerate(rows):
            if lo<=r['round']<=hi:
                vals=[q[key]*scale for q in rows[max(0,i-smooth+1):i+1]]
                points.append((r['round'],sum(vals)/len(vals)))
        series[n]=points
    vals=[y for pts in series.values() for _,y in pts]; ymin,ymax=min(vals),max(vals)
    pad=(ymax-ymin)*.09 or 1; ymin=max(0,ymin-pad); ymax+=pad
    X=lambda x:left+(x-lo)/(hi-lo)*(width-left-right)
    Y=lambda y:height-bottom-(y-ymin)/(ymax-ymin)*(height-top-bottom)
    parts=[f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" role="img" aria-label="{html.escape(title)}"><rect width="100%" height="100%" fill="white"/><g font-family="Segoe UI,Arial,sans-serif" font-size="13" fill="#26333e">',f'<text x="{left}" y="22" font-size="16">{html.escape(title)}</text>']
    for i in range(5):
        v=ymin+(ymax-ymin)*i/4; yy=Y(v)
        parts.append(f'<line x1="{left}" x2="{width-right}" y1="{yy}" y2="{yy}" stroke="#e0e5e9"/><text x="{left-9}" y="{yy+4}" text-anchor="end">{v:.2f}</text>')
    ticks=sorted(set([lo,hi]+[x for x in [60,90,135,200,250] if lo<x<hi]))
    for r in ticks: parts.append(f'<text x="{X(r)}" y="{height-bottom+22}" text-anchor="middle">{r}</text>')
    for r in [90,135]:
        if lo<r<hi: parts.append(f'<line x1="{X(r)}" x2="{X(r)}" y1="{top}" y2="{height-bottom}" stroke="#aab3ba" stroke-dasharray="4 5"/>')
    for n,pts in series.items():
        path=' '.join(('M' if i==0 else 'L')+f'{X(x):.2f},{Y(y):.2f}' for i,(x,y) in enumerate(pts))
        parts.append(f'<path d="{path}" fill="none" stroke="{colors[n]}" stroke-width="2"/>')
    parts.extend([f'<text x="{width/2}" y="{height-6}" text-anchor="middle">Communication round</text>',f'<text transform="translate(17,{height/2}) rotate(-90)" text-anchor="middle">{html.escape(ylabel)}</text>','</g></svg>'])
    return ''.join(parts)
charts=[('accuracy','Global test accuracy · raw rounds','acc','acc',1,300,100,'Accuracy (%)',1),('accuracy-tail','Global test accuracy · last 100 rounds','acc','acc',201,300,100,'Accuracy (%)',1),('accuracy-smooth','Global test accuracy · trailing 10-round mean','acc','acc',1,300,100,'Accuracy (%)',10),('a-quality','A pseudo-label precision · trailing 10-round mean','metrics','a_prec',91,300,100,'Precision (%)',10),('a-mass','Effective A coefficient mass per unlabeled visit · trailing 10-round mean','dynamics','a_effective_mass_per_u',91,300,1,'Coefficient / visit',10),('a-wrong','A coefficient mass on incorrect argmax labels · trailing 10-round mean','dynamics','a_wrong_mass_per_u',91,300,1,'Coefficient / visit',10),('b-wrong','B coefficient mass on incorrect argmax labels · trailing 10-round mean','dynamics','b_wrong_mass_per_u',91,300,1,'Coefficient / visit',10),('c-ratio','C visit fraction · trailing 10-round mean','dynamics','c_ratio',91,300,100,'Visits (%)',10),('update-size','Aggregated parameter update norm · trailing 10-round mean','updates','params_global_norm',61,300,1,'L2 norm',10),('update-direction','Relative client-update disagreement · trailing 10-round mean','updates','params_disagreement_relative',61,300,1,'Ratio',10),('trust-valid','Valid geometry fraction · trusted only nonzero','dynamics','trust_valid_fraction',91,300,100,'Visits (%)',10)]
figures=[]
for filename,title,tab,key,lo,hi,scale,yl,sm in charts:
    svg=plot(title,tab,key,lo,hi,scale,yl,sm)
    (OUT/(filename+'.svg')).write_text(svg,encoding='utf-8')
    figures.append('<figure>'+svg+'<figcaption>来源：三组 REV13 '+tab+'.csv；轮次 '+str(lo)+'–'+str(hi)+'；'+('原始逐轮值' if sm==1 else f'向后 {sm} 轮滑动平均（仅作图，不用于汇总统计）')+'。</figcaption></figure>')
def table(headers,rows):
    return '<div class="table-wrap"><table><thead><tr>'+''.join('<th>'+html.escape(str(v))+'</th>' for v in headers)+'</tr></thead><tbody>'+''.join('<tr>'+''.join('<td>'+html.escape(str(v))+'</td>' for v in row)+'</tr>' for row in rows)+'</tbody></table></div>'
headline=table(['实验','最高准确率 / 轮次','最终准确率','末30轮平均','末10轮平均','末30轮波动标准差'],[[labels[n],f'{S[n]["best"]*100:.2f}% / {S[n]["best_round"]}',f'{S[n]["final"]*100:.2f}%',f'{S[n]["last30"]*100:.3f}%',f'{S[n]["last10"]*100:.3f}%',f'{S[n]["last30_sd"]*100:.3f} 个百分点'] for n in names])
phase=table(['轮次','对照平均准确率','高LR平均准确率','trusted平均准确率','高LR−对照','trusted−对照'],[[w]+[f'{S[n]["windows"][w]["acc"]*100:.3f}%' for n in names]+[f'{100*(S[n]["windows"][w]["acc"]-S["reference"]["windows"][w]["acc"]):+.3f} pp' for n in ['high_lr','trusted']] for w in ['1-60','61-90','91-135','136-200','201-270','271-300']])
tail=table(['末30轮指标','对照','高LR','trusted'],[[label]+[f'{S[n]["windows"]["271-300"][key]*scale:.3f}'+unit for n in names] for key,label,scale,unit in [('a_ratio','A访问比例',100,'%'),('a_precision','A伪标签正确率',100,'%'),('b_ratio','B访问比例',100,'%'),('b_precision','B argmax正确率',100,'%'),('c_ratio','C访问比例',100,'%'),('a_mass_per_u','A有效系数 / 无标签访问',1,''),('a_wrong_per_u','A错误系数 / 无标签访问',1,''),('b_mass_per_u','B有效系数 / 无标签访问',1,''),('b_wrong_per_u','B错误argmax系数 / 无标签访问',1,''),('a_weight_correct','A正确样本平均权重',1,''),('a_weight_wrong','A错误样本平均权重',1,''),('geom_drop','通过置信度后被几何拒绝比例',100,'%')]])
matched=table(['轮次','对照：正确−错误权重','高LR','trusted'],[[w]+[f'{A[n][w]["matched_A_weight_gap"]:.4f}' for n in names] for w in ['136-200','201-270','271-300']])
sources=''.join('<li><a href="'+Path(S[n]['folder']).as_uri()+'">'+html.escape(Path(S[n]['folder']).name)+'</a></li>' for n in names)
report='''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>REV13 三组实验复核</title><style>body{font:16px/1.75 "Segoe UI","Microsoft YaHei",sans-serif;color:#24313c;background:#fff;max-width:1120px;margin:36px auto;padding:0 24px}h1{font-size:28px}h2{font-size:21px;margin-top:34px}p{max-width:1000px}table{border-collapse:collapse;width:100%;font-variant-numeric:tabular-nums;font-size:14px}th,td{padding:10px 12px;text-align:left;border-bottom:1px solid #dce2e7}th{background:#f3f5f7}.table-wrap{overflow:auto}figure{margin:24px 0}figure svg{width:100%;height:auto}figcaption,.note{font-size:13px;color:#54616d}.legend{display:flex;gap:22px;flex-wrap:wrap}.legend span:before{content:"";display:inline-block;width:24px;height:3px;background:var(--line);vertical-align:middle;margin-right:8px}.lead{font-size:19px}a{color:#1267b1}details{margin:18px 0}summary{cursor:pointer;font-weight:600}</style><h1>REV13：高学习率提高最终成绩，trusted改善中期稳定性</h1><p class="note">来源：2026-09-12 完成的三组 CIFAR-10、α=0.1 实验；每组300轮，seed=7，partition seed=0。逐轮配对分析，不把相邻轮当作独立重复实验。</p>'''
report+='<p class="lead">高LR末30轮均值 84.852%，比对照高 2.074 个百分点；trusted为83.370%，高0.592个百分点。高LR的优势在后期扩大，并没有解决中期波动。下一步优先补齐 trusted + LR=0.15 的第四组，检验两种改动是否互补。</p>'+headline
report+='<div class="legend">'+''.join(f'<span style="--line:{colors[n]}">{labels[n]}</span>' for n in names)+'</div>'+figures[0]+figures[1]
report+='<h2>1. 对照条件与完整性</h2><p>三组acc、metrics、updates、dynamics均包含连续1–300轮，准确率文件相互一致。配置快照显示：高LR与reference只差初始LR（0.10→0.15）及run_id；trusted只差几何模式及run_id。随附完整YAML支持同样的控制设计，但快照未记录Git提交和全部环境参数，不能证明机器环境完全相同。三组每轮客户端ID序列、FedAvg权重和本地步数相同，每客户端95步，每轮760步；阶段均为1–90 / 91–135 / 136–300。trusted与reference前90轮准确率逐轮完全相同。</p><p>高LR改变的是余弦学习率曲线的整体尺度，不只是第1轮；在触及相同0.0001下限前通常为对照的1.5倍。trusted模式同时改变参考构建、路由、权重及原型监督目标，所以不能将其收益单独归因于某一项。</p>'
report+='<h2>2. 收益在什么时候出现</h2>'+phase+figures[2]+'<p>136–200轮trusted平均79.804%，高LR78.402%，对照78.105%；到末30轮高LR领先。高LR136–200轮平均相邻轮绝对变化3.326pp，对照2.558pp，trusted1.837pp；高LR这一窗口最大单轮跌幅10.21pp。最后30轮相邻波动高LR0.258pp、对照0.257pp、trusted0.173pp。标准差同时包含趋势，不应单独等同训练噪声。</p>'
report+='<h2>3. A/B/C与实际学习强度</h2>'+tail+'<p>高LR后期A更多且更准确，A错误系数质量略降，B错误argmax系数质量也下降。但136–200轮高LR的A正确率78.573%略低于对照79.032%，所以并非从训练开始就产生更干净的伪标签。这里的系数质量是权重之和除以无标签访问数，不是损失、梯度范数或因果伤害；B采用软KL，argmax错误也不等于整个软目标毫无价值。访问数包含同一样本的重复训练以及并入无标签池的有标签样本。</p>'+figures[3]+figures[4]+figures[5]+figures[6]+figures[7]
report+='<h2>4. trusted确实会区分正确和错误，但并非没有副作用</h2><p>201–270轮A正确/错误平均权重：对照0.971/0.940，高LR0.974/0.945，trusted0.935/0.718。trusted保留较多正确监督，同时显著降低错误监督。为减轻置信度和类别构成干扰，按通信轮×客户端×预测类别×置信度区间分层，每层正确与错误A访问均至少5次，以二者较小计数加权，结果如下。</p>'+matched+'<p>这些是在各自训练轨迹内的描述性比较，分层后仍非跨实验同一样本配对，不能当成因果效应或显著性检验。</p><p>trusted有效几何覆盖率从136–200轮85.69%降至末30轮71.17%，平均authority从0.453降至0.373；几何并非没有参与。84,000条客户端×epoch×类别记录中64.6%具有正trust，但重复记录不是独立证据。</p><p>末30轮trusted的未知几何A访问正确率93.52%，已知且支持A为97.44%，已知但反对A为79.92%。因此未知原型不能直接当坏样本；反对组也含大量正确样本，尚不宜把软降权改为硬拒绝。三个组的置信度与类别分布不同，不作直接因果比较。legacy与trusted的几何分数单位不同，不能合并它们的support/oppose门槛。</p>'+figures[10]
report+='<h2>5. 更新幅度变大，不等于方向冲突变严重</h2><p>136–200轮聚合参数更新L2范数：对照1.338，高LR1.751（约+30.9%），trusted1.228。相对客户端分歧分别0.9270、0.9254、0.9283；保留比例分别0.3760、0.3798、0.3729。高LR主要增大更新幅度，没有显示相对方向冲突明显恶化。BN运行统计变化也更大，但这些观察不能证明BN或FedAvg导致准确率下跌。</p>'+figures[8]+figures[9]+'<p>所有客户端均95步，FedAvg权重均1/8，本轮实验不支持把不同客户端步数作为FedNova的优先动机。较低保留比例说明存在抵消，不证明抵消有害或更换聚合器会提高准确率。</p>'
report+='<h2>6. 历史结果与耗时边界</h2><p>在当前α=0.1、300轮的本地run记录中，高LR具有最高峰值和末30轮均值。500轮历史base500峰值85.19%、末30轮84.964%，与高LR300轮峰值85.31%、末30轮84.852%接近；不同修订和训练预算只能作背景，不能严格断言同等计算效率提升。α=1的86.31%属于不同数据异质性设置，不应混为同条件排名。</p><p>按第1轮至第300轮日志时间跨度，对照12.91小时、高LR12.48小时、trusted12.82小时；缺少受控硬件与运行竞争条件，不能据此声称trusted没有额外开销或高LR计算更快。未包含第1轮之前的启动与首轮时间，也不是纯GPU计算计时。</p>'
report+='<h2>7. 下一步实验优先级</h2><ol><li><strong>补齐trusted + LR0.15。</strong>其余配置完全保持trusted配方。现有三组构成2×2设计的三个格子；第四组可检验提高LR与可信几何是否互补，或二者组合抵消收益。不要承诺收益相加。</li><li><strong>配对重复seed17、27。</strong>固定partition_seed=0，至少对reference/high_lr重复，理想情况下四组均重复。用预先约定的末30轮均值作主要终点，报告种子间差异；更广泛结论再更换划分种子。</li><li><strong>如需机制解释，后做同路由下几何权重消融。</strong>在trusted的参考与损失目标均不变时仅将修正权重设为1，以判断有多少收益来自可靠性降权，而非替换原型损失或路由。此项不是首个优先训练任务。</li><li><strong>暂缓多原型、C新损失和FedAdam同时上。</strong>目前只有约8–10%的C访问，trusted已更好利用数据却未获得最高终点成绩。先解决LR与几何搭配，后续再依据失败类别和参考覆盖率选择大改方向。</li></ol><p>若后续探索LR0.20，应作为新的小范围搜索，不能从0.15优于0.10推断越大越好。当前中期波动已提示稳定性代价。调参使用验证集；本次测试集比较若用于选择方案，最终需新的重复验证。</p>'
report+='<h2>来源与复算</h2><ul>'+sources+'</ul><p>复算脚本：scripts/analyze_rev13_results.py、scripts/audit_rev13_results.py、scripts/render_rev13_report.py。输出：summary.json、audit_summary.json、series.json及SVG图。无训练代码或实验配置被修改，未启动新的训练。</p></html>'
(OUT/'report.html').write_text(report,encoding='utf-8')
print(OUT/'report.html')
