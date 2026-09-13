"""Extend the existing HTML analysis with the verified follow-up snapshot."""
import html, json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'results/analysis_rev13/followup'
S=json.loads((OUT/'summary.json').read_text(encoding='utf-8'))
D=json.loads((OUT/'series.json').read_text(encoding='utf-8'))
names=['reference','high_lr','trusted','trusted_lr015','legacy_lr020','legacy_lr015_500']
labels=dict(reference='老几何 LR0.10 / 300轮',high_lr='老几何 LR0.15 / 300轮',trusted='Trusted LR0.10 / 300轮',trusted_lr015='Trusted LR0.15 / 300轮',legacy_lr020='老几何 LR0.20 / 300轮',legacy_lr015_500='老几何 LR0.15 / 500轮（仅到412）')
colors=dict(reference='#777777',high_lr='#2267a1',trusted='#777777',trusted_lr015='#087d64',legacy_lr020='#ac5436',legacy_lr015_500='#777777')

def table(headers,rows):
    return '<div class="scroll"><table><thead><tr>'+''.join('<th>'+html.escape(str(x))+'</th>' for x in headers)+'</tr></thead><tbody>'+''.join('<tr>'+''.join('<td>'+html.escape(str(x))+'</td>' for x in r)+'</tr>' for r in rows)+'</tbody></table></div>'

def plot(title,selected,tab,key,lo,hi,scale=1,smooth=1,ylabel='准确率（%）'):
    pts={n:[(r['round'],sum(q[key]*scale for q in D[n][tab][max(0,i-smooth+1):i+1])/len(D[n][tab][max(0,i-smooth+1):i+1])) for i,r in enumerate(D[n][tab]) if lo<=r['round']<=hi] for n in selected}
    ys=[y for p in pts.values() for x,y in p]; low,high=min(ys),max(ys);pad=(high-low)*.08 or .01;low=max(0,low-pad);high+=pad
    X=lambda x:75+(x-lo)/(hi-lo)*850
    Y=lambda y:290-(y-low)/(high-low)*240
    svg=[f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 960 345" role="img" aria-label="{html.escape(title)}"><rect width="960" height="345" fill="white"/><g font-family="Segoe UI,Microsoft YaHei,sans-serif" font-size="13" fill="#26333e">',f'<text x="75" y="24" font-size="17">{html.escape(title)}</text>']
    for i in range(5):
        v=low+(high-low)*i/4;y=Y(v)
        svg.append(f'<line x1="75" x2="925" y1="{y}" y2="{y}" stroke="#e1e5e8"/><text x="67" y="{y+4}" text-anchor="end">{v:.3f}</text>')
    for x in sorted({lo,hi,*[x for x in [60,90,135,200,250,300,350,400] if lo<x<hi]}):
        svg.append(f'<text x="{X(x)}" y="314" text-anchor="middle">{x}</text>')
    for n,p in pts.items():
        path=' '.join(('M' if i==0 else 'L')+f'{X(x):.2f},{Y(y):.2f}' for i,(x,y) in enumerate(p))
        svg.append(f'<path d="{path}" fill="none" stroke="{colors[n]}" stroke-width="2"/>')
    svg.extend([f'<text x="500" y="339" text-anchor="middle">通信轮次（轮）</text><text transform="translate(18,175) rotate(-90)" text-anchor="middle">{ylabel}</text>','</g></svg>'])
    legend=' · '.join(f'<span style="color:{colors[n]}">{labels[n]}</span>' for n in selected)
    return '<figure>'+''.join(svg)+'<figcaption>'+legend+f'<br>来源：各运行的 {tab}.csv，第 {lo}–{hi} 轮；'+('原始逐轮值' if smooth==1 else f'向后 {smooth} 轮均值')+'。曲线仅延伸至本地已有记录。</figcaption></figure>'

report='''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>REV13 新实验复核</title><style>body{font:16px/1.8 "Segoe UI","Microsoft YaHei",sans-serif;color:#202c35;max-width:1080px;margin:36px auto;padding:0 24px;background:#fff}h1{font-size:25px}h2{font-size:21px;margin-top:34px}h3{font-size:18px}a{color:#2267a1}table{border-collapse:collapse;width:100%;font-size:14px}td,th{padding:10px 12px;text-align:left;border-bottom:1px solid #dce2e5;white-space:nowrap}th{background:#f4f6f7}.scroll{overflow:auto}figure{margin:24px 0}svg{width:100%;height:auto}figcaption,.note{font-size:13px;color:#56616a}.lead{border-left:4px solid #087d64;padding-left:18px}li{margin:8px 0}</style><h1>REV13：两组新实验与组合效果复核</h1><p class="note">本地快照：2026-09-13；CIFAR10、α=0.1、模型/抽样种子7、划分种子0。五组300轮已完成，500轮记录仅到412。</p><p class="lead"><strong>Trusted＋LR0.15 是这批同条件300轮实验的最佳方案。</strong>末30轮均值85.507%，比老几何＋0.15高0.655个百分点，且中期更稳定。LR0.20退步。组合大致呈加性收益，尚无明显额外协同的证据。</p>'''
report+='<h2>1. 终点表现：同时看峰值和持续表现</h2>'+table(['实验','最高准确率 / 轮次','末次准确率','末30轮平均','末10轮平均','末30轮标准差（百分点）'],[[labels[n],f'{S[n]["best"]*100:.2f}% / {S[n]["best_round"]}',f'{S[n]["final"]*100:.2f}%',f'{S[n]["last30"]*100:.3f}%',f'{S[n]["last10"]*100:.3f}%',f'{S[n]["last30_sd"]*100:.3f}'] for n in names])
report+='<p>500轮行中的末30轮为383–412轮，不是471–500轮，不能据此判定完整实验胜负。其他五组末30轮均为271–300轮。</p>'
report+=plot('第201–300轮全局测试准确率',['high_lr','trusted_lr015','legacy_lr020'],'acc','acc',201,300,100)
report+='<h2>2. 组合有收益，但不是收益突然倍增</h2><p>以末30轮均值计算：老几何0.10→0.15提升2.074个百分点；Trusted0.10→0.15提升2.137个百分点。Trusted在LR0.10下贡献0.592个百分点，在LR0.15下贡献0.655个百分点。差分中的差分仅+0.063个百分点，峰值口径为−0.050。说明本次两个改动能够兼容，表现近似相加；不能声称显著超加性协同。</p><p>已核对保存的配置、轮次连续性、acc/metrics一致性、损失重构和前300轮客户端抽样；各客户端均95步、FedAvg权重1/8。Trusted0.15与老几何0.15的前90轮准确率完全一致，符合Trusted从第91轮才开启的实现。配置快照并非全部运行环境，缺少多种子复现仍限制结论。</p>'
report+='<h2>3. 稳定性改善了多少</h2>'+table(['区间','方案','平均准确率','平均相邻绝对变化（百分点）','区间内最差单轮变化（百分点）','下降超过2百分点次数'],[[w,labels[n],f'{S[n]["windows"][w]["acc"]*100:.3f}%',f'{S[n]["windows"][w]["step_abs_pp"]:.3f}',f'{S[n]["windows"][w]["worst_step_pp"]:.2f}',S[n]['windows'][w]['drops_gt2pp']] for w in ['136-200','201-270','271-300'] for n in ['high_lr','trusted_lr015','legacy_lr020']])
report+='<p>Trusted0.15在136–200轮把波动从3.326降到2.359个百分点（−29.1%），201–270轮从1.385降到0.750（−45.8%）。但第83轮仍下降13.39个百分点，因为前90轮没有启用Trusted；136–200轮仍有16次下降超过2个百分点。这里相邻变化只统计两个端点都在窗口内的差分。</p>'
report+=plot('全程准确率：相同早期轨迹与后续分离',['high_lr','trusted_lr015','legacy_lr020'],'acc','acc',1,300,100,10)
report+='<h2>4. 权重确实更能区分对错</h2>'+table(['区间','方案','A正确率','A正确/错误平均权重','A错误权重总和/无标签访问','B错误argmax权重/无标签访问','C访问比例'],[[w,labels[n],f'{S[n]["windows"][w]["a_precision"]*100:.2f}%',f'{S[n]["windows"][w]["a_weight_correct"]:.3f} / {S[n]["windows"][w]["a_weight_wrong"]:.3f}',f'{S[n]["windows"][w]["a_wrong_per_u"]:.4f}',f'{S[n]["windows"][w]["b_wrong_per_u"]:.4f}',f'{S[n]["windows"][w]["c_ratio"]*100:.2f}%'] for w in ['136-200','201-270','271-300'] for n in ['high_lr','trusted_lr015']])
report+='<p>201–270轮A错误权重总和/无标签访问下降38.8%，正确A的总权重/无标签访问反而略增（0.581→0.597），因此并非把一切监督都压低。按轮次×客户端×预测类别×置信度区间分层，正确减错误的A权重差从0.0098增至0.1360；每层对错访问均至少5次，以较小计数加权。它支持选择性降权的描述，但不是跨实验同一样本的因果配对。</p><p>注意：权重和不是梯度或损失贡献；B使用软KL，argmax错误不能等同于整个软目标无效；访问包括重复训练和并入无标签池的有标签样本。</p>'
report+=plot('错误A的有效系数总和 / 无标签访问',['high_lr','trusted_lr015','legacy_lr020'],'dynamics','a_wrong_mass_per_u',91,300,1,10,'系数 / 访问')
report+='<h2>5. 仍然存在的具体问题</h2><h3>中期伪标签依然不够可靠</h3><p>Trusted0.15在136–200轮A正确率79.88%、B argmax正确率48.98%；201–270轮分别87.71%、55.47%。因此不能把后期A达到93.47%理解成整个训练过程都很干净。中期仍有错误硬伪标签以平均约0.740的权重参与训练。</p><h3>Trusted覆盖不足，但不能把未知样本直接剔除</h3><p>有效几何访问覆盖从136–200轮87.28%降至末30轮71.11%；平均authority从0.461降至0.373。末段约28.89%访问没有有效几何判据。可未知几何A的正确率95.11%，支持组98.11%，反对组82.33%。因此unknown不是坏样本，反对也不等于错误；现阶段硬拒绝可能损失正确监督。</p><h3>原型写入仍有污染，且应区分哪套原型在生效</h3><p>Trusted0.15在136–200轮记录的A原型写入权重中16.75%来自错误伪标签，末30轮降至5.29%；老几何0.15分别20.94%、6.76%。但Trusted路由和有标原型损失实际使用每epoch从有标签数据刷新出来的参考中心，不能把该写入污染直接认作Trusted loss的污染来源。</p><h3>现有实验没有单独验证哪一部分Trusted贡献最大</h3><p>代码同时改变了路由、A/B权重、几何参考和有标原型损失的参考中心。末30轮有效L_B从0.10265降至0.07664，有效L_proto从0.03726降至0.02496；损失更小不能单独证明表征更好，也可能受参考难度和有效覆盖影响。需同路由、同参考下的权重消融，以及独立lambda_proto=0实验。</p>'
report+='<h2>6. 为什么LR0.20不值得继续作为主线</h2><p>比老几何0.15：峰值−1.04、末30轮−1.035个百分点。136–200轮聚合参数更新范数1.751→2.117（+20.9%），相对客户端分歧0.9254→0.9250没有增加，而平均波动3.326→3.888；A正确率78.57%→76.62%，错误A权重/访问0.1261→0.1347。更符合更新步幅过大与伪标签质量变差相伴的现象，而非支持把问题归因于FedAvg方向冲突。不能由此断言唯一原因或0.15为全局最优。</p>'
report+='<h2>7. 500轮快照：尚未证实延长训练有效</h2><p>当前最高85.47%@339，第412轮85.42%，但383–412轮平均只有82.984%，相邻波动1.498个百分点；这一阶段A正确率83.08%，A写入错误权重占16.56%。它还没有稳定站在85%以上。</p><p>这次500轮从头训练会同时拉长余弦LR与阶段安排：第300轮LR约0.05182，第412轮约0.01118；阶段2从151轮开始，阶段3从226轮开始。原300轮方案在300轮LR已到0.0001。因此它检验的是更长总预算及更慢的日程，不是保留原前300轮再微调200轮。待500轮完整结果再判断，当前不能宣布过拟合或最终失败。</p>'
report+=plot('更长训练预算：向后10轮平均准确率',['high_lr','trusted_lr015','legacy_lr015_500'],'acc','acc',1,412,100,10)
report+='<h2>8. 建议按优先级安排后续实验</h2><ol><li><strong>复现当前胜者。</strong>固定划分0，老几何0.15与Trusted0.15配对增加seed17、27，沿用当前抽样种子随seed变化的约定；以末30轮均值为主要终点。当前单seed+0.655个百分点仍需确认。</li><li><strong>验证选择性降权的贡献。</strong>保持Trusted路由、参考和其他设置不变，只关闭几何权重修正，保留B基础置信度权重。更严格的后续对照可匹配每批平均权重，区分选择性与整体缩放收益。</li><li><strong>若优先解决早期震荡，单独提前启用Trusted。</strong>只移动Trusted参考与软权重开始时点，保持LR、原阶段日程及其他loss不变，需要拆开目前与phase>=2绑定的开关。早期参考差，保留authority保护；不预先保证涨点。</li><li><strong>若优先优化loss，先做B权重消融。</strong>Trusted0.15下仅lambda_B从1降到0.5；另一个独立实验仅lambda_proto=0。它们分别检验B监督强度和有标原型损失是否必要，避免同时改动。</li><li><strong>B/C对比学习放在这些之后。</strong>当前新结果未测试对比学习，C末段访问占7.61%；先确认已有改善机制，再逐项引入新目标。</li></ol><p>训练曲线各轮相关，末30轮标准差不能当成30个独立实验的不确定性；测试集若用于选择方案，后续正式比较应使用独立验证流程和新的重复实验。当前日志没有完整逐类测试结果及逐loss梯度诊断，不据此定位某类退化或断言某一loss主导梯度。</p>'
report+='<h2>来源与复算</h2><ul>'+''.join('<li>'+html.escape(S[n]['folder'])+'</li>' for n in names)+'</ul><p>复算：scripts/analyze_rev13_followup.py → scripts/render_rev13_followup.py。输出summary.json、series.json和本报告；只添加分析文件，未修改训练代码或配置。</p></html>'
(OUT/'report.html').write_text(report,encoding='utf-8')
old=ROOT/'results/analysis_rev13/report.html'
if old.exists():
    text=old.read_text(encoding='utf-8')
    marker='<!-- REV13 FOLLOWUP LINK -->'
    if marker not in text:
        note=marker+'<p><strong>2026-09-13 更新：</strong><a href="followup/report.html">两组新实验及500轮快照分析</a>。下文为旧三组分析，其最优方案和待做实验已由新报告更新。</p>'
        pos=text.find('</h1>')
        text=text[:pos+5]+note+text[pos+5:] if pos>=0 else text+note
        old.write_text(text,encoding='utf-8')
print(OUT/'report.html')
