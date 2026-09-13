"""Trusted multi-prototype primitives. No unlabeled ground truth enters this module."""
import math
from contextlib import contextmanager
import torch
from torch import nn
import torch.nn.functional as F
from training_dynamics import cosine_learning_rate


def schedule(r, args):
    # Tail length never changes the first main_rounds updates or ramps.
    ramp = min(1., max(0., (r-args.tm_start)/args.tm_ramp))
    if r <= args.tm_main_rounds:
        lr = cosine_learning_rate(r, args.lr_local_training, args.tm_main_rounds, args.lr_min)
    else:
        t = (r-args.tm_main_rounds)/max(args.tm_tail_rounds, 1)
        lr = max(args.lr_min, args.tm_tail_lr * .5 * (1+math.cos(math.pi*t)))
    return lr, ramp


class Learner(nn.Module):
    def __init__(self, backbone, dim):
        super().__init__()
        self.backbone = backbone
        # No BN in auxiliary heads; small/non-IID batches do not define their statistics.
        self.projector = nn.Sequential(nn.Linear(dim, dim), nn.LayerNorm(dim), nn.ReLU(), nn.Linear(dim, 128))
        self.predictor = nn.Sequential(nn.Linear(128, 256), nn.LayerNorm(256), nn.ReLU(), nn.Linear(256, 128))

    def forward(self, x):
        z, logits = self.backbone(x)
        z = F.normalize(z, dim=-1)
        projection = self.projector(z)
        return z, logits, projection


@torch.no_grad()
def ema_update(target, source, decay):
    for t, s in zip(target.parameters(), source.parameters()):
        t.mul_(decay).add_(s, alpha=1-decay)
    # Copy current clean-view running statistics; never EMA integer counters.
    for t, s in zip(target.buffers(), source.buffers()):
        t.copy_(s)
    target.eval()


@contextmanager
def no_bn_tracking(model):
    """Strong views use batch statistics without writing inference running statistics."""
    modules = [(m, m.track_running_stats) for m in model.modules() if isinstance(m, nn.modules.batchnorm._BatchNorm)]
    try:
        for m, _ in modules: m.track_running_stats = False
        yield
    finally:
        for m, flag in modules: m.track_running_stats = flag


@torch.no_grad()
def cluster(z, max_k, minimum):
    """Deterministic spherical k-means, reducing K when clusters have weak support."""
    k = max(1, min(max_k, len(z)//minimum))
    while True:
        centers = [z[0]]
        for _ in range(1, k):
            distance = 1-(z @ torch.stack(centers).T).max(1).values
            centers.append(z[distance.argmax()])
        centers = torch.stack(centers)
        for _ in range(8):
            assignment = (z @ centers.T).argmax(1)
            centers = torch.stack([F.normalize(z[assignment==j].mean(0), dim=0) if (assignment==j).any() else centers[j] for j in range(k)])
        counts = torch.bincount((z @ centers.T).argmax(1), minlength=k)
        if k==1 or int(counts.min())>=minimum: return centers, counts
        k-=1


def bank_logits(z, bank, temperature=.2):
    similarities = torch.einsum('nd,ckd->nck', z, bank['centers'])/temperature
    weights = bank['mass'] / bank['mass'].sum(1, keepdim=True).clamp_min(1)
    logits = torch.logsumexp(similarities + weights.clamp_min(1e-30).log()[None], dim=2)
    return logits.masked_fill(~bank['valid'][None], -1e4)


@torch.no_grad()
def fit_bank(z, labels, classes, max_k, minimum):
    centers = z.new_zeros(classes, max_k, z.shape[1]); mass=z.new_zeros(classes,max_k)
    query_z=[]; query_y=[]
    for c in range(classes):
        rows=z[labels==c]
        if len(rows)<4: continue
        support, query=rows[::2], rows[1::2]
        p,n=cluster(support,max_k,minimum)
        centers[c,:len(p)]=p;mass[c,:len(p)]=n
        query_z.append(query);query_y.append(labels.new_full((len(query),),c))
    bank=dict(centers=centers,mass=mass,valid=mass.sum(1)>0,trust=z.new_zeros(classes))
    if sum(bank['valid']).item()<2 or not query_z: return bank
    qz,qy=torch.cat(query_z),torch.cat(query_y)
    pred=bank_logits(qz,bank).argmax(1)
    for c in range(classes):
        chosen=pred==c;n=int(chosen.sum())
        if n<2:continue
        p=float((qy[chosen]==c).float().mean()); a=1.96**2
        lower=(p+a/(2*n)-1.96*math.sqrt(p*(1-p)/n+a/(4*n*n)))/(1+a/n)
        bank['trust'][c]=lower*min(1.,float((qy==c).sum())/4)
    return bank


@torch.no_grad()
def merge_banks(banks, max_k):
    """Only same-round, same-encoder banks are accepted; no stale feature mixing."""
    out={k:torch.zeros_like(v) for k,v in banks[0].items()}
    for c in range(len(out['valid'])):
        points=[]; masses=[]; trusts=[]
        for b in banks:
            valid=b['mass'][c]>0
            points.extend(b['centers'][c,valid]); masses.extend(b['mass'][c,valid]);trusts.extend([b['trust'][c]]*int(valid.sum()))
        if not points:continue
        # Preserve distinct modes: greedily merge nearest modes using support weights.
        p=torch.stack(points);w=torch.stack(masses);t=torch.stack(trusts)
        while len(p)>max_k:
            sim=p@p.T;sim.fill_diagonal_(-2)
            ij=int(sim.argmax());i,j=divmod(ij,len(p))
            keep=torch.ones(len(p),dtype=torch.bool,device=p.device);keep[i]=False;keep[j]=False
            weight=w[i]+w[j]
            center=F.normalize(p[i]*w[i]+p[j]*w[j],dim=0)
            trust=(t[i]*w[i]+t[j]*w[j])/weight
            p=torch.cat([p[keep],center[None]]);t=torch.cat([t[keep],trust[None]]);w=torch.cat([w[keep],weight[None]])
        out['centers'][c,:len(p)]=p;out['mass'][c,:len(p)]=w
        out['valid'][c]=True;out['trust'][c]=(w*t).sum()/w.sum()
    return out


@torch.no_grad()
def make_targets(teacher_probs, geometry_z, local, global_bank, ramp, args, prior=None):
    def probabilities(bank):
        if int(bank['valid'].sum())<2:return torch.zeros_like(teacher_probs),torch.zeros_like(bank['trust'])
        return bank_logits(geometry_z,bank,args.pp_proto_T).softmax(-1),bank['trust']
    pl,tl=probabilities(local);pg,tg=probabilities(global_bank)
    # Confidence weighting is a heuristic; not a calibrated correctness probability.
    geo=pl*tl[None]+pg*tg[None]
    available=geo.sum(-1)>0
    geo=geo/geo.sum(-1,keepdim=True).clamp_min(1e-8)
    gp=geo.argmax(-1);trust=torch.maximum(tl,tg)[gp]*available
    gamma=args.tm_geometry_mix*ramp*trust
    q=(1-gamma[:,None])*teacher_probs+gamma[:,None]*geo
    if args.tm_distribution_align and prior is not None:
        # Local teacher marginal, compared with smoothed LOCAL labeled frequencies.
        target,marginal=prior
        correction=(target/marginal.clamp_min(.01)).clamp(.5,2).pow(args.tm_distribution_align)
        q=q*correction[None];q=q/q.sum(-1,keepdim=True)
    score,pred=q.max(-1);tp=teacher_probs.argmax(-1)
    agreement=(pred==tp)&((gp==pred)|(~available)|(trust<.2))
    a=(score>=args.tau_warmup)&agreement
    b=(~a)&(score>=args.eta_B)
    c=~(a|b)
    # Unsupported geometry remains a no-op; confident disagreement softens and downweights.
    weight=1-ramp*trust*(gp!=tp).float()
    hard=F.one_hot(pred,q.shape[-1]).to(q.dtype)
    rho=(ramp*score*agreement.float()).clamp(0,1)
    mixed=rho[:,None]*hard+(1-rho[:,None])*q
    return dict(q=q, a_target=mixed, a=a,b=b,c=c,weight=weight,score=score,pred=pred,gamma=gamma)


def objective(student, labeled_output, strong_output, teacher_projection, labels, target, bank, ramp, args):
    zl,ll,_=labeled_output;_,ls,ps=strong_output
    sup=F.cross_entropy(ll,labels)
    logp=F.log_softmax(ls,dim=-1)
    a=(-target['a_target']*logp).sum(-1)
    b=F.kl_div(logp,target['q'],reduction='none').sum(-1)
    la=(a*target['a']*target['weight']).mean()
    lb=(b*target['b']*target['score']*target['weight']).mean()
    valid=bank['valid'][labels]
    proto=zl.sum()*0
    if int(bank['valid'].sum())>=2 and valid.any():
        proto=F.cross_entropy(bank_logits(zl[valid],bank,args.pp_proto_T),labels[valid])
    prediction=F.normalize(student.predictor(ps),dim=-1)
    feature=2-2*(prediction*F.normalize(teacher_projection.detach(),dim=-1)).sum(-1)
    fw=.1*target['a'].float()+.5*target['b'].float()+target['c'].float()
    lf=(feature*fw).mean()
    terms=dict(L_sup=sup,L_A=ramp*args.lambda_A*la,L_B=ramp*args.lambda_B*lb,
               L_proto=ramp*args.lambda_proto*proto,L_feature=ramp*args.tm_feature_weight*lf)
    return sum(terms.values()),terms


@torch.no_grad()
def aggregate(model, states, weights, historical_norm, clip_factor):
    old=model.state_dict();param_names=set(dict(model.named_parameters()))
    norms=[math.sqrt(sum(float((s[k]-old[k]).float().square().sum()) for k in param_names)) for s in states]
    base=historical_norm if historical_norm is not None else sorted(norms)[len(norms)//2]
    cap=max(base*clip_factor,1e-8);scales=[min(1.,cap/max(n,1e-8)) for n in norms]
    result={}
    for k,v in old.items():
        if k in param_names:
            result[k]=v+sum((s[k]-v)*(w*scale/sum(weights)) for s,w,scale in zip(states,weights,scales))
        elif v.is_floating_point():
            result[k]=sum(s[k]*(w/sum(weights)) for s,w in zip(states,weights))
        else:result[k]=torch.stack([s[k] for s in states]).amax(0)
    model.load_state_dict(result)
    median=sorted(norms)[len(norms)//2]
    history=median if historical_norm is None else .9*historical_norm+.1*median
    return history,dict(update_norm_mean=sum(norms)/len(norms),clip_fraction=sum(s<1 for s in scales)/len(scales),clip_cap=cap)
