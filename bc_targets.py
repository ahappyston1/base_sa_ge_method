"""First-order target routing. Hidden labels are only used by audit()."""
import math
import torch
import torch.nn.functional as F


def controls():
    return dict(version=1, start=60, end=120, history_decay=.7, max_age=20,
                min_observations=2, candidate_mass=.8, max_candidates=3,
                teacher_conf=.85, history_conf=.75, mode='hard_set_soft',
                prototype='original_label_only_when_hard', calibration='reference_support_only')


def gate(r):
    t=min(1.,max(0.,(r-60)/60.))
    return .5-.5*math.cos(math.pi*t)


class History:
    """Prior history frozen for a whole client participation; duplicates count once."""
    def __init__(self, ids, classes, device, state=None):
        self.ids=torch.tensor(sorted(set(map(int,ids))),dtype=torch.long,device=device)
        n=len(self.ids)
        if state is None:
            self.q=torch.zeros(n,classes,device=device)
            self.count=torch.zeros(n,dtype=torch.long,device=device)
            self.last=torch.zeros_like(self.count)
        else:
            if not torch.equal(self.ids.cpu(),state['ids']):
                raise ValueError('Target history sample IDs differ from client data')
            self.q=state['q'].to(device).clone()
            self.count=state['count'].to(device).clone()
            self.last=state['last'].to(device).clone()
        self.sums=torch.zeros_like(self.q)
        self.visits=torch.zeros_like(self.count)

    def positions(self, ids):
        pos=torch.searchsorted(self.ids,ids)
        if (pos>=len(self.ids)).any() or not torch.equal(self.ids[pos],ids):
            raise ValueError('Unknown sample IDs in target history')
        return pos

    @torch.no_grad()
    def lookup(self, ids, round_idx):
        p=self.positions(ids)
        mature=(self.count[p]>=2)&((round_idx-self.last[p])<=20)&(self.last[p]<round_idx)
        return self.q[p].detach(),mature

    @torch.no_grad()
    def observe(self, ids, q):
        p=self.positions(ids)
        self.sums.index_add_(0,p,q.detach())
        self.visits.index_add_(0,p,torch.ones_like(p))

    @torch.no_grad()
    def finish(self, round_idx):
        used=self.visits>0
        if (self.last[used]>=round_idx).any():
            raise ValueError('History must advance to a new communication round')
        mean=self.sums/self.visits.clamp_min(1)[:,None]
        fresh=(self.count==0)|((round_idx-self.last)>20)
        updated=torch.where(fresh[:,None],mean,.7*self.q+.3*mean)
        self.q[used]=updated[used]
        self.count[used]=torch.where(fresh[used],torch.ones_like(self.count[used]),self.count[used]+1)
        self.last[used]=round_idx
        return {k:getattr(self,k).detach().cpu().clone() for k in ('ids','q','count','last')}


@torch.no_grad()
def decide(pred, a, b, q, p, scores, hist, mature, z, ref):
    target=torch.where(mature[:,None],.5*(q+p)/2+.5*hist,(q+p)/2)
    cq,yq=q.max(-1);cp,yp=p.max(-1);ch,yh=hist.max(-1)
    stable=mature&(yq==yp)&(yq==yh)&(cq>=.85)&(cp>=.85)&(ch>=.75)
    sims=z@ref['centers'].T
    valid=ref['valid']&(ref['trust']>0)
    ranked=sims.masked_fill(~valid[None,:],-1e9)
    best=ranked.argmax(-1)
    own=sims.gather(1,pred[:,None]).squeeze(1)
    others=ranked.clone();others.scatter_(1,pred[:,None],-1e9)
    supported=valid[pred]&(valid.sum()>=2)&((1-own)<=ref['radius'][pred])&((own-others.max(-1).values)>=ref['margin_floor'][pred])
    alt_supported=valid[yq]&(best==yq)&((1-sims.gather(1,yq[:,None]).squeeze(1))<=ref['radius'][yq])
    # 0: original hard A; 1: candidate set; 2: teacher/history soft; 3: feature only.
    state=torch.zeros_like(pred)
    reliable_a=stable&(yq==pred)
    suspicious=a&mature&~supported&~reliable_a
    state[suspicious]=1
    state[suspicious&stable&(yq!=pred)&alt_supported]=2
    state[b]=1
    state[b&stable]=2
    # At most three candidates, retaining current A label and teacher's two leading classes.
    candidates=torch.zeros_like(q,dtype=torch.bool)
    candidates.scatter_(1,target.topk(2,dim=-1).indices,True)
    candidates.scatter_(1,pred[:,None],True)
    mass=(target*candidates).sum(-1)
    confident_set=(mass>=.8)&(mature|(scores>=.6))
    state[(state==1)&~confident_set]=3
    reliability=((scores-.4)/.5).clamp(0,1)
    historical_agreement=1-.5*(target-hist).abs().sum(-1)
    reliability*=torch.where(mature,historical_agreement.clamp(0,1),torch.ones_like(scores))
    return dict(state=state,target=target,candidates=candidates,reliability=reliability,
                mature=mature, a_changed=a&(state!=0), soft_a=a&(state==2))


def objectives(logits,pred,a,b,wa,wb,result,strength):
    logp=F.log_softmax(logits,dim=-1)
    hard=F.cross_entropy(logits,pred,reduction='none')
    partial=-torch.logsumexp(logp.masked_fill(~result['candidates'],-torch.inf),dim=-1)
    soft=F.kl_div(logp,result['target'],reduction='none').sum(-1)
    s=result['state']; routed=torch.where(s==0,hard,torch.where(s==1,partial,torch.where(s==2,soft,torch.zeros_like(hard))))
    la=(wa.detach()*((1-strength)*hard+strength*routed)).mean()
    lb=(wb.detach()*b*result['reliability']*routed).mean()
    proto=1-strength*result['a_changed'].float()
    return la,lb,proto


FIELDS=('a_visits','b_visits','history_mature','a_changed','a_changed_correct',
        'candidate_visits','candidate_contains_truth','soft_visits','soft_correct',
        'soft_a_fix','soft_a_harm','feature_only_visits','a_hard_released_correct_mass',
        'a_hard_released_wrong_mass','b_old_mass','b_new_mass')


@torch.no_grad()
def audit(result,pred,a,b,truth,wa,wb,strength):
    s=result['state'];right=pred==truth;teacher_right=result['target'].argmax(-1)==truth
    changed=result['a_changed'];candidate=(a|b)&(s==1);soft=(a|b)&(s==2)
    removed=wa.detach()*strength*changed
    values=(a,b,result['mature']&(a|b),changed,changed&right,candidate,
            candidate&result['candidates'].gather(1,truth[:,None]).squeeze(1),soft,soft&teacher_right,
            result['soft_a']&~right&teacher_right,result['soft_a']&right&~teacher_right,
            (a|b)&(s==3),removed*right,removed*~right,wb.detach()*b,
            wb.detach()*b*((1-strength)+strength*result['reliability']*(s!=3)))
    out=torch.zeros(result['target'].shape[1],len(FIELDS),device=pred.device,dtype=torch.float64)
    out.index_add_(0,pred,torch.stack([v.double() for v in values],1))
    return out
