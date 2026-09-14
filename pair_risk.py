"""Directional, confidence-matched query evidence for classrisk v2.

Only labeled query observations are counted. Each reference refresh replaces
the statistics; repeated epochs are never counted as independent evidence.
"""
import torch

MIN_PAIR = 4
MIN_REST = 8
MAX_ADJUSTMENT = .20
MAX_PENALTY = .85


@torch.no_grad()
def rival_details(z, pred, ref):
    sims=z @ ref['centers'].T
    own=sims.gather(1,pred[:,None]).squeeze(1)
    other=sims.masked_fill(~ref['valid'][None,:],-1e9).clone()
    other.scatter_(1,pred[:,None],-1e9)
    value,rival=other.max(1)
    strength=((value-own)/ref['margin_scale'][pred]>=2.).long()
    return rival,strength


@torch.no_grad()
def fit(z, pred, y, confidence, selected, group, ref):
    c=len(ref['valid']);rival,strength=rival_details(z,pred,ref)
    band=(confidence>=.99).long()
    count=z.new_zeros((c,c,2,2));errors=torch.zeros_like(count)
    mask=selected & (group==2)
    key=(((pred[mask]*c+rival[mask])*2+band[mask])*2+strength[mask])
    count.view(-1).index_add_(0,key,torch.ones_like(confidence[mask]))
    errors.view(-1).index_add_(0,key,(pred[mask]!=y[mask]).to(z.dtype))
    return dict(count=count,errors=errors)


def wilson(errors,n):
    safe=n.clamp_min(1.);p=errors/safe;z=1.96
    denominator=1+z*z/safe
    center=(p+z*z/(2*safe))/denominator
    radius=z*torch.sqrt((p*(1-p)/safe+z*z/(4*safe.square())).clamp_min(0))/denominator
    return (center-radius).clamp_min(0),(center+radius).clamp_max(1)


@torch.no_grad()
def adjust(z,pred,ref,band,group,usable,base_penalty):
    rival,strength=rival_details(z,pred,ref)
    stats=ref['risk_calibration'];pairs=stats['pairs']
    n=pairs['count'][pred,rival,band,strength]
    e=pairs['errors'][pred,rival,band,strength]
    # Same predicted class/confidence, excluding this exact pair/strength cell.
    # This comparator includes both conforming and conflicting observations.
    rest_n=(stats['count'].sum(-1)[pred,band]-n).clamp_min(0)
    rest_e=(stats['errors'].sum(-1)[pred,band]-e).clamp_min(0)
    lo,hi=wilson(e,n);rest_lo,rest_hi=wilson(rest_e,rest_n)
    eligible=usable & (group==2) & (n>=MIN_PAIR) & (rest_n>=MIN_REST)
    # Non-overlapping intervals provide a conservative evidence margin. These
    # are operational thresholds, not formal repeated-testing significance.
    delta=(lo-rest_hi).clamp_min(0)-(rest_lo-hi).clamp_min(0)
    delta=torch.where(eligible,delta.clamp(-MAX_ADJUSTMENT,MAX_ADJUSTMENT),torch.zeros_like(delta))
    penalty=torch.where(delta!=0,(base_penalty+delta).clamp(0,MAX_PENALTY),base_penalty)
    return dict(penalty=penalty,rival=rival,strength=strength,band=band,
                pair_n=n,rest_n=rest_n,eligible=eligible,delta=penalty-base_penalty)


FIELDS=('a_visits','a_correct','risk_weight','new_weight','risk_wrong_weight',
        'new_wrong_weight','eligible_visits','strengthened_visits','relaxed_visits')


def new_audit(c,device):
    return torch.zeros(c*c*4,len(FIELDS),device=device,dtype=torch.float64)


@torch.no_grad()
def audit(buffer,result,pred,in_a,gt):
    p=result['pair'];m=(in_a & (result['group']==2)).double();correct=(pred==gt).double()
    old=result['base_weight'].double()*m;new=result['weight'].double()*m
    vals=torch.stack((m,m*correct,old,new,old*(1-correct),new*(1-correct),
                      m*p['eligible'],m*(p['delta']>0),m*(p['delta']<0)),1)
    c=int((buffer.shape[0]/4)**.5)
    key=((pred*c+p['rival'])*2+p['band'])*2+p['strength']
    buffer.index_add_(0,key,vals)


def rows(buffer):
    c=int((buffer.shape[0]/4)**.5);out=[]
    for i,row in enumerate(buffer.cpu().tolist()):
        if row[0]:
            out.append(dict(pred_class=i//(c*4),rival_class=(i//4)%c,
                            confidence_band=(i//2)%2,strength_band=i%2,**dict(zip(FIELDS,row))))
    return out
