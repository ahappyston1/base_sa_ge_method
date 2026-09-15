"""Mutually exclusive A recovery/correction; no rewards, policy or hidden labels."""
import math
import torch
import torch.nn.functional as F


def controls(mode):
    return dict(version=1, mode=mode, recover_start=30, recover_end=90,
                correct_start=60, correct_end=120, recover_cap=.5,
                correction_cap=.5, recover_confidence=.95,
                correction_confidence=.90, recovery_radius_slack=1.,
                teacher_target='mean_two_weak', prototype_write='taper_corrected_only')


def gates(round_idx):
    def ramp(start, end):
        t = min(1., max(0., (round_idx-start)/(end-start)))
        return .5-.5*math.cos(math.pi*t)
    return ramp(30,90), ramp(60,120)


@torch.no_grad()
def decisions(old_weight, in_a, pred, q, p, scores, z, ref, recover_strength, correct_strength, mode):
    q=q.detach();p=p.detach();scores=scores.detach()
    cq, yq=q.max(-1);cp, yp=p.max(-1)
    agree=yq==yp
    own=(z*ref['centers'][pred]).sum(-1)
    fits=ref['valid'][pred] & (ref['trust'][pred] > 0) & ((1-own)<=ref['radius'][pred]+ref['distance_scale'][pred])
    recover=in_a & agree & (yq==pred) & (cq>=.95) & (cp>=.95) & fits & (old_weight<1.)
    correct=in_a & agree & (yq!=pred) & (cq>=.90) & (cp>=.90)
    recover = recover & (mode in ("recover", "joint")) & (recover_strength > 0)
    correct = correct & (mode in ("correct", "joint")) & (correct_strength > 0)
    weight=old_weight.detach()+.5*recover_strength*scores*ref['trust'][pred]*recover*(1-old_weight.detach())
    rho=.5*correct_strength*scores*correct
    # The original classification weight is retained on the correction branch.
    # No hard label is switched, and no weight is recovered on this branch.
    proto_factor=1-correct_strength*correct.float()
    return dict(weight=weight, rho=rho, recover=recover, correct=correct,
                target=(q+p)/2, proto_factor=proto_factor)


def loss(logits, pred, result):
    hard=F.cross_entropy(logits,pred,reduction='none')
    soft=F.kl_div(F.log_softmax(logits,dim=-1),result['target'],reduction='none').sum(-1)
    return (result['weight']*((1-result['rho'])*hard+result['rho']*soft)).mean()


FIELDS=('a_visits','recover_visits','recover_correct','added_correct_weight','added_wrong_weight',
        'correction_visits','correction_student_correct','correction_teacher_correct',
        'correction_fix_visits','correction_harm_visits','old_wrong_weight',
        'new_target_wrong_mass','prototype_removed_correct','prototype_removed_wrong')


@torch.no_grad()
def audit(result,old_weight,in_a,pred,gt,proto_weight):
    """Read-only per-predicted-class counters; sums are repeated sample visits."""
    truth=pred==gt;teacher_truth=result['target'].argmax(-1)==gt
    extra=result['weight']-old_weight
    true_prob=result['target'].gather(1,gt[:,None]).squeeze(1)
    soft_wrong=(1-result['rho'])*(~truth).float()+result['rho']*(1-true_prob)
    removed=proto_weight*(1-result['proto_factor'])
    values=(in_a,result['recover'],result['recover']&truth,extra*truth,extra*(~truth),
            result['correct'],result['correct']&truth,result['correct']&teacher_truth,
            result['correct']&~truth&teacher_truth,result['correct']&truth&~teacher_truth,
            old_weight*(~truth),result['weight']*soft_wrong,removed*truth,removed*(~truth))
    buffer=torch.zeros(10,len(FIELDS),device=pred.device,dtype=torch.float64)
    buffer.index_add_(0,pred,torch.stack([v.double() for v in values],dim=1))
    return buffer
