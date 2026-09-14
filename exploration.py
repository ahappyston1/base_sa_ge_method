"""Independent teacher/B-C and mid-training proximal experiment primitives."""
import copy
import math
import torch
from torch import nn
import torch.nn.functional as F


def ramp(r, start, end):
    t=min(1.,max(0.,(r-start)/max(1,end-start)))
    return .5-.5*math.cos(math.pi*t)


def prox_coefficient(r, args):
    return args.mid_prox_mu*ramp(r,30,60)*(1-ramp(r,200,250))


def proximal_loss(model, anchor, coefficient):
    # Raw sum over trainable parameters. BN affine included; running buffers excluded.
    return .5*coefficient*sum((p-anchor[n]).square().sum() for n,p in model.named_parameters())


class FeatureHeads(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.projector=nn.Sequential(nn.Linear(dim,dim),nn.LayerNorm(dim),nn.ReLU(),nn.Linear(dim,128))
        self.predictor=nn.Sequential(nn.Linear(128,256),nn.LayerNorm(256),nn.ReLU(),nn.Linear(256,128))


def initialize_heads(dim, device):
    # Fixed initialization isolated from data augmentation and client sampling RNG.
    with torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(1729)
        heads=FeatureHeads(dim)
    return heads.to(device)


@torch.no_grad()
def ema(target, source, decay):
    for t,s in zip(target.parameters(),source.parameters()):t.mul_(decay).add_(s,alpha=1-decay)
    for t,s in zip(target.buffers(),source.buffers()):t.copy_(s)
    target.eval()


def teacher_objectives(student_z, student_logits, teacher_z, teacher_logits,
                       heads, target_projector, b_mask, c_mask, b_weights, temperature):
    # Masking retains the baseline whole-unlabeled-batch denominator.
    q=F.softmax(teacher_logits.detach()/temperature,dim=-1)
    kl=F.kl_div(F.log_softmax(student_logits/temperature,dim=-1),q,reduction='none').sum(-1)
    b=(kl*b_mask.float()*b_weights.detach()).mean()
    pred=F.normalize(heads.predictor(heads.projector(student_z)),dim=-1)
    with torch.no_grad():target=F.normalize(target_projector(teacher_z.detach()),dim=-1)
    feature=((2-2*(pred*target).sum(-1))*(b_mask|c_mask).float()).mean()
    return b,feature
