"""BC-only masked feature reconstruction; diagnostic labels never enter objectives.

This is training-time representation learning, not test-time adaptation. The
decoder predicts the EMA teacher's raw normalized backbone feature, without a
learned target projector. Reconstruction error is NOT a correctness probability.
"""
import torch
from torch import nn
import torch.nn.functional as F


VERSION = 1
BINS = 40


def controls(args):
    return dict(version=VERSION, mode=args.bc_reconstruction,
                mask_ratio=args.bc_rec_mask, bottleneck=args.bc_rec_bottleneck,
                seed=2718, score_masks=4, bins=BINS,
                objective="masked_student_to_detached_teacher_2minus2cos",
                population="BC", coefficient="bc_gate*bc_feature_weight")


def initialize(dim, bottleneck, device):
    # Separate initialization stream: baseline model/data RNG is untouched.
    with torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(2718)
        head = nn.Sequential(nn.Linear(dim, dim), nn.LayerNorm(dim), nn.GELU(),
                             nn.Linear(dim, bottleneck), nn.GELU(),
                             nn.Linear(bottleneck, dim), nn.LayerNorm(dim),
                             nn.GELU(), nn.Linear(dim, dim))
    return head.to(device)


def corrupt(z, ratio, seed):
    """Exactly floor(ratio*d) masked channels, retaining at least one per row.

Explicit CPU generator avoids consuming any global CPU/CUDA/augmentation RNG.
Seed is a function of round/epoch/step, so round-boundary resume is reproducible.
No inverted-dropout scaling: normalized masked inputs retain their magnitudes.
"""
    count = min(z.shape[-1] - 1, int(ratio * z.shape[-1]))
    if count <= 0:
        return z
    generator = torch.Generator(device='cpu').manual_seed(int(seed) % (2**63 - 1))
    scores = torch.rand(z.shape, generator=generator, device='cpu')
    indices = scores.argsort(dim=-1)[:, :count]
    mask = torch.ones(z.shape, dtype=z.dtype, device='cpu')
    mask.scatter_(1, indices, 0)
    return z * mask.to(z.device)


def errors(head, source, target, ratio, seed):
    prediction = F.normalize(head(corrupt(source, ratio, seed)), dim=-1)
    clean = F.normalize(target.detach(), dim=-1)
    return (2 - 2 * (prediction * clean).sum(-1)).clamp(0, 4)


def objective(head, student, teacher, mask, ratio, seed, detached=False):
    source = student.detach() if detached else student
    error = errors(head, source, teacher, ratio, seed)
    return (error * mask.detach().to(error.dtype)).mean()


@torch.no_grad()
def score(head, student, teacher, ratio):
    # Common masks across modes, averaged before binning. Both scores are
    # measured BEFORE learning from the current batch. Existing samples may
    # have been seen on earlier visits: this is not a held-out error estimate.
    seeds = (104729, 130363, 155921, 181081)
    return dict(cross_view=sum(errors(head, student, teacher, ratio, s) for s in seeds) / len(seeds),
                teacher_self=sum(errors(head, teacher, teacher, ratio, s) for s in seeds) / len(seeds))


def new_audit(classes, device):
    # score kind, predicted class, ABC, confidence band, error bin, correct/wrong
    return torch.zeros(2, classes, 3, 4, BINS, 2, device=device, dtype=torch.float64)


@torch.no_grad()
def accumulate(buffer, scores, predictions, confidence, in_a, in_b, truth):
    bucket = torch.where(in_a, 0, torch.where(in_b, 1, 2))
    band = (confidence >= .6).long() + (confidence >= .95).long() + (confidence >= .99).long()
    wrong = (predictions != truth).long()
    classes = buffer.shape[1]
    for kind, name in enumerate(('cross_view', 'teacher_self')):
        error_bin = (scores[name] * BINS / 4).long().clamp(0, BINS-1)
        index = ((((predictions * 3 + bucket) * 4 + band) * BINS + error_bin) * 2 + wrong)
        counts = torch.bincount(index, minlength=classes*3*4*BINS*2)
        buffer[kind].add_(counts.reshape_as(buffer[kind]))


def audit_rows(buffer):
    data = buffer.detach().cpu()
    rows = []
    for kind, cls, bucket, band, bin_index in (data.sum(-1) > 0).nonzero().tolist():
        good, bad = data[kind, cls, bucket, band, bin_index].tolist()
        rows.append(dict(score=('cross_view', 'teacher_self')[kind], predicted_class=cls,
                         bucket='ABC'[bucket], confidence_band=band,
                         error_bin=bin_index, error_low=4*bin_index/BINS,
                         error_high=4*(bin_index+1)/BINS, correct=good, wrong=bad))
    return rows


@torch.no_grad()
def reference_audit(z, y, reference):
    """Same support/query split as trusted_geometry; queries excluded from fit.

They ARE used in supervised backbone training, hence not an independent
    validation set. Only valid reference classes with at least one valid rival
    are scored below.
"""
    if int(reference['valid'].sum()) < 2:
        return []
    rows = []
    for cls in range(len(reference['valid'])):
        if not reference['valid'][cls]:
            continue
        indices = torch.where(y == cls)[0][1::2]
        similarities = (z[indices] @ reference['centers'].T).masked_fill(~reference['valid'][None], -1e9)
        own = similarities[:, cls]
        rival = similarities.clone()
        rival[:, cls] = -1e9
        rows.append(dict(cls=cls, query_count=len(indices),
                         query_correct=int((similarities.argmax(-1) == cls).sum()),
                         own_distance_sum=float((1-own).sum()),
                         margin_sum=float((own-rival.max(-1).values).sum())))
    return rows
