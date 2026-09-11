"""Read-only, client/class/confidence-stratified geometry audit (Phase2/3).

Counters are sample visits, not independent examples. GT only enters counters.
"""
import csv
import os
import torch

EDGES = (0.0, 0.60, 0.90, 0.95, 0.97, 0.99, 0.995, 1.000001)
GROUPS = ("support", "oppose", "unknown")
FIELDS = ("visits", "correct", "confidence_sum", "a_visits", "a_correct",
          "a_weight", "a_wrong_weight")


@torch.no_grad()
def accumulate(buffer, s, yhat, gt, margin, threshold, valid, in_a, a_weight):
    edges = s.new_tensor(EDGES[1:-1])
    band = torch.bucketize(s.contiguous(), edges, right=True)
    group = torch.where(valid, (margin < threshold).long(), torch.full_like(yhat, 2))
    key = (yhat * (len(EDGES) - 1) + band) * len(GROUPS) + group
    correct = (yhat == gt).double()
    a = in_a.double()
    w = a_weight.detach().double() * a
    values = torch.stack((torch.ones_like(correct), correct, s.double(), a,
                          a * correct, w, w * (1.0 - correct)), dim=1)
    buffer.index_add_(0, key, values)


def new_buffer(num_classes, device):
    return torch.zeros(num_classes * (len(EDGES) - 1) * len(GROUPS), len(FIELDS),
                       dtype=torch.float64, device=device)


def append_rows(path, round_idx, phase, client_id, counters):
    names = ("round", "phase", "client", "pred_class", "s_lo", "s_hi", "geometry_group") + FIELDS
    new = not os.path.exists(path)
    with open(path, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if new:
            writer.writerow(names)
        for key, values in enumerate(counters):
            if not values[0]:
                continue
            cls_band, group = divmod(key, len(GROUPS))
            cls, band = divmod(cls_band, len(EDGES) - 1)
            writer.writerow((round_idx, phase, client_id, cls, EDGES[band],
                             min(EDGES[band + 1], 1.0), GROUPS[group], *values))
