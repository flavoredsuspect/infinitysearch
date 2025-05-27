# utils.py
import numpy as np
import torch
import torch.nn.functional as F

def rel(true_neighbors, run_neighbors, metrics=None, *args, **kwargs):
    all_deltas = []
    for gt, pred in zip(true_neighbors, run_neighbors):
        gt_list = list(gt)
        deltas = []
        for i, p in enumerate(pred):
            try:
                true_rank = gt_list.index(p)
            except ValueError:
                true_rank = len(gt_list)
            deltas.append(true_rank - i)
        all_deltas.append(deltas)
    flat = [x for row in all_deltas for x in row]
    rel_signed = float(np.mean(flat))
    rel_abs = float(np.mean(np.abs(flat)))
    if metrics is not None:
        attr = metrics.attrs if hasattr(metrics, 'attrs') else metrics
        attr['rel'] = rel_signed
        attr['rel_abs'] = rel_abs
    return rel_abs

def emb_dist(a, b=None):
    return torch.cdist(a, a if b is None else b, p=2)
