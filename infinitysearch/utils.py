import numpy as np
import torch
import vp_tree
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

def emb_dist(a: torch.Tensor, b: torch.Tensor = None, metric: str = "euclidean") -> torch.Tensor:
    if b is None:
        b = a

    a = torch.nn.functional.normalize(a, dim=-1) if metric == "cosine" else a
    b = torch.nn.functional.normalize(b, dim=-1) if metric == "cosine" else b

    if metric == "euclidean":
        return torch.cdist(a, b, p=2)
    elif metric == "manhattan":
        return torch.cdist(a, b, p=1)
    elif metric == "cosine":
        return 1 - torch.matmul(a, b.transpose(0, 1))
    elif metric == "correlation":
        a_centered = a - a.mean(dim=1, keepdim=True)
        b_centered = b - b.mean(dim=1, keepdim=True)
        a_norm = a_centered / a_centered.norm(dim=1, keepdim=True)
        b_norm = b_centered / b_centered.norm(dim=1, keepdim=True)
        return 1 - torch.matmul(a_norm, b_norm.transpose(0, 1))
    else:
        raise ValueError(f"Unknown embedding metric: {metric}")
def lambda_to_cpp_metric(pyfunc):
    if not callable(pyfunc):
        raise ValueError("Expected a callable for custom distance function.")
    def wrapper(a, b):
        return float(pyfunc(np.asarray(a), np.asarray(b)))
    return wrapper


metric_enum_map = {
    "euclidean":  vp_tree.Metric.Euclidean,
    "manhattan":  vp_tree.Metric.Manhattan,
    "cosine":     vp_tree.Metric.Cosine,
    # "jaccard": not supported or removed
    "correlation": vp_tree.Metric.Cosine,  # if approximation ok
}
