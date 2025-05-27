# fermat.py
import torch
import numpy as np

def fermat_gpu_exact(D: torch.Tensor, q=3.0) -> torch.Tensor:
    D = D.to(torch.float32)
    n = D.size(0)
    if q == float('inf') or q == np.inf:
        M = D.clone()
        for w in range(n):
            via_w = torch.max(M[:, w].unsqueeze(1), M[w, :].unsqueeze(0))
            M = torch.min(M, via_w)
        return M
    M = D.pow(q)
    for w in range(n):
        via_w = M[:, w].unsqueeze(1) + M[w, :].unsqueeze(0)
        M = torch.min(M, via_w)
    return M.pow(1.0 / q)

def fermat_gpu_approx(D, q=3.0, k=20, num_iters=2000, lr=0.05):
    """
    Fermat approximation using in-place updates.
    Supports q ∈ (0, ∞] — for q=inf, uses max-path instead of q-power path sum.
    """
    import math

    D = D.to(torch.float16).to(torch.float32)
    is_inf = q == float('inf') or q == np.inf

    if not is_inf:
        D.pow_(q)

    n, dev = D.size(0), D.device
    _, topk_idx = torch.topk(D, k=k + 1, dim=1, largest=False)
    topk_idx = topk_idx[:, 1:]
    row_idx = torch.arange(n, device=dev).unsqueeze(1)

    for _ in range(num_iters):
        edge_vals = D[row_idx, topk_idx]
        min_update = D

        for j in range(k):
            v = topk_idx[:, j]
            dv = D[v]
            if is_inf:
                cost = torch.maximum(edge_vals[:, j].unsqueeze(1), dv)
            else:
                cost = edge_vals[:, j].unsqueeze(1) + dv

            D = torch.minimum(D, cost)

        D.mul_(1.0 - lr).add_(lr * min_update)
        D.copy_(torch.minimum(D, D.T))

    if not is_inf:
        D.pow_(1.0 / q)

    return D.to(torch.float16)