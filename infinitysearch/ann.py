# Ann.py
import numpy as np
import torch
import torch.nn.functional as F
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from scipy.spatial.distance import cdist
import vp_tree

from .Models import EmbNet
from .fermat import fermat_gpu_exact
from .utils import rel, emb_dist


class BaseANN:
    def fit(self, X: np.ndarray):
        raise NotImplementedError

    def query(self, v: np.ndarray, k: int = 1):
        raise NotImplementedError


class InfinitySearch(BaseANN):
    def __init__(self, q: int = 3):
        self._metric = 'euclidean'
        self._object_type = 'Float'
        self._epsilon = 0.0
        self._q = q

    def train_inductive_model(
        self, X: torch.Tensor, q=3.0, k_neighbors=10,
        epochs=500, batch_size=1024, lr=1e-3,
        lambda_stress=1.0, lambda_triplet=0.0, lambda_triangle=0,
        val=False, val_points=None, verbose=False,
        metric='euclidean', emb_metric='euclidean'
    ):
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        X = X.to(device)
        n = X.size(0)
        D0 = torch.cdist(X, X, p=2)
        M = fermat_gpu_exact(D0, q)
        M = (M - M.min()) / (M.max() - M.min())

        model = EmbNet(X.size(1), output_dim=300).to(device)
        opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-3)
        sched = CosineAnnealingWarmRestarts(opt, T_0=50, T_mult=2)

        for ep in range(epochs):
            model.train()
            emb = model(X)
            D_emb = emb_dist(emb)
            mask = M.float()
            loss_s = torch.sqrt(((D_emb - M) ** 2 * mask).sum() / mask.sum())
            idx = torch.randint(0, n, (batch_size, 3), device=device)
            i, j, k = idx.t()
            raw = emb_dist(emb[i], emb[j]) + emb_dist(emb[j], emb[k]) - emb_dist(emb[i], emb[k])
            loss_t = F.relu(raw).min()
            loss = lambda_stress * loss_s + lambda_triangle * loss_t
            opt.zero_grad(); loss.backward(); opt.step(); sched.step(ep)
            if verbose and ep % 100 == 0:
                print(f"Epoch {ep} | stress={loss_s:.4f} | tri={loss_t:.4f}")
        final_emb = model(X)
        return model, final_emb, D0, device, None

    def fit(self, X: np.ndarray):
        X = X.astype(np.float32)
        X_tensor = torch.tensor(X)
        model, _, D0, device, _ = self.train_inductive_model(
            X_tensor, q=self._q, k_neighbors=50,
            epochs=200, batch_size=1024, lr=1e-3,
            lambda_stress=1.0, lambda_triangle=0,
            verbose=False
        )
        model.eval()
        emb = model(X_tensor.to(device)).cpu().detach().numpy().astype(np.float32)
        self.index = vp_tree.VpTree(self._q, vp_tree.Metric.Euclidean, vp_tree.Metric.Euclidean)
        self.index.create_numpy(X, emb, list(range(len(X))))
        print("✔ InfinitySearch fit & index done")

    def query(self, v: np.ndarray, k: int = 1):
        v = v.astype(np.float32)
        res = self.index.search(k, v, v, returnDistances=False)
        return res.ids[:k]

    def prepare_batch_query(self, X: np.ndarray, n: int):
        self.queries_np = X.astype(np.float32)
        self.kk = n

    def run_batch_query(self):
        return self.index.search_batch(1, self.kk, self.queries_np, self.queries_np, False)
