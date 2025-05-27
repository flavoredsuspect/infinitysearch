# ann.py
import numpy as np
import torch
import torch.nn.functional as F
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
import vp_tree
import json
import os

from .models import EmbNet, run_optuna_search
from .fermat import fermat_gpu_exact
from .utils import rel, emb_dist, metric_enum_map, lambda_to_cpp_metric


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
        lambda_stress=1.0, lambda_triangle=0,
        val=False, val_points=None, verbose=False,
        metric='euclidean', emb_metric='euclidean',
        metric_fn=None, emb_metric_fn=None,
        model=None
    ):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        X = X.to(self.device)
        n = X.size(0)
        full_X = X
        D0 = metric_fn(X) if metric_fn else emb_dist(X, metric=metric)

        try:
            M = fermat_gpu_exact(D0, q)
        except RuntimeError as e:
            if "CUDA out of memory" in str(e):
                print("⚠ GPU memory insufficient for exact Fermat. Using approximation.")
                from .fermat import fermat_gpu_approx
                M = fermat_gpu_approx(D0, q=q, k=k_neighbors, num_iters=2000, lr=0.05)
            else:
                raise
        M = (M - M.min()) / (M.max() - M.min())

        if model is None:
            model = EmbNet(X.size(1), output_dim=128).to(self.device)

        opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-3)
        sched = CosineAnnealingWarmRestarts(opt, T_0=50, T_mult=2)

        for ep in range(epochs):
            model.train()
            emb = model(X)
            D_emb = emb_metric_fn(emb) if emb_metric_fn else emb_dist(emb, metric=emb_metric)
            mask = M.float()
            loss_s = torch.sqrt(((D_emb - M) ** 2 * mask).sum() / mask.sum())
            idx = torch.randint(0, n, (batch_size, 3), device=self.device)
            i, j, k = idx.t()
            raw = emb_dist(emb[i], emb[j], metric=emb_metric) + emb_dist(emb[j], emb[k], metric=emb_metric) - emb_dist(emb[i], emb[k], metric=emb_metric)
            loss_t = F.relu(raw).min()
            loss = lambda_stress * loss_s + lambda_triangle * loss_t
            opt.zero_grad(); loss.backward(); opt.step(); sched.step(ep)
            if verbose and ep % 100 == 0:
                print(f"Epoch {ep} | stress={loss_s:.4f} | tri={loss_t:.4f}")
        final_emb = model(full_X)
        return model, final_emb, D0, self.device, None

    def fit(self, X: np.ndarray, config: dict | str | torch.nn.Module | None = "optuna"):
        X = X.astype(np.float32)
        X_tensor = torch.tensor(X)

        if isinstance(config, str):
            if config == "optuna":
                print("ℹ Using Optuna to find best hyperparameters...")
                config_dict = run_optuna_search(X_tensor, self._q)
            elif config == "last":
                cache_path = os.path.expanduser("~/.cache/infinitysearch/last_config.json")
                if os.path.exists(cache_path):
                    with open(cache_path, 'r') as f:
                        config_dict = json.load(f)
                    print("✔ Loaded last configuration from cache.")
                else:
                    print("⚠ No previous config found. Running Optuna instead...")
                    config_dict = run_optuna_search(X_tensor, self._q)
            else:
                raise ValueError("Invalid string option for config. Use 'optuna', 'last', a config dictionary, or a torch.nn.Module.")
        elif isinstance(config, dict):
            if "model" in config:
                print("✔ Using user-provided model and config dictionary.")
                config_dict = config
            else:
                print("ℹ Using Optuna with fixed parameters from config...")
                config_dict = run_optuna_search(X_tensor, self._q, fixed=config)
        elif isinstance(config, torch.nn.Module):
            config_dict = {"model": config}
        else:
            raise ValueError("Invalid config type. Must be 'optuna', 'last', a config dictionary, or a torch.nn.Module.")

        metric_fn = config_dict.get("metric_fn", None)
        emb_metric_fn = config_dict.get("emb_metric_fn", None)

        model = config_dict.get("model", None)
        model, _, D0, device, _ = self.train_inductive_model(
            X_tensor,
            q=config_dict.get("q", self._q),
            k_neighbors=config_dict.get("k_neighbors", 50),
            epochs=config_dict.get("epochs", 200),
            batch_size=config_dict.get("batch_size", 1024),
            lr=config_dict.get("lr", 1e-3),
            lambda_stress=config_dict.get("lambda_stress", 1.0),
            lambda_triangle=config_dict.get("lambda_triangle", 0.0),
            val=config_dict.get("val", False),
            val_points=config_dict.get("val_points", None),
            verbose=config_dict.get("verbose", False),
            metric=config_dict.get("metric", 'euclidean'),
            emb_metric=config_dict.get("emb_metric", 'euclidean'),
            metric_fn=metric_fn,
            emb_metric_fn=emb_metric_fn,
            model=model
        )

        self.config = config_dict
        model.eval()
        emb = model(X_tensor.to(self.device)).cpu().detach().numpy().astype(np.float32)

        metric_raw = config_dict.get("metric", 'euclidean')
        emb_metric_raw = config_dict.get("emb_metric", 'euclidean')

        if callable(metric_raw) or callable(emb_metric_raw):
            print(
                "⚠ Warning: Using custom distance functions may significantly reduce "
                "performance due to lack of C++ optimization.")

            self.index = vp_tree.VpTree(
                self._q,
                lambda_to_cpp_metric(emb_metric_raw) if callable(emb_metric_raw) else metric_enum_map.get(emb_metric_raw, -1),
                lambda_to_cpp_metric(metric_raw) if callable(metric_raw) else metric_enum_map.get(metric_raw, -1)
            )
        else:
            self.index = vp_tree.VpTree(
                self._q,
                metric_enum_map.get(emb_metric_raw, -1),
                metric_enum_map.get(metric_raw, -1)
            )
        self.model=model
        self.index.create_numpy(X, emb, list(range(len(X))))
        print("✔ InfinitySearch fit & index done")

    def query(self, v: np.ndarray, k: int = 1):
        return self.index.search(self.totalk, self.topk, self.query_embed, self.queries_np, False)

    def prepare_query(self, X: np.ndarray, n: int =1, k:int = 1):
        self.queries_np = X.astype(np.float32)

        with torch.no_grad():
            self.query_embed = self.model(
                torch.tensor(X, dtype=torch.float32).to(self.device)).cpu().numpy()

        self.topk = k
        self.totalk= max(n,k)

    def run_batch_query(self):
        return self.index.search_batch(self.totalk, self.topk, self.query_embed, self.queries_np, False)