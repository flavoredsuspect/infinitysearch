import torch
import torch.nn as nn
import torch.nn.functional as F


class ResidualBlock(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, dim),
            nn.GELU(),
            nn.BatchNorm1d(dim),
        )

    def forward(self, x):
        return x + self.net(x)

class EmbNet(nn.Module):
    def __init__(self, input_dim=784, output_dim=300, hidden_dim=256, num_layers=1, dropout=0.1, activation="gelu"):
        super().__init__()
        act_fn = {"relu": nn.ReLU(), "gelu": nn.GELU()}[activation]
        layers = [
            nn.Linear(input_dim, hidden_dim),
            act_fn,
            nn.BatchNorm1d(hidden_dim)
        ]
        for _ in range(num_layers):
            layers.append(ResidualBlock(hidden_dim))
        layers.extend([
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim),
        ])
        self.net = nn.Sequential(*layers)
        self.alpha = nn.Parameter(torch.zeros(1))
        self.max_scale = 2

    def forward(self, x):
        emb = F.normalize(self.net(x), p=2, dim=-1)
        scale = 1.0 + (self.max_scale - 1.0) * torch.sigmoid(self.alpha)
        return emb * scale



import optuna
def run_optuna_search(X: torch.Tensor, q: float, fixed: dict = None, verbose: bool = True):
    import optuna
    import torch
    import os
    import json
    import numpy as np
    from torch.utils.data import DataLoader, TensorDataset
    from torch import nn
    from torch.nn import functional as F
    from .utils import rel, emb_dist
    from .models import EmbNet
    from .fermat import fermat_gpu_exact, fermat_gpu_approx

    if fixed is None:
        fixed = {}
    X = torch.as_tensor(X, dtype=torch.float32) if not isinstance(X, torch.Tensor) else X
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    def objective(trial):
        try:
            dim_in = X.shape[1]
            n_total = min(1000, X.shape[0])
            X_sample = X[:n_total].to(device)
            val_split = int(0.2 * n_total)
            train_X, val_X = X_sample[:-val_split], X_sample[-val_split:]

            output_dim = fixed.get("output_dim") or trial.suggest_int("output_dim", dim_in // 3, int(dim_in * 1.5))
            hidden_dim = fixed.get("hidden_dim") or trial.suggest_int("hidden_dim", 128, 512)
            num_layers = fixed.get("num_layers") or trial.suggest_int("num_layers", 0, 4)
            activation = fixed.get("activation") or trial.suggest_categorical("activation", ["relu", "gelu"])
            emb_metric = fixed.get("emb_metric") or trial.suggest_categorical("emb_metric", ["euclidean", "manhattan"])
            metric = fixed.get("metric", "euclidean")
            batch_size = fixed.get("batch_size") or trial.suggest_categorical("batch_size", [128, 256, 512])
            epochs = fixed.get("epochs") or trial.suggest_int("epochs", 50, 300)
            lr = fixed.get("lr") or trial.suggest_float("lr", 1e-4, 1e-2, log=True)
            lambda_stress = fixed.get("lambda_stress") or trial.suggest_float("lambda_stress", 0.1, 5.0)

            # Ground truth distance matrix
            D0 = emb_dist(train_X, metric=metric)
            try:
                M = fermat_gpu_exact(D0, q)
            except RuntimeError as e:
                if "CUDA out of memory" in str(e):
                    print("⚠ Optuna: switching to approximate Fermat due to OOM.")
                    M = fermat_gpu_approx(D0, q=q, k=20, num_iters=1000, lr=0.05)
                else:
                    raise
            M = (M - M.min()) / (M.max() - M.min())

            model = EmbNet(
                input_dim=dim_in,
                output_dim=output_dim,
                hidden_dim=hidden_dim,
                num_layers=num_layers,
                activation=activation
            ).to(device)

            opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-3)
            dataset = DataLoader(train_X, batch_size=batch_size, shuffle=True)

            for ep in range(epochs):
                model.train()
                for batch in dataset:
                    emb = model(batch)
                    D_emb = emb_dist(emb, metric=emb_metric)
                    with torch.no_grad():
                        idx = torch.arange(batch.shape[0], device=device)
                        full_idx = torch.arange(train_X.shape[0], device=device)
                        M_sub = emb_dist(batch, train_X, metric=metric)
                        M_sub = (M_sub - M_sub.min()) / (M_sub.max() - M_sub.min())
                    mask = M_sub.float()
                    loss_s = torch.sqrt(((D_emb - mask[:, :batch.shape[0]]) ** 2 * mask[:, :batch.shape[0]]).sum() / mask[:, :batch.shape[0]].sum())
                    opt.zero_grad()
                    loss_s.backward()
                    opt.step()

            with torch.no_grad():
                emb_train = model(train_X)
                emb_val = model(val_X)
                pred_dists = emb_dist(emb_val, emb_train, metric=emb_metric).cpu().numpy()
                gt_dists = emb_dist(val_X, train_X, metric=metric).cpu().numpy()
                knn = np.argsort(pred_dists, axis=1)[:, :1]
                true_knn = np.argsort(gt_dists, axis=1)[:, :100]
                rel_error = rel(true_knn, knn)
                return rel_error
        except RuntimeError as e:
            if "CUDA out of memory" in str(e):
                torch.cuda.empty_cache()
                raise optuna.TrialPruned()
            raise

    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=50)

    best = study.best_trial.params
    best["model"] = EmbNet(
        input_dim=X.shape[1],
        output_dim=best["output_dim"],
        hidden_dim=best["hidden_dim"],
        num_layers=best["num_layers"],
        activation=best["activation"]
    ).to(device)

    cache_path = os.path.expanduser("~/.cache/infinitysearch/last_config.json")
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    json_ready = {k: v for k, v in best.items() if k != "model"}
    with open(cache_path, "w") as f:
        json.dump(json_ready, f)

    return best


