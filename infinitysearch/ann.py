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
import joblib
import shutil
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
        self._prepared = False

    def _build_model(self):
        input_dim = self.config.get("input_dim", 784)  # default for Fashion-MNIST
        output_dim = self.config.get("output_dim", 128)
        self.model = EmbNet(input_dim=input_dim, output_dim=output_dim)


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

    def fit(self, X: np.ndarray, config: dict | str | torch.nn.Module | None = "optuna", verbose: bool = True):
        X = X.astype(np.float32)
        X_tensor = torch.tensor(X)

        if isinstance(config, dict):
            print("🔧 Running Optuna with fixed parameters...")
            config_dict = run_optuna_search(X, self._q, fixed=config, verbose=verbose)

        else:
            cache_dir = os.path.expanduser("~/.cache/infinitysearch/")
            os.makedirs(cache_dir, exist_ok=True)
            cache_file = os.path.join(cache_dir, "configs.json")

            if os.path.exists(cache_file):
                with open(cache_file, "r") as f:
                    all_configs = json.load(f)
            else:
                all_configs = {}

            if config == "last":
                config_dict = all_configs.get("last")
                if config_dict is None:
                    print("⚠ No 'last' config found. Running Optuna...")
                    config_dict = run_optuna_search(X, self._q, verbose=verbose)
                    all_configs["last"] = {k: v for k, v in config_dict.items() if k != "model"}

            elif config == "optuna":
                print("🔍 Running full Optuna search...")
                config_dict = run_optuna_search(X, self._q, verbose=verbose)
                all_configs["last"] = {k: v for k, v in config_dict.items() if k != "model"}

            elif isinstance(config, str):
                config_dict = all_configs.get(config)
                if config_dict is None:
                    print(f"⚠ No config named '{config}' found. Running Optuna and saving it...")
                    config_dict = run_optuna_search(X, self._q, verbose=verbose)
                    all_configs[config] = {k: v for k, v in config_dict.items() if k != "model"}
                    all_configs["last"] = all_configs[config]
            else:
                raise ValueError(
                    "Invalid config type. Must be 'optuna', 'last', a name string, or a config dictionary.")

            # Save updated cache
            with open(cache_file, "w") as f:
                json.dump(all_configs, f)

        if verbose:
            print(f"Training model with config: {config}")
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
            verbose=verbose,
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

    def prepare_query(self, X: np.ndarray, n: int = 1, k: int = 1):
        self.queries_np = X.astype(np.float32)
        with torch.no_grad():
            self.query_embed = self.model(
                torch.tensor(X, dtype=torch.float32).to(self.device)
            ).cpu().numpy()
        self.topk = k
        self.totalk = max(n, k)
        self._prepared = True

    def query_one(self, x: np.ndarray, n: int = 1, k: int = 1):
        """
        Query a single 1D input vector `x`, safely preparing it and returning the top-k result.
        """
        if x.ndim != 1:
            raise ValueError("Expected a single query vector with shape (D,), got shape {}".format(x.shape))

        # Prepare a single-vector batch (shape: (1, D))
        X = x[None]
        self.prepare_query(X, n=n, k=k)

        # Run and return the first result
        result = self.index.search_batch(
            self.totalk,
            self.topk,
            self.query_embed,
            self.queries_np,
            False
        )
        self._prepared = False
        return result[0]

    def query(self, X: np.ndarray = None, n: int = 1, k: int = 1):
        if not self._prepared:
            if X is None:
                raise ValueError("No query data provided for automatic preparation.")
            self.prepare_query(X, n=n, k=k)
        result = self.index.search_batch(self.totalk, self.topk, self.query_embed, self.queries_np, False)
        self._prepared = False
        return result

    def save(self, name="last"):
        cache_root = os.path.expanduser("~/.cache/infinitysearch/")
        index_dir = os.path.join(cache_root, "indices")
        model_dir = os.path.join(cache_root, "models")
        config_path = os.path.join(cache_root, "configs.json")

        os.makedirs(index_dir, exist_ok=True)
        os.makedirs(model_dir, exist_ok=True)

        # Save model weights
        model_path = os.path.join(model_dir, f"{name}_model.pt")
        torch.save(self.model.state_dict(), model_path)

        # Save config with input/output dim and q
        config_copy = {k: v for k, v in self.config.items() if k != "model"}
        config_copy["input_dim"] = self.model.net[0].in_features
        config_copy["output_dim"] = self.model.net[-1].out_features
        config_copy["q"] = self._q

        if os.path.exists(config_path):
            with open(config_path, "r") as f:
                all_configs = json.load(f)
        else:
            all_configs = {}

        all_configs[name] = config_copy
        all_configs["last"] = config_copy

        with open(config_path, "w") as f:
            json.dump(all_configs, f)

        # Save index
        self.index.save(os.path.join(index_dir, name))

    @classmethod
    def load(cls, name="last"):
        cache_root = os.path.expanduser("~/.cache/infinitysearch/")
        index_path = os.path.join(cache_root, "indices", name)
        model_path = os.path.join(cache_root, "models", f"{name}_model.pt")
        config_path = os.path.join(cache_root, "configs.json")

        # Check config existence
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"❌ Config file not found: {config_path}")

        with open(config_path, "r") as f:
            all_configs = json.load(f)

        if name not in all_configs:
            raise ValueError(f"❌ No config named '{name}' found in {config_path}")

        # Check model and index existence
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"❌ Model file not found: {model_path}")
        if not os.path.exists(index_path):
            raise FileNotFoundError(f"❌ Index file not found: {index_path}")

        config = all_configs[name]
        q = config.get("q", 3)

        obj = cls(q=q)
        obj.config = config
        obj.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Build and load model
        obj._build_model()
        obj.model.to(obj.device)
        obj.model.load_state_dict(torch.load(model_path, map_location=obj.device))
        obj.model.eval()

        # Load index
        metric_raw = config.get("metric", 'euclidean')
        emb_metric_raw = config.get("emb_metric", 'euclidean')
        obj.index = vp_tree.VpTree(
            obj._q,
            lambda_to_cpp_metric(emb_metric_raw) if callable(emb_metric_raw) else metric_enum_map.get(emb_metric_raw,
                                                                                                      -1),
            lambda_to_cpp_metric(metric_raw) if callable(metric_raw) else metric_enum_map.get(metric_raw, -1)
        )
        obj.index.load(index_path)

        return obj

    @staticmethod
    def remove(name="all"):
        """
        Remove cached config, model, and index files for a given name.
        If name == "all", clear the entire InfinitySearch cache (with confirmation).
        """
        cache_root = os.path.expanduser("~/.cache/infinitysearch/")
        paths = {
            "metadata": os.path.join(cache_root, "metadata", f"{name}.json"),
            "models": os.path.join(cache_root, "models", f"{name}_model.pt"),
            "indices": os.path.join(cache_root, "indices", name),
        }

        if name == "all":
            confirm = input("⚠ This will delete the entire InfinitySearch cache. Proceed? (y/n): ")
            if confirm.lower() == 'y':
                shutil.rmtree(cache_root, ignore_errors=True)
                print("🗑️ All cache cleared.")
            else:
                print("❌ Cancelled.")
            return

        removed = False
        for label, path in paths.items():
            if os.path.exists(path):
                if os.path.isdir(path):
                    shutil.rmtree(path)
                else:
                    os.remove(path)
                print(f"✅ Removed {label} cache: {path}")
                removed = True
        if not removed:
            print(f"⚠ No cache found for name '{name}'.")
