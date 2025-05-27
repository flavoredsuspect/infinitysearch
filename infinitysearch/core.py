#!/usr/bin/env python3
# coding: utf-8
from __future__ import annotations

import os
import time
import random
import math
import argparse

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
import sklearn.preprocessing
from scipy.spatial.distance import cdist
from tensorflow.keras.datasets import fashion_mnist
import vp_tree

# -------------------- BaseANN stub --------------------
class BaseANN:
    def fit(self, X: np.ndarray):
        raise NotImplementedError
    def query(self, v: np.ndarray, k: int = 1):
        raise NotImplementedError

# -------------------- InfinitySearch class --------------------
class InfinitySearch(BaseANN):
    def __init__(self, q: int = 3):
        self._metric = 'euclidean'
        self._object_type = 'Float'
        self._epsilon = 0.0
        self._q = q

    # Fermat exact
    def fermat_gpu_exact(self, D: torch.Tensor, q=3.0) -> torch.Tensor:
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

    # Residual block
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

    # Embedding network
    class EmbNet(nn.Module):
        def __init__(self, input_dim=784, output_dim=300):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(input_dim, 256),
                nn.GELU(),
                nn.BatchNorm1d(256),
                InfinitySearch.ResidualBlock(256),
                nn.Dropout(0.1),
                nn.Linear(256, output_dim),
            )
            self.alpha = nn.Parameter(torch.zeros(1))
            self.max_scale = 2
        def forward(self, x):
            emb = F.normalize(self.net(x), p=2, dim=-1)
            scale = 1.0 + (self.max_scale - 1.0) * torch.sigmoid(self.alpha)
            return emb * scale

    # Compute relative error
    @staticmethod
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

    # Training
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
        M = self.fermat_gpu_exact(D0, q)
        M = (M - M.min())/(M.max()-M.min())

        model = InfinitySearch.EmbNet(X.size(1), output_dim=300).to(device)
        opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-3)
        sched = CosineAnnealingWarmRestarts(opt, T_0=50, T_mult=2)

        def emb_dist(a, b=None): return torch.cdist(a, a if b is None else b, p=2)

        for ep in range(epochs):
            model.train()
            emb = model(X)
            D_emb = emb_dist(emb)
            mask = M.float()
            loss_s = torch.sqrt(((D_emb - M)**2 * mask).sum() / mask.sum())
            idx = torch.randint(0, n, (batch_size, 3), device=device)
            i,j,k = idx.t()
            raw = emb_dist(emb[i], emb[j]) + emb_dist(emb[j], emb[k]) - emb_dist(emb[i], emb[k])
            loss_t = F.relu(raw).min()
            loss = lambda_stress * loss_s + lambda_triangle * loss_t
            opt.zero_grad(); loss.backward(); opt.step(); sched.step(ep)
            if verbose and ep%100==0:
                print(f"Epoch {ep} | stress={loss_s:.4f} | tri={loss_t:.4f}")
        final_emb = model(X)
        return model, final_emb, D0, device, None

    # Fit & index
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

    # Query
    def query(self, v: np.ndarray, k: int = 1):
        v = v.astype(np.float32)
        res = self.index.search(k, v, v, returnDistances=False)
        return res.ids[:k]

    def prepare_batch_query(self, X: np.ndarray, n: int):
        self.queries_np = X.astype(np.float32)
        self.kk = n

    def run_batch_query(self):
        return self.index.search_batch(1, self.kk, self.queries_np, self.queries_np, False)

# -------------------- Main --------------------
if __name__ == '__main__':
    (xtr,_),(xte,_) = fashion_mnist.load_data()
    data = np.concatenate((xtr, xte), axis=0).reshape(-1,28*28)/255.0
    data = data[:10000]
    split = int(0.8 * len(data))
    train, query = data[:split], data[split:]

    infsearch = InfinitySearch(q=5)
    infsearch.fit(train)

    infsearch.prepare_batch_query(query, n=1)
    start = time.time()
    results = infsearch.run_batch_query()
    elapsed = time.time() - start
    qps = len(query) / elapsed
    print(f"Queried {len(query)} points in {elapsed:.2f}s ({qps:.1f} q/s)")

    # compute true neighbors for evaluation
    true_nn = np.argsort(cdist(query, train), axis=1)[:, :10]
    rel_err = InfinitySearch.rel(true_nn, results)
    print(f"Mean absolute relative error: {rel_err:.4f}")
