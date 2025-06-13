
from __future__ import annotations
import itertools, os, tempfile, time, argparse
from infinitysearch.utils import rel
import torch
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist
from tensorflow.keras.datasets import fashion_mnist
from collections import defaultdict
from infinitysearch import InfinitySearch


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--q', type=int, default=20, help='q-metric')
    parser.add_argument('--n', type=int, default=10000, help='Total number of points')
    # Default is now 'last'
    parser.add_argument('--config', type=str, default="last", help="'optuna', 'last', or leave empty for manual config")
    args = parser.parse_args()

    (xtr, _), (xte, _) = fashion_mnist.load_data()
    data = np.concatenate((xtr, xte), axis=0).reshape(-1, 28 * 28) / 255.0
    data = data[:args.n]
    split = int(0.8 * len(data))
    train, query = data[:split], data[split:]
    I = InfinitySearch.remove("all")
    infsearch = InfinitySearch(q=args.q)
    infsearch.fit(train, config=args.config)

    infsearch.prepare_query(query, n=1)
    start = time.time()
    results = infsearch.query()
    elapsed = time.time() - start
    qps = len(query) / elapsed
    print(f"Queried {len(query)} points in {elapsed:.8f}s ({qps:.8f} q/s)")

    true_nn = np.argsort(cdist(query, train), axis=1)[:, :100]
    rel_err = rel(true_nn, results)
    print(f"Mean absolute relative error: {rel_err:.4f}")

    # Save the model and config
    infsearch.save("test")


def test(quick=False, verbose=True):
    print_banner("🏁 Starting Validation Test")

    (xtr, _), (xte, _) = fashion_mnist.load_data()
    data = np.concatenate([xtr, xte], axis=0).reshape(-1, 28 * 28) / 255.0
    data = data[:1000] if quick else data[:10000]
    split = int(0.8 * len(data))
    train, qry = data[:split], data[split:]

    qs = [2] if quick else [1, 2, 10, 20]
    metrics = ["euclidean", "cosine"] if quick else ["euclidean", "manhattan", "cosine", "correlation"]
    configs = ["optuna", "last"]

    for q in qs:
        for me in metrics:
            for mr in metrics:
                for cfg in configs:
                    tag = f"q{q}_{me[:3]}_{mr[:3]}_{cfg}"
                    try:
                        _one_pass(q, me, mr, cfg, train, qry, tag, verbose)
                    except Exception as e:
                        print(f"❌ Failed {tag}: {e}")

    print_banner("✅ Finished All Tests")

if __name__ == "__main__":
    main()


