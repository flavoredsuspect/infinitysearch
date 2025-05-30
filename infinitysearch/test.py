
from __future__ import annotations
import itertools, os, tempfile, time, argparse
from pathlib import Path

import numpy as np
import numpy.linalg as npl
from scipy.spatial.distance import cdist
from tensorflow.keras.datasets import fashion_mnist

from infinitysearch.ann import InfinitySearch
from infinitysearch.utils import rel


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

    infsearch = InfinitySearch(q=args.q)
    infsearch.fit(train, config=args.config)

    infsearch.prepare_query(query, n=1)
    start = time.time()
    results = infsearch.query()
    elapsed = time.time() - start
    qps = len(query) / elapsed
    print(f"Queried {len(query)} points in {elapsed:.8f}s ({qps:.8f} q/s)")

    true_nn = np.argsort(cdist(query, train), axis=1)[:, :1]
    rel_err = rel(true_nn, results)
    print(f"Mean absolute relative error: {rel_err:.4f}")

    # Save the model and config
    infsearch.save("test")

# --------------------------------------------------------------------------- #
def _custom_cosine(a: np.ndarray, b: np.ndarray) -> float:
    denom = (np.linalg.norm(a) * np.linalg.norm(b)) + 1e-9
    return 1.0 - float(np.dot(a, b) / denom)


_Q_VALUES  = [1, 2, 5, np.inf]
_BUILTINS  = ["euclidean", "manhattan", "cosine", "correlation"]
_DICT_CFG  = {
    "output_dim": 64,
    "emb_metric": "correlation",
    "batch_size": 128,
    "epochs": 5,
    "lr": 1e-3,
    "lambda_stress": 2.0,
    "input_dim": 784,
}

# --------------------------------------------------------------------------- #
def _prepare_data(n_total: int):
    (xtr, _), (xte, _) = fashion_mnist.load_data()
    data = (
        np.concatenate([xtr, xte]).astype("float32").reshape(-1, 28 * 28) / 255.0
    )[: n_total]
    split = int(0.8 * n_total)
    return data[:split], data[split:]


def _exact_nn(train, qry, k=1):
    return np.argsort(cdist(qry, train), axis=1)[:, :k]


def print_banner(title):
    print("\n" + "=" * 80)
    print(f"▶ {title}")
    print("=" * 80)

def print_header(tag, q, me, mr, cfg):
    print_banner(tag)
    print(f"  q = {q}")
    print(f"  metric_embed = {me}")
    print(f"  metric_real  = {mr}")
    print(f"  config spec  = {cfg}")

def _one_pass(q, me, mr, cfg, train, qry, tag, verbose=False):
    if isinstance(cfg, dict):
        cfg = dict(cfg)
        cfg["metric"] = mr
        cfg["emb_metric"] = me

    if verbose:
        print_header(tag, q, me, mr, cfg)

    t0 = time.time()
    search = InfinitySearch(q=q)
    search.fit(train, config=cfg, verbose=False)

    # Prepare and run full batch query
    search.prepare_query(qry, n=1)
    t1 = time.time()
    results = search.query()
    t2 = time.time()

    qps = len(qry) / (t2 - t1)
    err = rel(np.argsort(cdist(qry, train), axis=1)[:, :1], results)
    print(f"🧪 query@1 | rel_err = {err:.4f} | qps = {qps:.2f}")

    # Show an example query result
    first = search.query_one(qry[0])
    print(f"🔍 example query: {first}")

    # Save and reload test
    search.save(tag)
    loaded = InfinitySearch.load(tag)
    loaded.prepare_query(qry, n=1)
    results2 = loaded.query()
    err2 = rel(np.argsort(cdist(qry, train), axis=1)[:, :1], results2)
    assert np.allclose(err, err2, atol=1e-3), "Loaded model mismatch!"

    InfinitySearch.remove(tag)
    return err, qps


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


