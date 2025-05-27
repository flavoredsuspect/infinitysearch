# main.py
import time
import argparse
import numpy as np
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
    results = infsearch.run_batch_query()
    elapsed = time.time() - start
    qps = len(query) / elapsed
    print(f"Queried {len(query)} points in {elapsed:.8f}s ({qps:.8f} q/s)")

    true_nn = np.argsort(cdist(query, train), axis=1)[:, :1]
    rel_err = rel(true_nn, results)
    print(f"Mean absolute relative error: {rel_err:.4f}")


if __name__ == '__main__':
    main()
