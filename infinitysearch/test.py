
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
    import torch
    import numpy as np
    import matplotlib.pyplot as plt
    from tensorflow.keras.datasets import fashion_mnist
    from collections import defaultdict
    from infinitysearch import InfinitySearch

    # Load Fashion-MNIST
    (xtr, ytr), (xte, yte) = fashion_mnist.load_data()
    raw_data = np.concatenate((xtr, xte), axis=0)[:10000]
    labels = np.concatenate((ytr, yte), axis=0)[:10000]

    data = raw_data.reshape(-1, 784) / 255.0
    split = int(0.8 * len(data))
    train, query = data[:split], data[split:]
    train_labels, query_labels = labels[:split], labels[split:]
    raw_train = raw_data[:split]
    raw_query = raw_data[split:]

    train_torch = torch.tensor(train, dtype=torch.float32)
    query_torch = torch.tensor(query, dtype=torch.float32)

    def emb_dist(a, b=None, metric="euclidean"):
        if b is None:
            b = a
        if metric == "cosine":
            a = torch.nn.functional.normalize(a, dim=-1)
            b = torch.nn.functional.normalize(b, dim=-1)
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
            raise ValueError(f"Unknown metric: {metric}")

    metric = "euclidean"
    dist = emb_dist(query_torch, train_torch, metric=metric).numpy()
    true_nn = np.argsort(dist, axis=1)[:, :100]

    search = InfinitySearch(q=5)
    search.fit(train, config={"metric": metric, "name": metric})
    search.prepare_query(query, n=1)
    pred = search.query()

    # Compute rank error for each query
    rank_errors = []
    for i in range(len(query)):
        try:
            rank = list(true_nn[i]).index(pred[i][0])
        except ValueError:
            rank = 100  # not found
        rank_errors.append(rank)
    rank_errors = np.array(rank_errors)

    # Group queries by label and compute average error
    per_class_errors = defaultdict(list)
    for i, label in enumerate(query_labels):
        per_class_errors[label].append((rank_errors[i], i))  # (error, idx)

    # Compute average per-class error and sort
    avg_per_class = [(lbl, np.mean([e for e, _ in v])) for lbl, v in per_class_errors.items()]
    top_classes = sorted(avg_per_class, key=lambda x: x[1])[:4]  # best 4 classes

    # Pick best query from each of those classes
    best_queries = []
    for lbl, _ in top_classes:
        best_idx = min(per_class_errors[lbl], key=lambda x: x[0])[1]
        best_queries.append((lbl, best_idx))

    class_names = {
        0: "T-shirt", 1: "Trouser", 2: "Pullover", 3: "Dress", 4: "Coat",
        5: "Sandal", 6: "Shirt", 7: "Sneaker", 8: "Bag", 9: "Ankle boot"
    }

    # Build image triples
    images = []
    row_labels = []
    for label, q_idx in best_queries:
        row_labels.append(class_names[label])
        true_idx = true_nn[q_idx][0]
        ret_idx = pred[q_idx][0]

        q_img = np.stack([raw_query[q_idx]] * 3, axis=-1)
        true_img = np.stack([raw_train[true_idx]] * 3, axis=-1)
        ret_img = np.stack([raw_train[ret_idx]] * 3, axis=-1)

        images.append((q_img, true_img, ret_img))

    # Plot
    fig, axes = plt.subplots(4, 3, figsize=(12, 16))
    titles = ["Query", "True NN", "Returned NN"]

    for i in range(4):
        for j in range(3):
            ax = axes[i, j]
            ax.imshow(images[i][j])
            ax.axis('off')
            if i == 0:
                ax.set_title(titles[j], fontsize=24)
            if j == 0:
                ax.text(-0.5, 14, row_labels[i], fontsize=24,
                        rotation=90, va='center', ha='right', transform=ax.transData)

    plt.savefig("/home/antonio/h.svg", bbox_inches="tight")
    plt.savefig("/home/antonio/h.png", bbox_inches="tight")
    print("✅ Saved 4-class figure as 'h.svg' and 'h.png'")


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
    main_kosarak()


