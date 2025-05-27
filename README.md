# InfinitySearch

**InfinitySearch** is a Python package for fast nearest neighbor search using an inductive embedding model and a highly optimized VP-Tree backend.

It supports custom metrics (including Python lambdas), multi-metric search (original vs. embedded distances), and includes automatic configuration via Optuna.

Infinity Search: Approximate Vector Search with Projections on q-Metric Spaces introduces a novel projection method using Fermat distances in q-metric spaces. This allows embedding into structured manifolds while preserving key nearest neighbor relationships, offering efficiency and precision in high-dimensional search problems.

---

## 🚀 Installation

Install with pip:

```bash
pip install .
```

---

## 🧪 Quick Start

```python
from infinitysearch.test import main
main()
```

---

## 🧠 Class: `InfinitySearch`

```python
InfinitySearch(q=2.0, metric_embed="euclidean", metric_real="euclidean")
```

### Parameters:
- **q**: float
  - Exponent used in Fermat distance graph.
- **metric_embed**: str or callable
  - Distance in embedding space. Supported:
    - "euclidean"
    - "cosine"
    - "manhattan"
    - Callable: `f(a: np.ndarray, b: np.ndarray) -> float`
- **metric_real**: str or callable
  - Distance in original/real space. Supported:
    - "euclidean"
    - "cosine"
    - "manhattan"
    - Callable: `f(a: np.ndarray, b: np.ndarray) -> float`

---

## 🔍 Methods

### `fit(X: np.ndarray, config: str | dict = "optuna", verbose: bool = True)`
Trains the embedding model and builds the VP-tree index.

- **X**: ndarray of shape (n_samples, n_features)
- **config**:
  - "optuna" → run Optuna hyperparameter search
  - "last" → use the most recent configuration from cache
  - `str` → named config key stored in cache
  - `dict` → partial user-defined config. Remaining parameters are optimized

#### Config Dictionary Parameters:
- `output_dim`: int — Final embedding dimensionality
- `batch_size`: int — Training batch size (e.g., 128, 256, 512)
- `epochs`: int — Training epochs (e.g., 50–200)
- `lr`: float — Learning rate (log-uniform range)
- `lambda_stress`: float — Weight of the stress loss
- `emb_metric`: str — Embedding space distance

### `prepare_query(X: np.ndarray, n: int = 1, k: int = 1)`
Embeds and stores queries for batch search.

- **X**: 2D array of queries
- **n**: Number of candidates to fetch (≥ k)
- **k**: Final top-k neighbors to return

### `query(v: np.ndarray, n: int = 1, k: int = 1)`
Searches a single query.

- **X**: 1D array of query
- **n**: Number of candidates to fetch (≥ k)
- **k**: Final top-k neighbors to return

- Automatically calls `prepare_query` if needed

### `run_batch_query(X: np.ndarray, n: int = 1, k: int = 1)`
Returns the top-k neighbors for the last batch of queries.

- **X**: 2D array of queries
- **n**: Number of candidates to fetch (≥ k)
- **k**: Final top-k neighbors to return

- Automatically calls `prepare_query` if needed

---

## 📁 Caching & Configurations

- Configurations are cached in `~/.cache/infinitysearch/named_configs.json`
- The most recent run is saved under key `last`
- Named configs are stored and retrieved using their respective keys
- If a named config is not found, Optuna will run and save the result with that name

---

## 🧪 Test Example

This test can be run via:
```python
from infinitysearch.test import main
main()
```

Which is equivalent to:
```python
import numpy as np
from infinitysearch.ann import InfinitySearch

X = np.random.rand(500, 32).astype(np.float32)
ann = InfinitySearch(q=1.5)
ann.fit(X, config="test_config", verbose=False)
ann.prepare_query(X[:5], n=10, k=1)
res = ann.run_batch_query()
assert len(res[0]) == 5
```

---

## 📜 License

This package is distributed for **non-commercial research purposes** only. See `LICENSE` for details.

---

## ✉ Contact

For questions or contributions, please contact: `antonio@yourdomain.com`

---

## 📚 Citation

If you use InfinitySearch in your research, please cite:

**Infinity Search: Approximate Vector Search with Projections on q-Metric Spaces**

\[Insert link here\]
