# InfinitySearch

**InfinitySearch** is a Python package for fast nearest neighbor search using an inductive embedding model and an optimized VP-Tree backend.

It is especially well-suited for **large datasets** or applications involving **custom dissimilarity measures**. Its main strength is **speed**, measured in queries per second (qps).

It supports custom metrics (including Python lambdas), multi-metric search (original vs. embedded distances), and includes automatic configuration via Optuna.

Infinity Search: Approximate Vector Search with Projections on q-Metric Spaces introduces a novel projection method using distances in q-metric spaces. This allows embedding into structured manifolds while preserving key nearest neighbor relationships, offering efficiency and precision in high-dimensional search problems.

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

This runs a quick evaluation using the Fashion-MNIST dataset (grayscale images of clothing, 28×28). It:

1. Loads and flattens the Fashion-MNIST dataset to 784-dimensional vectors.
2. Initializes an `InfinitySearch` instance with a default q-metric.
3. Trains the embedding model and builds a VP-tree index using either cached or optimized configuration.
4. Prepares a batch of queries from the hold-out set.
5. Runs the batch nearest neighbor search.
6. Prints the query throughput (queries per second) and the mean relative rank error.

---

## 🧠 Class: `InfinitySearch`

```python
InfinitySearch(q=2.0, metric_embed="euclidean", metric_real="euclidean")
```

This is the main entry point to the library. It allows you to embed data, build an index, and efficiently search for nearest neighbors.

### Parameters:

- **q**: float

  - The exponent in the Fermat distance used for constructing the metric space. The higher it is, the faster will be the search, with a lower bound of $log2(n)$ for $q=np.inf$. Increasing q too much will decrease accuracy. `q` can range between 1 and `np.inf`.

- **metric_embed**: str or callable

  - Distance metric to use in the embedding space. Can be one of: "euclidean", "manhattan", "cosine", "correlation". Alternatively, it can be a custom Python callable, e.g.:
    ```python
    lambda a, b: 1 - np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
    ```

- **metric_real**: str or callable

  - Distance metric to use in the original input space. Same options as `metric_embed`.

> **Warning**: Any custom distance function not among the implemented options will be considerably slower due to the lack of AVX optimization.

### Intended Usage:

The `InfinitySearch` class is designed to provide a complete pipeline for nearest neighbor search on high-dimensional datasets. It includes:

- **Model Training**: Learns a neural embedding that maps the data into a q-metric space that facilitates efficient search.
- **Automatic Configuration**: Supports automated hyperparameter tuning (e.g., embedding dimension, learning rate) via Optuna.
- **Efficient Indexing**: Uses a VP-tree backend optimized in C++ to index the embedded data for a fast retrieval.
- **Query Interface**: Includes methods to run both single and batch nearest neighbor queries.

All relevant artifacts (model weights, VP-tree index, configuration dictionary) are automatically cached on disk for reproducibility and reusability. This means you can train once, save the results, and later reload everything with minimal overhead using the `load` method.

## 🔍 Methods

### `fit(X: np.ndarray, config: str | dict = "optuna", verbose: bool = True)`

Learns an embedding of your data and builds a VP-tree index for efficient nearest neighbor search.

**Parameters:**

- `X` (`np.ndarray`): Input data of shape `(n_samples, n_features)`.

- `config` (`str` or `dict`):

  - If "optuna", runs an Optuna search for best hyperparameters.
  - If "last", loads the most recently used configuration.
  - If `str`, looks up a named configuration from cache.
  - If `dict`, uses the given parameters (with missing values optimized).

    Example config:
    ```python
    {
      "output_dim": 128,
      "emb_metric": "correlation",
      "batch_size": 256,
      "epochs": 168,
      "lr": 0.002605583546753248,
      "lambda_stress": 4.603993679780747,
      "input_dim": 784
    }
    ```

- `verbose` (`bool`): Whether to print progress during training.

**Returns:**

- None. The model is trained and the index is built internally.

```python
model = InfinitySearch(q=3)
model.fit(X_train, config="optuna")
```

---

### `prepare_query(X: np.ndarray, n: int = 1, k: int = 1)`

Transforms and caches a batch of queries for fast retrieval.

**Parameters:**

- `X` (`np.ndarray`): Array of queries with shape `(n_queries, n_features)`.
- `n` (`int`): Number of candidates to retrieve from the index. Must be >= `k`. Increases accuracy but decreases speed.
- `k` (`int`): Number of final top-k neighbors to return.

**Returns:**

- None. Stores embedded queries internally for later search.

```python
model.prepare_query(X_query, n=10, k=5)
```

---

### `query_one(v: np.ndarray, n: int = 1, k: int = 1)`

Searches for the nearest neighbors of a single input vector `v`.

**Parameters:**

- `v` (`np.ndarray`): A single query vector with shape `(n_features,)`.
- `n` (`int`): Number of candidates to retrieve from the index. Must be >= `k`. Increases accuracy but decreases speed.
- `k` (`int`): Number of top-k neighbors to return.

**Returns:**

- `List[int]`: Indices of the top-k neighbors in the dataset.

```python
neighbors = model.query(v_single, n=10, k=3)
```

---

### `query(X: np.ndarray, n: int = 1, k: int = 1)`

Retrieves the top-k neighbors for a batch of preprocessed queries.

**Parameters:**

- `X` (`np.ndarray`): Batch of queries with shape `(n_queries, n_features)`.
- `n` (`int`): Number of candidates to retrieve from the index. Must be >= `k`.
- `k` (`int`): Number of top-k neighbors to return.

**Returns:**

- `np.ndarray`: Array of neighbor indices with shape `(n_queries, k)`.

```python
results = model.run_batch_query(X_query, n=10, k=5)
```

---

### `save(name: str = "last")`

Persists the trained model, its configuration, and the index to the local cache.

**Parameters:**

- `name` (`str`): Name under which to save the artifacts. Defaults to "last".

**Returns:**

- None. Files are saved to `~/.cache/infinitysearch/`.

```python
model.save(name="mnist_model")
```

---

### `load(name: str = "last")`

Loads a previously saved model, its configuration, and index from the cache.

**Parameters:**

- `name` (`str`): Identifier of the saved model and index to load.

**Returns:**

- `InfinitySearch`: A fully restored instance ready for querying.

```python
model = InfinitySearch.load(name="mnist_model")
```

---

### `remove(name: str = "all")`

Deletes one or all cached configurations, models, and indices.

**Parameters:**

- `name` (`str`):
  - If a name is provided, removes files associated with it.
  - If "all", deletes the entire InfinitySearch cache after confirmation.

**Returns:**

- None.

```python
InfinitySearch.remove(name="mnist_model")  # Remove specific
InfinitySearch.remove(name="all")          # Remove all (with confirmation)
```

---

## 📁 Caching & Configurations

- Configurations are stored in `~/.cache/infinitysearch/configs.json`
- The latest run is saved under key `last`
- Models and indices are stored under `~/.cache/infinitysearch/models/` and `~/.cache/infinitysearch/indices/`

---

## 🧪 Test Example

```python
from infinitysearch.test import main
main()
```

---

## 📊 Benchmarks


The following plots show the speed–accuracy tradeoff of InfinitySearch (yellow) compared to other popular vector search algorithms across different datasets.


<p align="center">
  <img src="benchmarks/legend_only-ann.png" style="background:white; padding:10px; border:1px solid #ccc;"/>
</p>


| Dataset           | k = 1 | k = 5 | k = 10 |
| ----------------- | ----- | ----- | ------ |
| **Fashion-MNIST** | ![](benchmarks/mnist-batch1.png) | ![](benchmarks/mnist-batch5.png) | ![](benchmarks/mnist-batch10.png) |
| **GIST**          | ![](benchmarks/gist-batch1.png)  | ![](benchmarks/gist-batch5.png)  | ![](benchmarks/gist-batch10.png)  |
| **Kosarak**       | ![](benchmarks/kosarak-batch1.png) | ![](benchmarks/kosarak-batch5.png) | ![](benchmarks/kosarak-batch10.png) |

---

## 📜 License

This package is distributed for **non-commercial research purposes** only. See `LICENSE` for details.

---

## ✉ Contact

For questions or contributions, please contact: `pariente@seas.upenn.edu`

---

## 📚 Citation

If you use InfinitySearch in your research, please cite:

**Infinity Search: Approximate Vector Search with Projections on q-Metric Spaces**

$Insert link here$
