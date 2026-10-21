---
title: "Retrieval backend"
icon: lucide/scan-search
---


# Retrieval Search backend

!!! note "Available since 2.3.0"
    This feature was introduced in version **2.3.0**.

For some large dataset search can take a lot of time and memory. To reduce this you can wrap the model in a `SearchEncoderWrapper` with an `index_backend` that matches its embeddings. For dense models use [`FaissSearchIndex`][mteb.models.search_encoder_index.search_indexes.faiss_search_index.FaissSearchIndex]. To work with it install:

```bash
# pip
pip install mteb[faiss-cpu]

# or uv
uv add "mteb[faiss-cpu]"
```

Usage example:
```python
import mteb
from mteb.models import SearchEncoderWrapper
from mteb.models.search_encoder_index import FaissSearchIndex

model = mteb.get_model(...)
index_backend = FaissSearchIndex(model)
model = SearchEncoderWrapper(model, index_backend=index_backend)
...
```

For example running `minishlab/potion-base-2M` on `SWEbenchVerifiedRR` took 694 seconds instead of 769.

## Multi-vector models (PLAID)

For multi-vector (late interaction) models, such as ColBERT-style models from sentence-transformers or ColPali models, you can use [`FastPlaidSearchIndex`][mteb.models.search_encoder_index.search_indexes.fast_plaid_search_index.FastPlaidSearchIndex], which is built on [fast-plaid](https://github.com/lightonai/fast-plaid). To work with it install:

```bash
# pip
pip install mteb[fast-plaid]

# or uv
uv add "mteb[fast-plaid]"
```

Usage example:
```python
import mteb
from mteb.models import SearchEncoderWrapper
from mteb.models.search_encoder_index import FastPlaidSearchIndex

model = mteb.get_model("perplexity-ai/pplx-embed-v2-late-0.6b")
model = SearchEncoderWrapper(model, index_backend=FastPlaidSearchIndex(model))
results = mteb.evaluate(model, mteb.get_tasks(["NanoSciFactRetrieval"]))
```

PLAID is approximate: scores can differ slightly from exact MaxSim, and this also holds for reranking tasks, where PLAID only restricts the search to each query's candidates. The trade-off can be tuned with `nbits`, `n_ivf_probe` and `n_full_scores`. Because of this, the index parameters are saved as the model's `experiment_kwargs`, so PLAID results are stored as a separate experiment and don't overwrite exact MaxSim results.
