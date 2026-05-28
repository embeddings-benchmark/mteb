## Using a Hybrid Model

MTEB provides a unified [`mteb.HybridSearch`][mteb.HybridSearch] wrapper that allows you to combine multiple retrievers and cross-encoders using different fusion strategies (e.g. Reciprocal Rank Fusion, Distribution-Based Score Fusion, or custom fusion functions).

You can define a hybrid model as follows:

```python
import mteb
from mteb import HybridSearch

# Load individual sub-models (lexical/sparse and dense)
bm25 = mteb.get_model("mteb/baseline-bm25s")
dense = mteb.get_model("intfloat/multilingual-e5-small")

# Create the hybrid model combining both sub-models
hybrid_model = HybridSearch(
    models=[bm25, dense],
    fusion_strategy="rrf",  # Options: "rrf", "dbsf", "relative-score-fusion", or a custom Callable
    weights=[0.5, 0.5],     # Optional: weight assigned to each model
)

# Evaluate the hybrid model on your selected tasks
tasks = mteb.get_tasks(tasks=["Banking77Classification"])
results = mteb.evaluate(hybrid_model, tasks=tasks)
```

### Registered Hybrid Models

You can also register pre-configured hybrid models by adding a custom loader function and model metadata under `mteb/models/model_implementations/hybrid_models.py`. Once registered, the hybrid model can be loaded by name:

```python
import mteb

# Load pre-registered hybrid model directly
hybrid_model = mteb.get_model("mteb/hybrid-bm25s-e5-small")
```
