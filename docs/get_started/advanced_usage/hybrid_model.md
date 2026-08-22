---
title: "Hybrid model"
icon: lucide/merge
---

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
    fusion_strategy="rrf",
    weights=[0.5, 0.5],
)

# Evaluate the hybrid model on your selected tasks
tasks = mteb.get_tasks(tasks=["NFCorpus"])
results = mteb.evaluate(hybrid_model, tasks=tasks)
```

### Performance Comparison

To demonstrate the effectiveness of combining different retrieval paradigms, the table below compares the performance (NDCG@10) of individual sub-models against their hybrid combinations using Reciprocal Rank Fusion (RRF), Distribution-Based Score Fusion (DBSF), and Relative Score Fusion (RSF):

| Task | mteb/baseline-bm25s | intfloat/multilingual-e5-small | hybrid using `rrf` | hybrid using `dbsf` | hybrid using 'rsf` |
|---|---|---|---|---|---|
| NanoSciFactRetrieval | 0.710 | 0.725 | *0.754* | 0.538 | **0.767** |
| NanoNFCorpusRetrieval | 0.325 | 0.288 | 0.329 | *0.338* | **0.359** |
| NanoSCIDOCSRetrieval | 0.335 | 0.344 | *0.369* | 0.344 | **0.372** |

*Note: Hybrid models combine the lexical `mteb/baseline-bm25s` and dense `intfloat/multilingual-e5-small` models using equal weights (0.5/0.5).*
