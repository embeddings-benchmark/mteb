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
    weights=[0.5, 0.5],     # Optional: weight assigned to each model
)

# Evaluate the hybrid model on your selected tasks
tasks = mteb.get_tasks(tasks=["NFCorpus"])
results = mteb.evaluate(hybrid_model, tasks=tasks)
```

### Performance Comparison

To demonstrate the effectiveness of combining different retrieval paradigms, the table below compares the performance (NDCG@10) of individual sub-models against their hybrid combinations using Reciprocal Rank Fusion (RRF), Distribution-Based Score Fusion (DBSF), and Relative Score Fusion (RSF):

| Task | mteb/baseline-bm25s | intfloat/multilingual-e5-small | mteb/baseline-hybrid-rrf (me5-small+bm25) | mteb/baseline-hybrid-dbsf (me5-small+bm25) | mteb/baseline-hybrid-rsf (me5-small+bm25) |
|---|---|---|---|---|---|
| NanoSciFactRetrieval | 0.70991 | 0.72458 | 0.75391 | 0.53757 | 0.76656 |
| NanoNFCorpusRetrieval | 0.32504 | 0.28820 | 0.32870 | 0.33824 | 0.35915 |
| NanoSCIDOCSRetrieval | 0.33512 | 0.34378 | 0.36856 | 0.34413 | 0.37242 |

*Note: Hybrid models combine the lexical `mteb/baseline-bm25s` and dense `intfloat/multilingual-e5-small` models using equal weights (0.5/0.5).*
