# ViDoRe(v3.1, reranking, beta): example summary table

Produced by `Benchmark._create_summary_table` (the function the API's
`build_benchmark_summary` uses) on synthetic results: each fake reranker has a
fixed score per first-stage retriever on every domain, so the per-retriever
columns should equal those inputs.

| Rank (Mean Task) | Model | Mean (Task) | Mean (Public) | Mean (Private) | First-stage retriever: BM25s | First-stage retriever: BGE-M3 | First-stage retriever: Qwen3-VL-Embedding-2B |
|---|---|---|---|---|---|---|---|
| 1 | reranker A | 0.500 | 0.500 | 0.500 | 0.30 | 0.50 | 0.70 |
| 2 | reranker B | 0.483 | 0.483 | 0.483 | 0.40 | 0.45 | 0.60 |

- **Mean (Task)**: mean over all 30 tasks (10 domains x 3 first-stage retrievers).
- **Mean (Public) / Mean (Private)**: mean over the 24 public tasks / the 6 private tasks (Telecom, Nuclear).
- **First-stage retriever columns**: mean over the 10 domains when reranking the top-50 candidates from that retriever
  (`mteb/baseline-bm25s`, `BAAI/bge-m3`, `Qwen/Qwen3-VL-Embedding-2B`).
- Ranking is by Mean (Task), tie-broken by Mean (Public), then Mean (Private).
