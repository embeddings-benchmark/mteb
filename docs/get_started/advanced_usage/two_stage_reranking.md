---
title: "Two stage reranking"
icon: lucide/list-ordered
---


## Two stage reranking

To use a cross encoder for reranking on a retrieval task you first need a task, a stage 1 model and our cross-encoder.

```python
import mteb

task = mteb.get_task("NanoArguAnaRetrieval")
# stage 1 model:
encoder = mteb.get_model("sentence-transformers/static-similarity-mrl-multilingual-v1")
# stage 2 model:
cross_encoder = mteb.get_model("cross-encoder/ms-marco-TinyBERT-L2-v2")  # (1)!
```

1.  You can also directly use `CrossEncoder` from [sentence transformers](https://www.sbert.net/).

Once we have that we can perform stage 1 retrieval, followed by a stage 2 reranking (call [`convert_to_reranking`][mteb.abstasks.retrieval.AbsTaskRetrieval.convert_to_reranking] to convert the task to a reranking task, which will use the predictions from stage 1 as input for stage 2):

```python
prediction_folder = "model_predictions"

# stage 1: retrieval
res = mteb.evaluate(
    encoder,
    task,
    prediction_folder=prediction_folder,
)

# convert task to retrieval
task = task.convert_to_reranking(prediction_folder, top_k=100)

# stage 2: reranking
cross_enc_results = mteb.evaluate(cross_encoder, task)

print(task.metadata.main_score)  # NDCG@10
res[0].get_score()  # 0.286
cross_enc_results[0].get_score()  # 0.338
```

## Reranking tasks with fixed first stages

A prepared reranking task defines its candidate sources and depth. All first-stage
subsets run by default, using the normal task API:

```python
# Each language is evaluated with BM25, BGE-M3 and Qwen candidates at top-50.
task = mteb.get_task("Vidore3HrReranking")
model = mteb.get_model("Qwen/Qwen3-VL-Reranker-2B")
cache = mteb.ResultCache("mteb-cache")
results = mteb.evaluate(model, task, cache=cache)

# All three first stages, English only.
task = mteb.get_task("Vidore3HrReranking", languages=["eng"])

# One condition. Subset names are listed in task.metadata.hf_subsets.
task = mteb.get_task("Vidore3HrReranking", hf_subsets=["bge-text__top50__english"])
```

A single task result file contains separate scores for each first-stage/depth/language
subset, such as `bm25-text__top50__english`, `bge-text__top50__english` and
`qwen-text-image__top50__english`. The existing `previous_results_model_meta` field records
the retriever for each subset. `reranking_top_k` records the maximum number of
first-stage candidates supplied per query (50 for these tasks). Metrics include
@50 alongside the usual cutoffs; cutoffs above the candidate depth evaluate the
returned list without appending any remaining first-stage documents. A later
evaluation of more subsets fills in the same result file using the normal cache
behaviour. Export and benchmark grouping use these subsets just as they do for
other tasks.

```text
results/<reranker>/<revision>/
  model_meta.json
  Vidore3HrReranking.json
```

First stages are task conditions, so they do not create model experiments. Explicit
model experiments still use MTEB's normal experiment directories. Candidate depth
and pinned prediction sources are fixed by the task; changing an existing
condition requires a new task version. Depth is included in subset names so future
depths can have distinct identities without renaming the top-50 results. For an
exploratory two-stage run, use `convert_to_reranking` as above.

Task authors can subclass
[`AbsTaskRerankingFromPredictions`][mteb.abstasks.reranking_from_predictions.AbsTaskRerankingFromPredictions]
to reuse an existing retrieval task's corpus, queries and qrels. Declare the
`retrieval_task` class, a `first_stage_predictions` mapping of source names to
pinned `FirstStagePredictionSource` objects (or local paths), and `top_k`. Declare
`eval_langs` with `{first_stage}__top{top_k}__{retrieval_subset}` keys mapped to
the original language codes. Only selected languages and prediction sources are
loaded; corpus, queries and qrels are shared across first stages. Candidate sets
remain separate.
