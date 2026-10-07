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

## One task with candidates from multiple retrievers

A reranking task can use named subsets for different first-stage retrievers. Each
reranker then evaluates the same frozen candidates for each retriever, with
separate scores for BM25, Qwen, BGE, or other candidate sources.

Set `TaskMetadata.reranking_subsets` to map each evaluation subset to its shared
data subset. The keys must match the subsets in `eval_langs`. For example:

```python
from mteb.abstasks.retrieval import AbsTaskRetrieval
from mteb.abstasks.task_metadata import TaskMetadata


class MultiRetrieverReranking(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="MultiRetrieverReranking",
        description="Frozen candidates from three retrievers over a shared corpus.",
        dataset={
            "path": "your-org/multi-retriever-reranking",
            "revision": "<dataset-commit-sha>",
        },
        type="Reranking",
        category="t2t",
        eval_splits=["test"],
        eval_langs={
            "bm25": ["eng-Latn"],
            "qwen": ["eng-Latn"],
            "bge": ["eng-Latn"],
        },
        reranking_subsets={
            "bm25": "default",
            "qwen": "default",
            "bge": "default",
        },
        main_score="ndcg_at_10",
    )
```

This task reads the following configurations from **one** Hugging Face dataset
repository:

| Configuration | Contents |
| --- | --- |
| `corpus` | Shared documents, including text and/or image columns |
| `queries` | Shared evaluation queries |
| `qrels` | Full relevance judgments (`query-id`, `corpus-id`, `score`) |
| `bm25-top_ranked` | BM25 candidates (`query-id`, ordered `corpus-ids`) |
| `qwen-top_ranked` | Qwen candidates in the same format |
| `bge-top_ranked` | BGE candidates in the same format |

The corpus, queries, and judgments are loaded once per shared data subset and
split. Each retriever subset receives its own candidate list. For multilingual
or multi-domain tasks, use names such as `bm25-english` and `qwen-english`, both
mapped to `english`; the shared configurations are then `english-corpus`,
`english-queries`, and `english-qrels`. Map the French subsets to `french`, and
so on. These are explicit mappings, not a naming convention parsed by MTEB.

### Upload shared data and separate candidates

Populate `task.dataset` using the usual retrieval data format. Reuse the same
`Dataset` objects for corpus and queries in subsets mapped to the same data
subset. Supply the same full relevance judgments and different ordered document
IDs for each retriever:

```python
# shared_split contains corpus, queries, relevant_docs, and top_ranked, as in
# an existing loaded retrieval task's dataset["default"]["test"].
# candidates_by_retriever maps "bm25", "qwen", and "bge" to
# {query_id: [doc_id, ...]}, with each list already truncated to the chosen top-k.
task = MultiRetrieverReranking()
task.dataset = {
    retriever: {
        "test": {**shared_split, "top_ranked": candidates}
    }
    for retriever, candidates in candidates_by_retriever.items()
}
task.data_loaded = True
task.push_dataset_to_hub("your-org/multi-retriever-reranking", private=True)
```

The uploader writes shared corpus, queries, and qrels once, then writes each
retriever's candidate configuration. It rejects conflicting shared data before
uploading. Update the task's dataset revision to the resulting commit before
running evaluations. Keep the `reranking_subsets` mapping in the task definition
that readers use; the loader does not infer it from the repository files.

Both upload and load validate that candidate lists cover exactly the evaluated
queries, contain no duplicate documents, and refer to existing corpus IDs. This
layout currently requires at least one candidate per query. A query
whose candidates contain **no relevant document** is valid and remains in the
evaluation. Never add known positives to a retriever's candidates or restrict
qrels to the retrieved documents.

Record each retriever's exact model revision, input representation, search
settings, and candidate depth in the dataset card. Keep the original retrieval
scores or ranks to compute the baseline. This layout also supports multimodal
corpora: retain stable page IDs across images and extracted text, and declare
the corresponding task category and modalities.

### Evaluate and inspect each retriever

```python
import mteb

reranker = mteb.get_model("cross-encoder/ms-marco-TinyBERT-L2-v2")
result = mteb.evaluate(reranker, MultiRetrieverReranking())[0]

for retriever in ("bm25", "qwen", "bge"):
    print(retriever, result.get_score(subsets=[retriever]))
```

To run just one subset, call
`task.filter_languages(languages=None, hf_subsets=["qwen"])` before evaluation, or use
`mteb.get_task(..., hf_subsets=["qwen"])` once the task is registered. Saved results
retain each `hf_subset`. Calling `get_score()` without a subset filter still
averages all selected subsets; define the benchmark's aggregation policy before
using that average to rank models. Baseline scores, improvement calculations,
and a leaderboard retriever selector are separate reporting work.

Tasks without `reranking_subsets` continue to use the existing dataset layout.

### Combine existing per-retriever tasks

Existing reranking tasks can be repackaged without regenerating candidates when
their corpus, queries, and full relevance judgments match. Keep one combined task
per source dataset or domain, and add the retriever name to each language subset:

| Existing configuration | Combined configuration |
| --- | --- |
| Each retriever's `english-corpus`, `english-queries`, `english-qrels` | One shared copy of each |
| BM25's `english-top_ranked` | `bm25-english-top_ranked` |
| Qwen's `english-top_ranked` | `qwen-english-top_ranked` |
| BGE's `english-top_ranked` | `bge-english-top_ranked` |

Declare `bm25-english`, `qwen-english`, and `bge-english` in `eval_langs`, each with
`["eng-Latn"]`, and map all three to `english` in `reranking_subsets`. Repeat for
the other languages. Preserve each existing candidate list's order and depth.

Compare relevance judgments by query ID, document ID, and score; different row
orders in the source files do not change their meaning. When using the Python
uploader, verify the shared data first, then reuse the corpus and query `Dataset`
objects from one source task in every corresponding retriever subset. Independently
loaded datasets can have different fingerprints even when their contents match.

Publish the combined layout under a new dataset revision and task name. Existing
results remain attached to their original task names; they are not automatically
migrated to the new subsets. Rerunning the rerankers against the saved candidates
does not require rerunning first-stage retrieval.
