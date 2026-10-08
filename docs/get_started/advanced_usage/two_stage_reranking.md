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

## Named first-stage predictions

A retrieval task can declare reusable candidate sources on the task class or an
instance. Corpus, queries, full qrels, and language subsets stay in the existing
retrieval dataset. Prediction files use MTEB's existing JSON format.

For example, configure a published prediction file on an existing task (replace
the repository, filename, and commit SHA with those of your predictions):

```python
from mteb.abstasks.first_stage_predictions import FirstStagePredictionSource

task = mteb.get_task("NanoArguAnaRetrieval")
task.first_stage_predictions = {
    "bm25": FirstStagePredictionSource(
        repo_id="your-org/first-stage-predictions",
        filename="bm25/NanoArguAnaRetrieval_predictions.json",
        revision="<full-40-character-dataset-commit-sha>",
    ),
    "local": prediction_folder,  # The directory saved in stage 1 above.
}
task.convert_to_reranking(first_stage="bm25", top_k=50)
result = mteb.evaluate(cross_encoder, task, cache=mteb.ResultCache("mteb-cache"))
```

The new source resolver supports local files/directories and pinned Hugging Face
dataset files. The existing
`convert_to_reranking(local_path, top_k=...)` call still works. The positional
argument always selects a local path. Use `first_stage="bm25"` to select a declared
source, supplying exactly one of the path or the source name.

Each prediction file contains one retriever's metadata and query/document scores:

```json
{
  "mteb_model_meta": {"model_name": "your-org/retriever", "revision": "model-sha"},
  "english": {
    "test": {
      "query-1": {"document-7": 0.91, "document-2": 0.83}
    }
  }
}
```

Conversion uses the existing candidate selection: descending scores, up to
`top_k` documents per query, with JSON order retained for ties. The corpus,
queries, and full qrels stay in the original retrieval task.

### Results and candidate provenance

Converted evaluations have a separate result directory:

```text
<cache>/results/<reranker>/<revision>/
  model_meta.json
  reranking/<configuration-id>/<task-name>.json
```

A task result has an optional top-level `reranking` object with `first_stage`,
`top_k`, `document_representation`, and
`predictions` (the file SHA256 and, for Hub sources, repository, revision and
filename). Source declarations can set `document_representation` to `text`,
`image`, or `text-image`; this describes the retriever's inputs, not the reranker's.
The producer model name and revision remain in the existing
`previous_results_model_meta` field on each score row. They are retained when
loading only main scores and included in dataset exports. Language subsets
remain unchanged inside `scores["test"]`.

The configuration ID groups tasks using the same named first stage,
representation, pinned Hub directory and top-k. Each directory should contain
one retriever and document representation across tasks. The filename and
checksum remain per-task provenance and must match when reusing or merging
results. Local files use their content hash in the configuration ID. Different
first stages, depths and pinned collection revisions select different folders.

Actual model experiments remain independent: when present, the reranking folder
is nested under `experiments/<experiment-name>/`. Model loader arguments never
include first-stage provenance. Raw reranker predictions use the corresponding
`prediction_folder/[experiments/<experiment-name>/]reranking/<configuration-id>/`
namespace.

```python
cache = mteb.ResultCache("mteb-cache")
result = mteb.evaluate(cross_encoder, task, cache=cache)
configuration = result[0].reranking
print(configuration.model_dump())
restored = cache.load_task_result(
    task.metadata.name, result.model_meta, reranking=configuration
)
```

Instantiate a task for each source. Task lists can contain several sources and
domains. Reloading, revision joining and submission retain each configuration:

```python
results = cache.load_results(include_remote=False)
rows = results._to_dataset()  # Includes `reranking_id`, `reranking`, and `previous_results_model_meta`.
rows.to_parquet("reranking-results.parquet")

# Summary helpers require an explicit first-stage selection.
selected = results.filter_reranking(configuration.configuration_id)
print(selected.to_dataframe())
```

Use `filter_reranking(None)` to select ordinary results. Summary methods reject
mixed configurations instead of averaging first stages or selecting the best
score. This draft does not define an aggregate-across-retrievers policy. For
`AbsTaskAggregate` containing converted tasks, evaluate the child tasks as a list
and select a configuration before summarising their results.

This directory and export schema is a proposal. The results repository's layout
checks and the leaderboard's grouping/filters need corresponding support before
these files can be published as leaderboard results. The MTEB loader, submission
copying and dataset export support the proposed format in this branch.

Historical ordinary results and the earlier experiment-based prototype are not
automatically migrated. They require an explicit, provenance-checked migration;
first-stage prediction files do not need to be regenerated.
