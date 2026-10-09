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


## Reuse prepared first-stage predictions

Tasks can declare named local files or predictions from a pinned Hugging Face dataset:

```python
from mteb.abstasks.first_stage_predictions import FirstStagePredictionSource

task.first_stage_predictions = {
    "bm25": FirstStagePredictionSource(
        repo_id="your-org/first-stage-predictions",
        filename="bm25/NanoArguAnaRetrieval_predictions.json",
        revision="<full-40-character-dataset-commit-sha>",
    ),
    "local": prediction_folder,
}
task.convert_to_reranking(first_stage="bm25", top_k=50)
cache = mteb.ResultCache("mteb-cache")
result = mteb.evaluate(cross_encoder, task, cache=cache)
```

The positional argument remains a local file or directory. Use the `first_stage`
keyword to select a declared source. Prediction files use the existing MTEB format,
including `mteb_model_meta`. Conversion preserves the corpus, queries, relevance
labels and language subsets; only the candidate lists change. Instantiate a fresh
task for ordinary retrieval or another first-stage source.

## Results use model experiments

Converted runs use the existing experiment directories:

```text
<cache>/results/<reranker>/<revision>/experiments/<experiment-id>/
  model_meta.json
  <task-name>.json
  run_settings.jsonl
```

The result's model metadata contains `experiment_kwargs["first_stage"]`: the
source name, retriever metadata, top-k, prediction identity and an identity digest.
Existing reranker experiment settings are preserved. There is no additional
reranking result schema. The task JSON retains the existing
`previous_results_model_meta` score field; dataset exports identify the first
stage through the existing `experiments` column.

Pinned Hub sources use repository, revision and filename. Standard task filenames
are stored as `{task}_predictions.json` templates so domains from the same
collection share an experiment. Other filenames remain literal. Local sources use
a hash of the prediction file's bytes, so changed files select a different
experiment. Older results are not migrated automatically.

One `evaluate` call returns one model experiment. Evaluate different first stages,
top-k values, or ordinary retrieval separately. Tasks from the same pinned
collection and candidate depth can be evaluated together. First-stage settings
are evaluation context: they are derived from the tasks and are not forwarded
to model loaders. If `prediction_folder` is supplied, predictions are also placed
under `experiments/<experiment-id>/`.

Use the returned metadata to reload exactly that experiment:

```python
results = cache.load_results(models=[result.model_meta], include_remote=False)
rows = results._to_dataset()
rows.to_parquet("reranking-results.parquet")
```

To inspect all experiments for a reranker, use `load_experiments="match_name"`.
Select one experimental condition before benchmark summaries. Leaderboard
presentation and aggregation across first stages are separate follow-up work.
