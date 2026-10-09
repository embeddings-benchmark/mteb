---
title: "Reranking ViDoRe with prepared predictions"
icon: lucide/list-ordered
---

# Reranking ViDoRe with prepared predictions

The existing `Vidore3<Domain>Retrieval.v2` tasks declare prepared first-stage
predictions for reranking. They retain their task names, pinned datasets, corpus,
queries, full qrels, prompts and language subsets. Domains are ComputerScience,
Energy, FinanceEn, FinanceFr, Hr, Industrial, Pharmaceuticals and Physics.

Candidates come from [mteb/Vidore3RetrievalPredictions](https://huggingface.co/datasets/mteb/Vidore3RetrievalPredictions/tree/3d6834bc0d3aded9de65eb2e431d875f654c96e8),
pinned to `3d6834bc0d3aded9de65eb2e431d875f654c96e8`:

| Source | Retriever | Retriever document inputs |
| --- | --- | --- |
| `bm25-text` | `mteb/baseline-bm25s` | Text |
| `bge-text` | `BAAI/bge-m3` | Text |
| `qwen-text-image` | `Qwen/Qwen3-VL-Embedding-2B` | Text + image |

Both images and OCR text remain available to the reranker. The first-stage
representation does not select the reranker's inputs.

## Run and inspect the format

```bash
# From a checkout containing the prepared ViDoRe prediction sources:
pip install -e .

# Two domains, three first stages, English subset, top-50 candidates.
# The default random encoder is a CPU workflow check, not a quality evaluation.
python -m scripts.run_vidore_reranking --output vidore-format-demo

# Run a real reranker in an environment supporting that model.
python -m scripts.run_vidore_reranking \
  --model Qwen/Qwen3-VL-Reranker-2B --output vidore-qwen-results
```

The helper downloads pinned predictions through the Hub cache, evaluates each
configuration, reloads the saved results and exports `reranking-results.parquet`
and `reranking-results.jsonl`. Use `--domains Hr` for a smaller check, or pass all
eight domain names and the desired `--subsets`. Repeating a command reuses cached
metrics. `--first-stages` and `--top-k` select candidates explicitly.

The equivalent single-task API is:

```python
import mteb

task = mteb.get_task("Vidore3HrRetrieval.v2", hf_subsets=["english"])
task.convert_to_reranking(first_stage="qwen-text-image", top_k=50)
reranker = mteb.get_model("Qwen/Qwen3-VL-Reranker-2B")
result = mteb.evaluate(reranker, task, cache=mteb.ResultCache("mteb-cache"))
print(result[0].reranking.model_dump())
```

Call `convert_to_reranking` to select candidates before a reranking evaluation.
Without conversion, the same task runs its usual full-corpus retrieval evaluation.
Instantiate a fresh task for each first-stage source or ordinary retrieval run.

## Select the reranker's document inputs

For models using `CrossEncoderWrapper`, including Qwen3-VL, set the document
modalities when loading the model:

```python
reranker = mteb.get_model("Qwen/Qwen3-VL-Reranker-2B", document_modalities=["text"])
```

Use `["image"]` or `["text", "image"]` for the other document representations.
This selects the actual document columns passed to the model; query inputs stay
unchanged. Missing or unsupported modalities raise an error. Omitting the option
preserves the wrapper's default handling and does not assert which inputs were
used. Other model wrappers must support this option before it can be used with them.

The choice is saved in `model_meta.json` under
`experiment_kwargs.document_modalities`, retained in the exported `experiments`
column, and selects a separate `experiments/<experiment-id>/` directory above
`reranking/<configuration-id>/`. `ModelMeta.modalities` continues to describe
model capabilities. The first-stage `document_modalities` describes the
retriever's inputs independently of this reranker setting.

For a small text-only run:

```bash
python -m scripts.run_vidore_reranking \
  --model cross-encoder/ms-marco-MiniLM-L6-v2 \
  --document-modalities text --domains Hr --output vidore-text-reranker
```

## Proposed results layout

```text
results/<reranker>/<revision>/
  model_meta.json
  Vidore3HrRetrieval.v2.json  # Ordinary retrieval, if evaluated with this model.
  reranking/
    <bm25-configuration-id>/
      Vidore3HrRetrieval.v2.json
      Vidore3EnergyRetrieval.v2.json
    <bge-configuration-id>/
      Vidore3HrRetrieval.v2.json
      Vidore3EnergyRetrieval.v2.json
    <qwen-configuration-id>/
      Vidore3HrRetrieval.v2.json
      Vidore3EnergyRetrieval.v2.json
```

Each task JSON retains normal scores and language subsets. Its top-level
`reranking` object records the source name, representation, top-k and the exact
pinned prediction filename/checksum. The first-stage model and revision remain
in the existing `previous_results_model_meta` score field, also retained in
exports. The exported `reranking_id`
groups domains evaluated with the same configuration, independently of the
reranker model. `reranking` contains the structured provenance for each row.
Model experiment settings, if used, retain their own namespace above `reranking/`.

Filter by `results.filter_reranking(configuration_id)` before creating a summary.
The proposed leaderboard can then compare rerankers within each configuration;
no average across retrievers or best-retriever selection is implied. Results-repo
validation and leaderboard grouping/UI need corresponding changes before
upstream publication. Task metadata remains `DocumentUnderstanding`; consumers
should use the result's `reranking` configuration to identify converted runs.

The existing retrieval task prompt is preserved. It differs from the prompt on
the separate reranking tasks in [PR #5584](https://github.com/embeddings-benchmark/mteb/pull/5584).
For comparisons, match the actual model instructions as well as candidate depth,
language subset and reranker inputs/settings. This change does not migrate
historical metrics or require regenerating first-stage predictions. A random-model smoke run does
not establish parity with an earlier model evaluation.

Focused task checks:

```bash
python -m pytest tests/test_tasks/test_vidore3_prepared_reranking.py -q
```
