---
title: "Reranking ViDoRe with prepared predictions"
icon: lucide/list-ordered
---

# Reranking ViDoRe with prepared predictions

The existing `Vidore3<Domain>Retrieval.v2` tasks declare prepared first-stage
predictions. Their corpus, queries, relevance labels, prompts and language subsets
remain unchanged. Domains are ComputerScience, Energy, FinanceEn, FinanceFr, Hr,
Industrial, Pharmaceuticals and Physics.

Candidates come from [mteb/Vidore3RetrievalPredictions](https://huggingface.co/datasets/mteb/Vidore3RetrievalPredictions/tree/3d6834bc0d3aded9de65eb2e431d875f654c96e8),
pinned to `3d6834bc0d3aded9de65eb2e431d875f654c96e8`:

| Source | Retriever | Retriever document inputs |
| --- | --- | --- |
| `bm25-text` | `mteb/baseline-bm25s` | Text |
| `bge-text` | `BAAI/bge-m3` | Text |
| `qwen-text-image` | `Qwen/Qwen3-VL-Embedding-2B` | Text + image |

The source names and paths distinguish the retriever's document inputs. They do
not select the reranker's inputs: the task still exposes images and OCR text, and
the reranker must support those inputs. A general document-modality selector is
outside this change.

## Evaluate one source

```python
import mteb

task = mteb.get_task("Vidore3HrRetrieval.v2", hf_subsets=["english"])
task.convert_to_reranking(first_stage="qwen-text-image", top_k=50)
reranker = mteb.get_model("Qwen/Qwen3-VL-Reranker-2B")
cache = mteb.ResultCache("mteb-cache")
result = mteb.evaluate(reranker, task, cache=cache)
print(result.model_meta.experiment_kwargs["first_stage"])

# Reload this experiment using the existing cache and export APIs.
results = cache.load_results(models=[result.model_meta], include_remote=False)
results._to_dataset().to_json("reranking-results.jsonl")
```

Without conversion, the same task performs ordinary retrieval. Use a fresh task
for each first-stage source. Results use the existing experiment hierarchy:

```text
results/<reranker>/<revision>/experiments/<experiment-id>/
  model_meta.json
  Vidore3HrRetrieval.v2.json
  Vidore3EnergyRetrieval.v2.json
```

Different sources, candidate depths and pinned prediction versions produce
separate experiments. Domains from one pinned collection share an experiment;
its metadata contains a task filename template. The ordinary result JSON format
is unchanged. The exported `experiments` column retains the first-stage settings.
See [two-stage reranking](two_stage_reranking.md#results-use-model-experiments) for
local-file identities and loading multiple experiments.

## Run several sources

```bash
# Two domains, three sources, English, top-50. The default random encoder
# is a CPU workflow check, not a quality evaluation.
python -m scripts.run_vidore_reranking --output vidore-format-demo

# A real reranker, in an environment supporting this model.
python -m scripts.run_vidore_reranking \
  --model Qwen/Qwen3-VL-Reranker-2B --domains Hr --top-k 10 \
  --output vidore-qwen-results
```

The helper evaluates each source separately, reloads the selected experiments,
and writes JSONL and Parquet exports. Repeating a command reuses cached metrics.
Use `--first-stages bm25-text bge-text` to select sources, and `--subsets` to select
languages. Choose one experimental condition before producing benchmark summaries;
leaderboard presentation and comparisons across first stages remain follow-up work.

The existing retrieval prompt differs from the separate reranking tasks in
[PR #5584](https://github.com/embeddings-benchmark/mteb/pull/5584). Comparisons should
match the actual instructions, candidate depth, language subset and reranker
inputs. No first-stage predictions need regenerating for this change.
