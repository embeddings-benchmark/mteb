---
title: "Downsampling a Dataset"
---

# Downsampling a dataset

A smaller dataset is useful for debugging an evaluation or developing a new task.
It is a different evaluation, however: fewer queries increase uncertainty, and
removing corpus documents changes the retrieval problem. Do not submit these
scores under the original task's name or compare them as equivalent benchmark scores.

For a row-based dataset, selecting rows with a fixed seed may be sufficient.
Retrieval needs extra care because queries, corpus documents, and relevance
judgments (qrels) reference one another by ID. Never sample those independently.

## Choose what to reduce

- **Queries only:** keep the full corpus and retain the sampled queries' qrels.
  This reduces query encoding and scoring work, but not corpus encoding or indexing.
- **Queries and corpus:** keep all judged documents for the selected queries,
  then sample additional documents as distractors. This reduces indexing work too,
  but may make retrieval easier. Unjudged documents are not necessarily irrelevant.

Sampling after loading does not reduce the initial dataset download. For large
datasets, prepare and publish a smaller derived dataset instead. For classification,
consider label stratification rather than using this retrieval-specific recipe.

## A reproducible retrieval recipe

The following helper is example code, not a new MTEB API. It operates on a single
current-format retrieval split: `task.dataset[subset][split]`, containing Hugging
Face `Dataset` objects for `queries` and `corpus`, a `relevant_docs` dictionary,
and optional `top_ranked` candidate lists. It supports ordinary full-corpus text
retrieval only, and rejects candidate-restricted tasks rather than silently
changing their candidates. IDs must be unique strings.

The seed and sorted IDs make the selection independent of input row order. Keep
the dataset revision and Python version fixed too, and record the selected IDs
when publishing a derived task. Relevance grades, including zero, are preserved.

```python
import random


def sample_retrieval(data, *, n_queries, corpus_size=None, seed=42):
    if data.get("top_ranked") is not None:
        raise ValueError("This recipe does not support candidate-restricted retrieval")
    queries, corpus = data["queries"], data["corpus"]
    qrels = data["relevant_docs"]
    query_ids, document_ids = list(queries["id"]), list(corpus["id"])
    for ids in (query_ids, document_ids):
        if not all(isinstance(item, str) for item in ids) or len(set(ids)) != len(ids):
            raise ValueError("Expected unique string IDs")
    if not 1 <= n_queries <= len(query_ids):
        raise ValueError("n_queries must be between 1 and the number of queries")
    document_set = set(document_ids)
    for query_id in query_ids:
        judgments = qrels.get(query_id, {})
        if not any(score > 0 for score in judgments.values()):
            raise ValueError(f"Query {query_id} has no positive judgment")
        if not set(judgments) <= document_set:
            raise ValueError(f"Query {query_id} references missing documents")

    rng = random.Random(seed)
    selected_queries = sorted(rng.sample(sorted(query_ids), n_queries))
    selected_qrels = {qid: dict(qrels[qid]) for qid in selected_queries}
    required = {doc for judgments in selected_qrels.values() for doc in judgments}
    if corpus_size is None:
        selected_documents = sorted(document_ids)
    else:
        if not len(required) <= corpus_size <= len(document_ids):
            raise ValueError(
                "corpus_size must fit all judged documents and not exceed the corpus"
            )
        extras = rng.sample(
            sorted(document_set - required), corpus_size - len(required)
        )
        selected_documents = sorted(required | set(extras))

    query_rows = {qid: index for index, qid in enumerate(query_ids)}
    document_rows = {doc: index for index, doc in enumerate(document_ids)}
    return {
        **data,
        "queries": queries.select([query_rows[qid] for qid in selected_queries]),
        "corpus": corpus.select([document_rows[doc] for doc in selected_documents]),
        "relevant_docs": selected_qrels,
    }
```

If the requested corpus size cannot hold every judged document, increase it or
select fewer queries. Do not silently drop positive judgments to meet a size limit.

## Run and validate without downloading a dataset

After installing MTEB, run this block after the helper above. No model or network
access is required. The assertions check referential integrity and repeatability.

```python
from datasets import Dataset

data = {
    "queries": Dataset.from_dict(
        {"id": ["q1", "q2", "q3"], "text": ["one", "two", "three"]}
    ),
    "corpus": Dataset.from_dict(
        {
            "id": ["d1", "d2", "d3", "d4", "d5", "d6"],
            "text": ["one", "two", "three", "four", "five", "six"],
        }
    ),
    "relevant_docs": {"q1": {"d1": 2, "d4": 0}, "q2": {"d2": 1}, "q3": {"d3": 1}},
    "top_ranked": None,
}
sample = sample_retrieval(data, n_queries=2, corpus_size=4, seed=42)
query_ids = set(sample["queries"]["id"])
document_ids = set(sample["corpus"]["id"])
assert len(query_ids) == 2 and len(document_ids) == 4
assert set(sample["relevant_docs"]) == query_ids
for qid, judgments in sample["relevant_docs"].items():
    assert judgments == data["relevant_docs"][qid]
    assert set(judgments) <= document_ids
    assert any(score > 0 for score in judgments.values())
again = sample_retrieval(data, n_queries=2, corpus_size=4, seed=42)
assert list(sample["queries"]["id"]) == list(again["queries"]["id"])
assert list(sample["corpus"]["id"]) == list(again["corpus"]["id"])
```

## Use it in a separate MTEB task

This example downloads the small NanoSciFact dataset and defines a new task name.
Run it after the helper above. The existing task's pinned dataset revision is
reused, but the derived task has its own identity. Loading is completed before
sampling, and repeated `load_data()` calls do not resample the task.

```python
import mteb
from mteb.abstasks import AbsTaskRetrieval


class SampledSciFact(AbsTaskRetrieval):
    metadata = mteb.get_task("NanoSciFactRetrieval").metadata.model_copy(
        update={
            "name": "LocalSampledSciFact",
            "description": "Local development sample: 10 queries, full NanoSciFact corpus, seed 42.",
        }
    )

    def load_data(self, **kwargs):
        if self.data_loaded:
            return
        super().load_data(**kwargs)
        self.dataset["default"]["train"] = sample_retrieval(
            self.dataset["default"]["train"], n_queries=10, seed=42
        )


task = SampledSciFact()
task.load_data()
assert len(task.dataset["default"]["train"]["queries"]) == 10
```

Use a separate `mteb.ResultCache("local-downsampling-results")` if evaluating this
task. Give each changed sampling configuration a distinct task name or fresh cache
to avoid reusing results from a different sample. For multilingual datasets, apply
and validate a sampling policy separately for each intended subset and split;
do not accidentally pool languages or mix training and evaluation data.

Before proposing a derived task, follow [Adding a Task](adding_a_dataset.md):
document the source revision, seed, selected IDs, sampling algorithm, Python and
MTEB versions, and per-subset/split sizes. Recompute descriptive statistics, choose
a distinct name and appropriate metadata, and discuss the changed evaluation with
maintainers. Do not reuse the parent task's statistics or claim equivalent coverage.
