# Selecting Tasks or Benchmarks

This section describes how to select benchmarks and tasks to evaluate, including selecting specific subsets or splits to run.

## Selecting a Benchmark

`mteb` comes with a set of predefined benchmarks. These can be fetched using [`get_benchmark`](../api/benchmark.md#mteb.get_benchmark) or [`get_benchmarks`](../api/benchmark.md#mteb.get_benchmarks) and run in a similar fashion to other sets of tasks.
For instance to select the English benchmark that forms the English leaderboard:

```python
import mteb
benchmark = mteb.get_benchmark("MTEB(eng, v2)")
model = ...
results = mteb.evaluate(model, tasks=benchmark)
```

The benchmark specifies not only a list of tasks, but also what splits and language to run on.

!!! note
    Generally we use the naming scheme for benchmarks `MTEB(*)`, where the "*" denotes the target of the benchmark.
    In the case of a language, we use the three-letter language code.
    For large groups of languages, we use the group notation, e.g., `MTEB(Scandinavian, v1)` for Scandinavian languages.
    External benchmarks implemented in MTEB like `CoIR`[@coir] use their original name.

To get an overview of all available benchmarks, simply run:

```python
import mteb
benchmarks = mteb.get_benchmarks()
```

When using a benchmark from MTEB please cite `mteb` along with the citations of the benchmark which you can access using [`benchmark.citation`](../api/benchmark.md#mteb.Benchmark).

## Selecting a Task

`mteb` comes with the utility function [`get_task`](../api/task.md#mteb.get_task) and [`get_tasks`](../api/task.md#mteb.get_tasks) for fetching and analysing the tasks of interest.

This can be done in multiple ways, e.g.:

* by the task name
* by their type (e.g. "Clustering" or "Classification")
* by their languages (specified as a three letter code)
* by their domains
* by their modalities
* and many more

```python
# by name
tasks = mteb.get_tasks(tasks=["Banking77Classification"])
# by type
tasks = mteb.get_tasks(task_types=["Clustering", "Retrieval"]) # Only select clustering and retrieval tasks
# by language
tasks = mteb.get_tasks(languages=["eng", "deu"]) # Only select datasets which contain "eng" or "deu" (iso 639-3 codes)
# by domain
tasks = get_tasks(domains=["Legal"])
# by modality
tasks = mteb.get_tasks(modalities=["text", "image"]) # Only select tasks with text or image modalities
# or using multiple
tasks = get_tasks(languages=["eng", "deu"], script=["Latn"], domains=["Legal"])
```


You can also specify which languages to load for multilingual/cross-lingual tasks like below:

```python
import mteb

tasks = [
    mteb.get_task("AmazonReviewsClassification", languages = ["eng", "fra"]),
    mteb.get_task("BUCCBitextMining", languages = ["deu"]), # all subsets containing "deu"
]
```
For more information see the documentation for [`get_tasks`](../api/task.md#mteb.get_tasks) and [`get_task`](../api/task.md#mteb.get_task).

### Selecting Evaluation Split or Subsets
A task in `mteb` mirrors the structure of a dataset on Huggingface. It includes a splits (i.e. "test") and a subset.

```python
# selecting an evaluation split
task = mteb.get_task("Banking77Classification", eval_splits=["test"])
# selecting a Huggingface subset
task = mteb.get_task("AmazonReviewsClassification", hf_subsets=["en", "fr"])
```

!!! question "What is a subset?"
    A subset on a Huggingface dataset is what you specify after the dataset name, e.g. `datasets.load_dataset("nyu-mll/glue", "cola")`.
    Often the subset does not need to be defined and is left as "default". The subset is however useful, especially for multilingual datasets to specify the
    desired language or language pair e.g. in [`mteb/bucc-bitext-mining`](https://huggingface.co/datasets/mteb/bucc-bitext-mining) we might want to evaluate only on the French-English subset `"fr-en"`.


### Using a Custom Task

To evaluate on a custom task, you can run the following code on your custom task.
See [how to add a new task](../contributing/adding_a_dataset.md), for how to create a new task in MTEB.


```python
import mteb
from mteb.abstasks.retrieval import AbsTaskRetrieval
from mteb.abstasks.task_metadata import TaskMetadata


class MyCustomTask(AbsTaskRetrieval):
    metadata = TaskMetadata(...)

model = mteb.get_model(...)
results = mteb.evaluate(model, tasks=[MyCustomTask()])
```
