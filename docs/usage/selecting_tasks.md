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

### Filtering Benchmark Tasks by Type

You can filter a benchmark to only include specific task types. This is useful when you want to evaluate your model on a subset of tasks, such as only retrieval tasks:

```python
import mteb

# Get the full English benchmark
benchmark = mteb.get_benchmark("MTEB(eng, v2)")

# Filter to only retrieval tasks
retrieval_tasks = [task for task in benchmark.tasks if task.metadata.type == "Retrieval"]
print(f"Found {len(retrieval_tasks)} retrieval tasks")

# Run evaluation on only retrieval tasks
model = mteb.get_model(...)
results = mteb.evaluate(model, tasks=retrieval_tasks)
```

You can filter by any task type, including:
- `"Retrieval"` - Information retrieval tasks
- `"Classification"` - Text classification tasks
- `"Clustering"` - Document clustering tasks
- `"STS"` - Semantic textual similarity tasks
- `"PairClassification"` - Pair classification tasks
- `"Reranking"` - Reranking tasks
- `"Summarization"` - Text summarization tasks
- `"InstructionRetrieval"` - Instruction-based retrieval tasks

For multiple task types, you can combine filters:

```python
# Get retrieval and reranking tasks from a benchmark
filtered_tasks = [
    task for task in benchmark.tasks
    if task.metadata.type in ["Retrieval", "Reranking"]
]
```

### Creating Custom Benchmarks from Filtered Tasks

You can create a custom benchmark with filtered tasks:

```python
import mteb
from mteb.benchmarks.benchmark import Benchmark

# Get the full English benchmark
original = mteb.get_benchmark("MTEB(eng, v2)")

# Create a retrieval-only benchmark
retrieval_benchmark = Benchmark(
    name="MTEB(eng, retrieval-only)",
    tasks=[task for task in original.tasks if task.metadata.type == "Retrieval"],
    description="English retrieval tasks from MTEB"
)

# Use it for evaluation
results = mteb.evaluate(model, tasks=retrieval_benchmark)
```

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

### Filtering Tasks from Existing Benchmarks

You can also filter existing benchmarks to get specific task types. For example, to get all retrieval tasks from multiple benchmarks:

```python
import mteb

# Get retrieval tasks from all English tasks
retrieval_tasks = mteb.get_tasks(
    task_types=["Retrieval"],
    languages=["eng"]
)

# Get retrieval tasks that are also part of the main English benchmark
eng_benchmark = mteb.get_benchmark("MTEB(eng, v2)")
benchmark_task_names = [task.metadata.name for task in eng_benchmark.tasks]

retrieval_from_benchmark = mteb.get_tasks(
    task_types=["Retrieval"],
    tasks=benchmark_task_names  # Only tasks from the benchmark
)

print(f"Found {len(retrieval_from_benchmark)} retrieval tasks in MTEB(eng, v2)")
```

### Advanced Filtering with Multiple Criteria

You can combine multiple filtering criteria to get very specific subsets:

```python
import mteb

# Get English retrieval tasks in specific domains
specialized_tasks = mteb.get_tasks(
    task_types=["Retrieval", "InstructionRetrieval"],
    languages=["eng"],
    domains=["Scientific", "Medical", "Legal"]
)

# Filter multilingual retrieval tasks
multilingual_retrieval = mteb.get_tasks(
    task_types=["Retrieval"],
    modalities=["text"]
)
# Then filter for tasks supporting multiple languages
multilingual_retrieval = [
    task for task in multilingual_retrieval
    if len(task.metadata.languages) > 1
]
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
