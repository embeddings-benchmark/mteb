## Adding a benchmark

The MTEB has a growing list of benchmarks, and we are always looking to add more. MTEB include both benchmark that displayed on the [leaderboard](https://huggingface.co/spaces/mteb/leaderboard) and benchmarks that are not displayed on the leaderboard but are still available for evaluation. These non-leaderboard benchmarks are available in the 
`mteb.get_benchmark(s)` function and are e.g. useful for evaluating models during development or benchmark that are too specific to be added to the leaderboard.

### Implement a new benchmark

To implement a new benchmark [`Benchmark`][mteb.Benchmark] object, and select the MTEB tasks that will be in the benchmark. If some of the tasks do not exist in MTEB, follow the ["add a dataset"](adding_a_dataset.md) instructions to add them.

Once you have selected the tasks, you can create a new benchmark as follows:

```python
import mteb

amozon_bench = Benchmark(
    name="MTEB(custom, v1)", # set the name 
    tasks=mteb.get_tasks( # (1)
        tasks=["AmazonCounterfactualClassification", "AmazonPolarityClassification"],
        languages=["eng"],
    ),
    # give a short description of the benchmark of what the benchmarks 
    # seeks to test for:
    description=(
        "My custom Amazon benchmark, "
        "which seeks to test for the ability of models "
        "to classify Amazon reviews based on their embeddings."
    ),
)
```

1. Select the tasks that will be in the benchmark. See [selecting tasks](../usage/selecting_tasks.md) for more details on how to select tasks.

??? info "Selecting high-quality tasks"
    When selecting tasks for a benchmark, it is important to select high-quality tasks that reflects what you seeks to measure. To facilitate this process each
    task in mteb comes with metadata (`task.metadata`) that includes a description of the task, the
    construction and annotation process, licensing and more. We additionally also
    include descriptive statistics (`task.metadata.descriptive_stats`) which includes information about the number of samples, minimum length and other statistics that can be useful to select the right tasks for your benchmark.
    
    Generally we recommend selecting tasks that are well established in the community, are not machine translated, are not too small and that are not too similar to other tasks in the benchmark. However, the selection of tasks will depend on what you seek to measure with the benchmark, and thus we recommend carefully reading the metadata of the tasks and selecting the ones that best fit your needs.

## Submitting a Benchmark

To submit a benchmark to MTEB, you need to add your benchmark to [benchmarks.py](https://github.com/embeddings-benchmark/mteb/blob/main/mteb/benchmarks/benchmarks/benchmarks.py) and then open a pull request (PR).

Once submitted, the PR will be reviewed by one of the organizers or contributors, who might ask you to make changes. The reviewer reviews both the implementation of the benchmark and the quality and relevance of the tasks.
Once the PR is approved, the benchmark will be added to MTEB and will be available via [`mteb.get_benchmark(name)`][mteb.get_benchmark]. Note this does not automatically add the benchmark to the leaderboard; see the next section for instructions on how to do that.

## Submitting a Benchmark to the Leaderboard

To submit a benchmark to the leaderboard, you need to:

1. Have added the benchmark to MTEB as described in the previous section
2. Evalaute a set of models on the benchmark and submit a PR with the results to the [results repository](https://github.com/embeddings-benchmark/results) with the results of the models on the benchmark.
3. When your PR with benchmarks results is merged, you can add your benchmark to the most fitting section in [benchmark_selector.py](https://github.com/embeddings-benchmark/mteb/blob/main/mteb/leaderboard/benchmark_selector.py) to be shown on the leaderboard. You can check that the leaderboard looks correctly by [running the leaderboard locally](../usage/leaderboard.md#running-the-leaderboard-locally).
4. When PRs are merged, your benchmark will be added to the leaderboard automatically after the next workflow trigger (every day at midnight Pacific Time (8 AM UTC)).


!!! note "Not all benchmarks becomes leaderboards"
    A benchmark is a selection of tasks that intends to test for a specific purpose.
    Some benchmarks are very specific, are intended for development or are superseeded by newer benchmark. We continually try to keep the benchmarks on the leaderboard relevant and thus we might remove benchmarks from the leaderboard if they are no longer relevant. However, these benchmarks will still be available in MTEB and can be used for evaluation, but they just won't be shown on the leaderboard.