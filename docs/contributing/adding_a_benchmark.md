## Adding a benchmark

The MTEB Leaderboard is available [here](https://huggingface.co/spaces/mteb/leaderboard) and we encourage additions of new benchmarks.

To add a new benchmark:

1. Add your benchmark to [benchmarks.py](https://github.com/embeddings-benchmark/mteb/blob/main/mteb/benchmarks/benchmarks/benchmarks.py) as a [`Benchmark`][mteb.Benchmark] object, and select the MTEB tasks that will be in the benchmark. If some of the tasks do not exist in MTEB, follow the ["add a dataset"](adding_a_dataset.md) instructions to add them.
2. Open a PR at [results repository](https://github.com/embeddings-benchmark/results) with results of models on your benchmark.
3. [optional] When your PR with benchmarks results is merged, you can add your benchmark to the most fitting section in [benchmark_selector.py](https://github.com/embeddings-benchmark/mteb/blob/main/mteb/leaderboard/benchmark_selector.py) to be shown on the leaderboard.
4. When PRs are merged, your benchmark will be added to the leaderboard automatically after the next workflow trigger (every day at midnight Pacific Time (8 AM UTC)).
