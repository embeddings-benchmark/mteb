## Adding a benchmark

The MTEB Leaderboard is available [here](https://huggingface.co/spaces/mteb/leaderboard) and we encourage additions of new benchmarks. To add a new benchmark:

1. Add your benchmark to [benchmarks.py](../mteb/benchmarks/benchmarks/benchmarks.py) as a `Benchmark` object, and select the MTEB tasks that will be in the benchmark. If some of the tasks do not exist in MTEB, follow the ["add a dataset"](./adding_a_dataset.md) instructions to add them.
2. Add your benchmark to the most fitting section in [benchmark_selector.py](../mteb/leaderboard/benchmark_selector.py).
3. Open a PR at https://github.com/embeddings-benchmark/results with results of models on your benchmark.
4. When PRs are merged, your benchmark will be added to the leaderboard automatically after the next workflow trigger.
