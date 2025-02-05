## Adding a benchmark

The MTEB Leaderboard is available [here](https://huggingface.co/spaces/mteb/leaderboard) and we encourage additions of new benchmarks. To add a new benchmark:

1. Add your benchmark to [benchmark.py](../mteb/benchmarks/benchmarks.py) as a `Benchmark` object, and select the MTEB tasks that will be in the benchmark. If some of the tasks do not exist in MTEB, follow the "add a dataset" instructions to add them.
2. Open a PR at https://github.com/embedding-benchmark/results with results of models on your benchmark.
3. When PRs are merged, your benchmark will be added to the leaderboard automatically after the next workflow trigger.