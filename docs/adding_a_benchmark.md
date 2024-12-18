## Adding a new Benchmark 

MTEB covers a wide variety of benchmarks that are all presented in the public [leaderboard](https://huggingface.co/spaces/mteb/leaderboard). However, many languages or domains are still missing, and we welcome contributions.

To add a new benchmark, you will need to:

1) [Implement the tasks](adding_a_dataset.md) that you want to include in the benchmark, or find them in the existing list of tasks.
2) Implement the benchmark in the [`benchmark.py`](https://github.com/embeddings-benchmark/mteb/blob/main/mteb/benchmarks/benchmarks.py) file and submit your changes as a single PR.

This is easy to do 
```python
tasks = mteb.get_tasks(tasks=[] ...) # fetch the tasks you want to include in your benchmark

MY_BENCHMARK = Benchmark(
    name="Name of your benchmark",
    tasks=tasks,
    description="This benchmark tests y, which is important because of X",
    reference="https://relevant_link_eg_to_paper.com",
    citation="A bibtex citation if relevant",
)
```

3) Run a representative set of models on benchmark. To submit the results: 
<!-- TODO: we should probably create seperate page for how to submit results -->
1. Open a PR on the result [repository](https://github.com/embeddings-benchmark/results) with:
- All results added in existing model folders or new folders
- Updated paths.json (see snippet results.py)
<!-- TODO: ^check if this is still required. If so, we should probably update it. If not, we should remove it once the new leaderboard is live -->
- If any new models are added, add their names to `results.py`
- If you have access to all models you are adding, you can also [add results via the metadata](https://github.com/embeddings-benchmark/mteb/blob/main/docs/adding_a_model.md) for all of them / some of them
1. Open a PR at https://huggingface.co/spaces/mteb/leaderboard modifying app.py to add your tab:
- Add any new models & their specs to the global lists
- Add your tab, credits etc to where the other tabs are defined
- If you're adding new results to existing models, remove those models from `EXTERNAL_MODEL_RESULTS.json` such that they can be reloaded with the new results and are not cached.
- You may also have to uncomment `, download_mode='force_redownload', verification_mode="no_checks")` where the datasets are loaded to experiment locally without caching of results
- Test that it runs & works locally as you desire with python app.py, **please add screenshots to the PR**

1) Wait for the automatic update

Once the review from (3) is done the benchmark should appear on the leaderboard once it automatically updated (might take a day).