# Running the Leaderboard

This section contains information on how to interact with the leaderboard including running it locally, analysing the results, annotating contamination and more.

### Running the Leaderboard Locally

It is possible to completely deploy the leaderboard locally or self-host it. This can e.g. be relevant for companies that might want to
integrate build their own benchmarks or integrate custom tasks into existing benchmarks.

The leaderboard can be run in two ways:

#### Using the CLI Command

The easiest way to run the leaderboard is using the MTEB CLI:

```bash
mteb leaderboard
```

You can also specify a custom cache path for model results:

```bash
mteb leaderboard --cache-path results
```

Additional options:
- `--host HOST`: Specify the host to run the server on (default: 0.0.0.0)
- `--port PORT`: Specify the port to run the server on (default: 7860)
- `--share`: Create a public URL for the leaderboard
- `--rebuild`: Force rebuild from full results repository, bypassing cached JSON. This is useful when you suspect the cache is stale or corrupted.

Example with all options:
```bash
mteb leaderboard --cache-path results --port 8080 --share --rebuild
```

#### Cache Directory Structure

The leaderboard stores data in the cache directory (default: `~/.cache/mteb` or set via `MTEB_CACHE` environment variable):

- `results/` - Your locally run evaluation results
- `remote/` - Cloned results from the [results repository](https://github.com/embeddings-benchmark/results)
- `leaderboard/` - Pre-computed leaderboard cache for fast loading

#### Using Make Command

Alternatively, you can use the Makefile:
```bash
make run-leaderboard
```

The leaderboard requires gradio install, which can be installed using `pip install mteb[leaderboard]` or `uv add "mteb[leaderboard]"` and requires python >3.10.

### Annotate Contamination

have your found contamination in the training data of a model? Please let us know, either by opening an [issue](https://github.com/embeddings-benchmark/mteb/issues) or ideally by submitting a PR annotating the training datasets of the model:

```python
model_w_contamination = ModelMeta(
    name = "model-with-contamination"
    ...
    training_datasets = {"ArguAna", ...} # name of dataset within MTEB
    ...
)
```
