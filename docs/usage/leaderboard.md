# Running the Leaderboard

This section contains information on how to interact with the leaderboard including running it locally, analysing the results, annotating contamination and more.

### Running the Leaderboard Locally

It is possible to completely deploy the leaderboard locally or self-host it. This can e.g. be relevant for companies that might want to
integrate build their own benchmarks or integrate custom tasks into existing benchmarks.

Running the leaderboard is quite easy. Simply run:
```bash
make run-leaderboard
```

The leaderboard requires gradio install, which can be installed using `pip install mteb[leaderboard]` and requires python >3.10.

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
