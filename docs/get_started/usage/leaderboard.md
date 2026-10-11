---
title: "Run the Leaderboard"
icon: lucide/list-ordered
---


# Run the Leaderboard

This section contains information on how to interact with the leaderboard including running it locally and annotating contamination.

The public leaderboard is available at [leaderboard.mteb.org](https://leaderboard.mteb.org/).

### Running the Leaderboard Locally

It is possible to deploy the leaderboard locally or self-host it. This can e.g. be relevant to check how a new benchmark looks before submitting it, or for companies that want to build their own benchmarks or integrate custom tasks into existing benchmarks.

The leaderboard consists of two parts:

- **Backend** — a [FastAPI](https://fastapi.tiangolo.com/) service that lives in this repository under [`mteb/api`](https://github.com/embeddings-benchmark/mteb/tree/main/mteb/api). It loads the results, aggregates the scores and serves them as JSON.
- **Frontend** — a SvelteKit app in the [leaderboard-frontend](https://github.com/embeddings-benchmark/leaderboard-frontend) repository that renders the data from the backend.

#### Running the backend

Install `mteb` with the `api` extra:

=== "pip"
    ```bash
    pip install "mteb[api]"
    ```

=== "uv"
    ```bash
    uv add "mteb[api]"
    ```

Then start the service:

```bash
make serve-api
# or directly
uvicorn mteb.api.app:app --port 8000
```

The API is now available on `http://localhost:8000` (e.g. `http://localhost:8000/v1/benchmarks`). An overview of the endpoints can be found in the [API README](https://github.com/embeddings-benchmark/mteb/blob/main/mteb/api/README.md) or in the interactive docs at `http://localhost:8000/docs`.

By default, the results are downloaded from the [`mteb/results`](https://huggingface.co/datasets/mteb/results) dataset on the Hugging Face Hub and stored in the cache directory (default: `~/.cache/mteb`, can be changed via the `MTEB_CACHE` environment variable), so only the first start is slow.

The service is configured using environment variables:

| Variable       | Description                                                                                                                                                                                   |
|----------------|-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| `CACHE_REPO`   | Hugging Face dataset to load results from (default: `mteb/results`). Set it to an empty string to build the results from the local results cache instead, e.g. to include your own results. |
| `DISK_CACHE`   | `1` (default) to persist the processed results on disk between restarts, `0` to always rebuild.                                                                                               |
| `PRELOAD`      | `1` to pre-compute all benchmark summaries on startup in the background.                                                                                                                      |
| `HTTP_MAX_AGE` | `Cache-Control` max-age in seconds for JSON responses (default: 4 hours). Set it to `0` during development to always get fresh data.                                                          |
| `CORS_ORIGINS` | Comma-separated list of allowed origins (default: `*`).                                                                                                                                       |

For example, to serve the leaderboard using the results in your local cache:

```bash
CACHE_REPO="" DISK_CACHE=0 HTTP_MAX_AGE=0 uvicorn mteb.api.app:app --port 8000
```

#### Running the frontend

Clone the [leaderboard-frontend](https://github.com/embeddings-benchmark/leaderboard-frontend) repository and point it to your backend:

```bash
git clone https://github.com/embeddings-benchmark/leaderboard-frontend
cd leaderboard-frontend
make setup  # installs npm dependencies
cp .env.example .env.local  # sets PUBLIC_API_URL=http://localhost:8000
make dev  # starts the app on http://localhost:5173
```

If you only want to work on the frontend, `make dev-mock` starts it against a mock API, so no backend is needed.

See the frontend [README](https://github.com/embeddings-benchmark/leaderboard-frontend#readme) for building and deploying it.

#### Legacy Gradio leaderboard

The previous Gradio-based leaderboard is still available through the `mteb leaderboard` CLI command (requires `pip install "mteb[leaderboard]"`):

```bash
mteb leaderboard --cache-path results --port 7860
```

!!! warning
    The Gradio leaderboard is no longer actively maintained and is slow to start, as it has to load and process all results. It may not reflect all features of the current leaderboard. We recommend using the setup described above instead.

### Annotate Contamination

The leaderboard shows how zero-shot each model is on a benchmark, i.e. the percentage of the benchmark's tasks that are not part of the model's training data. This is computed from the `training_datasets` field of the model's [`ModelMeta`][mteb.models.model_meta.ModelMeta].

Have you found that a model was trained on data from an MTEB task? Please let us know, either by opening an [issue](https://github.com/embeddings-benchmark/mteb/issues) or ideally by submitting a PR that adds the dataset to the model's `training_datasets`:

```python
model_meta = ModelMeta(
    name="org/model-with-contamination",
    ...,
    training_datasets={"ArguAna", "MSMARCO"},  # names of the tasks in MTEB
    ...,
)
```

Use the task names as they appear in MTEB (e.g. `mteb.get_task("ArguAna").metadata.name`). Datasets of models listed in `adapted_from` are included automatically, so for fine-tuned models you only need to add the additional training data.
