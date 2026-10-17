---
title: "Command Line"
icon: lucide/terminal
---

# Command Line

This described the is the command line interface for `mteb`.

`mteb` is a toolkit for evaluating the quality of embedding models on various benchmarks. It supports the following commands:

- [`mteb run`](#running-models-on-tasks): Runs a model on a set of tasks
- [`mteb available_tasks`](#listing-available-tasks): Lists the available tasks within MTEB
- [`mteb available_benchmarks`](#listing-available-benchmarks): Lists the available benchmarks
- [`mteb submit-results`](#submitting-results): Submits results to the results repository
- [`mteb create_meta`](#creating-model-metadata): Creates the metadata for a model card from a folder of results
- [`mteb leaderboard`](#running-the-leaderboard): Runs the MTEB leaderboard locally
- [`mteb mock-run`](#checking-model-implementations): Sanity checks a model implementation using mock tasks

In the following we outline some sample use cases, but if you want to learn more about the arguments for each command you can run:

```bash
mteb {command} --help
```

## Running Models on Tasks

To run a model on a set of tasks, use the `mteb run` command. For example:

```bash
mteb run -m sentence-transformers/average_word_embeddings_komninos \
         -t Banking77Classification EmotionClassification \
         --output-folder mteb_output
```

This will create a folder `mteb_output/{model_name}/{model_revision}` containing the results of the model on the specified tasks supplied as a json
file; `{task_name}.json`.

### Running models served by an OpenAI-compatible API

Embedding models served by an OpenAI-compatible `/v1/embeddings` endpoint (e.g. [vLLM](https://docs.vllm.ai/) or the [OpenAI API](https://platform.openai.com/)) can be evaluated with `--endpoint-url`. `--model` is the name of the served model:

```bash
# start a server: vllm serve BAAI/bge-small-en-v1.5 --runner pooling --port 8000
mteb run -m BAAI/bge-small-en-v1.5 \
         -t Banking77Classification \
         --endpoint-url http://localhost:8000 \
         --modalities text \
         --no-chat-template
```

The API key is read from the `OPENAI_API_KEY` environment variable, use `--api-key-env` to read it from a different variable. For example, to evaluate an OpenAI model:

```bash
export OPENAI_API_KEY=sk-...
mteb run -m text-embedding-3-small \
         -t Banking77Classification \
         --endpoint-url https://api.openai.com/v1 \
         --modalities text \
         --no-chat-template
```

Other options:

- `--timeout SECONDS`: Request timeout (default: 300)
- `--max-retries N`: Maximum number of attempts for a failed request (default: 3)
- `--no-verify-ssl`: Disable SSL certificate verification
- `--max-length N`: Maximum number of tokens per input. Read from the server's model metadata if not set
- `--modalities MODALITY [MODALITY ...]`: Modalities supported by the served model (default: text, image, audio and video)
- `--no-chat-template`: Send text through the plain `input` field instead of vLLM's chat embeddings API (`messages`). Required for the OpenAI API, other non-vLLM servers and vLLM models without a chat template

Reranking and token-level (ColBERT-style) models served this way are only supported through the [Python API](../advanced_usage/openai_api_wrappers.md).

## Submitting Results

To submit results to the [results repository](https://github.com/embeddings-benchmark/results), use the `mteb submit-results` command. It submits the results that the repository does not have yet, from the same folder that `mteb run` writes to (`--output-folder`, `results` by default). For example:

```bash
# run task
mteb run -m sentence-transformers/all-MiniLM-L6-v2 -t ArguAna

# submit all unsubmitted results
mteb submit-results --create-pr

# submit specific results, with a title describing the submission
mteb submit-results --create-pr -m sentence-transformers/all-MiniLM-L6-v2 -t ArguAna \
    --message "Add ArguAna results for all-MiniLM-L6-v2"
```

Without `--create-pr`, the results are committed to a new branch in a local clone of the repository, to push and open a PR manually.
`--create-pr` requires the `mteb[github]` extra and GitHub authentication, see [Submit Results](../../contributing/submitting_results.md) for the requirements.


## Listing Available Tasks

To list the available tasks within MTEB, use the `mteb available-tasks` command. For example:

```bash
mteb available-tasks # list _all_ available tasks
```

You can also use the multiple arguments for filtering:
```
mteb available-tasks --task-types Retrieval --languages eng # list all English (eng) retrieval tasks
```

## Listing Available Benchmarks

To list the available benchmarks within MTEB:

```bash
mteb available-benchmarks # list all available benchmarks
```


## Creating Model Metadata

Once a model is run you can create the metadata for a model card from a folder of results, use the `mteb create-meta` command. For example:

```bash
mteb create-meta --results-folder mteb_output/sentence-transformers__average_word_embeddings_komninos/{revision} \
                 --output-path model_card.md
```

This will create a model card at `model_card.md` containing the metadata for the model on MTEB within the YAML frontmatter. This will make the model
discoverable on the MTEB leaderboard.

## Running the Leaderboard

!!! warning
    `mteb leaderboard` launches the legacy Gradio leaderboard, which is no longer actively maintained. See the [leaderboard documentation](leaderboard.md#running-the-leaderboard-locally) for how to run the current leaderboard.

To run the legacy MTEB leaderboard locally, use the `mteb leaderboard` command. For example:

```bash
mteb leaderboard
```

You can specify a custom cache path and other options:

```bash
mteb leaderboard --cache-path results --port 8080 --share
```

Available options:
- `--cache-path PATH`: Custom path for model results cache
- `--host HOST`: Host to run the server on (default: 0.0.0.0)
- `--port PORT`: Port to run the server on (default: 7860)
- `--share`: Create a public URL for the leaderboard
- `--rebuild`: Force rebuild from full results repository, bypassing cached JSON

For more details on running the leaderboard, see the [leaderboard documentation](leaderboard.md).

## Checking Model Implementations

To sanity check a model implementation using a set of example test tasks, use the `mteb mock-run` command. This command evaluates the model on a set of tasks covering various modalities and task types, while not requiring any external datasets, making it a fast way to test that the implementation works as intended before running larger benchmarks or when implementing a model.

For example:

```bash
mteb mock-run -m sentence-transformers/average_word_embeddings_komninos
```

This will run the model on all compatible mock tasks, print a Markdown summary table to stdout, and save the markdown results to `mteb_mock_run_results.md`.

Available options:
- `-m, --model MODEL`: The model to use. Prioritizes the model implementation from MTEB's model registry, or defaults to loading via `sentence-transformers`.
- `--model-revision REVISION`: Revision of the model to load.
- `--device DEVICE`: Device to use for computation (e.g. `cpu`, `cuda`).
- `-v, --verbosity VERBOSITY`: Verbosity level (0 to 4, default: 2).

The same checks are available from Python using [`mteb.mock_run`](../../contributing/adding_a_model.md#local-model-verification-using-mock-tasks), which returns the per-task status instead of writing a file.
