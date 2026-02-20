# Command Line Interface

This described the is the command line interface for `mteb`.

`mteb` is a toolkit for evaluating the quality of embedding models on various benchmarks. It supports the following commands:

- [`mteb run`](#running-models-on-tasks): Runs a model on a set of tasks
- [`mteb available_tasks`](#listing-available-tasks): Lists the available tasks within MTEB
- [`mteb available_benchmarks`](#listing-available-benchmarks): Lists the available benchmarks
- [`mteb create_meta`](#creating-model-metadata): Creates the metadata for a model card from a folder of results
- [`mteb leaderboard`](#running-the-leaderboard): Runs the MTEB leaderboard locally

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

An example frontmatter for a model card is shown below:

```yaml
---

## Running the Leaderboard

To run the MTEB leaderboard locally, use the `mteb leaderboard` command. For example:

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
tags:
- mteb
model-index:
- name: SGPT-5.8B-weightedmean-msmarco-specb-bitfit
  results:
  - task:
      type: classification
    dataset:
      type: mteb/banking77
      name: MTEB Banking77
      config: default
      split: test
      revision: 44fa15921b4c889113cc5df03dd4901b49161ab7
    metrics:
    - type: accuracy
      value: 84.49350649350649
---
```
