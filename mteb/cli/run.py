"""This is the command line interface for `mteb`.

`mteb` is a toolkit for evaluating the quality of embedding models on various benchmarks. It supports the following commands:

- `mteb run`: Runs a model on a set of tasks
- `mteb available_tasks`: Lists the available tasks within MTEB
- `mteb available_benchmarks`: Lists the available benchmarks
- `mteb create_meta`: Creates the metadata for a model card from a folder of results

In the following we outline some sample use cases, but if you want to learn more about the arguments for each command you can run:

```
mteb {command} --help
```

## Running Models on Tasks

To run a model on a set of tasks, use the `mteb run` command. For example:

```bash
mteb run -m sentence-transformers/average_word_embeddings_komninos \
         -t Banking77Classification EmotionClassification \
         --output_folder mteb_output \
          --verbosity 3
```

This will create a folder `mteb_output/{model_name}/{model_revision}` containing the results of the model on the specified tasks supplied as a json
file: "{task_name}.json".


## Listing Available Tasks

To list the available tasks within MTEB, use the `mteb available_tasks` command. For example:

```bash
mteb available_tasks # list all available tasks
mteb available_tasks --task_types Clustering # list tasks of type Clustering
```

## Listing Available Benchmarks

To list the available benchmarks within MTEB, use the `mteb available_benchmarks` command. For example:

```bash
mteb available_benchmarks # list all available benchmarks
```


## Creating Model Metadata

Once a model is run you can create the metadata for a model card from a folder of results, use the `mteb create_meta` command. For example:

```bash
mteb create_meta --results_folder mteb_output/sentence-transformers__average_word_embeddings_komninos/{revision} \
                 --output_path model_card.md
```

This will create a model card at `model_card.md` containing the metadata for the model on MTEB within the YAML frontmatter. This will make the model
discoverable on the MTEB leaderboard.

An example frontmatter for a model card is shown below:

```
---
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
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import torch

import mteb

logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)


def _save_model_metadata(model: mteb.Encoder, output_folder: Path) -> None:
    meta = model.mteb_model_meta  # type: ignore

    revision = meta.revision if meta.revision is not None else "no_revision_available"

    save_path = output_folder / meta.model_name_as_path() / revision / "model_meta.json"

    with save_path.open("w") as f:
        json.dump(meta.to_dict(), f)


def run(args: argparse.Namespace) -> None:
    # set logging based on verbosity level
    if args.verbosity == 0:
        logging.getLogger("mteb").setLevel(logging.CRITICAL)
    elif args.verbosity == 1:
        logging.getLogger("mteb").setLevel(logging.WARNING)
    elif args.verbosity == 2:
        logging.getLogger("mteb").setLevel(logging.INFO)
    elif args.verbosity == 3:
        logging.getLogger("mteb").setLevel(logging.DEBUG)

    logger.info("Running with parameters: %s", args)

    if args.device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device

    model = mteb.get_model(args.model, args.model_revision, device=device)

    if args.benchmarks:
        tasks = mteb.get_benchmarks(names=args.benchmarks)
    else:
        tasks = mteb.get_tasks(
            categories=args.categories,
            task_types=args.task_types,
            languages=args.languages,
            tasks=args.tasks,
        )

    eval = mteb.MTEB(tasks=tasks)

    encode_kwargs = {}
    if args.batch_size is not None:
        encode_kwargs["batch_size"] = args.batch_size

    save_predictions = (
        args.save_predictions if hasattr(args, "save_predictions") else False
    )

    enable_co2_tracker = not args.disable_co2_tracker

    eval.run(
        model,
        verbosity=args.verbosity,
        output_folder=args.output_folder,
        eval_splits=args.eval_splits,
        co2_tracker=enable_co2_tracker,
        overwrite_results=args.overwrite,
        encode_kwargs=encode_kwargs,
        save_predictions=save_predictions,
    )

    _save_model_metadata(model, Path(args.output_folder))
