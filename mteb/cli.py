"""Command line interface for various MTEB.

MTEB is a benchmark for evaluating the quality of embeddings in various tasks. It supports the following commands:

- mteb run: Runs a model on a set of tasks
- mteb available_tasks: Lists the available tasks within MTEB
- mteb create_meta: Creates the metadata for a model card from a folder of results

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
import warnings
from pathlib import Path

import torch

import mteb
from mteb.create_meta import generate_readme

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

    eval.run(
        model,
        verbosity=args.verbosity,
        output_folder=args.output_folder,
        eval_splits=args.eval_splits,
        co2_tracker=args.co2_tracker,
        overwrite_results=args.overwrite,
        encode_kwargs=encode_kwargs,
        save_predictions=save_predictions,
    )

    _save_model_metadata(model, Path(args.output_folder))


def available_benchmarks(args: argparse.Namespace) -> None:
    benchmarks = mteb.get_benchmarks(names=args.benchmarks)
    eval = mteb.MTEB(tasks=benchmarks)
    eval.mteb_benchmarks()


def available_tasks(args: argparse.Namespace) -> None:
    tasks = mteb.get_tasks(
        categories=args.categories,
        task_types=args.task_types,
        languages=args.languages,
        tasks=args.tasks,
    )
    eval = mteb.MTEB(tasks=tasks)
    eval.mteb_tasks()


def add_benchmark_selection_args(parser: argparse.ArgumentParser) -> None:
    """Adds arguments to the parser for filtering benchmarks by name."""
    parser.add_argument(
        "-b",
        "--benchmarks",
        nargs="+",
        type=str,
        default=None,
        help="List of benchmark to be evaluated.",
    )


def add_task_selection_args(parser: argparse.ArgumentParser) -> None:
    """Adds arguments to the parser for filtering tasks by type, category, language, and task name."""
    parser.add_argument(
        "--task_types",
        nargs="+",
        type=str,
        default=None,
        help="List of task types (Clustering, Retrieval..) to be evaluated. If None, the filter is not applied",
    )
    parser.add_argument(
        "--categories",
        nargs="+",
        type=str,
        default=None,
        help="List of task categories (s2s, p2p..) to be evaluated. If None the filter is not applied",
    )
    parser.add_argument(
        "-t",
        "--tasks",
        nargs="+",
        type=str,
        default=None,
        help="List of tasks to be evaluated. If specified, the other arguments are ignored.",
    )
    parser.add_argument(
        "-l",
        "--languages",
        nargs="*",
        type=str,
        default=None,
        help="List of languages to be evaluated. if not set, all languages will be evaluated. Specified as ISO 639-3 codes (e.g. eng, deu, fra).",
    )


def add_available_tasks_parser(subparsers) -> None:
    parser = subparsers.add_parser(
        "available_tasks", help="List the available tasks within MTEB"
    )
    add_task_selection_args(parser)

    parser.set_defaults(func=available_tasks)


def add_available_benchmarks_parser(subparsers) -> None:
    parser = subparsers.add_parser(
        "available_benchmarks", help="List the available benchmarks within MTEB"
    )
    add_benchmark_selection_args(parser)

    parser.set_defaults(func=available_benchmarks)


def add_run_parser(subparsers) -> None:
    parser = subparsers.add_parser("run", help="Run a model on a set of tasks")

    parser.add_argument(
        "-m",
        "--model",
        type=str,
        help="Model to use. Will priotize model implementation in MTEB's model registry, but default to loading the model using sentence-transformers.",
    )

    add_task_selection_args(parser)
    add_benchmark_selection_args(parser)

    parser.add_argument(
        "--device", type=int, default=None, help="Device to use for computation"
    )
    parser.add_argument(
        "--output_folder",
        type=str,
        default="results",
        help="Output directory for results. Will default to `results` if not set.",
    )
    parser.add_argument(
        "-v", "--verbosity", type=int, default=2, help="Verbosity level"
    )
    parser.add_argument(
        "--co2_tracker",
        type=bool,
        default=False,
        help="Enable CO₂ tracker, disabled by default",
    )
    parser.add_argument(
        "--eval_splits",
        nargs="+",
        type=str,
        default=None,
        help="Evaluation splits to use (train, dev, test..). If None, all splits will be used",
    )
    parser.add_argument(
        "--model_revision",
        type=str,
        default=None,
        help="Revision of the model to be loaded. Revisions are automatically read if the model is loaded from huggingface.",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=None,
        help="Batch size of the encode. Will be passed to the MTEB as MTEB.evaluate(model, encode_kwargs = {'batch_size': value}).",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        default=False,
        help="Overwrite the output file if it already exists",
    )
    parser.add_argument(
        "--save_predictions",
        action="store_true",
        default=False,
        help="For retrieval tasks. Saves the predictions file in output_folder.",
    )

    parser.set_defaults(func=run)


def create_meta(args: argparse.Namespace) -> None:
    results_folder = Path(args.results_folder)
    output_path = Path(args.output_path)
    overwrite = args.overwrite
    from_existing = Path(args.from_existing) if args.from_existing else None
    if output_path.exists() and overwrite:
        logger.warning("Output path already exists, overwriting.")
    elif output_path.exists():
        raise FileExistsError(
            "Output path already exists, use --overwrite to overwrite."
        )

    frontmatter = generate_readme(results_folder, from_existing)

    with output_path.open("w") as f:
        f.write(frontmatter)


def add_create_meta_parser(subparsers) -> None:
    parser = subparsers.add_parser(
        "create_meta", help="Create model metadata from a folder of results"
    )

    parser.add_argument(
        "--results_folder",
        type=str,
        help="Folder containing the results of a model run",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default="model_card.md",
        help="Output path for the model metadata",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        default=False,
        help="Overwrite the output file if it already exists",
    )
    parser.add_argument(
        "--from_existing",
        type=str,
        required=False,
        help="Merge results with existing README.md",
    )

    parser.set_defaults(func=create_meta)


def main():
    parser = argparse.ArgumentParser(description="The MTEB Command line interface.")

    subparsers = parser.add_subparsers(
        title="subcommands", description="valid subcommands", help="additional help"
    )
    add_run_parser(subparsers)
    add_available_tasks_parser(subparsers)
    add_available_benchmarks_parser(subparsers)
    add_create_meta_parser(subparsers)

    args = parser.parse_args()

    # If no subcommand is provided, default to run with a deprecation warning
    if not hasattr(args, "func"):
        warnings.warn(
            "Using `mteb` without a subcommand is deprecated. Use `mteb run` instead.",
            DeprecationWarning,
        )
        # Set default arguments for 'run' if no subcommand is provided
        default_args = parser.parse_args(
            ["run"]
            + list(map(str, args._get_args()))
            + [
                f"--{k}" if v is None else f"--{k}={v}"
                for k, v in vars(args).items()
                if k != "func"
            ]
        )
        default_args.func(default_args)
    else:
        args.func(args)


if __name__ == "__main__":
    main()
