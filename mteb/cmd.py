"""entry point for the library

example call:
  pip install git+https://github.com/embeddings-benchmark/mteb-draft.git
  mteb -m average_word_embeddings_komninos \
       -t Banking77Classification EmotionClassification \
       --output_folder mteb_output \
       --verbosity 3
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

from sentence_transformers import SentenceTransformer

import mteb
from mteb import MTEB

logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)


def name_to_path(name: str) -> str:
    return name.replace("/", "__").replace(" ", "_")


def _save_model_metadata(
    model: SentenceTransformer, model_name: str, output_folder: Path
) -> None:
    save_path = output_folder / "model_meta.json"

    model_meta = {
        "model_name": model_name,
        "versions": model._model_config.get("__version__", None),
    }

    with save_path.open("w") as f:
        json.dump(model_meta, f)


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "-m",
        "--model",
        type=str,
        default=None,
        help="Model to use. Use pre-trained model name from https://huggingface.co/models",
    )
    parser.add_argument(
        "--task_types",
        nargs="+",
        type=str,
        default=None,
        help="List of task types (Clustering, Retrieval..) to be evaluated. If None, all tasks will be evaluated",
    )
    parser.add_argument(
        "--categories",
        nargs="+",
        type=str,
        default=None,
        help="List of task categories (s2s, p2p..) to be evaluated. If None, all tasks will be evaluated",
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
        help="List of languages to be evaluated. if not set, all languages will be evaluated.",
    )
    parser.add_argument(
        "--device", type=int, default=None, help="Device to use for computation"
    )
    parser.add_argument(
        "--output_folder",
        type=str,
        default=None,
        help="Output directory for results. Will default to results/{model_name} if not set.",
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

    ## evaluation params
    parser.add_argument(
        "--eval_splits",
        nargs="+",
        type=str,
        default=None,
        help="Evaluation splits to use (train, dev, test..). If None, all splits will be used",
    )

    ## display tasks
    parser.add_argument(
        "--available_tasks",
        action="store_true",
        default=False,
        help="Display the available tasks",
    )

    args = parser.parse_args()

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

    if args.available_tasks:
        MTEB.mteb_tasks()
        return

    if args.model is None:
        raise ValueError("Please specify a model using the -m or --model argument")

    if args.output_folder is None:
        args.output_folder = f"results/{name_to_path(args.model)}"

    model = SentenceTransformer(args.model, device=args.device)

    tasks = mteb.get_tasks(
        categories=args.categories,
        task_types=args.task_types,
        languages=args.languages,
        tasks=args.tasks,
    )
    eval = MTEB(tasks=tasks)

    eval.run(
        model,
        verbosity=args.verbosity,
        output_folder=args.output_folder,
        eval_splits=args.eval_splits,
        co2_tracker=args.co2_tracker,
    )

    _save_model_metadata(model, args.model, Path(args.output_folder))


if __name__ == "__main__":
    main()
