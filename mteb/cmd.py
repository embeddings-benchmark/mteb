"""Entry point for the library.

example call:
  pip install git+https://github.com/embeddings-benchmark/mteb-draft.git
  mteb -m average_word_embeddings_komninos \
       -t Banking77Classification EmotionClassification \
       --output_folder mteb_output \
       --verbosity 3.
"""

from __future__ import annotations

import argparse
import datetime
import json
import logging
from pathlib import Path

from sentence_transformers import SentenceTransformer

from mteb import MTEB

logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)


def _name_to_path(name: str) -> str:
    return name.replace("/", "__").replace(" ", "_")


def _save_model_metadata(
    model: SentenceTransformer, model_name: str, output_folder: Path
) -> None:
    save_path = output_folder / "model_meta.json"

    model_meta = {
        "model_name": model_name,
        "time_of_run": str(datetime.datetime.today()),
        "versions": model._model_config["__version__"],
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
        "--task_categories",
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
        "--task-langs",
        nargs="*",
        type=str,
        default=None,
        help="List of languages to be evaluated. if not set, all languages will be evaluated.",
    )
    parser.add_argument(
        "--device", type=int, default=None, help="Device to use for computation"
    )
    parser.add_argument(
        "--batch_size", type=int, default=32, help="Batch size for computation"
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="Random seed for computation"
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

    ## evaluation params
    parser.add_argument(
        "--eval_splits",
        nargs="+",
        type=str,
        default=None,
        help="Evaluation splits to use (train, dev, test..). If None, all splits will be used",
    )

    ## classification params
    parser.add_argument(
        "--k",
        type=int,
        default=None,
        help="Number of nearest neighbors to use for classification",
    )
    parser.add_argument(
        "--n_experiments",
        type=int,
        default=None,
        help="Number of splits for bootstrapping",
    )
    parser.add_argument(
        "--samples_per_label",
        type=int,
        default=None,
        help="Number of samples per label for bootstrapping",
    )

    ## retrieval params
    parser.add_argument(
        "--corpus_chunk_size",
        type=int,
        default=None,
        help="Number of sentences to use for each corpus chunk. If None, a convenient number is suggested",
    )

    ## display tasks
    parser.add_argument(
        "--available_tasks",
        action="store_true",
        default=False,
        help="Display the available tasks",
    )

    # TODO: check what prams are useful to add
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
        args.output_folder = f"results/{_name_to_path(args.model)}"

    model = SentenceTransformer(args.model, device=args.device)
    eval = MTEB(
        task_categories=args.task_categories,
        task_types=args.task_types,
        task_langs=args.task_langs,
        tasks=args.tasks,
    )

    eval.run(
        model,
        verbosity=args.verbosity,
        output_folder=args.output_folder,
        eval_splits=args.eval_splits,
    )

    _save_model_metadata(model, args.model, Path(args.output_folder))


if __name__ == "__main__":
    main()
