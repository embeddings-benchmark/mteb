"""
entry point for the library
example call:
  pip install git+https://github.com/embeddings-benchmark/mteb-draft.git
  mteb -m average_word_embeddings_komninos \
       -t Banking77Classification EmotionClassification \
       --output_folder mteb_output \
       --verbosity 3
"""


import argparse
import logging

from mteb import MTEB
from sentence_transformers import SentenceTransformer


logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)


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
    parser.add_argument("--device", type=int, default=None, help="Device to use for computation")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for computation")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for computation")
    parser.add_argument("--output_folder", type=str, default="results", help="Output directory for results")
    parser.add_argument("-v", "--verbosity", type=int, default=2, help="Verbosity level")

    ## classification params
    # parser.add_argument(
    #     "--method",
    #     type=str,
    #     default=None,
    #     help="Method to use for evaluation. Can be 'kNN' or 'logReg'",
    # )
    parser.add_argument("--k", type=int, default=None, help="Number of nearest neighbors to use for classification")
    parser.add_argument("--n_splits", type=int, default=None, help="Number of splits for bootstrapping")
    parser.add_argument(
        "--samples_per_label", type=int, default=None, help="Number of samples per label for bootstrapping"
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
    del args.available_tasks

    if args.model is None:
        raise ValueError("Please specify a model using the -m or --model argument")
    # delete None values
    for key in [k for k in args.__dict__ if args.__dict__[k] is None]:
        del args.__dict__[key]

    model = SentenceTransformer(args.model, device=args.device if "device" in args else None)
    eval = MTEB(**vars(args))
    del args.model
    eval.run(model, **vars(args))


if __name__ == "__main__":
    main()
