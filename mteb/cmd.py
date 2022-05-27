# entry point for the library
# example call:
#   mteb --model average_word_embeddings_komninos --tasks Banking77Classification EmotionClassification  --k 5 --device 0 --batch_size 32 --seed 42 --output_folder /tmp/mteb_output --n_splits 5 --samples_per_label 8 --verbosity 3

import argparse
import logging
from sentence_transformers import SentenceTransformer
from mteb import MTEB

logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        type=str,
        required=True,
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
    parser.add_argument("--output_folder", type=str, default="results/res", help="Output directory for results")
    parser.add_argument(
        "--method",
        type=str,
        default=None,
        help="Method to use for evaluation. Can be 'kNN' or 'logReg'",
    )
    parser.add_argument("--k", type=int, default=None, help="Number of nearest neighbors to use for classification")
    parser.add_argument("--n_splits", type=int, default=None, help="Number of splits for bootstrapping")
    parser.add_argument(
        "--samples_per_label", type=int, default=None, help="Number of samples per label for bootstrapping"
    )
    parser.add_argument("-v", "--verbosity", type=int, default=1, help="Verbosity level")
    args = parser.parse_args()

    logger.info("Running with parameters: %s", args)

    model = SentenceTransformer(args.model, device=args.device)
    eval = MTEB(**vars(args))
    eval.run(model, verbosity=args.verbosity, output_folder=args.output_folder)


if __name__ == "__main__":
    main()
