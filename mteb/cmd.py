# entry point for the library
# example call:
#   mteb --model average_word_embeddings_komninos --tasks Banking77Classification EmotionClassification  --k 5 --device 0 --batch_size 32 --seed 42 --output_dir /tmp/mteb_output --n_splits 5 --samples_per_label 8 --verbose 3

import argparse
import logging
from sentence_transformers import SentenceTransformer
from mteb import MTEB
from mteb.tasks.kNNClassification import MassiveIntentClassification

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
    parser.add_argument("--tasks", nargs="+", type=str, default=None, help="Tasks to run. Use task name from #TODO")
    parser.add_argument("--k", type=int, default=5, help="Number of nearest neighbors to use for classification")
    parser.add_argument("--device", type=int, default=0, help="Device to use for computation")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for computation")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for computation")
    parser.add_argument("--output_dir", type=str, default="mteb_output", help="Output directory for results")
    parser.add_argument("--n_splits", type=int, default=5, help="Number of splits for bootstrapping")
    parser.add_argument(
        "--samples_per_label", type=int, default=8, help="Number of samples per label for bootstrapping"
    )
    parser.add_argument("--verbose", type=int, default=1, help="Verbosity level")
    args = parser.parse_args()

    logger.info("Running with parameters: %s", args)

    model = SentenceTransformer(args.model)
    eval = MTEB(task_list=args.tasks)
    eval.run(model)


if __name__ == "__main__":
    main()
