from argparse import ArgumentParser, Namespace
import logging
from sentence_transformers import SentenceTransformer
from mteb import MTEB

logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)


class EvalCommand():
    @classmethod
    def register_subcommand(cls, parser: ArgumentParser):
        eval_parser = parser.add_parser("eval")

        eval_parser.add_argument(
            "--model",
            type=str,
            required=True,
            help="Model to use. Use pre-trained model name from https://huggingface.co/models",
        )
        eval_parser.add_argument(
            "--task_types",
            nargs="+",
            type=str,
            default=None,
            help="List of task types (Clustering, Retrieval..) to be evaluated. If None, all tasks will be evaluated",
        )
        eval_parser.add_argument(
            "--task_categories",
            nargs="+",
            type=str,
            default=None,
            help="List of task categories (s2s, p2p..) to be evaluated. If None, all tasks will be evaluated",
        )
        eval_parser.add_argument(
            "-t",
            "--tasks",
            nargs="+",
            type=str,
            default=None,
            help="List of tasks to be evaluated. If specified, the other arguments are ignored.",
        )
        eval_parser.add_argument("--device", type=int, default=None, help="Device to use for computation")
        eval_parser.add_argument("--batch_size", type=int, default=32, help="Batch size for computation")
        eval_parser.add_argument("--seed", type=int, default=42, help="Random seed for computation")
        eval_parser.add_argument("--output_folder", type=str, default="results/res", help="Output directory for results")
        eval_parser.add_argument(
            "--method",
            type=str,
            default=None,
            help="Method to use for evaluation. Can be 'kNN' or 'logReg'",
        )
        eval_parser.add_argument("--k", type=int, default=None, help="Number of nearest neighbors to use for classification")
        eval_parser.add_argument("--n_splits", type=int, default=None, help="Number of splits for bootstrapping")
        eval_parser.add_argument(
            "--samples_per_label", type=int, default=None, help="Number of samples per label for bootstrapping"
        )
        eval_parser.add_argument("-v", "--verbosity", type=int, default=1, help="Verbosity level")

        eval_parser.set_defaults(func=lambda args: cls(args))

    def __init__(self, args):
        self.args = args

    def run(self):
        logger.info("Running with parameters: %s", self.args)

        model = SentenceTransformer(self.args.model, device=self.args.device)
        eval = MTEB(**vars(self.args))
        eval.run(model, verbosity=self.args.verbosity, output_folder=self.args.output_folder)

