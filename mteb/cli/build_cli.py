import argparse
import logging
import os
import warnings
from pathlib import Path

import torch
from rich.logging import RichHandler

import mteb
from mteb.abstasks.abstask import AbsTask
from mteb.cache import ResultCache
from mteb.cli._display_tasks import _display_benchmarks, _display_tasks
from mteb.cli.generate_model_card import generate_model_card
from mteb.evaluate import OverwriteStrategy

logger = logging.getLogger(__name__)


def run(args: argparse.Namespace) -> None:
    """Run a model on a set of tasks."""
    # set logging based on verbosity level
    if args.verbosity == 0:
        logging.getLogger("mteb").setLevel(logging.CRITICAL)
    elif args.verbosity == 1:
        logging.getLogger("mteb").setLevel(logging.WARNING)
    elif args.verbosity == 2:
        logging.getLogger("mteb").setLevel(logging.INFO)
    elif args.verbosity == 3:
        logging.getLogger("mteb").setLevel(logging.DEBUG)

    if args.benchmarks and (
        args.tasks
        or args.eval_splits
        or args.languages
        or args.task_types
        or args.categories
    ):
        logger.warning(
            "`benchmarks` is specified but so is one or more of `tasks`, `eval_splits`, `languages`, `task_types` and `categories`. These will be ignored."
        )

    logger.debug("Setting environment variable TOKENIZERS_PARALLELISM to false")
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    logger.info("Running with parameters: %s", args)

    if args.device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device

    model = mteb.get_model(args.model, args.model_revision, device=device)

    if args.benchmarks:
        benchmarks = mteb.get_benchmarks(names=args.benchmarks)
        tasks = tuple(t for b in benchmarks for t in b.tasks)
    else:
        tasks = mteb.get_tasks(
            categories=args.categories,
            task_types=args.task_types,
            languages=args.languages,
            tasks=args.tasks,
            eval_splits=args.eval_splits,
        )

    encode_kwargs = {}
    if args.batch_size is not None:
        encode_kwargs["batch_size"] = args.batch_size

    overwrite_strategy = args.overwrite_strategy
    if args.overwrite:
        warnings.warn(
            "`--overwrite` is deprecated, please use `--overwrite-strategy 'always'` instead.",
            DeprecationWarning,
        )
        overwrite_strategy = OverwriteStrategy.ALWAYS.value

    prediction_folder = args.prediction_folder
    if args.save_predictions:
        warnings.warn(
            "`--save_predictions` is deprecated, please use `--prediction-folder` instead.",
            DeprecationWarning,
        )
        prediction_folder = args.output_folder

    mteb.evaluate(
        model,
        tasks,
        cache=ResultCache(args.output_folder),
        co2_tracker=args.co2_tracker,
        overwrite_strategy=overwrite_strategy,
        encode_kwargs=encode_kwargs,
        prediction_folder=prediction_folder,
    )


def _available_benchmarks(args: argparse.Namespace) -> None:
    benchmarks = mteb.get_benchmarks(names=args.benchmarks)
    _display_benchmarks(benchmarks)


def _available_tasks(args: argparse.Namespace) -> None:
    tasks = mteb.get_tasks(
        categories=args.categories,
        task_types=args.task_types,
        languages=args.languages,
        tasks=args.tasks,
    )
    _display_tasks(tasks)


def _add_benchmark_selection_args(parser: argparse.ArgumentParser) -> None:
    """Adds arguments to the parser for filtering benchmarks by name."""
    parser.add_argument(
        "-b",
        "--benchmarks",
        nargs="+",
        type=str,
        default=None,
        help="List of benchmark to be evaluated.",
    )


def _add_task_selection_args(parser: argparse.ArgumentParser) -> None:
    """Adds arguments to the parser for filtering tasks by type, category, language, and task name."""
    parser.add_argument(
        "--task-types",
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


def _add_available_tasks_parser(subparsers: argparse._SubParsersAction) -> None:
    parser = subparsers.add_parser(
        "available-tasks", help="List the available tasks within MTEB"
    )
    _add_task_selection_args(parser)

    parser.set_defaults(func=_available_tasks)


def _add_available_benchmarks_parser(subparsers: argparse._SubParsersAction) -> None:
    parser = subparsers.add_parser(
        "available-benchmarks", help="List the available benchmarks within MTEB"
    )
    _add_benchmark_selection_args(parser)

    parser.set_defaults(func=_available_benchmarks)


def _add_run_parser(subparsers: argparse._SubParsersAction) -> None:
    parser = subparsers.add_parser("run", help="Run a model on a set of tasks")

    parser.add_argument(
        "-m",
        "--model",
        type=str,
        help="Model to use. Will prioritize model implementation in MTEB's model registry, but default to loading the model using sentence-transformers.",
    )

    _add_task_selection_args(parser)
    _add_benchmark_selection_args(parser)

    parser.add_argument(
        "--device", type=int, default=None, help="Device to use for computation."
    )
    parser.add_argument(
        "--output-folder",
        "--output_folder",  # for backward compatibility
        type=str,
        default="results",
        help="Output directory for results. Will default to `results` if not set.",
    )
    parser.add_argument(
        "-v", "--verbosity", type=int, default=2, help="Verbosity level"
    )
    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        "--co2-tracker",
        dest="co2_tracker",
        action="store_true",
        help="Enable CO₂ tracker. If no argument is passed, the CO₂ tracker will be run if installed.",
    )
    group.add_argument(
        "--no-co2-tracker",
        dest="co2_tracker",
        action="store_false",
        help="Disable CO₂ tracker. If no argument is passed, the CO₂ tracker will be run if installed.",
    )
    parser.set_defaults(co2_tracker=None)

    parser.add_argument(
        "--eval-splits",
        "--eval_splits",  # for backward compatibility
        nargs="+",
        type=str,
        default=None,
        help="Evaluation splits to use (train, dev, test..). If None, all splits will be used.",
    )
    parser.add_argument(
        "--model-revision",
        "--model_revision",  # for backward compatibility
        type=str,
        default=None,
        help="Revision of the model to be loaded. Revisions are automatically read if the model is loaded from huggingface.",
    )
    parser.add_argument(
        "--batch-size",
        "--batch_size",  # for backward compatibility
        type=int,
        default=None,
        help="Batch size of the encode. Will be passed to the MTEB as `mteb.evaluate(model, task, encode_kwargs={'batch_size': value})`.",
    )
    # for backward compatibility
    parser.add_argument(
        "--overwrite",
        action="store_true",
        default=False,
        help="Whether to overwrite existing results. Deprecated, use `--overwrite-strategy 'always'` instead.",
    )
    parser.add_argument(
        "--overwrite-strategy",
        type=str,
        default="only-missing",
        choices=[val.value for val in OverwriteStrategy],
        help=(
            "Strategy for when to overwrite. Can be 'always', 'never', 'only-missing'. 'only-missing' will only rerun the missing splits of a task."
            + " It will not rerun the splits if the dataset revision or mteb version has changed."
        ),
    )
    parser.add_argument(
        "--save_predictions",
        action="store_true",
        default=False,
        help="Saves the predictions file in output_folder. Deprecated, use `--prediction-folder` instead.",
    )
    parser.add_argument(
        "--prediction-folder",
        type=str,
        default=None,
        help="Folder to save the model predictions in. If None, predictions will not be saved.",
    )

    parser.set_defaults(func=run)


def _create_meta(args: argparse.Namespace) -> None:
    model_name = args.model_name
    tasks_names = args.tasks
    benchmarks = args.benchmarks
    results_folder = Path(args.results_folder) if args.results_folder else None
    output_path = Path(args.output_path)
    overwrite = args.overwrite
    from_existing = args.from_existing if args.from_existing else None
    if from_existing is not None and from_existing.endswith(".md"):
        from_existing = Path(from_existing)

    if output_path.exists() and overwrite:
        msg = "Output path already exists, overwriting."
        logger.warning(msg)
        warnings.warn(msg)
    elif output_path.exists():
        raise FileExistsError(
            "Output path already exists, use --overwrite to overwrite."
        )

    benchmarks = None
    tasks: list[AbsTask] = []
    if tasks_names is not None:
        tasks = list(mteb.get_tasks(tasks_names))
    if benchmarks is not None:
        benchmarks = mteb.get_benchmarks(benchmarks)

    generate_model_card(
        model_name,
        tasks,
        benchmarks,
        existing_model_card_id_or_path=from_existing,
        results_cache=ResultCache(results_folder),
        output_path=output_path,
    )


def _add_create_meta_parser(subparsers) -> None:
    parser = subparsers.add_parser("create-model-results", help="Create model results")

    parser.add_argument(
        "--model-name",
        type=str,
        help="Name of the model",
    )
    parser.add_argument(
        "--tasks",
        type=str,
        nargs="+",
        help="Name of the tasks to use. By default, all tasks results will be used.",
        required=False,
        default=None,
    )
    parser.add_argument(
        "--benchmarks",
        type=str,
        nargs="+",
        help="Name of the benchmarks to use",
        required=False,
        default=None,
    )
    parser.add_argument(
        "--results-folder",
        type=str,
        help="Folder containing the results of a model run",
        required=False,
        default=None,
    )
    parser.add_argument(
        "--output-path",
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
        "--from-existing",
        type=str,
        required=False,
        help="Merge results with existing README.md. Can be path to file or Huggingface model_id",
        default=None,
    )

    parser.set_defaults(func=_create_meta)


def _add_leaderboard_parser(subparsers) -> None:
    parser = subparsers.add_parser("leaderboard", help="Launch the MTEB leaderboard")

    parser.add_argument(
        "--cache-path",
        type=str,
        help="Path to the cache folder containing model results",
        required=False,
        default=None,
    )
    parser.add_argument(
        "--host",
        type=str,
        default="0.0.0.0",
        help="Host to run the leaderboard server on",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=7860,
        help="Port to run the leaderboard server on",
    )
    parser.add_argument(
        "--share",
        action="store_true",
        default=False,
        help="Create a public URL for the leaderboard",
    )

    parser.set_defaults(func=_leaderboard)


def _leaderboard(args: argparse.Namespace) -> None:
    """Launch the MTEB leaderboard with specified cache path."""
    # Import leaderboard module only when needed to avoid requiring leaderboard dependencies
    # for other CLI commands
    try:
        import gradio as gr

        from mteb.leaderboard import get_leaderboard_app
    except ImportError as e:
        raise ImportError(
            "Seems like some dependencies are not installed. "
            + "You can likely install these using: `pip install mteb[leaderboard]`. "
            + f"{e}"
        )

    cache_path = args.cache_path

    if cache_path:
        logger.info(f"Using cache path: {cache_path}")
        cache = ResultCache(cache_path)
    else:
        cache = ResultCache()
        logger.info(f"Using default cache path: {cache.cache_path}")

    app = get_leaderboard_app(cache)

    logger.info(f"Starting leaderboard on {args.host}:{args.port}")
    if args.share:
        logger.info("Creating public URL...")

    logging.getLogger("mteb.load_results.task_results").setLevel(
        logging.ERROR
    )  # Warnings related to task split
    logging.getLogger("mteb.model_meta").setLevel(
        logging.ERROR
    )  # Warning related to model metadata (fetch_from_hf=False)
    logging.getLogger("mteb.load_results.benchmark_results").setLevel(
        logging.ERROR
    )  # Warning related to model metadata (fetch_from_hf=False)
    warnings.filterwarnings("ignore", message="Couldn't get scores for .* due to .*")

    # Head content for Tailwind CSS
    head = """
    <link href="https://cdn.jsdelivr.net/npm/tailwindcss@2.2.19/dist/tailwind.min.css" rel="stylesheet">
    """

    app.launch(
        server_name=args.host,
        server_port=args.port,
        share=args.share,
        theme=gr.themes.Soft(
            font=[gr.themes.GoogleFont("Roboto Mono"), "Arial", "sans-serif"],
        ),
        head=head,
    )


def build_cli() -> argparse.ArgumentParser:
    """Builds the argument parser for the MTEB CLI.

    Returns:
        An argparse.ArgumentParser object configured with subcommands and options.
    """
    parser = argparse.ArgumentParser(
        description="MTEB Command Line Interface",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    subparsers = parser.add_subparsers(
        title="subcommands", description="valid subcommands", help="additional help"
    )

    _add_run_parser(subparsers)
    _add_available_tasks_parser(subparsers)
    _add_available_benchmarks_parser(subparsers)
    _add_create_meta_parser(subparsers)
    _add_leaderboard_parser(subparsers)

    return parser


def main() -> None:
    """Main entry point for the MTEB CLI."""
    logging.basicConfig(
        level=logging.WARNING,
        format="%(message)s",
        handlers=[RichHandler(rich_tracebacks=True)],
    )

    parser = build_cli()
    args = parser.parse_args()
    args.func(args)
