import logging
import warnings
from collections.abc import Sequence
from pathlib import Path

from huggingface_hub import ModelCard, ModelCardData, repo_exists

from mteb import BenchmarkResults
from mteb.abstasks.abstask import AbsTask
from mteb.benchmarks.benchmark import Benchmark
from mteb.cache import ResultCache

logger = logging.getLogger(__name__)


def generate_model_card(
    model_name: str,
    tasks: Sequence[AbsTask] | None = None,
    benchmark: Benchmark | list[Benchmark] | None = None,
    existing_model_card_id_or_path: str | Path | None = None,
    results_cache: ResultCache = ResultCache(),
    output_path: Path = Path("model_card.md"),
    add_table_to_model_card: bool = False,
    models_to_compare: list[str] | None = None,
    token: str | None = None,
    push_to_hub: bool = False,
) -> None:
    """Generate or update a model card with evaluation results from MTEB.

    Args:
        model_name: Name of the model.
        tasks: List of tasks to generate results for.
        benchmark: A Benchmark or list of benchmarks to generate results for.
        existing_model_card_id_or_path: Path or ID of an existing model card to update.
        results_cache: Instance of ResultCache to load results from.
        output_path: Path to save the generated model card.
        models_to_compare: List of models to add to results table.
        add_table_to_model_card: Whether to add a results table to the model card. There will be a table with model
         (with `models_to_compare` if provided) and tasks.
        token: Optional token for pushing to Hugging Face Hub.
        push_to_hub: Whether to push the updated model card to the Hub if it exists there.
    """
    existing_model_card: ModelCard | None = None
    if existing_model_card_id_or_path:
        existing_model_card = ModelCard.load(existing_model_card_id_or_path)

    all_tasks = []
    benchmark_list: list[Benchmark] = []
    if tasks is not None:
        all_tasks.extend(tasks)

    if benchmark is not None:
        if isinstance(benchmark, Benchmark):
            benchmark_list = [benchmark]
        else:
            benchmark_list = benchmark

        for b in benchmark_list:
            all_tasks.extend(b.tasks)

    benchmark_results = results_cache.load_results(
        [model_name], all_tasks if all_tasks else None, only_main_score=True
    )
    eval_results = []
    for models_results in benchmark_results.model_results:
        for task_result in models_results.task_results:
            eval_results.extend(task_result.get_hf_eval_results())

    existing_model_card_data: ModelCardData = (
        existing_model_card.data if existing_model_card else ModelCardData()  # type: ignore[assignment]
    )

    if existing_model_card_data.eval_results is None:
        existing_model_card_data.model_name = (
            existing_model_card_data.model_name or model_name
        )
    else:
        unique_eval_results = {
            eval_result.unique_identifier: eval_result
            for eval_result in existing_model_card_data.eval_results + eval_results
        }
        eval_results = list(unique_eval_results.values())

    existing_model_card_data.eval_results = sorted(
        eval_results, key=lambda x: x.unique_identifier
    )

    if existing_model_card_data.tags is None:
        existing_model_card_data.tags = ["mteb"]
    else:
        existing_model_card_data.tags.append("mteb")

    if existing_model_card:
        existing_model_card.data = existing_model_card_data
    else:
        existing_model_card = ModelCard.from_template(
            card_data=existing_model_card_data
        )

    if models_to_compare:
        benchmark_results = results_cache.load_results(
            [model_name, *models_to_compare],
            all_tasks if all_tasks else None,
            only_main_score=True,
        )

    if add_table_to_model_card:
        existing_model_card = _add_table_to_model_card(
            benchmark_results, existing_model_card, benchmark_list
        )

    if push_to_hub and existing_model_card_id_or_path:
        existing_model_card_id_or_path = str(existing_model_card_id_or_path)
        if repo_exists(existing_model_card_id_or_path):
            existing_model_card.push_to_hub(existing_model_card_id_or_path, token=token)
        else:
            msg = f"Repository {existing_model_card_id_or_path} does not exist on the Hub. Skipping push to hub."
            logger.warning(msg)
            warnings.warn(msg)
    existing_model_card.save(output_path)


def _add_table_to_model_card(
    results: BenchmarkResults,
    model_card: ModelCard,
    benchmark_list: list[Benchmark],
) -> ModelCard:
    original_content = model_card.content
    mteb_content = "# MTEB Results\n\n"

    task_to_benchmark = {}
    for b in benchmark_list:
        for task in b.tasks:
            task_to_benchmark[task.metadata.name] = b.name

    for b in benchmark_list:
        try:
            # Filter results to only include tasks from this benchmark
            benchmark_tasks = list(b.tasks)
            filtered_results = results.select_tasks(benchmark_tasks)

            if filtered_results.model_results:
                results_df = filtered_results.get_benchmark_result()

                mteb_content += f"## {b.name}\n\n"
                mteb_content += results_df.to_markdown()
                mteb_content += "\n\n"
        except Exception as e:
            logger.warning(f"Could not create table for benchmark {b.name}: {e}")
            continue
    model_card.content = original_content + "\n\n" + mteb_content
    return model_card
