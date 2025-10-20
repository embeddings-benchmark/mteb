import logging
from pathlib import Path

from huggingface_hub import ModelCard, ModelCardData, repo_exists

from mteb import BenchmarkResults
from mteb.abstasks.abstask import AbsTask
from mteb.cache import ResultCache

logger = logging.getLogger(__name__)


def generate_model_card(
    model_name: str,
    tasks: list[AbsTask] | None = None,
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

    benchmark_results = results_cache.load_results(
        [model_name], tasks, only_main_score=True
    )
    eval_results = []
    for models_results in benchmark_results.model_results:
        for task_result in models_results.task_results:
            eval_results.extend(task_result.get_hf_eval_results())

    existing_model_card_data = (
        existing_model_card.data if existing_model_card else ModelCardData()
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
            [model_name, *models_to_compare], tasks, only_main_score=True
        )

    if add_table_to_model_card:
        existing_model_card = _add_table_to_model_card(
            benchmark_results, existing_model_card
        )

    if push_to_hub:
        if repo_exists(existing_model_card_id_or_path):
            existing_model_card.push_to_hub(existing_model_card_id_or_path, token=token)
        else:
            logger.warning(
                f"Repository {existing_model_card_id_or_path} does not exist on the Hub. Skipping push to hub."
            )
    existing_model_card.save(output_path)


def _add_table_to_model_card(
    results: BenchmarkResults, model_card: ModelCard
) -> ModelCard:
    original_content = model_card.content
    results_df = results.to_dataframe()
    results_df = results_df.set_index("task_name")
    mteb_content = f"""
# MTEB results
{results_df.to_markdown()}
"""
    model_card.content = original_content + "\n\n" + mteb_content
    return model_card
