from __future__ import annotations

from pathlib import Path

from huggingface_hub import EvalResult, ModelCard, ModelCardData, repo_exists

from mteb.abstasks.AbsTask import AbsTask
from mteb.cache import ResultCache


def generate_model_card(
    model_name: str,
    tasks: list[AbsTask] | None = None,
    existing_model_card_id_or_path: str | Path | None = None,
    results_cache: ResultCache = ResultCache(),
    output_path: Path = Path("model_card.md"),
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
        existing_model_card_data.eval_results = eval_results
    else:

        def _eval_result_key(result: EvalResult) -> str:
            return result.dataset_name + result.dataset_revision + result.dataset_config

        unique_eval_results = {
            _eval_result_key(eval_result): eval_result
            for eval_result in existing_model_card_data.eval_results + eval_results
        }

        existing_model_card_data.eval_results = list(unique_eval_results.values())

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

    if push_to_hub and repo_exists(existing_model_card_id_or_path):
        existing_model_card.push_to_hub(existing_model_card_id_or_path, token=token)
    existing_model_card.save(output_path)
