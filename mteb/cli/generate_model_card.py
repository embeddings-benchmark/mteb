from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from mteb.cache import ResultCache
from mteb.models.model_meta import ModelMeta

if TYPE_CHECKING:
    from collections.abc import Sequence

    from mteb.abstasks.abstask import AbsTask
    from mteb.benchmarks.benchmark import Benchmark


def generate_model_card(
    model_name: str,
    tasks: Sequence[AbsTask] | None = None,
    benchmarks: Sequence[Benchmark] | None = None,
    existing_model_card_id_or_path: str | Path | None = None,
    results_cache: ResultCache | None = None,
    output_path: Path = Path("model_card.md"),
    add_table_to_model_card: bool = False,
    models_to_compare: Sequence[str] | None = None,
    token: str | None = None,
    push_to_hub: bool = False,
) -> None:
    """Generate or update a model card with evaluation results from MTEB.

    Args:
        model_name: Name of the model.
        tasks: List of tasks to generate results for.
        benchmarks: A Benchmark or list of benchmarks to generate results for.
        existing_model_card_id_or_path: Path or ID of an existing model card to update.
        results_cache: Instance of ResultCache to load results from.
        output_path: Path to save the generated model card.
        models_to_compare: List of models to add to results table.
        add_table_to_model_card: Whether to add a results table to the model card.
        token: Optional token for pushing to Hugging Face Hub.
        push_to_hub: Whether to push the updated model card to the Hub if it exists there.
    """
    if results_cache is None:
        results_cache = ResultCache()
    meta = ModelMeta.create_empty(overwrites={"name": model_name})
    meta.generate_model_card(
        tasks=tasks,
        benchmarks=benchmarks,
        existing_model_card_id_or_path=existing_model_card_id_or_path,
        results_cache=results_cache,
        output_path=output_path,
        add_table_to_model_card=add_table_to_model_card,
        models_to_compare=models_to_compare,
        token=token,
        push_to_hub=push_to_hub,
    )
