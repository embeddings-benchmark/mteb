import json
from pathlib import Path
from typing import Literal

import mteb

# Model sizes in million parameters
MIN_MODEL_SIZE, MAX_MODEL_SIZE = 0, 100_000


def filter_models(
    model_names: list[str],
    task_select: list[str],
    availability: bool | None,
    compatibility: list[str],
    instructions: bool | None,
    model_size: tuple[int | None, int | None],
    zero_shot_setting: Literal["hard", "soft", "off"],
):
    lower, upper = model_size
    # Setting to None, when the user doesn't specify anything
    if (lower == MIN_MODEL_SIZE) or (lower is None):
        lower = None
    else:
        # Multiplying by millions
        lower = lower * 1e6
    if (upper == MAX_MODEL_SIZE) or (upper is None):
        upper = None
    else:
        upper = upper * 1e6
    model_metas = mteb.get_model_metas(
        model_names=model_names,
        open_weights=availability,
        use_instructions=instructions,
        frameworks=compatibility,
        n_parameters_range=(lower, upper),
    )
    models_to_keep = set()
    for model_meta in model_metas:
        is_model_zero_shot = model_meta.is_zero_shot_on(task_select)
        if is_model_zero_shot is None:
            if zero_shot_setting == "hard":
                continue
        elif not is_model_zero_shot:
            if zero_shot_setting != "off":
                continue
        models_to_keep.add(model_meta.name)
    return list(models_to_keep)


ALL_MODELS = {meta.name for meta in mteb.get_model_metas()}


def load_results():
    results_cache_path = Path(__file__).parent.joinpath("__cached_results.json")
    if not results_cache_path.exists():
        all_results = mteb.load_results(
            only_main_score=True, require_model_meta=False, models=ALL_MODELS
        ).filter_models()
        all_results.to_disk(results_cache_path)
        return all_results
    else:
        with results_cache_path.open() as cache_file:
            return mteb.BenchmarkResults.from_validated(**json.load(cache_file))
