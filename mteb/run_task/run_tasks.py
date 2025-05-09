from __future__ import annotations

import logging
from pathlib import Path
from time import time
from typing import Any, Literal, cast

from sentence_transformers import CrossEncoder, SentenceTransformer

import mteb.cache as cache
from mteb import SentenceTransformerWrapper
from mteb.abstasks.AbsTask import AbsTask
from mteb.abstasks.TaskMetadata import HFSubset, Splitname
from mteb.encoder_interface import Encoder
from mteb.load_results import TaskResult

from .get_model_meta import get_model_meta, get_output_folder

logger = logging.getLogger(__name__)


def _run_task(
    model: Encoder,
    task: AbsTask,
    *,
    splits: dict[Splitname, list[HFSubset]],
    co2_tracker: bool | None,
    encode_kwargs: dict[str, Any],
    **kwargs: Any,
) -> TaskResult:
    """The core logic to run a model on a given task. See `run_task` for more details."""
    if co2_tracker is None or co2_tracker is True:
        try:
            from codecarbon import EmissionsTracker  # type: ignore[import]
        except ImportError:
            if co2_tracker is True:
                raise ImportError(
                    "Codecarbon is required when co2_tracker=True. Please install it using `pip install mteb[codecarbon]` to track CO₂ emissions."
                )
            co2_tracker = False
        else:
            co2_tracker = True

    if co2_tracker:
        if co2_tracker:
            logger.warning(
                "Evaluating multiple MTEB runs simultaniously will produce incorrect CO₂ results"
            )
            with EmissionsTracker(
                save_to_file=False,
                save_to_api=False,
                logging_logger=logger,
                allow_multiple_runs=True,
            ) as tracker:
                result = _run_task(
                    model,
                    task,
                    splits=splits,
                    encode_kwargs=encode_kwargs,
                    co2_tracker=False,
                    **kwargs,
                )
            result.kg_co2_emissions = tracker.final_emissions
            return result

    task_results = {}

    task.check_if_dataset_is_superseded()

    data_loaded = task.data_loaded
    if data_loaded:
        task.load_data(**kwargs)

    evaluation_time = 0

    for split, hf_subsets in splits.items():
        tick = time()
        task_results[split] = task.evaluate(
            model,
            split,
            subsets_to_run=hf_subsets,
            encode_kwargs=encode_kwargs,
            **kwargs,
        )
        tock = time()

        logger.info(
            f"Evaluation for {task.metadata.name} on {split} took {tick - tock:.2f} seconds"
        )
        evaluation_time += tock - tick

    result = TaskResult.from_task_results(
        task,
        task_results,
        evaluation_time=evaluation_time,
        kg_co2_emissions=None,
    )

    if data_loaded:  # only unload if we loaded the data
        task.unload_data()

    return result


def run_tasks(
    model: Encoder | SentenceTransformer | CrossEncoder,
    task: AbsTask,  # TODO: allow multiple tasks + benchmarks
    *,
    co2_tracker: bool | None = None,
    raise_error: bool = True,
    encode_kwargs: dict[str, Any] | None = None,
    output_folder: Path | str | None = "results",
    cache_strategy: Literal[
        "no_cache", "output_folder", "online_cache", "only_cache"
    ] = "output_folder",
    overwrite_strategy: Literal["always", "never", "only-missing"] = "only-missing",
    **kwargs: Any,
) -> TaskResult | None:
    """This function runs a model on a a given task and returns the results.

    Args:
        model: The model to use for encoding.
        task: A task to run.
        co2_tracker: If True, track the CO₂ emissions of the evaluation. If none is passed co2 tracking will be run if codecarbon is installed.
        encode_kwargs: Additional keyword arguments passed to the models `encode` method.
        raise_error: If True, raise an error if the task fails. If False, return an empty list.
        output_folder: The folder to save the results to. If None, the results will not be saved.
        cache_strategy: The strategy to use for loading existing the results. Can be:
            - "no_cache": will not load the results from the cache.
            - "output_folder": Will check if the results exist in the output folder and load them from there.
            - "online_cache": Will check if the results exist in the online cache and load them from there. This will download the results from the online cache.
            - "only_cache": Will only load the results from the cache folder and do not run the task. Useful for debugging and testing.
        overwrite_strategy: The strategy to use for overwriting the results. Can be:
            - "always": Always overwrite the results
            - "never": Never overwrite the results
            - "only-missing": Only rerun the missing splits of a run. It will not rerun the splits if the dataset revision or mteb version has changed.
        kwargs: Additional keyword arguments for the task.

    Returns:
        The results of the evaluation.
    """
    if encode_kwargs is None:
        encode_kwargs = {}

    if "batch_size" not in encode_kwargs:
        encode_kwargs["batch_size"] = 32
        logger.info(
            "No batch size defined in encode_kwargs. Setting `encode_kwargs['batch_size'] = 32`."
        )

    meta = get_model_meta(model)
    if isinstance(model, (SentenceTransformer, CrossEncoder)):
        model = SentenceTransformerWrapper(model)  # type: ignore[assignment] # TODO: SentenceTransformerWrapper should be a subclass of Encoder
    model = cast(Encoder, model)

    output_folder = get_output_folder(meta, output_folder)

    # figure out cache path
    cache_result_path = None
    if cache_strategy == "no_cache":
        pass
    elif cache_strategy == "output_folder":
        if output_folder is None:
            raise ValueError(
                "output_folder must be specified when cache_strategy is 'output_folder'"
            )
        cache_result_path = output_folder / f"{task.metadata.name}.json"
    elif cache_strategy == "online_cache":
        cache_path = cache.download_results_cache(download_latest=True)

    if cache_result_path:
        existing_results = TaskResult.from_disk(cache_path)
    else:
        existing_results = None

    if cache_strategy == "only_cache":
        if existing_results is None:
            logger.warning(
                f"Cache strategy is set to 'only_cache' but no results were found in {cache_path}. Rerun the task with alternative cache_strategy to generate the results."
            )
        return existing_results

    if existing_results and existing_results.is_mergeable(task):
        missing_eval = existing_results.get_missing_evaluations(task)
    else:
        missing_eval = {split: task.hf_subsets for split in task.eval_splits}

    if raise_error is False:
        try:
            result = _run_task(
                model=model,
                splits=missing_eval,
                task=task,
                subsets_to_run=task.hf_subsets,
                co2_tracker=co2_tracker,
                encode_kwargs=encode_kwargs,
                **kwargs,
            )
        except Exception as e:
            logger.error(
                f"Error while running task {task.metadata.name} on splits {list(missing_eval.keys())}: {e}"
            )
            return None

    result = _run_task(
        model=model,
        splits=missing_eval,
        task=task,
        subsets_to_run=task.hf_subsets,
        co2_tracker=co2_tracker,
        encode_kwargs=encode_kwargs,
        **kwargs,
    )

    if existing_results:
        result = result.merge(existing_results)

    if output_folder and overwrite_strategy != "never":
        save_path = output_folder / f"{task.metadata.name}.json"
        result.to_disk(save_path)

    return result


# TODO:

# Other:
# - Model handling for bm25
#   - Currently the MTEB runner checks for bm25 and retrieval task. We should move this check to bm25's encode.
# - Add Benchmark.run(model) method, which simply a wrapper around run_tasks
# - Redo AggregatedTask to use the new run_tasks method

# Figure out a good UI for running tasks (how should the TQDM progress bar look like) - how does it interact with the sentence TRF progress bar

# Add tests