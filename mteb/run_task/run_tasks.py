from __future__ import annotations

import logging
from collections.abc import Iterable
from pathlib import Path
from time import time
from typing import Any, Literal

from sentence_transformers import CrossEncoder, SentenceTransformer

from mteb import SentenceTransformerWrapper
from mteb.abstasks.AbsTask import AbsTask
from mteb.benchmarks import Benchmark
from mteb.encoder_interface import Encoder
from mteb.load_results import TaskResult

from .get_model_meta import get_model_meta, get_output_path

logger = logging.getLogger(__name__)


def _run_task(
    model: Encoder,
    task: AbsTask,
    *,
    splits: list[str],
    subsets_to_run: list[str] | None,  # TODO: Can this actualy be None?
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
                    subsets_to_run=subsets_to_run,
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

    for split in splits:
        tick = time()
        task_results[split] = task.evaluate(
            model,
            split,
            subsets_to_run=subsets_to_run,
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


def run_task(
    model: Encoder | SentenceTransformer | CrossEncoder,
    task: AbsTask,
    *,
    splits: list[str] | None = None,
    subsets_to_run: list[str] | None = None,
    co2_tracker: bool | None = None,
    raise_error: bool = True,
    encode_kwargs: dict[str, Any] | None = None,
    output_folder: Path | str | None = None,
    **kwargs: Any,
) -> TaskResult | None:
    """This function runs a model on a a given task and returns the results.

    Args:
        model: The model to use for encoding.
        tasks: A task to run.
        splits: The splits to evaluate on. If None, the default evaluation splits for the task will be used.
        subsets_to_run: The subsets to evaluate on. If None, all subsets will be evaluated.
        co2_tracker: If True, track the CO₂ emissions of the evaluation. If none is passed co2 tracking will be run if codecarbon is installed.
        encode_kwargs: Additional keyword arguments passed to the models `encode` method.
        raise_error: If True, raise an error if the task fails. If False, return an empty list.
        output_folder: The folder to save the results to. If None, the results will not be saved.
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

    splits_to_run = splits if splits is not None else task.eval_splits
    subsets_to_run = subsets_to_run if subsets_to_run is not None else task.hf_subsets

    meta = get_model_meta(model)
    if isinstance(model, (SentenceTransformer, CrossEncoder)):
        model: Encoder = SentenceTransformerWrapper(model)  # type: ignore[assignment] # TODO: SentenceTransformerWrapper should be a subclass of Encoder

    output_path = get_output_path(meta, output_folder)

    if raise_error is False:
        try:
            result = _run_task(
                model,
                task,
                splits=splits_to_run,
                subsets_to_run=task.hf_subsets,
                co2_tracker=co2_tracker,
                encode_kwargs=encode_kwargs,
                **kwargs,
            )
        except Exception as e:
            logger.error(
                f"Error while running task {task.metadata.name} on splits {splits_to_run}: {e}"
            )
            return None

    result = _run_task(
        model,
        task,
        splits=splits_to_run,
        subsets_to_run=task.hf_subsets,
        co2_tracker=co2_tracker,
        encode_kwargs=encode_kwargs,
        **kwargs,
    )

    if output_path:
        save_path = output_path / f"{task.metadata.name}.json"
        result.to_disk(save_path)

    return result


def run_tasks(
    model: Encoder | SentenceTransformer | CrossEncoder,
    tasks: Iterable[AbsTask] | Benchmark,
    *,
    co2_tracker: bool | None = None,
    raise_error: bool = True,
    encode_kwargs: dict[str, Any] | None = None,
    output_folder: str | None = None,
    cache_strategy: Literal[
        "no_cache", "output_folder", "online_cache"
    ] = "output_folder",
    overwrite_strategy: Literal[
        "always", "never", "only-missing", "if-version-changed"
    ] = "only-missing",
    **kwargs: Any,
) -> list[TaskResult]:
    """This function runs a model on a a given task and returns the results.

    Args:
        model: The model to use for encoding.
        tasks: A benhmark or a list of tasks to run.
        co2_tracker: If True, track the CO₂ emissions of the evaluation. If none is passed co2 tracking will be run if codecarbon is installed.
        encode_kwargs: Additional keyword arguments passed to the models `encode` method.
        raise_error: If True, raise an error if the task fails. If False, return an empty list.
        output_folder: The folder to save the results to. If None, the results will not be saved.
        cache_strategy: The strategy to use for loading existing the results. Can be:
            - "no_cache": will not load the results from the cache.
            - "output_folder": Will check if the results exist in the output folder and load them from there.
            - "online_cache": Will check if the results exist in the online cache and load them from there. This will download the results from the online cache.
            - "only_cache": Will only load the results from the cache folder and do not run the task. Useful for debugging and testing.
        overwrite_strategy: The strategy to use for overwriting the results. Can be "always", "never", "only-missing" or "on-version-mismatch".
        kwargs: Additional keyword arguments for the task.

    Returns:
        The results of the evaluation.
    """
    if isinstance(tasks, AbsTask):
        tasks = [tasks]

    if isinstance(tasks, Benchmark):
        tasks = tasks.tasks

    results = []

    for task in tasks:
        result = run_task(
            model,
            task,
            co2_tracker=co2_tracker,
            raise_error=raise_error,
            encode_kwargs=encode_kwargs,
            output_folder=output_folder,
            **kwargs,
        )
        if result is not None:
            results.append(result)

    return results


# TODO:
# - Add merging strategy for results
# - Add cache_strategy
#   - cache_strategy = "output_folder" (default) will load the results only if they exist in the output folder
#   - cache_stategy = "online_cache" will load the results from the online results folder. This will download the results from the online results folder and save them to the cache folder
#   - cache_strategy = "no_cache" will not load the results from the cache folder
# - Add overwrite_strategy
#   - "always" Always overwrite the results
#   - "never" Never overwrite the results
#   - "only-missing" (default) Overwrite the results only if it contains missing splits or results and in that case only rerun the missing splits
#   - "if-version-changed" Overwrite the results only if it hasn't been run using the latest mteb version
#   - "only-load": Only load the results from the cache folder and do not run the task. Useful for debugging and testing.

# Other:
# - Model handling for bm25
#   - Currently the MTEB runner checks for bm25 and retrieval task. We should move this check to bm25's encode.
# - Add Benchmark.run(model) method, which simply a wrapper around run_tasks
# - Redo AggregatedTask to use the new run_tasks method
