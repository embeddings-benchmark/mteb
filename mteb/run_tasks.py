from __future__ import annotations

import logging
from collections.abc import Iterable
from copy import deepcopy
from time import time
from typing import Any, Literal, cast

from sentence_transformers import CrossEncoder, SentenceTransformer

from mteb import SentenceTransformerWrapper
from mteb.abstasks.AbsTask import AbsTask
from mteb.abstasks.TaskMetadata import HFSubset, Splitname
from mteb.cache import ResultCache
from mteb.encoder_interface import Encoder
from mteb.load_results.task_results import TaskResult
from mteb.model_meta import ModelMeta
from mteb.models import (
    model_meta_from_cross_encoder,
    model_meta_from_sentence_transformers,
)

logger = logging.getLogger(__name__)

empty_model_meta = ModelMeta(
    loader=None,
    name=None,
    revision=None,
    release_date=None,
    languages=None,
    framework=[],
    similarity_fn_name=None,
    n_parameters=None,
    memory_usage_mb=None,
    max_tokens=None,
    embed_dim=None,
    license=None,
    open_weights=None,
    public_training_code=None,
    public_training_data=None,
    use_instructions=None,
    training_datasets=None,
)


# TODO: seems like we could avoid this if we know that we are getting
def _get_model_meta(model: Encoder | SentenceTransformer | CrossEncoder) -> ModelMeta:
    meta: ModelMeta | None = None
    if hasattr(model, "mteb_model_meta"):
        meta = model.mteb_model_meta  # type: ignore

    if meta is None:
        if isinstance(model, CrossEncoder):
            meta = model_meta_from_cross_encoder(model)
        elif isinstance(model, SentenceTransformer):
            meta = model_meta_from_sentence_transformers(model)
        else:
            meta = empty_model_meta

    # create a copy of the meta to avoid modifying the original object
    meta = deepcopy(meta)
    meta.revision = meta.revision or "no_revision_available"
    meta.name = meta.name or "no_model_name_available"

    return meta


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
    model: ModelMeta | Encoder | SentenceTransformer | CrossEncoder,
    tasks: AbsTask | Iterable[AbsTask],
    *,
    co2_tracker: bool | None = None,
    raise_error: bool = True,
    encode_kwargs: dict[str, Any] | None = None,
    cache: None | ResultCache = ResultCache(),
    overwrite_strategy: Literal[
        "always", "never", "only-missing", "only-cache"
    ] = "only-missing",
    **kwargs: Any,
) -> list[TaskResult]:
    """This function runs a model on a a given task and returns the results.

    Args:
        model: The model to use for encoding.
        tasks: A task to run.
        co2_tracker: If True, track the CO₂ emissions of the evaluation. If none is passed co2 tracking will be run if codecarbon is installed.
        encode_kwargs: Additional keyword arguments passed to the models `encode` method.
        raise_error: If True, raise an error if the task fails. If False, return an empty list.
        cache: The cache to use for loading the results. If None, the default cache will be used. The default cache saved the cache in the
            `~/.cache/mteb` directory. It can be overridden by setting the `MTEB_CACHE` environment variable to a different directory or by directly
            passing a `ResultCache` object.
        overwrite_strategy: The strategy to use for run a task and overwrite the results. Can be:
            - "always": Always run the task, overwriting the results
            - "never": Run the task only if the results are not found in the cache. If the results are found, it will not run the task.
            - "only-missing": Only rerun the missing splits of a task. It will not rerun the splits if the dataset revision or mteb version has
                changed.
            - "only-cache": Only load the results from the cache folder and do not run the task. Useful if you just want to load the results from the
                cache.
        kwargs: Additional keyword arguments for the task.

    Returns:
        The results of the evaluation.

    Example:
        >>> import mteb
        >>> model_meta = mteb.get_model_meta("sentence-transformers/all-MiniLM-L6-v2")
        >>> task = mteb.get_task("STS12")
        >>> result = mteb.run_tasks(ModelMeta, task)
        >>>
        >>> # with CO2 tracking
        >>> result = mteb.run_tasks(model_meta, task, co2_tracker=True)
        >>>
        >>> # with encode kwargs
        >>> result = mteb.run_tasks(model_meta, task, encode_kwargs={"batch_size": 16})
        >>>
        >>> # with online cache
        >>> cache = mteb.ResultCache(cache_path="~/.cache/mteb")
        >>>
        >>> cache.download_from_remote()
        >>> result = mteb.run_tasks(model_meta, task, cache=cache)
    """
    if isinstance(tasks, AbsTask):
        task = tasks
    else:
        results = []
        for task in tasks:
            results.extend(
                run_tasks(
                    model,
                    task,
                    co2_tracker=co2_tracker,
                    raise_error=raise_error,
                    encode_kwargs=encode_kwargs,
                    cache=cache,
                    overwrite_strategy=overwrite_strategy,
                    **kwargs,
                )
            )
        return results

    if encode_kwargs is None:
        encode_kwargs = {}

    if "batch_size" not in encode_kwargs:
        encode_kwargs["batch_size"] = 32
        logger.debug(
            "No batch size defined in encode_kwargs. Setting `encode_kwargs['batch_size'] = 32`."
        )

    meta = _get_model_meta(model) if not isinstance(model, ModelMeta) else model
    if isinstance(model, (SentenceTransformer, CrossEncoder)):
        model = SentenceTransformerWrapper(model)  # type: ignore[assignment] # TODO: SentenceTransformerWrapper should be a subclass of Encoder
    model = cast(Encoder, model)

    existing_results = None
    if cache:
        results = cache.load_from_cache(task.metadata.name, meta)
        if results:
            existing_results = results

    if existing_results and overwrite_strategy in ["never", "only-cache"]:
        logger.debug(
            f"Results for {task.metadata.name} already exist in cache. Skipping evaluation."
        )
        return [existing_results]

    if (
        existing_results
        and overwrite_strategy == "only-missing"
        and existing_results.is_mergeable(task)
    ):
        missing_eval = existing_results.get_missing_evaluations(task)
    else:
        missing_eval = {split: task.hf_subsets for split in task.eval_splits}

    if isinstance(model, ModelMeta):
        model = model.load_model()
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
            return []

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

    if cache:
        cache.save_to_cache(result, meta)

    return [result]


# TODO:
# Figure out a good UI for running tasks (how should the TQDM progress bar look like) - how does it interact with the sentence TRF progress bar
# Add tests


# PR notes:
# Notes to consider:
# - Currently MTEB checks for bm25 and retrieval task. I don't believe this is currently needed.
# - One of the reasons for doing this task was also to refactor AbsTaskAggregate. I haven't done this yet, as I think it might be worth refactoring this
#   part. We have also have people ask about getting result for a benchmark and I feel like the two are very close. I also don't like that AbsTaskAggregate
#   isn't really a AbsTask. I would love to brainstorm this idea a bit and hear what you guys think. (this means that I currently can't remove MTEB)
# - Currently the results is a list[TaskResult], but it could as well be ModelResults() as it also stores the metadata. WDYT?
# - I had a lot of issues with circular imports. A lot of this stems from types. I think an easy solution here is to create a types module (I didn't more everything here because it woudl inflate the div)

# Minor notes:
# - Previously mentioned that I wanted to add a benchmark.run() method, but I think it is better to just do mteb.run_tasks(benchmark)
# - Seems like we have a few attributes on the benchmark that are only for the leaderboard (e.g. icon). Not sure if those should be private or moved elsewhere (leaderboard code)
