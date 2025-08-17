from __future__ import annotations

import logging
from collections.abc import Iterable
from copy import deepcopy
from time import time
from typing import Any, cast

from sentence_transformers import CrossEncoder, SentenceTransformer

from mteb._helpful_enum import HelpfulStrEnum
from mteb.abstasks.AbsTask import AbsTask
from mteb.abstasks.aggregated_task import AbsTaskAggregate
from mteb.cache import ResultCache
from mteb.load_results.benchmark_results import ModelResult
from mteb.load_results.task_results import TaskResult
from mteb.models.get_model_meta import (
    _model_meta_from_cross_encoder,
    _model_meta_from_sentence_transformers,
)
from mteb.models.model_meta import ModelMeta
from mteb.models.models_protocols import (
    CrossEncoderProtocol,
    Encoder,
    MTEBModels,
)
from mteb.models.sentence_transformer_wrapper import (
    CrossEncoderWrapper,
    SentenceTransformerEncoderWrapper,
)
from mteb.types import HFSubset, SplitName

logger = logging.getLogger(__name__)


class OverwriteStrategy(HelpfulStrEnum):
    ALWAYS = "always"
    NEVER = "never"
    ONLY_MISSING = "only-missing"
    ONLY_CACHE = "only-cache"


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


# TODO: seems like we could avoid this by making the wrapper first
def _get_model_meta(model: Encoder | SentenceTransformer | CrossEncoder) -> ModelMeta:
    meta: ModelMeta | None = None
    if hasattr(model, "mteb_model_meta"):
        meta = model.mteb_model_meta  # type: ignore

    if meta is None:
        if isinstance(model, CrossEncoder):
            meta = _model_meta_from_cross_encoder(model)
        elif isinstance(model, SentenceTransformer):
            meta = _model_meta_from_sentence_transformers(model)
        else:
            meta = empty_model_meta

    # create a copy of the meta to avoid modifying the original object
    meta = deepcopy(meta)
    meta.revision = meta.revision or "no_revision_available"
    meta.name = meta.name or "no_model_name_available"

    return meta


def _evaluate(
    model: Encoder,
    task: AbsTask,
    *,
    splits: dict[SplitName, list[HFSubset]],
    co2_tracker: bool | None,
    encode_kwargs: dict[str, Any],
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
        with EmissionsTracker(
            save_to_file=False,
            save_to_api=False,
            logging_logger=logger,  # type: ignore[arg-type]
            allow_multiple_runs=False,
        ) as tracker:
            result = _evaluate(
                model,
                task,
                splits=splits,
                encode_kwargs=encode_kwargs,
                co2_tracker=False,
            )
        result.kg_co2_emissions = tracker.final_emissions
        return result

    task_results = {}

    task.check_if_dataset_is_superseded()

    data_loaded = task.data_loaded
    if data_loaded:
        task.load_data()

    evaluation_time = 0

    for split, hf_subsets in splits.items():
        tick = time()
        task_results[split] = task.evaluate(
            model,
            split,
            subsets_to_run=hf_subsets,
            encode_kwargs=encode_kwargs,
        )
        tock = time()

        logger.debug(
            f"Evaluation for {task.metadata.name} on {split} took {tock - tick:.2f} seconds"
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


def evaluate(
    model: ModelMeta | MTEBModels | SentenceTransformer | CrossEncoder,
    tasks: AbsTask | Iterable[AbsTask],
    *,
    co2_tracker: bool | None = None,
    raise_error: bool = True,
    encode_kwargs: dict[str, Any] | None = None,
    cache: ResultCache | None = ResultCache(),
    overwrite_strategy: str | OverwriteStrategy = "only-missing",
) -> ModelResult:
    """This function runs a model on a a given task and returns the results.

    Args:
        model: The model to use for encoding.
        tasks: A task to run.
        co2_tracker: If True, track the CO₂ emissions of the evaluation, required codecarbon to be installed, which can be installed using
            `pip install mteb[codecarbon]`. If none is passed co2 tracking will only be run if codecarbon is installed.
        encode_kwargs: Additional keyword arguments passed to the models `encode` method.
        raise_error: If True, raise an error if the task fails. If False, return an empty list.
        cache: The cache to use for loading the results. If None, then no cache will be used. The default cache saved the cache in the
            `~/.cache/mteb` directory. It can be overridden by setting the `MTEB_CACHE` environment variable to a different directory or by directly
            passing a `ResultCache` object.
        overwrite_strategy: The strategy to use for run a task and overwrite the results. Can be:
            - "always": Always run the task, overwriting the results
            - "never": Run the task only if the results are not found in the cache. If the results are found, it will not run the task.
            - "only-missing": Only rerun the missing splits of a task. It will not rerun the splits if the dataset revision or mteb version has
                changed.
            - "only-cache": Only load the results from the cache folder and do not run the task. Useful if you just want to load the results from the
                cache.

    Returns:
        The results of the evaluation.

    Example:
        >>> import mteb
        >>> model_meta = mteb.get_model_meta("sentence-transformers/all-MiniLM-L6-v2")
        >>> task = mteb.get_task("STS12")
        >>> result = mteb.evaluate(ModelMeta, task)
        >>>
        >>> # with CO2 tracking
        >>> result = mteb.evaluate(model_meta, task, co2_tracker=True)
        >>>
        >>> # with encode kwargs
        >>> result = mteb.evaluate(model_meta, task, encode_kwargs={"batch_size": 16})
        >>>
        >>> # with online cache
        >>> cache = mteb.ResultCache(cache_path="~/.cache/mteb")
        >>>
        >>> cache.download_from_remote()
        >>> result = mteb.evaluate(model_meta, task, cache=cache)
    """
    # AbsTaskAggregate is a special case where we have to run multiple tasks and combine the results
    if isinstance(tasks, AbsTaskAggregate):
        task = cast(AbsTaskAggregate, tasks)
        results = evaluate(
            model,
            task.metadata.tasks,
            co2_tracker=co2_tracker,
            raise_error=raise_error,
            encode_kwargs=encode_kwargs,
            cache=cache,
            overwrite_strategy=overwrite_strategy,
        )
        result = task.combine_task_results(results.task_results)
        return ModelResult(
            model_name=results.model_name,
            model_revision=results.model_revision,
            task_results=[result],
        )

    if isinstance(tasks, AbsTask):
        task = tasks
    else:
        results = []
        for task in tasks:
            _res = evaluate(
                model,
                task,
                co2_tracker=co2_tracker,
                raise_error=raise_error,
                encode_kwargs=encode_kwargs,
                cache=cache,
                overwrite_strategy=overwrite_strategy,
            )
            results.extend(_res.task_results)
        return ModelResult(
            model_name=_res.model_name,
            model_revision=_res.model_revision,
            task_results=results,
        )

    overwrite_strategy = OverwriteStrategy.from_str(overwrite_strategy)
    if encode_kwargs is None:
        encode_kwargs = {}

    if "batch_size" not in encode_kwargs:
        encode_kwargs["batch_size"] = 32
        logger.info(
            "No batch size defined in encode_kwargs. Setting `encode_kwargs['batch_size'] = 32`."
        )

    meta = _get_model_meta(model) if not isinstance(model, ModelMeta) else model
    model_name = cast(str, meta.name)
    model_revision = cast(str, meta.revision)
    if isinstance(model, SentenceTransformer):
        model = SentenceTransformerEncoderWrapper(model)
        model = cast(Encoder, model)
    elif isinstance(model, CrossEncoder):
        model = CrossEncoderWrapper(model)
        model = cast(CrossEncoderProtocol, model)

    existing_results = None
    if cache:
        results = cache.load_task_result(task.metadata.name, meta)
        if results:
            existing_results = results

    if (
        existing_results
        and overwrite_strategy == "only-missing"
        and overwrite_strategy == OverwriteStrategy.ONLY_MISSING
        and existing_results.is_mergeable(task)
    ):
        missing_eval = existing_results.get_missing_evaluations(task)
    else:
        missing_eval = dict.fromkeys(task.eval_splits, task.hf_subsets)

    if (
        existing_results
        and not missing_eval
        and overwrite_strategy != OverwriteStrategy.ALWAYS
    ):
        # if there are no missing evals we can just return the results
        logger.debug(
            f"Results for {task.metadata.name} already exist in cache. Skipping evaluation."
        )
        return ModelResult(
            model_name=model_name,
            model_revision=model_revision,
            task_results=[existing_results],
        )
    if missing_eval and overwrite_strategy in [
        OverwriteStrategy.NEVER,
        OverwriteStrategy.ONLY_CACHE,
    ]:
        raise ValueError(
            f"overwrite_strategy is set to '{overwrite_strategy.value}' and the results file exists. However there are the following missing splits (and subsets): {missing_eval}. To rerun these set overwrite_strategy to 'only-missing'."
        )

    if isinstance(model, ModelMeta):
        logger.info(
            f"Loading model {model_name} with revision {model_revision} from ModelMeta."
        )
        model = model.load_model()
    if raise_error is False:
        try:
            result = _evaluate(
                model=model,
                splits=missing_eval,
                task=task,
                co2_tracker=co2_tracker,
                encode_kwargs=encode_kwargs,
            )
            return ModelResult(
                model_name=model_name,
                model_revision=model_revision,
                task_results=[result],
            )
        except Exception as e:
            logger.error(
                f"Error while running task {task.metadata.name} on splits {list(missing_eval.keys())}: {e}"
            )
            return ModelResult(
                model_name=model_name,
                model_revision=model_revision,
                task_results=[],
            )
    result = _evaluate(
        model=model,
        splits=missing_eval,
        task=task,
        co2_tracker=False,
        encode_kwargs=encode_kwargs,
    )

    if existing_results:
        result = result.merge(existing_results)

    if cache:
        cache.save_to_cache(result, meta)

    return ModelResult(
        model_name=model_name,
        model_revision=model_revision,
        task_results=[result],
    )
