from __future__ import annotations

import logging
from collections.abc import Iterable
from copy import deepcopy
from pathlib import Path
from time import time
from typing import TYPE_CHECKING, Any, cast

from tqdm.auto import tqdm

from mteb._helpful_enum import HelpfulStrEnum
from mteb.abstasks import AbsTaskRetrieval
from mteb.abstasks.abstask import AbsTask
from mteb.abstasks.aggregated_task import AbsTaskAggregate
from mteb.cache import ResultCache
from mteb.models.model_meta import ModelMeta
from mteb.models.models_protocols import (
    CrossEncoderProtocol,
    EncoderProtocol,
    MTEBModels,
)
from mteb.models.sentence_transformer_wrapper import (
    CrossEncoderWrapper,
    SentenceTransformerEncoderWrapper,
)
from mteb.results import ModelResult, TaskResult
from mteb.types import HFSubset, PromptType, SplitName
from mteb.types._metadata import ModelName, Revision

if TYPE_CHECKING:
    from sentence_transformers import CrossEncoder, SentenceTransformer

logger = logging.getLogger(__name__)


class OverwriteStrategy(HelpfulStrEnum):
    """Enum for the overwrite strategy when running a task.

    - "always": Always run the task, overwriting the results
    - "never": Run the task only if the results are not found in the cache. If the results are found, it will not run the task.
    - "only-missing": Only rerun the missing splits of a task. It will not rerun the splits if the dataset revision or mteb version has
        changed.
    - "only-cache": Only load the results from the cache folder and do not run the task. Useful if you just want to load the results from the
        cache.
    """

    ALWAYS = "always"
    NEVER = "never"
    ONLY_MISSING = "only-missing"
    ONLY_CACHE = "only-cache"


_empty_model_meta = ModelMeta(
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
    modalities=[],
)


def _create_empty_model_meta() -> ModelMeta:
    logger.warning("Model metadata is missing. Using empty metadata.")
    meta = deepcopy(_empty_model_meta)
    meta.revision = "no_revision_available"
    meta.name = "no_model_name_available"
    return meta


def _sanitize_model(
    model: ModelMeta | MTEBModels | SentenceTransformer | CrossEncoder,
) -> tuple[MTEBModels | ModelMeta, ModelMeta, ModelName, Revision]:
    from sentence_transformers import CrossEncoder, SentenceTransformer

    if isinstance(model, SentenceTransformer):
        _mdl = SentenceTransformerEncoderWrapper(model)
        meta = _mdl.mteb_model_meta
        _mdl = cast(EncoderProtocol, _mdl)
        model = _mdl
    elif isinstance(model, CrossEncoder):
        _mdl = CrossEncoderWrapper(model)
        _mdl = cast(CrossEncoderProtocol, _mdl)
        meta = _mdl.mteb_model_meta
        model = _mdl
    elif hasattr(model, "mteb_model_meta"):
        meta = model.mteb_model_meta  # type: ignore[attr-defined]
        if not isinstance(meta, ModelMeta):
            meta = _create_empty_model_meta()
    else:
        meta = _create_empty_model_meta() if not isinstance(model, ModelMeta) else model

    model_name = cast(str, meta.name)
    model_revision = cast(str, meta.revision)

    return model, meta, model_name, model_revision


def _evaluate_task(
    model: MTEBModels,
    task: AbsTask,
    *,
    splits: dict[SplitName, list[HFSubset]],
    co2_tracker: bool | None,
    encode_kwargs: dict[str, Any],
    prediction_folder: Path | None,
) -> TaskResult:
    """The core logic to run a model on a given task. See `evaluate` for more details.

    Returns:
        The results of the evaluation.
    """
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
            result = _evaluate_task(
                model,
                task,
                splits=splits,
                encode_kwargs=encode_kwargs,
                co2_tracker=False,
                prediction_folder=prediction_folder,
            )
        result.kg_co2_emissions = tracker.final_emissions
        return result

    task_results = {}

    task.check_if_dataset_is_superseded()

    data_loaded = task.data_loaded
    if not data_loaded:
        task.load_data()

    evaluation_time = 0

    for split, hf_subsets in splits.items():
        tick = time()
        task_results[split] = task.evaluate(
            model,
            split,
            subsets_to_run=hf_subsets,
            encode_kwargs=encode_kwargs,
            prediction_folder=prediction_folder,
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


def _check_model_modalities(
    model: ModelMeta,
    tasks: AbsTask | Iterable[AbsTask],
) -> None:
    """Check that model modalities are compatible with task modalities.

    Logs warnings for partial overlaps and raises errors for mismatches.

    Args:
        model: The model metadata containing supported modalities.
        tasks: A single task or an iterable of tasks to check against the model.
    """
    if model.modalities is None or len(model.modalities) == 0:
        return

    model_modalities = set(model.modalities)
    if isinstance(tasks, AbsTask):
        tasks = [tasks]

    warnings, errors = [], []

    for task in tasks:
        # only retrieval tasks have different modalities for query and document and can be run with partial overlaps
        if isinstance(task, AbsTaskRetrieval):
            query_mods = set(task.metadata.get_modalities(PromptType.query))
            doc_mods = set(task.metadata.get_modalities(PromptType.document))

            query_overlap = model_modalities & query_mods
            doc_overlap = model_modalities & doc_mods

            if (
                # both query and document modalities are fully supported by the model
                doc_mods.issubset(model_modalities)
                and query_mods.issubset(model_modalities)
            ):
                continue
            elif query_overlap and doc_overlap:
                warnings.append(
                    f"Model {model.name} supports {model.modalities}, partially overlapping "
                    f"with task {task.metadata.name} query={sorted(query_mods)}, document={sorted(doc_mods)}. "
                    "Performance might be suboptimal."
                )
            else:
                errors.append(
                    f"Model {model.name} supports {model.modalities}, but none overlap with "
                    f"task {task.metadata.name} query={sorted(query_mods)}, document={sorted(doc_mods)}."
                )
        else:
            task_mods = set(task.metadata.modalities)

            if task_mods.issubset(model_modalities):
                continue
            else:
                errors.append(
                    f"Model {model.name} supports {model.modalities}, but none overlap with "
                    f"task {task.metadata.name} modalities={task.metadata.modalities}."
                )

    if errors:
        raise ValueError("\n".join(errors))
    for msg in warnings:
        logger.warning(msg)


def evaluate(
    model: ModelMeta | MTEBModels | SentenceTransformer | CrossEncoder,
    tasks: AbsTask | Iterable[AbsTask],
    *,
    co2_tracker: bool | None = None,
    raise_error: bool = True,
    encode_kwargs: dict[str, Any] | None = None,
    cache: ResultCache | None = ResultCache(),
    overwrite_strategy: str | OverwriteStrategy = "only-missing",
    prediction_folder: Path | str | None = None,
    show_progress_bar: bool = True,
) -> ModelResult:
    """This function runs a model on a given task and returns the results.

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
        prediction_folder: Optional folder in which to save model predictions for the task. Predictions of the tasks will be sabed in `prediction_folder/{task_name}_predictions.json`
        show_progress_bar: Whether to show a progress bar when running the evaluation. Default is True. Setting this to False will also set the
            `encode_kwargs['show_progress_bar']` to False if encode_kwargs is unspecified.

    Returns:
        The results of the evaluation.

    Examples:
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
    if isinstance(prediction_folder, str):
        prediction_folder = Path(prediction_folder)

    if encode_kwargs is None:
        encode_kwargs = (
            {"show_progress_bar": False} if show_progress_bar is False else {}
        )
    if "batch_size" not in encode_kwargs:
        encode_kwargs["batch_size"] = 32
        logger.info(
            "No batch size defined in encode_kwargs. Setting `encode_kwargs['batch_size'] = 32`. Explicitly set the batch size to silence this message."
        )

    model, meta, model_name, model_revision = _sanitize_model(model)
    _check_model_modalities(meta, tasks)

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
            prediction_folder=prediction_folder,
            show_progress_bar=show_progress_bar,
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
        tasks_tqdm = tqdm(
            tasks,
            desc="Evaluating tasks",
            disable=not show_progress_bar,
        )
        for i, task in enumerate(tasks_tqdm):
            tasks_tqdm.set_description(f"Evaluating task {task.metadata.name}")
            _res = evaluate(
                model,
                task,
                co2_tracker=co2_tracker,
                raise_error=raise_error,
                encode_kwargs=encode_kwargs,
                cache=cache,
                overwrite_strategy=overwrite_strategy,
                prediction_folder=prediction_folder,
                show_progress_bar=False,
            )
            results.extend(_res.task_results)
        return ModelResult(
            model_name=_res.model_name,
            model_revision=_res.model_revision,
            task_results=results,
        )

    overwrite_strategy = OverwriteStrategy.from_str(overwrite_strategy)

    existing_results = None
    if cache and overwrite_strategy != OverwriteStrategy.ALWAYS:
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
        logger.info(
            f"Results for {task.metadata.name} already exist in cache. Skipping evaluation and loading results."
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

    if existing_results:
        logger.info(
            f"Found existing results for {task.metadata.name}, only running missing splits: {list(missing_eval.keys())}"
        )

    if isinstance(model, ModelMeta):
        logger.info(
            f"Loading model {model_name} with revision {model_revision} from ModelMeta."
        )
        model = model.load_model()
        logger.info("✓ Model loaded")

    if raise_error is False:
        try:
            result = _evaluate_task(
                model=model,
                splits=missing_eval,
                task=task,
                co2_tracker=co2_tracker,
                encode_kwargs=encode_kwargs,
                prediction_folder=prediction_folder,
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
    else:
        result = _evaluate_task(
            model=model,
            splits=missing_eval,
            task=task,
            co2_tracker=False,
            encode_kwargs=encode_kwargs,
            prediction_folder=prediction_folder,
        )
    logger.info(f"✓ Finished evaluation for {task.metadata.name}")

    if existing_results:
        result = result.merge(existing_results)

    if cache:
        cache.save_to_cache(result, meta)

    return ModelResult(
        model_name=model_name,
        model_revision=model_revision,
        task_results=[result],
    )
