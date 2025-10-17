import json
import logging
import sys
from collections.abc import Sequence
from pathlib import Path

from mteb.abstasks.abstask import AbsTask
from mteb.models.model_meta import ModelMeta
from mteb.results import BenchmarkResults, ModelResult, TaskResult
from mteb.types import ModelName, Revision

if sys.version_info >= (3, 13):
    from warnings import deprecated
else:
    from typing_extensions import deprecated

logger = logging.getLogger(__name__)


def _model_name_and_revision(
    revision_path: Path, fallback_to_path: bool
) -> tuple[ModelName, Revision] | None:
    model_meta = revision_path / "model_meta.json"
    model_path = revision_path.parent
    if not model_meta.exists() and fallback_to_path:
        logger.info(
            f"model_meta.json not found in {revision_path}, extracting model_name and revision from the path"
        )
        model_name, revision = model_path.name, revision_path.name
    elif not model_meta.exists():
        return None
    else:
        with model_meta.open("r") as f:
            model_meta_json = json.load(f)
            model_name = model_meta_json["name"]
            revision = model_meta_json["revision"]

    return model_name, revision


@deprecated(
    "`load_results` is deprecated and will be removed in future versions. "
    + "Please use the `ResultCache`'s `.load_results` method instead."
)
def load_results(
    results_repo: str = "https://github.com/embeddings-benchmark/results",
    download_latest: bool = True,
    models: Sequence[ModelMeta] | Sequence[str] | None = None,
    tasks: Sequence[AbsTask] | Sequence[str] | None = None,
    validate_and_filter: bool = True,
    require_model_meta: bool = True,
    only_main_score: bool = False,
) -> BenchmarkResults:
    """Loads the results from the latest version of the [results](https://github.com/embeddings-benchmark/results) repository.

    The results are cached locally in the MTEB_CACHE directory.
    This directory can be set using the MTEB_CACHE environment variable or defaults to "~/.cache/mteb".

    Args:
        results_repo: The URL of the results repository on GitHub. Defaults to "https://github.com/embeddings-benchmark/results".
        download_latest: If True it will update the existing version of the results cache. Defaults to True.
        models: A list of model names to load the results for. If None it loads the results for all models. Defaults to None.
        tasks: A list of task names to load the results for. If None it loads the results for all tasks. Defaults to None.
        require_model_meta: If True it will ignore results that do not have a model_meta.json file. Defaults to True. If false it will
            extract the model name and revision from the path.
        validate_and_filter: If True it will validate that the results object for the task contains the correct splits and filter out
            splits from the results object that are not default in the task metadata. Defaults to True.
        only_main_score: If True, only the main score will be loaded.

    Returns:
        A BenchmarkResults object containing the results for the specified models and tasks.
    """
    from mteb.cache import ResultCache

    cache = ResultCache()
    if download_latest:
        cache.download_from_remote(remote=results_repo, download_latest=download_latest)
    repo_directory = cache.cache_path
    model_paths = [p for p in (repo_directory / "results").glob("*") if p.is_dir()]
    model_paths += [
        p for p in (repo_directory / "remote" / "results").glob("*") if p.is_dir()
    ]

    if models is not None:
        models_to_keep = {}
        for model_path in models:
            if isinstance(model_path, ModelMeta):
                models_to_keep[model_path.name] = model_path.revision
            else:
                models_to_keep[model_path] = None
    else:
        models_to_keep = None

    task_names = {}
    if tasks is not None:
        for task in tasks:
            if isinstance(task, AbsTask):
                task_names[task.metadata.name] = task
            else:
                task_names[task] = None

    model_results = []
    for model_path in model_paths:
        model_revisions = model_path.glob("*")

        for revision_path in model_revisions:
            model_name_and_revision = _model_name_and_revision(
                revision_path, fallback_to_path=(not require_model_meta)
            )
            if model_name_and_revision is None:
                continue
            model_name, revision = model_name_and_revision

            model_name = model_name.replace("__", "/")
            if models_to_keep is not None and model_name not in models_to_keep:
                continue
            elif models_to_keep is not None and models_to_keep[model_name] is not None:
                if models_to_keep[model_name] != revision:
                    continue

            task_json_files = [
                f for f in revision_path.glob("*.json") if "model_meta.json" != f.name
            ]
            _results = []
            for f in task_json_files:
                task_res = TaskResult.from_disk(f)
                if only_main_score:
                    task_res = task_res.only_main_score()
                _results.append(task_res)

            # filter out tasks that are not in the tasks list
            if tasks is not None:
                _results = [r for r in _results if r.task_name in task_names]

            if validate_and_filter:
                filtered_results = []
                for r in _results:
                    try:
                        if task_names:
                            task = task_names[r.task_name]
                        else:
                            task = None
                        r = r.validate_and_filter_scores(task=task)
                        filtered_results.append(r)
                    except Exception as e:
                        logger.info(
                            f"Validation failed for {r.task_name} in {model_name} {revision}: {e}"
                        )
                _results = filtered_results
            model_results.append(
                ModelResult(
                    model_name=model_name,
                    model_revision=revision,
                    task_results=_results,
                )
            )

    return BenchmarkResults(model_results=model_results)
