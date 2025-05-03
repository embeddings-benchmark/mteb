from __future__ import annotations

import json
import logging
import os
import subprocess
from collections.abc import Sequence
from pathlib import Path

from mteb.abstasks.AbsTask import AbsTask
from mteb.load_results.benchmark_results import BenchmarkResults, ModelResult
from mteb.load_results.task_results import TaskResult
from mteb.model_meta import ModelMeta

logger = logging.getLogger(__name__)
MODEL_NAME = str
REVISION = str


def download_of_results(
    results_repo: str, cache_directory: Path | None = None, download_latest: bool = True
) -> Path:
    """Downloads the latest version of the results repository from GitHub to a local cache directory. Required git to be installed.

    Args:
        results_repo: The URL of the results repository on GitHub.
        cache_directory: The directory where the repository should be cached. If None it will use the MTEB_CACHE environment variable or "~/.cache/mteb" by default.
        download_latest: If True it will download the latest version of the repository, otherwise it will only update the existing repository.

    Returns:
        The path to the local cache directory.
    """
    default_cache_directory = Path.home() / ".cache" / "mteb"

    if cache_directory is None:
        _cache_directory = os.environ.get("MTEB_CACHE", None)
        cache_directory = (
            Path(_cache_directory) if _cache_directory else default_cache_directory
        )

    if not cache_directory.exists():
        cache_directory.mkdir(parents=True)

    # if "results" folder already exists update it
    results_directory = cache_directory / os.path.basename(results_repo)
    if results_directory.exists():
        if download_latest:
            logger.info(
                f"Results repository already exists in {results_directory}, updating it using git pull"
            )
            subprocess.run(["git", "pull"], cwd=results_directory)
        else:
            logger.info(
                f"Results repository already exists in {results_directory}, skipping update, set download_latest=True to update it"
            )
    else:
        logger.info(
            f"No results repository found in {results_directory}, cloning it from {results_repo}"
        )
        subprocess.run(["git", "clone", results_repo], cwd=cache_directory)

    return results_directory


def _model_name_and_revision(
    revision_path: Path, fallback_to_path: bool
) -> tuple[MODEL_NAME, REVISION] | None:
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


def load_results(
    results_repo: str = "https://github.com/embeddings-benchmark/results",
    download_latest: bool = True,
    models: Sequence[ModelMeta] | Sequence[str] | None = None,
    tasks: Sequence[AbsTask] | Sequence[str] | None = None,
    validate_and_filter: bool = True,
    require_model_meta: bool = True,
    only_main_score: bool = False,
) -> BenchmarkResults:
    """Loads the results from the latest version of the results repository. The results are cached locally in the MTEB_CACHE directory.
    This directory can be set using the MTEB_CACHE environment variable or defaults to "~/.cache/mteb".

    Args:
        results_repo: The URL of the results repository on GitHub. Defaults to "https://github.com/embeddings-benchmark/results".
        download_latest: If True it will update the existing version of the results cache. Defaults to True.
        models: A list of model names to load the results for. If None it will load the results for all models. Defaults to None.
        tasks: A list of task names to load the results for. If None it will load the results for all tasks. Defaults to None.
        require_model_meta: If True it will ignore results that do not have a model_meta.json file. Defaults to True. If false it will
            extract the model name and revision from the path.
        validate_and_filter: If True it will validate that the results object for the task contains the correct splits and filter out
            splits from the results object that are not default in the task metadata. Defaults to True.
        only_main_score: If True, only the main score will be loaded.
    """
    # TODO: we want to allow results_repo (the first argument) to be a local path
    # TODO: in v2 we can rename it to "path" to align with load_dataset
    repo_directory = download_of_results(results_repo, download_latest=download_latest)
    model_paths = [p for p in (repo_directory / "results").glob("*") if p.is_dir()]

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
