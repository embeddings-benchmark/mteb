import json
import logging
import os
import shutil
import subprocess
import warnings
from collections import defaultdict
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import cast

import mteb
from mteb.abstasks import AbsTask
from mteb.benchmarks.benchmark import Benchmark
from mteb.models import ModelMeta
from mteb.results import BenchmarkResults, ModelResult, TaskResult
from mteb.types import ModelName, Revision

logger = logging.getLogger(__name__)


class ResultCache:
    """Class to handle the local cache of MTEB results.

    Examples:
        >>> from mteb.cache import ResultCache
        >>> cache = ResultCache(cache_path="~/.cache/mteb") # default
        >>> cache.download_from_remote() # download the latest results from the remote repository
        >>> result = cache.load_results("task_name", "model_name")
    """

    cache_path: Path

    def __init__(self, cache_path: Path | str | None = None) -> None:
        if cache_path is not None:
            self.cache_path = Path(cache_path)
        else:
            self.cache_path = self.default_cache_path
        self.cache_path.mkdir(parents=True, exist_ok=True)

    @property
    def has_remote(self) -> bool:
        """Check if the remote results repository exists in the cache directory.

        Returns:
            True if the remote results repository exists, False otherwise.
        """
        return (self.cache_path / "remote").exists()

    def get_task_result_path(
        self,
        task_name: str,
        model_name: str | ModelMeta,
        model_revision: str | None = None,
        remote: bool = False,
    ) -> Path:
        """Get the path to the results of a specific task for a specific model and revision.

        Args:
            task_name: The name of the task.
            model_name: The name of the model as a valid directory name or a ModelMeta object.
            model_revision: The revision of the model. Must be specified if model_name is a string.
            remote: If True, it will return the path to the remote results repository, otherwise it will return the path to the local results repository.

        Returns:
            The path to the results of the task.
        """
        results_folder = (
            self.cache_path / "results"
            if not remote
            else self.cache_path / "remote" / "results"
        )

        if isinstance(model_name, ModelMeta):
            if model_revision is not None:
                logger.warning(
                    "model_revision is ignored when model_name is a ModelMeta object"
                )
            model_revision = model_name.revision
            model_name = model_name.model_name_as_path()
        elif isinstance(model_name, str):
            model_name = model_name.replace("/", "__").replace(" ", "_")

        model_path = results_folder / model_name

        if model_revision is None:
            msg = "`model_revision` is not specified, attempting to load the latest revision. To disable this behavior, specify the 'model_revision` explicitly."
            logger.warning(msg)
            warnings.warn(msg)
            # get revs from paths
            revisions = [p for p in model_path.glob("*") if p.is_dir()]
            if not revisions:
                model_revision = "no_revision_available"
            else:
                if len(revisions) > 1:
                    logger.warning(
                        f"Multiple revisions found for model {model_name}: {revisions}. Using the latest one (according to latest edit)."
                    )
                    # sort folder by latest edit time
                    revisions.sort(key=lambda p: p.stat().st_mtime, reverse=True)
                model_revision = revisions[0].name

        return model_path / model_revision / f"{task_name}.json"

    def load_task_result(
        self,
        task_name: str,
        model_name: str | ModelMeta,
        model_revision: str | None = None,
        raise_if_not_found: bool = False,
        prioritize_remote: bool = False,
    ) -> TaskResult | None:
        """Load the results from the local cache directory.

        Args:
            task_name: The name of the task.
            model_name: The name of the model as a valid directory name or a ModelMeta object.
            model_revision: The revision of the model. Must be specified if model_name is a string.
            raise_if_not_found: If True, raise an error if the results are not found.
            prioritize_remote: If True, it will first try to load the results from the remote repository, if available.

        Returns:
            The results of the task, or None if not found.
        """
        result_path = self.get_task_result_path(
            model_name=model_name,
            model_revision=model_revision,
            task_name=task_name,
        )

        if self.has_remote:
            remote_result_path = self.get_task_result_path(
                model_name=model_name,
                model_revision=model_revision,
                task_name=task_name,
                remote=True,
            )
            if remote_result_path.exists() and prioritize_remote:
                result_path = remote_result_path
            elif not result_path.exists():
                result_path = remote_result_path

        if not result_path.exists():
            msg = f"Results for {model_name} on {task_name} not found in {result_path}"
            if raise_if_not_found:
                raise FileNotFoundError(msg)
            logger.debug(msg)
            return None

        return TaskResult.from_disk(result_path)

    def save_to_cache(
        self,
        task_result: TaskResult,
        model_name: str | ModelMeta,
        model_revision: str | None = None,
    ) -> None:
        """Save the task results to the local cache directory in the location {model_name}/{model_revision}/{task_name}.json.

        Where model_name is a path-normalized model name.
        In addition we also save a model_meta.json in the revision folder to preserve the model metadata.

        Args:
            task_result: The results of the task.
            model_name: The name of the model as a valid directory name or a ModelMeta object.
            model_revision: The revision of the model. Must be specified if model_name is a string.
        """
        result_path = self.get_task_result_path(
            model_name=model_name,
            model_revision=model_revision,
            task_name=task_result.task_name,
        )
        result_path.parent.mkdir(parents=True, exist_ok=True)
        task_result.to_disk(result_path)

        model_meta_path = result_path.parent / "model_meta.json"
        if isinstance(model_name, ModelMeta):
            meta = model_name
            with model_meta_path.open("w") as f:
                json.dump(meta.to_dict(), f, default=str)

    @property
    def default_cache_path(self) -> Path:
        """Get the local cache directory for MTEB results.

        Returns:
            The path to the local cache directory.
        """
        default_cache_directory = Path.home() / ".cache" / "mteb"

        _cache_directory = os.environ.get("MTEB_CACHE", None)
        cache_directory = (
            Path(_cache_directory) if _cache_directory else default_cache_directory
        )
        return cache_directory

    def download_from_remote(
        self,
        remote: str = "https://github.com/embeddings-benchmark/results",
        download_latest: bool = True,
        revision: str | None = None,
    ) -> Path:
        """Downloads the latest version of the results repository from GitHub to a local cache directory. Required git to be installed.

        Args:
            remote: The URL of the results repository on GitHub.
            download_latest: If True it will download the latest version of the repository, otherwise it will only update the existing repository.
            revision: If specified, it will checkout the given revision after cloning or pulling the repository.

        Returns:
            The path to the local cache directory.
        """
        if not self.cache_path.exists() and not self.cache_path.is_dir():
            logger.info(
                f"Cache directory {self.cache_path} does not exist, creating it"
            )

        # if "results" folder already exists update it
        results_directory = self.cache_path / "remote"

        if results_directory.exists():
            # check repository in the directory is the same as the remote
            remote_url = subprocess.run(
                ["git", "config", "--get", "remote.origin.url"],
                cwd=results_directory,
                capture_output=True,
                text=True,
            ).stdout.strip()
            if remote_url != remote:
                msg = (
                    f"remote repository '{remote}' does not match the one in {results_directory},  which is '{remote_url}'."
                    + " Please remove the directory and try again."
                )
                raise ValueError(msg)

            if revision or download_latest:
                logger.info(
                    f"remote repository already exists in {results_directory}, fetching updates"
                )
                subprocess.run(
                    ["git", "fetch", "--all", "--tags"],
                    cwd=results_directory,
                    check=True,
                )
            else:
                logger.debug(
                    f"Results repository already exists in {results_directory}, skipping update, "
                    f"set download_latest=True to update it"
                )

            if revision:
                logger.info(f"Checking out revision '{revision}'")
                subprocess.run(
                    ["git", "checkout", revision],
                    cwd=results_directory,
                    check=True,
                )
            return results_directory

        logger.info(
            f"No results repository found in {results_directory}, cloning it from {remote}"
        )

        clone_cmd = ["git", "clone", "--depth", "1"]

        if revision:
            logger.info(f"Cloning repository at revision '{revision}'")
            clone_cmd.append(f"--revision={revision}")
        clone_cmd.extend([remote, "remote"])

        subprocess.run(
            clone_cmd,
            cwd=self.cache_path,
            check=True,
        )

        return results_directory

    def clear_cache(self) -> None:
        """Clear the local cache directory."""
        if self.cache_path.exists() and self.cache_path.is_dir():
            shutil.rmtree(self.cache_path)
            logger.info(f"Cache directory {self.cache_path} cleared.")
        else:
            msg = f"Cache directory `{self.cache_path}` does not exist."
            logger.warning(msg)
            warnings.warn(msg)

    def __repr__(self) -> str:
        return f"ResultCache(cache_path={self.cache_path})"

    def get_cache_paths(
        self,
        models: Sequence[str] | Iterable[ModelMeta] | None = None,
        tasks: Sequence[str] | Iterable[AbsTask] | None = None,
        require_model_meta: bool = True,
        include_remote: bool = True,
    ) -> list[Path]:
        """Get all paths to result JSON files in the cache directory.

        These paths can then be used to fetch task results, like:
        ```python
        for path in paths:
            task_result = TaskResult.from_disk(path)
        ```

        Args:
            models: A list of model names or ModelMeta objects to filter the paths.
            tasks: A list of task names to filter the paths.
            require_model_meta: If True, only return paths that have a model_meta.json file.
            include_remote: If True, include remote results in the returned paths.

        Returns:
            A list of paths in the cache directory.

        Examples:
            >>> from mteb.cache import ResultCache
            >>> cache = ResultCache()
            >>>
            >>> # Get all cache paths
            >>> paths = cache.get_cache_paths()
            >>>
            >>> # Get all cache paths for a specific task
            >>> paths = cache.get_cache_paths(tasks=["STS12"])
            >>>
            >>> # Get all cache paths for a specific model
            >>> paths = cache.get_cache_paths(models=["sentence-transformers/all-MiniLM-L6-v2"])
            >>>
            >>> # Get all cache paths for a specific model and revision
            >>> model_meta = mteb.get_model_meta("sentence-transformers/all-MiniLM-L6-v2")
            >>> paths = cache.get_cache_paths(models=[model_meta])
        """
        cache_paths = [
            p
            for p in (self.cache_path / "results").glob("**/*.json")
            if p.name != "model_meta.json"
        ]
        if include_remote:
            cache_paths += [
                p
                for p in (self.cache_path / "remote" / "results").glob("**/*.json")
                if p.name != "model_meta.json"
            ]

        cache_paths = self._filter_paths_by_model_and_revision(
            cache_paths,
            models=models,
        )
        cache_paths = self._filter_paths_by_task(cache_paths, tasks=tasks)

        if require_model_meta:
            cache_paths = [
                p for p in cache_paths if (p.parent / "model_meta.json").exists()
            ]
        return cache_paths

    def get_models(
        self,
        tasks: Sequence[str] | None = None,
        require_model_meta: bool = True,
        include_remote: bool = True,
    ) -> list[tuple[ModelName, Revision]]:
        """Get all models in the cache directory.

        Args:
            tasks: A list of task names to filter the models.
            require_model_meta: If True, only return models that have a model_meta.json file.
            include_remote: If True, include remote results in the returned models.

        Returns:
            A list of tuples containing the model name and revision.
        """
        cache_paths = self.get_cache_paths(
            tasks=tasks,
            require_model_meta=require_model_meta,
            include_remote=include_remote,
        )
        models = [(p.parent.parent.name, p.parent.name) for p in cache_paths]
        return list(set(models))

    def get_task_names(
        self,
        models: list[str] | list[ModelMeta] | None = None,
        require_model_meta: bool = True,
        include_remote: bool = True,
    ) -> list[str]:
        """Get all task names in the cache directory.

        Args:
            models: A list of model names or ModelMeta objects to filter the task names.
            require_model_meta: If True, only return task names that have a model_meta.json file
            include_remote: If True, include remote results in the returned task names.

        Returns:
            A list of task names in the cache directory.
        """
        cache_paths = self.get_cache_paths(
            models=models,
            require_model_meta=require_model_meta,
            include_remote=include_remote,
        )
        tasks = [p.stem for p in cache_paths]
        return list(set(tasks))

    @staticmethod
    def _get_model_name_and_revision_from_path(
        revision_path: Path,
    ) -> tuple[ModelName, Revision]:
        model_meta = revision_path / "model_meta.json"
        model_path = revision_path.parent

        if not model_meta.exists():
            logger.debug(
                f"model_meta.json not found in {revision_path}, extracting model_name and revision from the path"
            )
            model_name = model_path.name.replace("__", "/")
            revision = revision_path.name
            return model_name, revision
        with model_meta.open("r") as f:
            model_meta_json = json.load(f)
            model_name = model_meta_json["name"]
            revision = model_meta_json["revision"]
        return model_name, revision

    @staticmethod
    def _filter_paths_by_model_and_revision(
        paths: list[Path],
        models: Sequence[str] | Iterable[ModelMeta] | None = None,
    ) -> list[Path]:
        """Filter a list of paths by model name and optional revision.

        Returns:
            A list of paths that match the specified model names and revisions.
        """
        if not models:
            return paths

        first_model = next(iter(models))
        if isinstance(first_model, ModelMeta):
            models = cast(Iterable[ModelMeta], models)
            name_and_revision = {
                (m.model_name_as_path(), m.revision or "no_revision_available")
                for m in models
            }
            return [
                p
                for p in paths
                if (p.parent.parent.name, p.parent.name) in name_and_revision
            ]

        str_models = cast(Sequence[str], models)
        model_names = {m.replace("/", "__").replace(" ", "_") for m in str_models}
        return [p for p in paths if p.parent.parent.name in model_names]

    @staticmethod
    def _filter_paths_by_task(
        paths: list[Path],
        tasks: Sequence[str] | Iterable[AbsTask] | None = None,
    ) -> list[Path]:
        if tasks is not None:
            task_names = set()

            for task in tasks:
                if isinstance(task, AbsTask):
                    task_names.add(task.metadata.name)
                else:
                    task_names.add(task)

            paths = [p for p in paths if p.stem in task_names]
        return paths

    def load_results(
        self,
        models: Sequence[str] | Iterable[ModelMeta] | None = None,
        tasks: Sequence[str] | Iterable[AbsTask] | Benchmark | str | None = None,
        require_model_meta: bool = True,
        include_remote: bool = True,
        validate_and_filter: bool = False,
        only_main_score: bool = False,
    ) -> BenchmarkResults:
        """Loads the results from the cache directory and returns a BenchmarkResults object.

        Args:
            models: A list of model names to load the results for. If None it will load the results for all models.
            tasks: A list of task names to load the results for. If str is passed, then benchmark will be loaded.
                If Benchmark is passed, then all tasks in the benchmark will be loaded.
                If None it will load the results for all tasks.
            require_model_meta: If True it will ignore results that do not have a model_meta.json file. If false it attempt to
                extract the model name and revision from the path.
            include_remote: If True, it will include results from the remote repository.
            validate_and_filter: If True it will validate that the results object for the task contains the correct splits and filter out
                splits from the results object that are not default in the task metadata.
            only_main_score: If True, only the main score will be loaded.

        Returns:
            A BenchmarkResults object containing the results for the specified models and tasks.

        Examples:
            >>> from mteb.cache import ResultCache
            >>> cache = ResultCache()
            >>>
            >>> # Load results for specific models and tasks
            >>> results = cache.load_results(
            ...     models=["sentence-transformers/all-MiniLM-L6-v2"],
            ...     tasks=["STS12"],
            ...     require_model_meta=True,
            ... )
        """
        if isinstance(tasks, str):
            tasks = mteb.get_benchmark(tasks)

        paths = self.get_cache_paths(
            models=models,
            tasks=tasks,
            require_model_meta=require_model_meta,
            include_remote=include_remote,
        )
        models_results = defaultdict(list)

        task_names: dict[str, AbsTask | None] = {}
        if tasks is not None:
            for task in tasks:
                if isinstance(task, AbsTask):
                    task_names[task.metadata.name] = task
                else:
                    task_names[task] = None

        for path in paths:
            task_result = TaskResult.from_disk(path)

            if only_main_score:
                task_result = task_result.only_main_score()
            model_name, revision = self._get_model_name_and_revision_from_path(
                path.parent
            )

            if validate_and_filter:
                task_instance = task_names[task_result.task_name]
                try:
                    task_result = task_result.validate_and_filter_scores(
                        task=task_instance
                    )
                except Exception as e:
                    logger.info(
                        f"Validation failed for {task_result.task_name} in {model_name} {revision}: {e}"
                    )
                    continue

            models_results[(model_name, revision)].append(task_result)

        # create BenchmarkResults object
        models_results_object = [
            ModelResult(
                model_name=model_name,
                model_revision=revision,
                task_results=task_results,
            )
            for (model_name, revision), task_results in models_results.items()
        ]

        return BenchmarkResults(
            model_results=models_results_object,
            benchmark=tasks if isinstance(tasks, Benchmark) else None,
        )
