from __future__ import annotations

import gzip
import io
import json
import logging
import os
import shutil
import subprocess
import warnings
from collections import defaultdict
from pathlib import Path
from typing import TYPE_CHECKING, cast

import requests
from pydantic import ValidationError

import mteb
from mteb.abstasks import AbsTask
from mteb.benchmarks.benchmark import Benchmark
from mteb.models import ModelMeta
from mteb.results import BenchmarkResults, ModelResult, TaskResult

if TYPE_CHECKING:
    from collections.abc import Iterable, Sequence

    from mteb.types import ModelName, Revision

logger = logging.getLogger(__name__)


class ResultCache:
    """Class to handle the local cache of MTEB results.

    Examples:
        >>> import mteb
        >>> cache = mteb.ResultCache(cache_path="~/.cache/mteb") # default
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

    def _download_cached_results_from_branch(
        self,
        branch: str = "cached-data",
        filename: str = "__cached_results.json.gz",
        output_path: Path | None = None,
        remote: str = "https://github.com/embeddings-benchmark/results",
        timeout: int = 60,
        max_size_mb: int = 500,
    ) -> Path:
        """Download pre-computed cached results from a specific branch.

        This is significantly faster than download_from_remote() since it downloads
        only a compressed cache file instead of cloning the entire repository.

        The method performs the following steps:
        1. Downloads a gzipped JSON file from the specified branch
        2. Validates file size and content type
        3. Decompresses the gzip content
        4. Writes the decompressed JSON to disk

        Args:
            branch: Branch name to download from (default: "cached-data")
            filename: Name of the cached results file (default: "__cached_results.json.gz")
            output_path: Where to save the file. If None, uses mteb/leaderboard/__cached_results.json
            remote: Base URL of the results repository
            timeout: Request timeout in seconds (default: 60)
            max_size_mb: Maximum allowed file size in megabytes (default: 500)

        Returns:
            Path to the downloaded and decompressed cache file

        Raises:
            requests.exceptions.RequestException: On HTTP errors
            ValueError: On validation failures (size, content-type)
            gzip.BadGzipFile: If content is not valid gzip
            UnicodeDecodeError: If content cannot be decoded as UTF-8
            PermissionError: If file cannot be written due to permissions
            OSError: On other file system errors

        Examples:
            >>> import mteb
            >>> cache = mteb.ResultCache()
            >>> # Download optimized cached results
            >>> cache_file = cache._download_cached_results_from_branch()
            >>> # Use custom output path
            >>> cache_file = cache._download_cached_results_from_branch(
            ...     output_path=Path("/tmp/my_cache.json")
            ... )
        """
        if output_path is None:
            # Default to saving in mteb/leaderboard/__cached_results.json
            # Get the mteb package directory (parent of this file)
            mteb_package_dir = Path(__file__).parent
            output_path = mteb_package_dir / "leaderboard" / "__cached_results.json"

        # Extract repository owner and name from the remote URL
        # e.g., "https://github.com/embeddings-benchmark/results" -> "embeddings-benchmark/results"
        repo_path = remote.replace("https://github.com/", "").replace(
            "http://github.com/", ""
        )

        url = f"https://raw.githubusercontent.com/{repo_path}/{branch}/{filename}"
        logger.info(f"Downloading cached results from {url}")

        # Step 1: Download with validation
        max_size_bytes = max_size_mb * 1024 * 1024

        try:
            response = requests.get(url, timeout=timeout)
            response.raise_for_status()

            # Check if this is a Git LFS pointer file
            content_type = response.headers.get("content-type", "").lower()
            if (
                content_type == "text/plain; charset=utf-8"
                and b"git-lfs" in response.content
            ):
                # Try Git LFS media URL instead
                media_url = f"https://media.githubusercontent.com/media/{repo_path}/{branch}/{filename}"
                logger.info(f"Detected Git LFS file, trying media URL: {media_url}")
                response = requests.get(media_url, timeout=timeout)
                response.raise_for_status()
                content_type = response.headers.get("content-type", "").lower()

            # Validate content-type header
            expected_content_types = [
                "application/gzip",
                "application/octet-stream",
                "application/x-gzip",
            ]
            if content_type and not any(
                ct in content_type for ct in expected_content_types
            ):
                raise Exception(
                    f"Unexpected content-type: {content_type}. Expected one of: {expected_content_types}"
                )

            # Validate file size
            content_length = len(response.content)
            if content_length > max_size_bytes:
                raise ValueError(
                    f"Downloaded file too large: {content_length} bytes (max: {max_size_bytes})"
                )

            logger.info(
                f"HTTP request successful, content length: {content_length} bytes"
            )
            content = response.content

        except Exception as e:
            logger.error(f"Unexpected HTTP error: {type(e).__name__}: {e}")
            raise e

        # Step 2: Decompress gzip data
        logger.info("Attempting gzip decompression...")

        try:
            with gzip.open(io.BytesIO(content), "rt", encoding="utf-8") as gz_file:
                data = gz_file.read()
            logger.info(f"Decompression successful, data length: {len(data)} chars")

        except Exception as e:
            logger.error(f"Unexpected decompression error: {type(e).__name__}: {e}")
            raise e

        # Step 3: Write to disk
        logger.info(f"Attempting to write to: {output_path}")

        # Check parent directory exists and is writable
        output_path.parent.mkdir(parents=True, exist_ok=True)

        try:
            output_path.write_text(data, encoding="utf-8")
            logger.info(
                f"File write successful, size: {output_path.stat().st_size} bytes"
            )
        except Exception as e:
            logger.error(f"Unexpected file write error: {type(e).__name__}: {e}")
            raise e

        return output_path

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
            >>> import mteb
            >>> cache = mteb.ResultCache()
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
            models = cast("Iterable[ModelMeta]", models)
            name_and_revision = {
                (m.model_name_as_path(), m.revision or "no_revision_available")
                for m in models
            }
            return [
                p
                for p in paths
                if (p.parent.parent.name, p.parent.name) in name_and_revision
            ]

        str_models = cast("Sequence[str]", models)
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
            >>> import mteb
            >>> cache = mteb.ResultCache()
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
                except ValidationError as e:
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
