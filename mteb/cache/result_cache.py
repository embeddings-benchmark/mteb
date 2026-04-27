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
from collections.abc import Mapping
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

import requests
from pydantic import ValidationError

from mteb._helpful_enum import HelpfulStrEnum
from mteb._reversible_workflow.git_actions import (
    CommitAction,
    CreateBranchAction,
)
from mteb._reversible_workflow.reversible_workflow import (
    ReversibleAction,
    ReversibleWorkflow,
)
from mteb.abstasks import AbsTask
from mteb.benchmarks.benchmark import Benchmark
from mteb.benchmarks.get_benchmark import get_benchmark
from mteb.models import ModelMeta
from mteb.models.get_model_meta import get_model_metas
from mteb.models.model_meta import _serialize_experiment_kwargs_to_name
from mteb.results import BenchmarkResults, ModelResult, TaskResult
from mteb.types import SubmitResultsResponse

if TYPE_CHECKING:
    from collections.abc import Iterable, Sequence

    from mteb.types import ModelName, Revision, SubmitResultsResponse

logger = logging.getLogger(__name__)
_EXPERIMENTS_FOLDER_NAME = "experiments"


class CopyResultsAction(ReversibleAction):
    """Copy selected result files and optional model metadata files to the remote repo."""

    def __init__(
        self, unsubmitted: dict[ModelMeta, list[Path]], remote_path: Path
    ) -> None:
        """Initialize the action.

        Args:
            unsubmitted: Dict mapping ModelMeta to list of result file paths.
            remote_path: Path to the remote repository.
        """
        self.unsubmitted = unsubmitted
        self.remote_path = remote_path
        self.copied_files: list[Path] = []

    def do(self) -> None:
        """Copy listed json result files and optional model_meta.json to remote paths."""
        for model_meta, result_files in self.unsubmitted.items():
            if model_meta.name is None or model_meta.revision is None:
                logger.warning(
                    f"Skipping model with None name or revision: {model_meta}"
                )
                continue

            model_name_path = model_meta.model_name_as_path()
            revision = model_meta.revision
            dest_dir = self.remote_path / model_name_path / revision
            dest_dir.mkdir(parents=True, exist_ok=True)

            for result_file in result_files:
                dest_file = dest_dir / result_file.name
                shutil.copy2(result_file, dest_file)
                self.copied_files.append(dest_file)
                logger.debug(f"Copied {result_file} to {dest_file}")

            # Copy model_meta.json if it exists in the source directory
            source_model_dir = result_files[0].parent if result_files else None
            if source_model_dir and source_model_dir.exists():
                model_meta_file = source_model_dir / "model_meta.json"
                if model_meta_file.exists():
                    dest_model_meta = dest_dir / "model_meta.json"
                    shutil.copy2(model_meta_file, dest_model_meta)
                    self.copied_files.append(dest_model_meta)
                    logger.debug(f"Copied {model_meta_file} to {dest_model_meta}")

        logger.info(f"Copied {len(self.copied_files)} files to remote")

    def undo(self) -> None:
        """Deletion of files copied during do()."""
        for file_path in self.copied_files:
            try:
                file_path.unlink()
                logger.debug(f"Deleted {file_path}")
            except Exception as e:
                logger.warning(f"Failed to delete {file_path}: {e}")

        logger.info(f"Deleted {len(self.copied_files)} copied files")


class LoadExperimentEnum(HelpfulStrEnum):
    """Enum to specify whether to load experiments when loading results from the cache.

    Attributes:
        MATCH_NAME: Will load everything that matches the name of the model, including experiments. If a revision is supplied using `ModelMeta` this will also match the revision.
        MATCH_KWARGS: Will load experiments that match the keyword arguments supplied in the`ModelMeta`. Assumes a `ModelMeta`s are supplied.
        NO_EXPERIMENTS: Will only load models with default keyword arguments, meaning that it will not include any experiments.
    """

    MATCH_NAME = "match_name"
    MATCH_KWARGS = "match_kwargs"
    NO_EXPERIMENTS = "no_experiments"


class ResultCache:
    """Class to handle the local cache of MTEB results.

    Examples:
        >>> import mteb
        >>> cache = mteb.ResultCache(cache_path="~/.cache/mteb")  # default
        >>> cache.download_from_remote()  # download the latest results from the remote repository
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

    @property
    def remote_results_path(self) -> Path:
        """Get the path to the remote results directory.

        Returns:
            The path to the remote results directory.
        """
        return self.cache_path / "remote" / "results"

    def get_task_result_path(
        self,
        task_name: str,
        model_name: str | ModelMeta,
        model_revision: str | None = None,
        remote: bool = False,
        experiment_name: str | None = None,
    ) -> Path:
        """Get the path to the results of a specific task for a specific model and revision.

        Args:
            task_name: The name of the task.
            model_name: The name of the model as a valid directory name or a ModelMeta object.
            model_revision: The revision of the model. Must be specified if model_name is a string.
            remote: If True, it will return the path to the remote results repository, otherwise it will return the path to the local results repository.
            experiment_name: The name of the experiment as a valid directory name. If model_name is a ModelMeta object, its experiment_name will be used.

        Returns:
            The path to the results of the task.
        """
        results_folder = (
            self.cache_path / "results" if not remote else self.remote_results_path
        )

        if isinstance(model_name, ModelMeta):
            if model_revision is not None:
                logger.warning(
                    "model_revision and experiment_name is ignored when model_name is a ModelMeta object"
                )
            model_revision = model_name.revision
            experiment_name = model_name.experiment_name
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

        if experiment_name:
            return (
                model_path
                / model_revision
                / _EXPERIMENTS_FOLDER_NAME
                / experiment_name
                / f"{task_name}.json"
            )
        return model_path / model_revision / f"{task_name}.json"

    def load_task_result(
        self,
        task_name: str,
        model_name: str | ModelMeta,
        model_revision: str | None = None,
        raise_if_not_found: bool = False,
        prioritize_remote: bool = False,
        experiment_name: str | None = None,
    ) -> TaskResult | None:
        """Load the results from the local cache directory.

        Args:
            task_name: The name of the task.
            model_name: The name of the model as a valid directory name or a ModelMeta object.
            model_revision: The revision of the model. Must be specified if model_name is a string.
            raise_if_not_found: If True, raise an error if the results are not found.
            prioritize_remote: If True, it will first try to load the results from the remote repository, if available.
            experiment_name: Optional experiment folder name (a valid directory name). If None, the default is used.

        Returns:
            The results of the task, or None if not found.
        """
        result_path = self.get_task_result_path(
            model_name=model_name,
            model_revision=model_revision,
            task_name=task_name,
            experiment_name=experiment_name,
        )

        if self.has_remote:
            remote_result_path = self.get_task_result_path(
                model_name=model_name,
                model_revision=model_revision,
                task_name=task_name,
                remote=True,
                experiment_name=experiment_name,
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
                json.dump(meta.to_dict(), f, default=str, indent=4)

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
        if not self.cache_path.exists():
            logger.info(
                f"Cache directory {self.cache_path} does not exist, creating it"
            )
            self.cache_path.mkdir(parents=True, exist_ok=True)

        # if "results" folder already exists update it
        results_directory = self.cache_path / "remote"

        if results_directory.exists():
            # check repository in the directory is the same as the remote
            remote_url = subprocess.run(
                ["git", "config", "--get", "remote.origin.url"],
                check=False,
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
                    text=True,
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
                    text=True,
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
            text=True,
        )

        return results_directory

    def _download_cached_results_from_branch(
        self,
        *,
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
            output_path: Where to save the file. If None, uses {cache_path}/leaderboard/__cached_results.json
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
            # Default to saving in {cache_path}/leaderboard/__cached_results.json
            output_path = self.cache_path / "leaderboard" / "__cached_results.json"

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

    def _load_from_cache(
        self,
        cache_filename: str = "__cached_results.json",
        rebuild: bool = False,
    ) -> BenchmarkResults:
        """Load benchmark results using the best available strategy.

        Args:
            cache_filename: Name of the cache file. The full path will be constructed as
                {cache_path}/leaderboard/{cache_filename}.
            rebuild: If True, force a full rebuild from the results repository, bypassing any
                     pre-computed JSON cache.

        Strategy:
            1. If rebuild=False and local cache exists at cache_path → load and return
            2. If rebuild=False, try downloading pre-computed cache from 'cached-data' branch
               → save to cache_path and return
            3. Fallback (or if rebuild=True): clone the full results repository, build from
               individual model files, call results.to_disk(cache_path), and return.

        Returns:
            BenchmarkResults ready for leaderboard display
        """
        cache_path = self.cache_path / "leaderboard" / cache_filename

        # If rebuild=True, skip directly to full repository rebuild
        if rebuild:
            logger.info(
                "Rebuild requested, forcing full repository clone and rebuild..."
            )
            return self._rebuild_from_full_repository(cache_path)

        # Strategy 1: Try loading from existing local quick cache
        if cache_path.exists():
            logger.info(f"Loading existing quick cache from {cache_path}")
            try:
                return BenchmarkResults.from_disk(cache_path)
            except Exception as e:
                logger.warning(
                    f"Failed to load quick cache: {e}. Trying other strategies..."
                )

        # Strategy 2: Try downloading from cached-data branch
        try:
            logger.info(
                "Attempting to download pre-computed cache from cached-data branch..."
            )
            downloaded_path = self._download_cached_results_from_branch(
                output_path=cache_path
            )
            logger.info(f"Downloaded cache to {downloaded_path}")
            return BenchmarkResults.from_disk(downloaded_path)
        except Exception as e:
            logger.warning(f"Failed to download from cached-data branch: {e}")

        # Strategy 3: Fallback to full repository clone
        logger.info("Falling back to full repository clone and rebuild...")
        return self._rebuild_from_full_repository(cache_path)

    def _rebuild_from_full_repository(self, quick_cache_path: Path) -> BenchmarkResults:
        """Clone/pull the full results repository and build BenchmarkResults from individual files.

        This method performs a full rebuild by:
        1. Downloading or updating the full results repository
        2. Loading results from all individual model files
        3. Saving the aggregated results to the quick cache path
        4. Returning the BenchmarkResults object

        Args:
            quick_cache_path: Path where the rebuilt cache should be saved

        Returns:
            BenchmarkResults built from the full repository
        """
        # Download or update the full repository
        self.download_from_remote()

        all_model_names = [
            model_meta.name
            for model_meta in get_model_metas()
            if model_meta.name is not None
        ]

        all_results = self.load_results(
            models=all_model_names,
            only_main_score=True,
            require_model_meta=False,
            include_remote=True,
        )

        # Save to disk for future use
        logger.info(f"Saving rebuilt cache to {quick_cache_path}")
        all_results.to_disk(quick_cache_path)

        return all_results

    def __repr__(self) -> str:
        return f"ResultCache(cache_path={self.cache_path})"

    def get_cache_paths(
        self,
        models: Sequence[str] | Iterable[ModelMeta] | None = None,
        tasks: Sequence[str] | Iterable[AbsTask] | None = None,
        require_model_meta: bool = True,
        include_remote: bool = True,
        load_experiments: LoadExperimentEnum | str = LoadExperimentEnum.NO_EXPERIMENTS,
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
            load_experiments: If True, include experiments in the returned paths.

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
        if isinstance(load_experiments, str):
            load_experiments = LoadExperimentEnum.from_str(load_experiments)

        def _cache_paths(base_path: Path) -> list[Path]:
            return [
                p
                for p in base_path.glob("*/*/*.json")  # model/revision/task.json
                if p.name != "model_meta.json"
            ]

        def _experiments_paths(base_path: Path) -> list[Path]:
            return [
                p
                for p in base_path.glob(f"*/*/{_EXPERIMENTS_FOLDER_NAME}/*/*.json")
                if p.name != "model_meta.json"
            ]

        def _get_paths(base_path: Path, experiments: LoadExperimentEnum) -> list[Path]:
            paths = _cache_paths(base_path)
            if not experiments == LoadExperimentEnum.NO_EXPERIMENTS:
                paths += _experiments_paths(base_path)
            return paths

        results_path = self.cache_path / "results"
        remote_path = self.remote_results_path

        cache_paths = _get_paths(results_path, load_experiments)

        if include_remote:
            cache_paths += _get_paths(remote_path, load_experiments)

        cache_paths = self._filter_paths_by_model_and_revision(
            cache_paths,
            models=models,
            load_experiments=load_experiments,
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
    ) -> tuple[ModelName, Revision, str | None]:
        """Get model name, revision and experiment name from the given path.

        Args:
            revision_path: The path to the revision folder, which should contain a model_meta.json file. If the file is not found, it will attempt to extract the model name and revision from the path.

        Returns:
            A tuple containing the model name, revision and experiment name (if available).

        """
        model_meta = revision_path / "model_meta.json"
        model_path = revision_path.parent

        if not model_meta.exists():
            logger.debug(
                f"model_meta.json not found in {revision_path}, extracting model_name and revision from the path"
            )
            if _EXPERIMENTS_FOLDER_NAME in revision_path.parts:
                logger.debug(
                    f"Path {revision_path} contains an experiment folder, extracting model_name and revision accordingly"
                )
                experiment_name = revision_path.name
                revision = revision_path.parent.parent.name
                model_name = revision_path.parent.parent.parent.name.replace("__", "/")
                return model_name, revision, experiment_name
            model_name = model_path.name.replace("__", "/")
            revision = revision_path.name
            return model_name, revision, None
        with model_meta.open("r") as f:
            model_meta_json = json.load(f)
        model_name = model_meta_json["name"]
        revision = model_meta_json["revision"]
        experiment_kwargs = model_meta_json.get("experiment_kwargs", None)
        experiment_name_ = _serialize_experiment_kwargs_to_name(experiment_kwargs)
        return model_name, revision, experiment_name_

    @staticmethod
    def _filter_paths_by_model_and_revision(
        paths: list[Path],
        models: Sequence[str] | Iterable[ModelMeta] | None = None,
        load_experiments: LoadExperimentEnum | None = None,
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
                (
                    m.model_name_as_path(),
                    m.revision or "no_revision_available",
                    m.experiment_name
                    if load_experiments is LoadExperimentEnum.MATCH_KWARGS
                    else None,
                )
                for m in models
            }
            model_name_and_revision = list()
            for path in paths:
                if _EXPERIMENTS_FOLDER_NAME in path.parts:
                    revision = path.parent.parent.parent.name
                    model_name = path.parent.parent.parent.parent.name
                    experiment_name = (
                        path.parent.name
                        if load_experiments is LoadExperimentEnum.MATCH_KWARGS
                        else None
                    )
                else:
                    revision = path.parent.name
                    model_name = path.parent.parent.name
                    experiment_name = None
                model_name_and_revision.append((model_name, revision, experiment_name))
            return [
                p
                for model_revision, p in zip(model_name_and_revision, paths)
                if model_revision in name_and_revision
            ]

        str_models = cast("Sequence[str]", models)
        model_names = {m.replace("/", "__").replace(" ", "_") for m in str_models}
        filtered_paths = []
        for p in paths:
            if _EXPERIMENTS_FOLDER_NAME in p.parts:
                model_name = p.parent.parent.parent.parent.name
            else:
                model_name = p.parent.parent.name
            if model_name in model_names:
                filtered_paths.append(p)
        return filtered_paths

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

    def _load_model_meta_from_cache(
        self,
        model_name: str,
        revision: str,
    ) -> ModelMeta | None:
        """Load ModelMeta from cache directory.

        Args:
            model_name: The model name.
            revision: The model revision.

        Returns:
            ModelMeta object if found, None otherwise.
        """
        model_name_path = model_name.replace("/", "__").replace(" ", "_")
        meta_file = (
            self.cache_path / "results" / model_name_path / revision / "model_meta.json"
        )

        if not meta_file.exists():
            logger.warning(
                f"model_meta.json not found for {model_name} (revision: {revision})"
            )
            return None

        try:
            with meta_file.open("r") as f:
                meta_dict = json.load(f)
            return ModelMeta(**meta_dict)
        except Exception as e:
            logger.warning(f"Failed to load ModelMeta from {meta_file}: {e}")
            return None

    def _normalize_models(
        self,
        models: list[str] | list[ModelMeta] | str | ModelMeta | None = None,
    ) -> list[ModelMeta]:
        """Normalize model input to list of ModelMeta objects.

        Args:
            models: Model(s) to normalize. Can be:
                - None: get all models from local cache
                - str: single model name
                - ModelMeta: single model metadata object
                - list[str]: list of model names
                - list[ModelMeta]: list of model metadata objects

        Returns:
            List of ModelMeta objects.

        Raises:
            ValueError: If no models found or invalid input.
        """
        if models is None:
            local_models = self.get_models(
                require_model_meta=True, include_remote=False
            )
            if not local_models:
                raise ValueError(
                    "No models found in local cache. Please evaluate models first."
                )
            normalized = []
            for model_name, revision in local_models:
                model_meta = self._load_model_meta_from_cache(model_name, revision)
                if model_meta:
                    normalized.append(model_meta)
            return normalized

        if isinstance(models, (str, ModelMeta)):
            models_to_process: list[str | ModelMeta] = [models]
        else:
            models_to_process = cast("list[str | ModelMeta]", models)

        normalized = []
        for model in models_to_process:
            if isinstance(model, ModelMeta):
                if model.revision is None or model.name is None:
                    raise ValueError(
                        f"ModelMeta {model.name} has no revision or name. "
                        "Cannot submit results without both."
                    )
                normalized.append(model)
            elif isinstance(model, str):
                local_models = self.get_models(
                    require_model_meta=False, include_remote=False
                )
                matching = [
                    (name, rev)
                    for name, rev in local_models
                    if name == model.replace("/", "__")
                ]
                if not matching:
                    raise ValueError(
                        f"Model '{model}' not found in local cache. "
                        "Please evaluate it first."
                    )
                for model_name, revision in matching:
                    model_meta = self._load_model_meta_from_cache(model_name, revision)
                    if model_meta:
                        normalized.append(model_meta)
            else:
                raise TypeError(f"Invalid model type: {type(model)}")

        if not normalized:
            raise ValueError("No valid models to submit.")

        return normalized

    def _get_unsubmitted_results(
        self,
        models: list[ModelMeta],
    ) -> dict[ModelMeta, list[Path]]:
        """Find unsubmitted results by comparing local vs remote.

        Args:
            models: List of ModelMeta objects.

        Returns:
            Dict mapping ModelMeta to list of unsubmitted result file paths.
        """
        unsubmitted: dict[ModelMeta, list[Path]] = {}

        local_paths = self.get_cache_paths(
            models=models,
            require_model_meta=False,
            include_remote=False,
        )
        remote_files_set: set[Path] = set()
        for model in models:
            if model.name is None or model.revision is None:
                logger.warning(f"Skipping model with None name or revision: {model}")
                continue

            model_name_path = model.model_name_as_path()
            remote_results_dir = (
                self.remote_results_path / model_name_path / model.revision
            )

            if remote_results_dir.exists():
                remote_files_set.update(
                    f.relative_to(remote_results_dir)
                    for f in remote_results_dir.rglob("*.json")
                    if f.name != "model_meta.json"
                )

        for local_path in local_paths:
            model_name_path = local_path.parent.parent.name
            revision = local_path.parent.name
            model_name = model_name_path.replace("__", "/").replace("_", " ")
            relative_path = local_path.name

            if relative_path not in remote_files_set:
                model_meta = None
                for m in models:
                    if m.name == model_name and m.revision == revision:
                        model_meta = m
                        break

                if model_meta is not None:
                    if model_meta not in unsubmitted:
                        unsubmitted[model_meta] = []
                    unsubmitted[model_meta].append(local_path)

        return unsubmitted

    @staticmethod
    def _prepare_pr_body(
        models: list[ModelMeta],
        unsubmitted: dict[ModelMeta, list[Path]],
    ) -> str:
        """Prepare the pull request body with results summary.

        Args:
            models: List of ModelMeta objects.
            unsubmitted: Dict mapping ModelMeta to list of result file paths.

        Returns:
            Formatted PR body string.
        """
        model_details = []
        total_results = 0

        for model in models:
            if model in unsubmitted:
                result_count = len(unsubmitted[model])
                total_results += result_count
                model_details.append(
                    f"- **{model.name}** (revision: `{model.revision}`): {result_count} results"
                )

        model_details_str = "\n".join(model_details)

        checklist = """### Checklist
- [ ] My model has a model sheet, report, or similar
- [ ] My model has a reference implementation in [`mteb/models/model_implementations/`](https://github.com/embeddings-benchmark/mteb/tree/main/mteb/models/model_implementations), this can be as an API. Instruction on how to add a model can be found [here](https://embeddings-benchmark.github.io/mteb/contributing/adding_a_model/)
  - [ ] No, but there is an existing PR ___
- [ ] The results submitted are obtained using the reference implementation
- [ ] My model is available, either as a publicly accessible API or publicly on e.g., Huggingface
- [ ] I *solemnly swear* that for all results submitted I have not trained on the evaluation dataset including training splits. If I have, I have disclosed it clearly."""

        body = f"""### Models Submitted
{model_details_str}

**Total Results:** {total_results}

---

*This PR was created automatically using [`ResultCache.submit_results()`](https://embeddings-benchmark.github.io/mteb/get_started/advanced_usage/result_cache/#submit_results). Please check the results carefully before merging.*

{checklist}"""

        logger.info("\n📋 Please complete the checklist in the PR body before merging.")
        return body

    @staticmethod
    def _check_uncommitted_changes(repo_path: Path) -> None:
        """Detect staged/uncommitted changes that would corrupt result submission."""
        try:
            result = subprocess.run(
                ["git", "status", "--porcelain"],
                cwd=repo_path,
                check=True,
                capture_output=True,
                text=True,
            )

            if result.stdout.strip():
                raise RuntimeError(
                    f"Repository has uncommitted changes:\n{result.stdout.strip()}\n"
                    "Please commit or clean these changes before submitting."
                )
        except subprocess.CalledProcessError as e:
            logger.warning(f"Could not check uncommitted changes: {e}")

    @staticmethod
    def _check_detached_head(repo_path: Path) -> None:
        """Check if repository is in detached HEAD state.

        In detached HEAD state, branch operations fail and state is confusing.

        Args:
            repo_path: Path to the git repository.

        Raises:
            RuntimeError: If in detached HEAD state.
        """
        try:
            result = subprocess.run(
                ["git", "symbolic-ref", "-q", "HEAD"],
                check=False,
                cwd=repo_path,
                capture_output=True,
                text=True,
            )

            if result.returncode != 0:
                # Non-zero return = detached HEAD
                raise RuntimeError(
                    "Repository is in detached HEAD state. "
                    "Please checkout a branch before submitting results:\n"
                    "  git checkout main    # Checkout main branch\n"
                    "  OR\n"
                    "  git checkout -b my-branch  # Create new branch"
                )
        except subprocess.CalledProcessError as e:
            logger.warning(f"Could not check HEAD state: {e}")

    def _run_preflight_checks(self, repo_path: Path) -> None:
        """Run all pre-flight validations before submission workflow.

        These checks prevent common issues like uncommitted changes or detached HEAD
        that would corrupt the repository or cause the workflow to fail mid-way without proper rollback.

        Args:
            repo_path: Path to the git repository.

        Raises:
            RuntimeError: If any validation fails.
        """
        logger.info("Running pre-flight checks...")
        self._check_uncommitted_changes(repo_path)
        self._check_detached_head(repo_path)
        logger.info("Pre-flight checks passed ✓")

    @staticmethod
    def _get_current_branch(repo_path: Path) -> str:
        """Get the current branch name.

        Args:
            repo_path: Path to the git repository.

        Returns:
            Current branch name.

        Raises:
            RuntimeError: If unable to determine current branch.
        """
        try:
            result = subprocess.run(
                ["git", "rev-parse", "--abbrev-ref", "HEAD"],
                cwd=repo_path,
                check=True,
                capture_output=True,
                text=True,
                timeout=10,
            )
            branch = result.stdout.strip()
            logger.debug(f"Current branch: {branch}")
            return branch
        except subprocess.CalledProcessError as e:
            raise RuntimeError(f"Failed to get current branch: {e}")

    @staticmethod
    def _restore_branch(repo_path: Path, original_branch: str) -> None:
        """Restore to the original branch after successful PR creation.

        Args:
            repo_path: Path to the git repository.
            original_branch: Name of the branch to restore to.

        Raises:
            RuntimeError: If restoration fails.
        """
        try:
            subprocess.run(
                ["git", "checkout", original_branch],
                cwd=repo_path,
                check=True,
                capture_output=True,
                text=True,
                timeout=10,
            )
            logger.info(f"Restored to original branch '{original_branch}'")
        except subprocess.CalledProcessError as e:
            raise RuntimeError(f"Failed to restore to branch '{original_branch}': {e}")

    @staticmethod
    def _delete_branch(repo_path: Path, branch_name: str) -> None:
        """Delete a git branch to clean up after failed PR creation.

        Args:
            repo_path: Path to the git repository.
            branch_name: Name of the branch to delete.
        """
        try:
            subprocess.run(
                ["git", "branch", "-D", branch_name],
                cwd=repo_path,
                check=True,
                capture_output=True,
                text=True,
                timeout=10,
            )
            logger.info(f"Deleted temporary branch '{branch_name}'")
        except subprocess.CalledProcessError as e:
            logger.warning(f"Failed to delete temporary branch '{branch_name}': {e}")

    @staticmethod
    def _build_manual_submission_message(
        commit_sha: str, remote_path: Path, result_count: int, model_count: int
    ) -> str:
        """Build the manual submission instructions message.

        Args:
            commit_sha: The git commit SHA.
            remote_path: Path to the remote repository.
            result_count: Number of result files submitted.
            model_count: Number of models submitted.

        Returns:
            Formatted submission instructions as a single string.
        """
        lines = [
            "\n" + "=" * 80,
            f"✓ Commit created with {result_count} results for {model_count} model(s)",
            "=" * 80,
            f"\nCommit SHA: {commit_sha}",
            f"Location: {remote_path}",
            "\n📋 To submit these results, follow these steps:\n",
            "1. Go to the remote repository:",
            f"   {remote_path}\n",
            "2. Create a fork (if you don't have one already):",
            "   gh repo fork --remote --remote-name fork --clone=false\n",
            "3. Push your changes to your fork:",
            "   git push fork\n",
            "4. Create a pull request:",
            "   gh pr create --base main --head <your-username>:main\n",
            "5. Provide details about your evaluation in the PR description\n",
            "=" * 80,
        ]
        return "\n".join(lines)

    def submit_results(
        self,
        models: list[str] | list[ModelMeta] | str | ModelMeta | None = None,
        *,
        push: bool = False,
    ) -> SubmitResultsResponse:
        """Create a commit of the results to the official MTEB results repository (https://github.com/embeddings-benchmark/results).

        It does this by downloading the remote (if not downloaded already) and
        submitting the diff from the local result to the repository. Requires PyGithub
        to be installed if `push=True`.

        Args:
            models: Model(s) whose results should be submitted. Can be:
                - None: submit all unsubmitted results
                - str or ModelMeta: single model
                - list[str] or list[ModelMeta]: multiple models
            push: If True, create a PR directly to the remote. If False, prints
                  instructions for manual submission.

        Returns:
            Dictionary containing submission metadata:
                - status: "ready_for_submission" or "pr_created"
                - models_submitted: list of (model_name, revision) tuples
                - result_count: number of result files submitted
                - commit_sha: git commit hash
                - pr_url: URL to created PR (only if push=True)
                - pr_number: PR number (only if push=True)
                - fork_url: URL to user's fork (only if push=True)

        Raises:
            ValueError: If no models found or invalid input.
            RuntimeError: If git operations fail.
            ImportError: If push=True and PyGithub is not installed.
            GithubException: If GitHub API operations fail.

        Examples:
            >>> import mteb
            >>> cache = mteb.ResultCache()
            >>> results = mteb.evaluate(model, tasks, cache=cache)
            >>>
            >>> # Manual submission (step-by-step)
            >>> submission = cache.submit_results(model, push=False)
            >>> # Follow printed instructions
            >>>
            >>> # Automated submission
            >>> submission = cache.submit_results(model, push=True)
            >>> print(f"PR created: {submission['pr_url']}")
        """
        branch_name = (
            f"mteb-results-{int(datetime.now().timestamp())}" if push else None
        )
        try:
            normalized_models = self._normalize_models(models)
            self.download_from_remote()
            unsubmitted = self._get_unsubmitted_results(normalized_models)

            if not unsubmitted:
                logger.warning("No unsubmitted results found.")
                return cast(
                    "SubmitResultsResponse",
                    {
                        "status": "no_changes",
                        "models_submitted": [
                            (m.name, m.revision) for m in normalized_models
                        ],
                        "result_count": 0,
                        "commit_sha": None,
                    },
                )

            remote_path = self.cache_path / "remote"
            self._run_preflight_checks(remote_path)

            # Capture original branch before making any changes
            original_branch = self._get_current_branch(remote_path)

        except RuntimeError as e:
            logger.error(f"Setup error during submit_results: {e}")
            raise
        except Exception as e:
            logger.error(f"Error during submit_results setup: {e}")
            raise

        actions: list[ReversibleAction] = []

        copy_action = CopyResultsAction(unsubmitted, self.remote_results_path)
        actions.append(copy_action)

        model_str = ", ".join(model.name for model in normalized_models if model.name)
        result_count = sum(len(files) for files in unsubmitted.values())
        commit_message = (
            f"Add MTEB evaluation results for {model_str}\n\n"
            f"Models: {model_str}\n"
            f"Total results: {result_count}\n"
            f"Submitted by MTEB ResultCache"
        )

        commit_action = CommitAction(remote_path, commit_message)
        actions.append(commit_action)

        if push and branch_name:
            actions.append(
                CreateBranchAction(remote_path, branch_name, original_branch)
            )

        workflow = ReversibleWorkflow(steps=actions)
        try:
            workflow.run()
        except Exception as e:
            logger.error(f"Workflow error during submit_results: {e}")
            raise

        commit_sha = commit_action.commit_sha
        if not commit_sha:
            raise RuntimeError("Failed to create commit: commit_sha is None")

        if not push:
            message = self._build_manual_submission_message(
                commit_sha, remote_path, result_count, len(normalized_models)
            )
            for line in message.split("\n"):
                logger.info(line)

            return cast(
                "SubmitResultsResponse",
                {
                    "status": "ready_for_submission",
                    "models_submitted": [
                        (m.name, m.revision) for m in normalized_models
                    ],
                    "result_count": result_count,
                    "commit_sha": commit_sha,
                    "path": str(remote_path),
                },
            )

        return self._handle_pr_creation_with_cleanup(
            commit_sha,
            normalized_models,
            unsubmitted,
            result_count,
            branch_name,
            remote_path,
            original_branch,
        )

    @staticmethod
    def _get_github_token() -> str:
        """Get GitHub token using gh CLI authentication.

        Returns:
            GitHub token string

        Raises:
            RuntimeError: If authentication fails.
        """
        try:
            result = subprocess.run(
                ["gh", "auth", "token"],
                check=False,
                capture_output=True,
                text=True,
                timeout=10,
            )
            if result.returncode == 0 and result.stdout.strip():
                logger.debug("Using token from gh auth")
                return result.stdout.strip()
        except FileNotFoundError:
            logger.debug("gh CLI not found, trying git credential helper")
        except Exception as e:
            logger.debug(f"Failed to get token from gh auth: {e}")

        try:
            result = subprocess.run(
                ["git", "credential", "fill"],
                check=False,
                input="protocol=https\nhost=github.com\n\n",
                capture_output=True,
                text=True,
                timeout=10,
            )
            if result.returncode == 0:
                for line in result.stdout.split("\n"):
                    if line.startswith("password="):
                        logger.debug("Using token from git credential helper")
                        return line.split("=", 1)[1]
        except Exception as e:
            logger.debug(f"Failed to get token from git credential: {e}")

        raise RuntimeError(
            "GitHub token not found. Please set up gh CLI (gh auth login) or "
            "configure git credential helper."
        )

    def _handle_pr_creation_with_cleanup(
        self,
        commit_sha: str,
        models: list[ModelMeta],
        unsubmitted: dict[ModelMeta, list[Path]],
        result_count: int,
        branch_name: str | None,
        remote_path: Path,
        original_branch: str,
    ) -> SubmitResultsResponse:
        """Create a pull request with proper cleanup on failure.

        Args:
            commit_sha: The commit SHA to reference.
            models: List of ModelMeta objects.
            unsubmitted: Dict mapping ModelMeta to list of result file paths.
            result_count: Total number of results.
            branch_name: Name of the branch to create.
            remote_path: Path to the remote repository.
            original_branch: Original branch name for restoration.

        Returns:
            Dictionary with PR information.

        Raises:
            RuntimeError: If PR creation or cleanup fails.
        """
        try:
            result = self._create_pull_request(
                commit_sha,
                models,
                unsubmitted,
                result_count,
                branch_name=branch_name,
            )
            # After successful PR, restore to original branch
            self._restore_branch(remote_path, original_branch)
            return result
        except Exception as e:
            # PR creation failed, but workflow.run() already completed
            # Restore to original branch and delete temporary branch to clean up
            logger.error(f"PR creation failed: {e}")

            try:
                self._restore_branch(remote_path, original_branch)
            except Exception as restore_error:
                logger.error(f"Failed to restore branch on error: {restore_error}")
                logger.warning(
                    f"You may be on branch '{branch_name}'. "
                    f"To restore, run: git checkout {original_branch}"
                )

            if branch_name:
                try:
                    self._delete_branch(remote_path, branch_name)
                except Exception as delete_error:
                    logger.error(f"Failed to delete branch: {delete_error}")

            raise

    def _create_pull_request(
        self,
        commit_sha: str,
        models: list[ModelMeta],
        unsubmitted: dict[ModelMeta, list[Path]],
        result_count: int,
        branch_name: str | None,
    ) -> SubmitResultsResponse:
        """Create a pull request on GitHub using PyGithub.

        Args:
            commit_sha: The commit SHA to reference.
            models: List of ModelMeta objects.
            unsubmitted: Dict mapping ModelMeta to list of result file paths.
            result_count: Total number of results.
            branch_name: Name of the branch to create.

        Returns:
            Dictionary with PR information.

        Raises:
            RuntimeError: If authentication fails.
            GithubException: If GitHub API call fails.
        """
        try:
            from github import (  # type: ignore[import-not-found]
                Auth,
                Github,
                GithubException,
            )
        except ImportError:
            raise ImportError(
                "PyGithub is not installed. Please install it using `pip install 'mteb[pygithub]'`"
            )

        logger.info("Creating PR using PyGithub")
        token = self._get_github_token()

        try:
            auth = Auth.Token(token)
            gh = Github(auth=auth)
            user = gh.get_user()
        except Exception as e:
            raise RuntimeError(f"Failed to authenticate with GitHub: {e}") from e

        try:
            upstream = gh.get_repo("embeddings-benchmark/results")
            logger.info("Connected to upstream: embeddings-benchmark/results")
        except Exception as e:
            raise RuntimeError(f"Failed to access upstream repository: {e}") from e

        remote_path = self.cache_path / "remote"
        fork_url = None
        try:
            logger.info("Creating/configuring fork using gh CLI...")
            subprocess.run(
                [
                    "gh",
                    "repo",
                    "fork",
                    "--remote",
                    "--remote-name",
                    "fork",
                ],
                cwd=remote_path,
                check=True,
                capture_output=True,
                text=True,
                timeout=30,
            )
            logger.info("Fork created/configured")

            # Get fork URL from gh CLI
            result = subprocess.run(
                ["gh", "repo", "view", "--json", "url"],
                check=False,
                capture_output=True,
                text=True,
                timeout=10,
            )
            if result.returncode == 0:
                fork_data = json.loads(result.stdout)
                fork_url = fork_data.get(
                    "url", f"https://github.com/{user.login}/results"
                )
            else:
                fork_url = f"https://github.com/{user.login}/results"

            logger.info(f"Using fork: {fork_url}")
        except subprocess.CalledProcessError as e:
            raise RuntimeError(
                f"Failed to create/configure fork: {e.stderr or e.stdout}"
            ) from e
        except Exception as e:
            raise RuntimeError(f"Failed to setup fork: {e}") from e

        try:
            rate_limit = gh.get_rate_limit()
            core_limit = rate_limit.raw_data["resources"]["core"]["limit"]
            core_remaining = rate_limit.raw_data["resources"]["core"]["remaining"]
            logger.info(
                f"GitHub API rate limit: {core_remaining}/{core_limit} remaining"
            )
            if core_remaining < 5:
                logger.warning(
                    f"GitHub API rate limit low ({core_remaining} remaining). "
                    "Consider waiting before submitting more PRs."
                )
        except Exception as e:
            logger.debug(f"Could not check rate limit (non-critical): {e}")

        try:
            logger.info(f"Pushing to fork branch '{branch_name}'...")
            subprocess.run(
                ["git", "push", "fork", f"HEAD:refs/heads/{branch_name}"],
                cwd=remote_path,
                check=True,
                capture_output=True,
                text=True,
                timeout=60,
            )
            logger.info("Push successful")
        except subprocess.CalledProcessError as e:
            raise RuntimeError(f"Failed to push to fork: {e.stderr or e.stdout}") from e

        try:
            pr_body = self._prepare_pr_body(models, unsubmitted)
            model_str = ", ".join(model.name for model in models if model.name)

            logger.info("Creating pull request...")
            pr = upstream.create_pull(
                title=f"MTEB Evaluation Results: {model_str}",
                body=pr_body,
                head=f"{user.login}:{branch_name}",
                base="main",
            )

            logger.info("\n" + "=" * 80)
            logger.info("✓ Pull request created successfully!")
            logger.info("=" * 80)
            logger.info(f"\nPR URL: {pr.html_url}")
            logger.info(f"PR Number: #{pr.number}")
            logger.info(f"Fork: {fork_url}")
            logger.info(f"\nModels: {model_str}")
            logger.info(f"Results: {result_count}")
            logger.info(f"Commit: {commit_sha}")
            logger.info("\n" + "=" * 80)

            return cast(
                "SubmitResultsResponse",
                {
                    "status": "pr_created",
                    "models_submitted": [(m.name, m.revision) for m in models],
                    "result_count": result_count,
                    "commit_sha": commit_sha,
                    "pr_url": pr.html_url,
                    "pr_number": pr.number,
                    "fork_url": fork_url,
                    "branch_name": branch_name,
                },
            )

        except GithubException as e:
            raise RuntimeError(
                f"Failed to create pull request: "
                f"Status {e.status}: {e.data.get('message', str(e))}"
            ) from e
        except Exception as e:
            raise RuntimeError(f"Unexpected error creating PR: {e}") from e

    def load_results(
        self,
        models: Sequence[str] | Iterable[ModelMeta] | None = None,
        tasks: Sequence[str] | Iterable[AbsTask] | Benchmark | str | None = None,
        *,
        require_model_meta: bool = True,
        include_remote: bool = True,
        validate_and_filter: bool = False,
        only_main_score: bool = False,
        load_experiments: LoadExperimentEnum | str = LoadExperimentEnum.MATCH_KWARGS,
        experiment_kwargs: Mapping[str, Any] | list[Mapping[str, Any]] | None = None,
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
            load_experiments: If True, it will also load results from experiment folders.
            experiment_kwargs: If specified, it will only load results from experiments with the specified kwargs. Only used if load_experiments is True.

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
            tasks = get_benchmark(tasks)

        if isinstance(load_experiments, str):
            load_experiments = LoadExperimentEnum.from_str(load_experiments)

        if (
            load_experiments is not LoadExperimentEnum.MATCH_KWARGS
            and experiment_kwargs is not None
        ):
            warnings.warn(
                "experiment_kwargs is specified but load_experiments is not set to MATCH_KWARGS."
                "No results will be loaded."
            )

        models_as_model_meta = models is not None and isinstance(
            next(iter(models)), ModelMeta
        )

        paths = self.get_cache_paths(
            models=models,
            tasks=tasks,
            require_model_meta=require_model_meta,
            include_remote=include_remote,
            load_experiments=load_experiments,
        )
        models_results = defaultdict(list)

        task_names: dict[str, AbsTask | None] = {}
        if tasks is not None:
            for task in tasks:
                if isinstance(task, AbsTask):
                    task_names[task.metadata.name] = task
                else:
                    task_names[task] = None

        experiment_names = set()
        if isinstance(experiment_kwargs, Mapping):
            experiment_kwargs = [experiment_kwargs]
        if isinstance(experiment_kwargs, list):
            experiment_names = {
                _serialize_experiment_kwargs_to_name(params)
                for params in experiment_kwargs
            }
        for path in paths:
            task_result = TaskResult.from_disk(path)

            if only_main_score:
                task_result = task_result.only_main_score()
            model_name, revision, experiment_name = (
                self._get_model_name_and_revision_from_path(path.parent)
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

            if len(experiment_names) > 0 and experiment_name not in experiment_names:
                logger.debug(
                    f"Skipping experiment {experiment_name} as it is not in the specified experiment names"
                )
                continue

            if (
                load_experiments is LoadExperimentEnum.MATCH_KWARGS
                and not models_as_model_meta  # for models meta path are prefiltered
                and len(experiment_names) == 0
                and experiment_name is not None
            ):
                continue

            models_results[(model_name, revision, experiment_name)].append(task_result)

        # create BenchmarkResults object
        models_results_object = [
            ModelResult(
                model_name=model_name,
                model_revision=revision,
                task_results=task_results,
                experiment_name=experiment_name,
            )
            for (
                model_name,
                revision,
                experiment_name,
            ), task_results in models_results.items()
        ]

        return BenchmarkResults(
            model_results=models_results_object,
            benchmark=tasks if isinstance(tasks, Benchmark) else None,
        )
