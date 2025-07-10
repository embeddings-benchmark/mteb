from __future__ import annotations

import logging
import os
import shutil
import subprocess
from pathlib import Path

from mteb.load_results.task_results import TaskResult
from mteb.models.model_meta import ModelMeta

logger = logging.getLogger(__name__)


class ResultCache:
    """Class to handle the local cache of MTEB results.

    Example:
        >>> from mteb.cache import ResultCache
        >>> cache = mteb.ResultCache(cache_path="~/.cache/mteb") # default
        >>> cache.download_from_remote() # download the latest results from the remote repository
        >>> result = cache.load_from_cache("task_name", "model_name", "model_revision")
    """

    cache_path: Path

    def __init__(self, cache_path: Path | str | None = None) -> None:
        if cache_path is not None:
            self.cache_path = Path(cache_path)
        else:
            self.cache_path = self.default_cache_path

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
        if isinstance(model_name, ModelMeta):
            if model_revision is not None:
                logger.warning(
                    "model_revision is ignored when model_name is a ModelMeta object"
                )
            model_revision = model_name.revision
            model_name = model_name.model_name_as_path()

        if model_revision is None:
            raise ValueError(
                "model_revision must be specified when model_name is a string"
            )

        results_folder = "results" if not remote else "remote"

        return (
            self.cache_path
            / results_folder
            / model_name
            / model_revision
            / f"{task_name}.json"
        )

    def load_from_cache(
        self,
        task_name: str,
        model_name: str | ModelMeta,
        model_revision: str | None = None,
        raise_if_not_found: bool = False,
    ) -> TaskResult | None:
        """Load the results from the local cache directory.

        Args:
            task_name: The name of the task.
            model_name: The name of the model as a valid directory name or a ModelMeta object.
            model_revision: The revision of the model. Must be specified if model_name is a string.
            raise_if_not_found: If True, raise an error if the results are not found.

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
            if remote_result_path.exists():
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
        """Save the results to the local cache directory.

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
    ) -> Path:
        """Downloads the latest version of the results repository from GitHub to a local cache directory. Required git to be installed.

        Args:
            remote: The URL of the results repository on GitHub.
            download_latest: If True it will download the latest version of the repository, otherwise it will only update the existing repository.

        Returns:
            The path to the local cache directory.
        """
        if not self.cache_path.exists() and not self.cache_path.is_dir():
            logger.debug(
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

            if download_latest:
                logger.info(
                    f"remote repository already exists in {results_directory}, updating it using git pull"
                )
                subprocess.run(["git", "pull"], cwd=results_directory)
            else:
                logger.debug(
                    f"Results repository already exists in {results_directory}, skipping update, set download_latest=True to update it"
                )
            return results_directory

        logger.info(
            f"No results repository found in {results_directory}, cloning it from {remote}"
        )

        subprocess.run(["git", "clone", remote, "remote"], cwd=self.cache_path)

        return results_directory

    def clear_cache(self) -> None:
        """Clear the local cache directory."""
        if self.cache_path.exists() and self.cache_path.is_dir():
            shutil.rmtree(self.cache_path)
            logger.info(f"Cache directory {self.cache_path} cleared.")
        else:
            logger.warning(f"Cache directory {self.cache_path} does not exist.")

    def __repr__(self) -> str:
        return f"ResultCache(cache_path={self.cache_path})"
