from __future__ import annotations

import logging
import os
import subprocess
from pathlib import Path

logger = logging.getLogger(__name__)


def get_cache_directory() -> Path:
    """Get the local cache directory for MTEB results.

    Returns:
        The path to the local cache directory.
    """
    default_cache_directory = Path.home() / ".cache" / "mteb"

    _cache_directory = os.environ.get("MTEB_CACHE", None)
    cache_directory = (
        Path(_cache_directory) if _cache_directory else default_cache_directory
    )

    if not cache_directory.exists():
        cache_directory.mkdir(parents=True)

    return cache_directory


def download_results_cache(
    results_repo: str = "https://github.com/embeddings-benchmark/results",
    cache_directory: Path | None = None,
    download_latest: bool = True,
) -> Path:
    """Downloads the latest version of the results repository from GitHub to a local cache directory. Required git to be installed.

    Args:
        results_repo: The URL of the results repository on GitHub.
        cache_directory: The directory where the repository should be cached. If None it will use the MTEB_CACHE environment variable or "~/.cache/mteb" by default.
        download_latest: If True it will download the latest version of the repository, otherwise it will only update the existing repository.

    Returns:
        The path to the local cache directory.
    """
    cache_directory = (
        get_cache_directory() if cache_directory is None else cache_directory
    )

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
