from __future__ import annotations

import json
import logging
import os
import subprocess
from collections import defaultdict
from pathlib import Path

import mteb

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
    results_directory = cache_directory / "results"
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


def _model_name_and_revision(revision_path: Path) -> tuple[MODEL_NAME, REVISION]:
    model_meta = revision_path / "model_meta.json"
    model_path = revision_path.parent
    if not model_meta.exists():
        logger.warning(
            f"model_meta.json not found in {revision_path}, extracting model_name and revision from the path"
        )
        model_name, revision = model_path.name, revision_path.name
    else:
        with model_meta.open("r") as f:
            model_meta_json = json.load(f)
            model_name = model_meta_json["name"]
            revision = model_meta_json["revision"]

    return model_name, revision


def load_results(
    results_repo: str = "https://github.com/embeddings-benchmark/results",
    download_latest: bool = True,
) -> dict[MODEL_NAME, dict[REVISION, list[mteb.MTEBResults]]]:
    """Loads the results from the latest version of the results repository. The results are cached locally in the MTEB_CACHE directory.
    This directory can be set using the MTEB_CACHE environment variable or defaults to "~/.cache/mteb".

    Args:
        results_repo: The URL of the results repository on GitHub. Defaults to "https://github.com/embeddings-benchmark/results".
        download_latest: If True it will update the existing version of the results cache. Defaults to True.

    Returns:
        A dictionary where the keys are the model names and the values are dictionaries where the keys are the revisions and the values are lists of MTEBResults objects.

    Example:
        >>> results = load_results()
        >>> results
        {'mixedbread-ai/mxbai-embed-large-v1':
            {'990580e27d329c7408b3741ecff85876e128e203': [
                MTEBResults(task_name=TwentyNewsgroupsClustering.v2, scores=...),
                MTEBResults(task_name=MedrxivClusteringP2P, scores=...),
                MTEBResults(task_name=StackExchangeClustering, scores=...),
                MTEBResults(task_name=BiorxivClusteringP2P.v2, scores=...),
                MTEBResults(task_name=MedrxivClusteringS2S.v2, scores=...),
                MTEBResults(task_name=MedrxivClusteringS2S, scores=...),
                ...
            ]},
         'intfloat/multilingual-e5-small':
            {'e4ce9877abf3edfe10b0d82785e83bdcb973e22e': [
                MTEBResults(task_name=IndicGenBenchFloresBitextMining, scores=...),
                MTEBResults(task_name=PpcPC, scores=...),
                MTEBResults(task_name=TwentyNewsgroupsClustering.v2, scores=...),
                ...
            ]},
        ...
    """
    repo_directory = download_of_results(results_repo, download_latest=download_latest)
    models = [p for p in (repo_directory / "results").glob("*") if p.is_dir()]

    results = defaultdict(dict)

    for model in models:
        model_revisions = model.glob("*")

        for revision_path in model_revisions:
            model_name, revision = _model_name_and_revision(revision_path)

            task_json_files = [
                f for f in revision_path.glob("*.json") if "model_meta.json" != f.name
            ]
            results[model_name][revision] = [
                mteb.MTEBResults.from_disk(f) for f in task_json_files
            ]

    return dict(results)
