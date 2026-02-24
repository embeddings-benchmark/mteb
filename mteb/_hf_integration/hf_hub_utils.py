from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any

from huggingface_hub import (
    hf_hub_download,
    list_repo_commits,
    repo_exists,
)
from huggingface_hub.errors import (
    EntryNotFoundError,
    GatedRepoError,
    HFValidationError,
    RepositoryNotFoundError,
)

if TYPE_CHECKING:
    from huggingface_hub import GitCommitInfo

logger = logging.getLogger(__name__)


def _get_repo_commits(repo_id: str, repo_type: str) -> list[GitCommitInfo] | None:
    try:
        return list_repo_commits(repo_id=repo_id, repo_type=repo_type)
    except (GatedRepoError, RepositoryNotFoundError) as e:
        logger.warning(f"Can't get commits of {repo_id}: {e}")
        return None


def _get_json_from_hub(
    repo_id: str, file_name: str, repo_type: str, revision: str | None = None
) -> dict[str, Any] | None:
    path = _get_file_on_hub(repo_id, file_name, repo_type, revision)
    if path is None:
        return None

    with Path(path).open() as f:
        js = json.load(f)
    return js


def _get_file_on_hub(
    repo_id: str, file_name: str, repo_type: str, revision: str | None = None
) -> str | None:
    try:
        return hf_hub_download(
            repo_id=repo_id, filename=file_name, repo_type=repo_type, revision=revision
        )
    except (GatedRepoError, RepositoryNotFoundError, EntryNotFoundError) as e:
        logger.warning(f"Can't get file {file_name} of {repo_id}: {e}")
        return None


def _repo_exists(repo_id: str, repo_type: str | None = None) -> bool:
    """Checks if a repository exists on HuggingFace Hub.

    Repo exists will raise HFValidationError for invalid local paths

    Args:
        repo_id: The repository ID.
        repo_type: The type of repository (e.g., "model", "dataset", "space").
    """
    try:
        return repo_exists(repo_id=repo_id, repo_type=repo_type)
    except HFValidationError as e:
        logger.warning(f"Can't check existence of {repo_id}: {e}")
        return False
