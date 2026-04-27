from __future__ import annotations

import logging
import subprocess
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pathlib import Path

logger = logging.getLogger(__name__)


def check_uncommitted_changes(repo_path: Path) -> None:
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


def check_detached_head(repo_path: Path) -> None:
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


def run_preflight_checks(repo_path: Path) -> None:
    """Run all pre-flight validations before submission workflow.

    These checks prevent common issues like uncommitted changes or detached HEAD
        that would corrupt the repository or cause the workflow to fail mid-way without proper rollback.

    Args:
            repo_path: Path to the git repository.

    Raises:
            RuntimeError: If any validation fails.

    """
    logger.info("Running pre-flight checks...")
    check_uncommitted_changes(repo_path)
    check_detached_head(repo_path)
    logger.info("Pre-flight checks passed ✓")


def get_current_branch(repo_path: Path) -> str:
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


def restore_branch(repo_path: Path, original_branch: str) -> None:
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


def delete_branch(repo_path: Path, branch_name: str) -> None:
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
