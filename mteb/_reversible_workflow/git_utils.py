from __future__ import annotations

import json
import logging
import subprocess
from typing import TYPE_CHECKING

from mteb.types import SubmitResultsResponse

if TYPE_CHECKING:
    from pathlib import Path

    from mteb.models import ModelMeta

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


def get_github_token() -> str:
    """Get a GitHub token using gh CLI authentication or git credential helper.

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


def handle_pr_creation_with_cleanup(
    *,
    remote_repo_path: Path,
    original_branch: str,
    branch_name: str | None,
    models: list[ModelMeta],
    unsubmitted: dict[ModelMeta, list[Path]],
    result_count: int,
    pr_body: str,
) -> SubmitResultsResponse:
    """Create a pull request and clean up branch state on success or failure.

    Args:
        remote_repo_path: Path to the remote repository.
        models: List of ModelMeta objects.
        unsubmitted: Dict mapping ModelMeta to list of result file paths.
        result_count: Total number of results.
        branch_name: Name of the branch to create.
        remote_path: Path to the remote repository.
        original_branch: Original branch name for restoration.
        pr_body: The body text for the pull request.

    Returns:
        Dictionary with PR information.

    Raises:
        RuntimeError: If PR creation or cleanup fails.
    """
    try:
        result = create_pull_request(
            remote_repo_path=remote_repo_path,
            models=models,
            unsubmitted=unsubmitted,
            result_count=result_count,
            branch_name=branch_name,
            pr_body=pr_body,
        )
        # After successful PR, restore to original branch
        restore_branch(remote_repo_path, original_branch)
        return result
    except Exception as e:
        # PR creation failed, but workflow.run() already completed
        # Restore to original branch and delete temporary branch to clean up
        logger.error(f"PR creation failed: {e}")

        try:
            restore_branch(remote_repo_path, original_branch)
        except Exception as restore_error:
            logger.error(f"Failed to restore branch on error: {restore_error}")
            logger.warning(
                f"You may be on branch '{branch_name}'. "
                f"To restore, run: git checkout {original_branch}"
            )

        if branch_name:
            try:
                delete_branch(remote_repo_path, branch_name)
            except Exception as delete_error:
                logger.error(f"Failed to delete branch: {delete_error}")

        raise


def create_pull_request(
    *,
    remote_repo_path: Path,
    models: list[ModelMeta],
    unsubmitted: dict[ModelMeta, list[Path]],
    result_count: int,
    branch_name: str | None,
    pr_body: str,
) -> SubmitResultsResponse:
    """Create a pull request on GitHub using PyGithub and gh CLI.

    Args:
        remote_repo_path: Path to the remote repository.
        models: List of ModelMeta objects.
        unsubmitted: Dict mapping ModelMeta to list of result file paths.
        result_count: Total number of results.
        branch_name: Name of the branch to create.
        pr_body: The body text for the pull request.

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
    token = get_github_token()

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
            cwd=remote_repo_path,
            check=True,
            capture_output=True,
            text=True,
            timeout=30,
        )
        logger.info("Fork created/configured")

        result = subprocess.run(
            ["gh", "repo", "view", "--json", "url"],
            check=False,
            capture_output=True,
            text=True,
            timeout=10,
        )
        if result.returncode == 0:
            fork_data = json.loads(result.stdout)
            fork_url = fork_data.get("url", f"https://github.com/{user.login}/results")
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
        logger.info(f"GitHub API rate limit: {core_remaining}/{core_limit} remaining")
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
            cwd=remote_repo_path,
            check=True,
            capture_output=True,
            text=True,
            timeout=60,
        )
        logger.info("Push successful")
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"Failed to push to fork: {e.stderr or e.stdout}") from e

    try:
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
        logger.info("\n" + "=" * 80)

        return SubmitResultsResponse(
            status="pr_created",
            models_submitted=[(m.name, m.revision) for m in models],
            result_count=result_count,
            pr_url=pr.html_url,
            pr_number=pr.number,
            fork_url=fork_url,
            branch_name=branch_name,
        )

    except GithubException as e:
        raise RuntimeError(
            f"Failed to create pull request: "
            f"Status {e.status}: {e.data.get('message', str(e))}"
        ) from e
    except Exception as e:
        raise RuntimeError(f"Unexpected error creating PR: {e}") from e
