from __future__ import annotations

import logging
import shutil
import subprocess
from typing import TYPE_CHECKING, Any

from mteb.workflow.reversible_workflow import ReversibleAction

if TYPE_CHECKING:
    from pathlib import Path

    from github.github import Github

    from mteb.models import ModelMeta

logger = logging.getLogger(__name__)


class CreateBranchAction(ReversibleAction):
    """Create a git branch. Undo by deleting the branch and restoring original."""

    def __init__(
        self, repo_path: Path, branch_name: str, original_branch: str = "main"
    ) -> None:
        """Initialize the action.

        Args:
            repo_path: Path to the git repository.
            branch_name: Name of the branch to create.
            original_branch: Name of the branch to restore to on undo (default: "main").
        """
        self.repo_path = repo_path
        self.branch_name = branch_name
        self.original_branch = original_branch

    def do(self) -> None:
        """Create the branch."""
        subprocess.run(
            ["git", "checkout", "-b", self.branch_name],
            cwd=self.repo_path,
            check=True,
            capture_output=True,
            text=True,
        )
        logger.info(f"Created branch '{self.branch_name}'")

    def undo(self) -> None:
        """Delete the branch and restore original branch."""
        try:
            subprocess.run(
                ["git", "checkout", self.original_branch],
                cwd=self.repo_path,
                check=True,
                capture_output=True,
                text=True,
            )
            subprocess.run(
                ["git", "branch", "-D", self.branch_name],
                cwd=self.repo_path,
                check=True,
                capture_output=True,
                text=True,
            )
            logger.info(
                f"Deleted branch '{self.branch_name}' and restored to '{self.original_branch}'"
            )
        except subprocess.CalledProcessError as e:
            logger.warning(f"Failed to cleanup branch '{self.branch_name}': {e}")


class CopyResultsAction(ReversibleAction):
    """Copy result files to the remote directory. Undo by deleting copied files.

    Handles all JSON files including model_meta.json and task result files.
    """

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
        """Copy result files to remote.

        Copies all JSON files from the source directories including:
        - Task result files (*.json)
        - model_meta.json
        """
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
        """Delete copied files."""
        for file_path in self.copied_files:
            try:
                file_path.unlink()
                logger.debug(f"Deleted {file_path}")
            except Exception as e:
                logger.warning(f"Failed to delete {file_path}: {e}")

        logger.info(f"Deleted {len(self.copied_files)} copied files")


class CommitAction(ReversibleAction):
    """Create a git commit. Undo by resetting to the previous commit."""

    def __init__(self, repo_path: Path, message: str) -> None:
        """Initialize the action.

        Args:
            repo_path: Path to the git repository.
            message: Commit message.
        """
        self.repo_path = repo_path
        self.message = message
        self.previous_sha: str | None = None
        self.commit_sha: str | None = None

    def do(self) -> None:
        """Stage and commit all changes."""
        # Save current HEAD for rollback
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=self.repo_path,
            check=True,
            capture_output=True,
            text=True,
        )
        self.previous_sha = result.stdout.strip()

        subprocess.run(
            ["git", "add", "-A"],
            cwd=self.repo_path,
            check=True,
            capture_output=True,
            text=True,
        )

        subprocess.run(
            ["git", "commit", "-m", self.message],
            cwd=self.repo_path,
            check=True,
            capture_output=True,
            text=True,
        )

        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=self.repo_path,
            check=True,
            capture_output=True,
            text=True,
        )
        self.commit_sha = result.stdout.strip()
        logger.info(f"Created commit {self.commit_sha}")

    def undo(self) -> None:
        """Reset to the previous commit."""
        if not self.previous_sha:
            logger.warning("No previous SHA saved, cannot undo commit")
            return

        try:
            subprocess.run(
                ["git", "reset", "--hard", self.previous_sha],
                cwd=self.repo_path,
                check=True,
                capture_output=True,
                text=True,
            )
            logger.info(f"Reset to previous commit {self.previous_sha}")
        except subprocess.CalledProcessError as e:
            logger.warning(f"Failed to reset to {self.previous_sha}: {e}")


class PushToForkAction(ReversibleAction):
    """Push commits to a fork remote. Undo by force-pushing origin branch back to rewind the remote branch."""

    def __init__(
        self,
        repo_path: Path,
        fork_remote: str,
        branch_name: str,
        origin_branch: str = "main",
    ) -> None:
        """Initialize the action.

        Args:
            repo_path: Path to the git repository.
            fork_remote: Name of the fork remote (e.g., "fork").
            branch_name: Name of the branch to push.
            origin_branch: Name of the origin branch to fall back to (default "main").
        """
        self.repo_path = repo_path
        self.fork_remote = fork_remote
        self.branch_name = branch_name
        self.origin_branch = origin_branch

    def do(self) -> None:
        """Push to fork remote."""
        subprocess.run(
            ["git", "push", self.fork_remote, f"HEAD:refs/heads/{self.branch_name}"],
            cwd=self.repo_path,
            check=True,
            capture_output=True,
            text=True,
            timeout=60,
        )
        logger.info(f"Pushed to {self.fork_remote}/{self.branch_name}")

    def undo(self) -> None:
        """Rewind the remote branch by force-pushing origin_branch to branch_name."""
        try:
            subprocess.run(
                [
                    "git",
                    "push",
                    "-f",
                    self.fork_remote,
                    f"{self.origin_branch}:refs/heads/{self.branch_name}",
                ],
                cwd=self.repo_path,
                check=True,
                capture_output=True,
                text=True,
                timeout=60,
            )
            logger.info(f"Reverted push to {self.fork_remote}/{self.branch_name}")
        except subprocess.CalledProcessError as e:
            logger.warning(f"Failed to revert push: {e}")


class CreatePRAction(ReversibleAction):
    """Create a GitHub pull request. Undo by closing the PR."""

    def __init__(
        self,
        gh: Github,
        upstream_repo_name: str,
        user_login: str,
        branch_name: str,
        title: str,
        pr_body: str,
    ) -> None:
        """Initialize the action.

        Args:
            gh: GitHub API client instance.
            upstream_repo_name: Name of upstream repo (e.g., "owner/repo").
            user_login: GitHub username (for PR head).
            branch_name: Name of the branch with changes.
            title: PR title.
            pr_body: PR description body.
        """
        self.gh = gh
        self.upstream_repo_name = upstream_repo_name
        self.user_login = user_login
        self.branch_name = branch_name
        self.title = title
        self.pr_body = pr_body
        self.pr: Any = None

    def do(self) -> None:
        """Create the pull request."""
        try:
            from github import GithubException  # type: ignore[import-not-found]
        except ImportError:
            raise ImportError("PyGithub is required for CreatePRAction")

        try:
            upstream = self.gh.get_repo(self.upstream_repo_name)
            self.pr = upstream.create_pull(
                title=self.title,
                body=self.pr_body,
                head=f"{self.user_login}:{self.branch_name}",
                base="main",
            )
            logger.info(f"Created PR #{self.pr.number}: {self.pr.html_url}")
        except GithubException as e:
            raise RuntimeError(
                f"Failed to create PR: Status {e.status}: {e.data.get('message', str(e))}"
            ) from e

    def undo(self) -> None:
        """Close the PR."""
        if not self.pr:
            logger.warning("No PR object saved, cannot close PR")
            return

        try:
            self.pr.edit(state="closed")
            logger.info(f"Closed PR #{self.pr.number}")
        except Exception as e:
            logger.warning(f"Failed to close PR: {e}")


class RestoreOriginalBranchAction(ReversibleAction):
    """Restore the original branch after successful PR creation."""

    def __init__(self, repo_path: Path, original_branch: str) -> None:
        """Initialize the action.

        Args:
            repo_path: Path to the git repository.
            original_branch: Name of the branch to restore to.
        """
        self.repo_path = repo_path
        self.original_branch = original_branch
        self.current_branch_before: str | None = None

    def do(self) -> None:
        """Save current branch for rollback and checkout the original branch."""
        result = subprocess.run(
            ["git", "rev-parse", "--abbrev-ref", "HEAD"],
            cwd=self.repo_path,
            check=True,
            capture_output=True,
            text=True,
        )
        self.current_branch_before = result.stdout.strip()

        subprocess.run(
            ["git", "checkout", self.original_branch],
            cwd=self.repo_path,
            check=True,
            capture_output=True,
            text=True,
        )
        logger.info(f"Restored to original branch '{self.original_branch}'")

    def undo(self) -> None:
        """Return to the branch we were on before (best effort)."""
        if not self.current_branch_before:
            logger.warning("No previous branch saved, skipping undo")
            return

        try:
            subprocess.run(
                ["git", "checkout", self.current_branch_before],
                cwd=self.repo_path,
                check=True,
                capture_output=True,
                text=True,
            )
            logger.info(f"Returned to branch '{self.current_branch_before}'")
        except subprocess.CalledProcessError as e:
            logger.warning(f"Failed to return to previous branch: {e}")
