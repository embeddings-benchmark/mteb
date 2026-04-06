from __future__ import annotations

import logging
import subprocess
from typing import TYPE_CHECKING, Any

from mteb._reversible_workflow import ReversibleAction

if TYPE_CHECKING:
    from pathlib import Path

    from github import Github


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
            from github import GithubException
        except ImportError:
            raise ImportError(
                "PyGithub is required for CreatePRAction. To install it run `pip install mteb[github]`"
            )

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
