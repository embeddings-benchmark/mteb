from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from mteb._reversible_workflow.git_actions import (
    CommitAction,
    CreateBranchAction,
    CreatePRAction,
    PushToForkAction,
    RestoreOriginalBranchAction,
)


@pytest.fixture
def git_repo(tmp_path: Path) -> Path:
    """Create a real git repository for testing."""
    repo_path = tmp_path / "repo"
    repo_path.mkdir()

    subprocess.run(["git", "init"], cwd=repo_path, check=True, capture_output=True)

    if (
        subprocess.run(
            ["git", "config", "user.email"],
            check=False,
            cwd=repo_path,
            capture_output=True,
        ).returncode
        != 0
    ):
        subprocess.run(
            ["git", "config", "user.email", "test@test.com"],
            cwd=repo_path,
            check=True,
            capture_output=True,
        )

    if (
        subprocess.run(
            ["git", "config", "user.name"],
            check=False,
            cwd=repo_path,
            capture_output=True,
        ).returncode
        != 0
    ):
        subprocess.run(
            ["git", "config", "user.name", "Test User"],
            cwd=repo_path,
            check=True,
            capture_output=True,
        )

    # Create initial commit
    (repo_path / "file.txt").write_text("content")
    subprocess.run(["git", "add", "."], cwd=repo_path, check=True, capture_output=True)
    subprocess.run(
        ["git", "commit", "-m", "initial"],
        cwd=repo_path,
        check=True,
        capture_output=True,
    )

    return repo_path


def _get_default_branch(repo_path: Path) -> str:
    """Get the default branch name of the repository."""
    result = subprocess.run(
        ["git", "rev-parse", "--abbrev-ref", "HEAD"],
        cwd=repo_path,
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def test_create_branch_action_do_and_undo(git_repo: Path) -> None:
    """Test creating (Do) and deleting (Undo) a git branch."""
    default_branch = _get_default_branch(git_repo)

    action = CreateBranchAction(
        repo_path=git_repo,
        branch_name="feature-1",
        original_branch=default_branch,
    )

    action.do()
    result = subprocess.run(
        ["git", "branch", "--show-current"],
        cwd=git_repo,
        check=True,
        capture_output=True,
        text=True,
    )
    assert result.stdout.strip() == "feature-1", "Should be on feature-1 branch"

    action.undo()
    result = subprocess.run(
        ["git", "branch", "--show-current"],
        cwd=git_repo,
        check=True,
        capture_output=True,
        text=True,
    )
    assert result.stdout.strip() == default_branch, (
        f"Should be back on {default_branch} branch"
    )

    result = subprocess.run(
        ["git", "branch", "-a"],
        cwd=git_repo,
        check=True,
        capture_output=True,
        text=True,
    )
    assert "feature-1" not in result.stdout, "feature-1 branch should be deleted"


def test_commit_action_saves_shas_and_undo_resets(git_repo: Path) -> None:
    """Test creating a commit and resetting to previous state."""
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=git_repo,
        check=True,
        capture_output=True,
        text=True,
    )
    initial_sha = result.stdout.strip()

    (git_repo / "new_file.txt").write_text("new content")

    action = CommitAction(repo_path=git_repo, message="Add new file")
    action.do()
    assert action.previous_sha == initial_sha

    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=git_repo,
        check=True,
        capture_output=True,
        text=True,
    )
    head_sha = result.stdout.strip()
    assert head_sha != initial_sha

    action.undo()
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=git_repo,
        check=True,
        capture_output=True,
        text=True,
    )
    assert result.stdout.strip() == initial_sha
    assert not (git_repo / "new_file.txt").exists()


def test_push_to_fork_action_do_and_undo(git_repo: Path) -> None:
    """Test pushing to a fork remote (Do) and reverting the push (Undo)."""
    default_branch = _get_default_branch(git_repo)
    fork_path = git_repo.parent / "fork"
    fork_path.mkdir()
    subprocess.run(
        ["git", "init", "--bare"], cwd=fork_path, check=True, capture_output=True
    )

    subprocess.run(
        ["git", "remote", "add", "fork", str(fork_path)],
        cwd=git_repo,
        check=True,
        capture_output=True,
    )
    subprocess.run(
        ["git", "checkout", "-b", "feature-1"],
        cwd=git_repo,
        check=True,
        capture_output=True,
    )
    (git_repo / "feature.txt").write_text("feature content")
    subprocess.run(["git", "add", "."], cwd=git_repo, check=True, capture_output=True)
    subprocess.run(
        ["git", "commit", "-m", "feature commit"],
        cwd=git_repo,
        check=True,
        capture_output=True,
    )

    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=git_repo,
        check=True,
        capture_output=True,
        text=True,
    )
    feature_sha = result.stdout.strip()
    action = PushToForkAction(
        repo_path=git_repo,
        fork_remote="fork",
        branch_name="feature-1",
        origin_branch=default_branch,
    )

    action.do()
    result = subprocess.run(
        ["git", "show-ref"],
        cwd=fork_path,
        check=True,
        capture_output=True,
        text=True,
    )
    assert feature_sha in result.stdout, "Commit should be in fork refs"
    assert "feature-1" in result.stdout, "feature-1 branch should exist in fork"

    action.undo()
    result = subprocess.run(
        ["git", "show-ref"],
        cwd=fork_path,
        check=True,
        capture_output=True,
        text=True,
    )

    assert "feature-1" in result.stdout, "feature-1 branch should still exist"
    result_main = subprocess.run(
        ["git", "rev-parse", default_branch],
        cwd=git_repo,
        check=True,
        capture_output=True,
        text=True,
    )

    main_sha = result_main.stdout.strip()
    result_fork_feature = subprocess.run(
        ["git", "show-ref", "refs/heads/feature-1"],
        cwd=fork_path,
        check=True,
        capture_output=True,
        text=True,
    )
    assert main_sha in result_fork_feature.stdout, (
        f"feature-1 should be reverted to {default_branch}'s SHA"
    )


def test_create_pr_action_do_and_undo() -> None:
    pytest.importorskip("github", reason="github is not installed")

    class FakePR:
        def __init__(self):
            self.number = 7
            self.html_url = "https://example/pr/7"
            self.closed = False

        def edit(self, *, state: str) -> None:
            if state == "closed":
                self.closed = True

    class FakeRepo:
        def __init__(self):
            self.pr = FakePR()
            self.last_args = None

        def create_pull(self, **kwargs):
            self.last_args = kwargs
            return self.pr

    class FakeGh:
        def __init__(self):
            self.repo = FakeRepo()

        def get_repo(self, _name: str):
            return self.repo

    gh = FakeGh()
    action = CreatePRAction(
        gh=gh,
        upstream_repo_name="org/repo",
        user_login="alice",
        branch_name="feature-1",
        title="Test PR",
        pr_body="Body",
    )

    action.do()
    assert gh.repo.last_args == {
        "title": "Test PR",
        "body": "Body",
        "head": "alice:feature-1",
        "base": "main",
    }

    action.undo()
    assert gh.repo.pr.closed is True


def test_restore_original_branch_action_do_and_undo(git_repo: Path) -> None:
    """Test detecting and restoring original branch."""
    default_branch = _get_default_branch(git_repo)

    subprocess.run(
        ["git", "checkout", "-b", "feature-1"],
        cwd=git_repo,
        check=True,
        capture_output=True,
    )

    result = subprocess.run(
        ["git", "rev-parse", "--abbrev-ref", "HEAD"],
        cwd=git_repo,
        check=True,
        capture_output=True,
        text=True,
    )
    assert result.stdout.strip() == "feature-1"
    action = RestoreOriginalBranchAction(
        repo_path=git_repo, original_branch=default_branch
    )
    action.do()
    assert action.current_branch_before == "feature-1"
    result = subprocess.run(
        ["git", "rev-parse", "--abbrev-ref", "HEAD"],
        cwd=git_repo,
        check=True,
        capture_output=True,
        text=True,
    )
    assert result.stdout.strip() == default_branch

    action.undo()
    result = subprocess.run(
        ["git", "rev-parse", "--abbrev-ref", "HEAD"],
        cwd=git_repo,
        check=True,
        capture_output=True,
        text=True,
    )
    assert result.stdout.strip() == "feature-1"
