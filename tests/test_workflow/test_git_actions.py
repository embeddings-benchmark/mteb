from __future__ import annotations

from pathlib import Path

from mteb._reversible_workflow.git_actions import (
    CommitAction,
    CreateBranchAction,
    CreatePRAction,
    PushToForkAction,
    RestoreOriginalBranchAction,
)


class _Completed:
    def __init__(self, stdout: str = "") -> None:
        self.stdout = stdout


def test_create_branch_action_do_and_undo(monkeypatch, tmp_path: Path) -> None:
    calls: list[list[str]] = []

    def fake_run(cmd, **kwargs):
        calls.append(cmd)
        return _Completed()

    monkeypatch.setattr("mteb.workflow.git_actions.subprocess.run", fake_run)

    action = CreateBranchAction(
        repo_path=tmp_path / "repo",
        branch_name="feature-1",
        original_branch="main",
    )
    action.do()
    action.undo()

    assert calls == [
        ["git", "checkout", "-b", "feature-1"],
        ["git", "checkout", "main"],
        ["git", "branch", "-D", "feature-1"],
    ]


def test_commit_action_saves_shas_and_undo_resets(monkeypatch, tmp_path: Path) -> None:
    calls: list[list[str]] = []
    rev_parse_outputs = ["sha-before\n", "sha-after\n"]

    def fake_run(cmd, **kwargs):
        calls.append(cmd)
        if cmd[:3] == ["git", "rev-parse", "HEAD"]:
            return _Completed(rev_parse_outputs.pop(0))
        return _Completed()

    monkeypatch.setattr("mteb.workflow.git_actions.subprocess.run", fake_run)

    action = CommitAction(repo_path=tmp_path / "repo", message="msg")
    action.do()

    assert action.previous_sha == "sha-before"
    assert action.commit_sha == "sha-after"

    action.undo()

    assert ["git", "add", "-A"] in calls
    assert ["git", "commit", "-m", "msg"] in calls
    assert ["git", "reset", "--hard", "sha-before"] in calls


def test_push_to_fork_action_do_and_undo(monkeypatch, tmp_path: Path) -> None:
    calls: list[list[str]] = []

    def fake_run(cmd, **kwargs):
        calls.append(cmd)
        return _Completed()

    monkeypatch.setattr("mteb.workflow.git_actions.subprocess.run", fake_run)

    action = PushToForkAction(
        repo_path=tmp_path / "repo",
        fork_remote="fork",
        branch_name="feature-1",
        origin_branch="main",
    )
    action.do()
    action.undo()

    assert [
        "git",
        "push",
        "fork",
        "HEAD:refs/heads/feature-1",
    ] in calls
    assert [
        "git",
        "push",
        "-f",
        "fork",
        "main:refs/heads/feature-1",
    ] in calls


def test_create_pr_action_do_and_undo(monkeypatch) -> None:
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


def test_restore_original_branch_action_do_and_undo(
    monkeypatch, tmp_path: Path
) -> None:
    calls: list[list[str]] = []

    def fake_run(cmd, **kwargs):
        calls.append(cmd)
        if cmd == ["git", "rev-parse", "--abbrev-ref", "HEAD"]:
            return _Completed("feature-1\n")
        return _Completed()

    monkeypatch.setattr("mteb.workflow.git_actions.subprocess.run", fake_run)

    action = RestoreOriginalBranchAction(
        repo_path=tmp_path / "repo", original_branch="main"
    )

    action.do()
    assert action.current_branch_before == "feature-1"

    action.undo()

    assert ["git", "checkout", "main"] in calls
    assert ["git", "checkout", "feature-1"] in calls
