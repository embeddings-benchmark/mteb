"""Test cases for the ResultCache class in the mteb.cache module."""

import subprocess
from pathlib import Path
from typing import Any, cast
from unittest.mock import patch

import numpy as np
import pytest

import mteb
from mteb.cache import LoadExperimentEnum, ResultCache
from mteb.mocks.mock_tasks import MockRetrievalTask
from mteb.mocks.mock_tasks.clustering import MockMultilingualClusteringTask
from mteb.models import ModelMeta
from mteb.results import TaskResult

test_cache_path = Path(__file__).parent / "mock_mteb_cache"


def test_result_cache() -> None:
    cache = ResultCache(cache_path=test_cache_path)

    assert cache.has_remote is True, "Cache should not have a remote repository"

    # load known results from the cache
    result = cache.load_task_result(
        "BornholmBitextMining",
        "sentence-transformers/all-MiniLM-L6-v2",
        model_revision="8b3219a92973c328a8e22fadcfa821b5dc75636a",
        raise_if_not_found=True,
    )
    result = cache.load_task_result(
        "BornholmBitextMining",
        "sentence-transformers/all-MiniLM-L6-v2",
        raise_if_not_found=True,
    )

    assert isinstance(result, TaskResult), "Loaded result should be a TaskResult"
    assert result.task_name == "BornholmBitextMining", "Task name should match"


def test_get_cache_path() -> None:
    cache = ResultCache(cache_path=test_cache_path)
    paths = cache.get_cache_paths(require_model_meta=False, include_remote=False)

    assert isinstance(paths, list), "Cache paths should be a list"
    assert isinstance(paths[0], Path), "Cache paths should be a list of Paths"

    paths_w_meta = cache.get_cache_paths(require_model_meta=True, include_remote=False)
    assert len(paths_w_meta) < len(paths), (
        "Paths with model meta should be fewer than without"
    )

    paths_w_remote = cache.get_cache_paths(
        include_remote=True, require_model_meta=False
    )

    assert len(paths_w_remote) > len(paths), (
        "Paths with remote should be at least as many as without"
    )

    known_model = "sentence-transformers/average_word_embeddings_levy_dependency"

    paths_for_model = cache.get_cache_paths(
        models=[known_model], require_model_meta=False
    )
    assert len(paths_for_model) > 0, "Should return paths for the specified model"


def test_get_models_and_tasks() -> None:
    cache = ResultCache(cache_path=test_cache_path)

    models = cache.get_models()
    assert isinstance(models, list), "Models should be a list"
    assert isinstance(models[0], tuple) and len(models[0]) == 2, (
        "Models should be a list of tuples (model_name, model_revision)"
    )

    tasks = cache.get_task_names()
    assert isinstance(tasks, list), "Tasks should be a list"
    assert isinstance(tasks[0], str), "Tasks should be a list of task names"

    known_model = "sentence-transformers__average_word_embeddings_levy_dependency"
    known_revision = "6d9c09a789ad5dd126b476323fccfeeafcd90509"

    assert known_model in [mdl[0] for mdl in models], (
        "Known model should be in the results"
    )
    assert known_revision in [mdl[1] for mdl in models if mdl[0] == known_model], (
        "Known revision should be in the results for the known model"
    )


def test_no_duplicates_in_models() -> None:
    """Test that get_models() returns no duplicates (issue #3173)."""
    cache = ResultCache(cache_path=test_cache_path)

    models = cache.get_models()

    # Check that there are no duplicates
    assert len(models) == len(set(models)), (
        f"get_models() returned {len(models)} models but {len(set(models))} unique models. "
        "There should be no duplicates."
    )


def test_no_duplicates_in_tasks() -> None:
    """Test that get_task_names() returns no duplicates (issue #3173)."""
    cache = ResultCache(cache_path=test_cache_path)

    tasks = cache.get_task_names()

    # Check that there are no duplicates
    assert len(tasks) == len(set(tasks)), (
        f"get_task_names() returned {len(tasks)} tasks but {len(set(tasks))} unique tasks. "
        "There should be no duplicates."
    )


def test_load_results():
    cache = ResultCache(cache_path=test_cache_path)

    results = cache.load_results()

    known_model = "sentence-transformers/average_word_embeddings_levy_dependency"
    known_revision = "6d9c09a789ad5dd126b476323fccfeeafcd90509"

    assert known_model in [res.model_name for res in results]
    assert known_revision in [
        res.model_revision for res in results if res.model_name == known_model
    ], "Known revision should be in the results"


def test_load_result_specific_model():
    cache = ResultCache(cache_path=test_cache_path)

    model = "sentence-transformers/average_word_embeddings_levy_dependency"
    results = cache.load_results(models=[model], require_model_meta=False)

    model_names = {mdl_res.model_name for mdl_res in results.model_results}
    assert len(model_names) == 1, "Should only have one model in the results"
    assert model in model_names, "Model should be in the results"


def test_filter_with_modelmeta():
    cache = ResultCache(cache_path=test_cache_path)

    base = test_cache_path / "results"
    model_meta = mteb.get_model_meta("sentence-transformers/all-MiniLM-L6-v2")

    model_name = model_meta.model_name_as_path()
    model_revision_1 = model_meta.revision
    model_revision_1 = cast("str", model_revision_1)
    sample_paths = [
        base / model_name / model_revision_1 / "task1.json",
        base / model_name / model_revision_1 / "task2.json",
        base / model_name / "revision" / "task1.json",
        base / "not_existing_model" / "revision" / "task2.json",
    ]

    filtered = cache._filter_paths_by_model_and_revision(sample_paths, [model_meta])

    expected = {
        (
            "sentence-transformers__all-MiniLM-L6-v2",
            "8b3219a92973c328a8e22fadcfa821b5dc75636a",
        )
    }
    actual = {(p.parent.parent.name, p.parent.name) for p in filtered}
    assert actual == expected


def test_filter_with_string_models():
    cache = ResultCache(cache_path=test_cache_path)

    base = test_cache_path / "results"
    model_meta = mteb.get_model_meta("sentence-transformers/all-MiniLM-L6-v2")

    model_name = model_meta.model_name_as_path()
    model_revision_1 = model_meta.revision
    model_revision_1 = cast("str", model_revision_1)
    sample_paths = [
        base / model_name / model_revision_1 / "task1.json",
        base / model_name / model_revision_1 / "task2.json",
        base / model_name / "revision" / "task1.json",
        base / "not_existing_model" / "revision" / "task2.json",
    ]

    filtered = cache._filter_paths_by_model_and_revision(sample_paths, [model_name])

    expected = {
        (
            "sentence-transformers__all-MiniLM-L6-v2",
            "8b3219a92973c328a8e22fadcfa821b5dc75636a",
        ),
        ("sentence-transformers__all-MiniLM-L6-v2", "revision"),
    }
    actual = {(p.parent.parent.name, p.parent.name) for p in filtered}
    assert actual == expected


def test_cache_filter_languages():
    cache = ResultCache(cache_path=test_cache_path)

    task = MockMultilingualClusteringTask()
    results = cache.load_results(
        tasks=[task],
        validate_and_filter=True,
    )
    assert len(results.model_results[0].task_results[0].scores["test"]) == 2
    task = task.filter_languages(["eng"])
    eng_results = cache.load_results(tasks=[task], validate_and_filter=True)
    assert len(eng_results.model_results[0].task_results[0].scores["test"]) == 1


def test_cache_load_different_subsets():
    cache = ResultCache(cache_path=test_cache_path)

    task = mteb.get_task(
        "BelebeleRetrieval", hf_subsets=["acm_Arab-acm_Arab", "nld_Latn-nld_Latn"]
    )
    model1 = mteb.get_model_meta(
        "sentence-transformers/all-MiniLM-L6-v2"
    )  # model have only arab subset results
    model2 = mteb.get_model_meta(
        "mteb/baseline-random-encoder"
    )  # model have all subsets results

    result1 = cache.load_results(
        models=[
            model1,
        ],
        tasks=[task],
    )
    result2 = cache.load_results(
        models=[
            model2,
        ],
        tasks=[task],
    )
    assert len(result1.model_results[0].task_results[0].scores["test"]) == 1
    assert len(result2.model_results[0].task_results[0].scores["test"]) == 2

    assert result1.model_results[0].task_results[0].get_score() == 0.01568
    assert result2.model_results[0].task_results[0].get_score() == 0.01035

    result1 = cache.load_results(
        models=[
            model1,
        ],
        tasks=[task],
        validate_and_filter=True,
    )
    result2 = cache.load_results(
        models=[
            model2,
        ],
        tasks=[task],
        validate_and_filter=True,
    )
    assert len(result1.model_results[0].task_results[0].scores["test"]) == 2
    assert len(result2.model_results[0].task_results[0].scores["test"]) == 2

    assert np.isnan(result1.model_results[0].task_results[0].get_score())
    assert result2.model_results[0].task_results[0].get_score() == 0.01035


def test_load_experiment_results(tmp_path: Path):
    """Test that results from an experiment can be loaded correctly."""
    model = mteb.get_model("mteb/baseline-random-encoder")
    task = MockRetrievalTask()
    cache = mteb.ResultCache(tmp_path)
    mteb.evaluate(model, task, cache=cache)

    params_1 = {"a": "test"}
    params_2 = {"a": "test", "b": "test2"}
    model = mteb.get_model(
        "mteb/baseline-random-encoder",
        **params_1,
    )
    mteb.evaluate(model, task, cache=cache)

    model = mteb.get_model(
        "mteb/baseline-random-encoder",
        **params_2,
    )
    mteb.evaluate(model, task, cache=cache)

    # load without experiments - should only get the first result
    base_res = cache.load_results()
    assert len(base_res.model_results) == 1
    assert base_res.model_results[0].experiment_name is None

    base_res = cache.load_results(experiment_kwargs=[params_1, params_2])
    assert len(base_res.model_results) == 2

    # load all experiments
    experiment_res = cache.load_results(load_experiments=LoadExperimentEnum.MATCH_NAME)
    assert len(experiment_res.model_results) == 3

    # don't load experiments
    experiment_res = cache.load_results(
        load_experiments=LoadExperimentEnum.NO_EXPERIMENTS
    )
    assert len(experiment_res.model_results) == 1

    # load **only** specific experiment by kwargs
    only_named_experiment_res = cache.load_results(
        load_experiments=LoadExperimentEnum.MATCH_KWARGS,
        experiment_kwargs=params_1,
    )
    assert len(only_named_experiment_res.model_results) == 1
    assert only_named_experiment_res.model_results[0].experiment_name == "a_test"

    model_meta_res = cache.load_results(
        models=[model.mteb_model_meta],
        load_experiments=LoadExperimentEnum.MATCH_NAME,
        experiment_kwargs=params_1,
    )
    assert len(model_meta_res.model_results) == 1

    # load specific experiment with model meta filter
    model_meta_res = cache.load_results(
        models=[model.mteb_model_meta.name],
        load_experiments=LoadExperimentEnum.MATCH_NAME,
        experiment_kwargs=params_2,
    )
    assert len(model_meta_res.model_results) == 1
    assert model_meta_res.model_results[0].experiment_name == "a_test__b_test2"

    model_meta_res = cache.load_results(
        models=[model.mteb_model_meta],
        load_experiments=LoadExperimentEnum.MATCH_KWARGS,
    )
    assert len(model_meta_res.model_results) == 1

    model_meta_res = cache.load_results(
        models=[model.mteb_model_meta],
        load_experiments=LoadExperimentEnum.NO_EXPERIMENTS,
    )
    assert len(model_meta_res.model_results) == 1

    model_meta_res = cache.load_results(
        models=[model.mteb_model_meta],
        load_experiments=LoadExperimentEnum.MATCH_KWARGS,
    )
    assert len(model_meta_res.model_results) == 1
    assert model_meta_res.model_results[0].experiment_name == "a_test__b_test2"

    # load experiments with model name filter
    model_meta_res = cache.load_results(
        models=[model.mteb_model_meta.name],
        load_experiments=LoadExperimentEnum.MATCH_NAME,
    )
    assert len(model_meta_res.model_results) == 3

    # load specific experiment with model name filter
    model_meta_res = cache.load_results(
        models=[model.mteb_model_meta.name],
        load_experiments=LoadExperimentEnum.MATCH_KWARGS,
        experiment_kwargs=[model.mteb_model_meta.experiment_kwargs],
    )
    assert len(model_meta_res.model_results) == 1
    assert model_meta_res.model_results[0].experiment_name == "a_test__b_test2"

    model_meta_res = cache.load_results(
        models=[model.mteb_model_meta],
        load_experiments=LoadExperimentEnum.MATCH_NAME,
    )
    assert len(model_meta_res.model_results) == 3

    model_meta_res = cache.load_results(
        models=[model.mteb_model_meta],
        load_experiments=LoadExperimentEnum.MATCH_KWARGS,
    )
    assert len(model_meta_res.model_results) == 1

    model_meta_res = cache.load_results(
        models=[model.mteb_model_meta.name],
        load_experiments=LoadExperimentEnum.MATCH_KWARGS,
    )
    assert len(model_meta_res.model_results) == 1

    model_meta_res = cache.load_results(
        experiment_kwargs=[model.mteb_model_meta.experiment_kwargs],
    )
    assert len(model_meta_res.model_results) == 1


def _setup_fake_remote(tmp_path: Path) -> tuple[Path, Path]:
    """Set up a fake remote git repository with initial commit."""
    cache_path = tmp_path / "cache"
    remote_path = cache_path / "remote"
    remote_path.mkdir(parents=True)

    # Initialize with explicit default branch to 'main' to avoids issues with git configurations that default to 'main' already
    subprocess.run(
        ["git", "init", "-b", "main"],
        cwd=remote_path,
        check=True,
        capture_output=True,
    )
    subprocess.run(
        ["git", "config", "user.email", "test@mteb.com"],
        cwd=remote_path,
        check=True,
        capture_output=True,
    )
    subprocess.run(
        ["git", "config", "user.name", "MTEB Test"],
        cwd=remote_path,
        check=True,
        capture_output=True,
    )
    subprocess.run(
        [
            "git",
            "remote",
            "add",
            "origin",
            "https://github.com/embeddings-benchmark/results",
        ],
        cwd=remote_path,
        check=True,
        capture_output=True,
    )

    (remote_path / "README.md").write_text("# MTEB Results\n")
    subprocess.run(
        ["git", "add", "README.md"],
        cwd=remote_path,
        check=True,
        capture_output=True,
    )
    subprocess.run(
        ["git", "commit", "-m", "Initial commit"],
        cwd=remote_path,
        check=True,
        capture_output=True,
    )

    return cache_path, remote_path


def _setup_test_model_results(cache_path: Path) -> tuple[ModelMeta, list[str]]:
    """Generate test results by evaluating the baseline random encoder on a mock task."""
    cache = ResultCache(cache_path=cache_path)
    task = MockRetrievalTask()
    model = mteb.get_model("mteb/baseline-random-encoder")

    mteb.evaluate(model, task, cache=cache)

    model_meta = model.mteb_model_meta
    model_name_path = model_meta.model_name_as_path()
    revision = cast("str", model_meta.revision)
    model_dir = cache.cache_path / "results" / model_name_path / revision
    result_files = [
        result_file.name
        for result_file in model_dir.glob("*.json")
        if result_file.name != "model_meta.json"
    ]
    return model_meta, result_files


def test_submit_results_with_fake_remote(tmp_path: Path):
    """Comprehensive test for submit_results workflow: verifies file copying, commit creation, branch restoration, and pre-flight checks."""
    cache_path, remote_path = _setup_fake_remote(tmp_path)
    test_model, result_files_copied = _setup_test_model_results(cache_path)

    revision = cast("str", test_model.revision)
    cache = ResultCache(cache_path=cache_path)

    # Avoid fetching from the remote so the test remains hermetic in CI.
    with patch.object(cache, "download_from_remote", return_value=None):
        # Verify whether pre-flight checks detect uncommitted changes (error path)
        unrelated_file = remote_path / "unrelated_staged_file.txt"
        unrelated_file.write_text("This should not be committed with results")

        subprocess.run(
            ["git", "add", "unrelated_staged_file.txt"],
            cwd=remote_path,
            check=True,
            capture_output=True,
        )

        with pytest.raises(RuntimeError, match="uncommitted changes"):
            cache.submit_results(models=[test_model], create_pr=False)

        subprocess.run(
            ["git", "reset", "HEAD", "unrelated_staged_file.txt"],
            cwd=remote_path,
            check=True,
            capture_output=True,
        )
        unrelated_file.unlink()

        # Verify successful submission workflow
        result = subprocess.run(
            ["git", "rev-parse", "--abbrev-ref", "HEAD"],
            check=False,
            cwd=remote_path,
            capture_output=True,
            text=True,
        )
        original_branch = result.stdout.strip()
        assert original_branch == "main"

        result = cache.submit_results(models=[test_model], create_pr=False)

        assert result["status"] == "ready_for_submission"
        assert result["result_count"] == len(result_files_copied)
        commit_sha = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=True,
            cwd=remote_path,
            capture_output=True,
            text=True,
        ).stdout.strip()
        assert commit_sha
        check = subprocess.run(
            ["git", "cat-file", "-t", commit_sha],
            check=False,
            cwd=remote_path,
            capture_output=True,
            text=True,
        )
        assert check.returncode == 0
        assert check.stdout.strip() == "commit"

        for filename in result_files_copied:
            check = subprocess.run(
                ["git", "ls-tree", "-r", "--name-only", commit_sha],
                check=False,
                cwd=remote_path,
                capture_output=True,
                text=True,
            )
            assert check.returncode == 0
            expected_path = f"{test_model.model_name_as_path()}/{revision}/{filename}"
            assert expected_path in check.stdout, (
                f"File {expected_path} not found in commit {commit_sha}"
            )

        expected_run_settings = (
            f"{test_model.model_name_as_path()}/{revision}/run_settings.jsonl"
        )
        assert expected_run_settings in check.stdout, (
            f"File {expected_run_settings} not found in commit {commit_sha}"
        )

        # For manual submission, verify we're on the submission branch with the commit
        result_after = subprocess.run(
            ["git", "rev-parse", "--abbrev-ref", "HEAD"],
            check=False,
            cwd=remote_path,
            capture_output=True,
            text=True,
        )
        current_branch = result_after.stdout.strip()
        assert current_branch.startswith("mteb-results-"), (
            f"Should be on submission branch but got '{current_branch}'"
        )


def test_submit_results(tmp_path: Path):
    cache_path, remote_path = _setup_fake_remote(tmp_path)
    test_model, result_files_copied = _setup_test_model_results(cache_path)

    cache = ResultCache(cache_path=cache_path)

    with patch.object(cache, "download_from_remote", return_value=None):
        initial_result = cache.submit_results(models=[test_model], create_pr=False)
        assert initial_result["status"] == "ready_for_submission"
        assert initial_result["result_count"] > 0

        submission_branch = subprocess.run(
            ["git", "rev-parse", "--abbrev-ref", "HEAD"],
            cwd=remote_path,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()

        commit_sha = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=True,
            cwd=remote_path,
            capture_output=True,
            text=True,
        ).stdout.strip()
        check = subprocess.run(
            ["git", "ls-tree", "-r", "--name-only", commit_sha],
            check=True,
            cwd=remote_path,
            capture_output=True,
            text=True,
        )
        expected_run_settings = f"{test_model.model_name_as_path()}/{test_model.revision}/run_settings.jsonl"
        assert expected_run_settings in check.stdout, (
            f"File {expected_run_settings} not found in commit {commit_sha}"
        )

        # Merge submission branch back to main to simulate files being integrated
        subprocess.run(
            ["git", "checkout", "main"],
            cwd=remote_path,
            check=True,
            capture_output=True,
        )
        subprocess.run(
            ["git", "merge", submission_branch, "-m", "Integrate results"],
            cwd=remote_path,
            check=True,
            capture_output=True,
        )


def test_pr_creation_failure_cleans_up_branch(tmp_path: Path):
    """Verify that failed PR creation cleans up temporary branch and restores original branch."""
    cache_path, remote_path = _setup_fake_remote(tmp_path)
    test_model, _ = _setup_test_model_results(cache_path)

    cache = ResultCache(cache_path=cache_path)
    result = subprocess.run(
        ["git", "rev-parse", "--abbrev-ref", "HEAD"],
        check=False,
        cwd=remote_path,
        capture_output=True,
        text=True,
    )
    original_branch = result.stdout.strip()

    # Avoid fetching from the remote so the test reliably reaches PR creation.
    # Mock _create_pull_request to fail.
    with (
        patch.object(cache, "download_from_remote", return_value=None),
        patch(
            "mteb._reversible_workflow.git_utils.create_pull_request",
            side_effect=Exception("GitHub API error"),
        ),
        pytest.raises(Exception, match="GitHub API error"),
    ):
        cache.submit_results(models=[test_model], create_pr=True)

    # Verify user is back on original branch
    result = subprocess.run(
        ["git", "rev-parse", "--abbrev-ref", "HEAD"],
        check=False,
        cwd=remote_path,
        capture_output=True,
        text=True,
    )
    current_branch = result.stdout.strip()
    assert current_branch == original_branch, (
        f"Should be on original branch '{original_branch}' but got '{current_branch}'"
    )

    # Verify temporary branch was deleted (should not appear in branch list)
    result = subprocess.run(
        ["git", "branch"],
        check=False,
        cwd=remote_path,
        capture_output=True,
        text=True,
    )
    # Strip whitespace and the leading * (for current branch indicator)
    branches = [
        b.strip().lstrip("*").strip() for b in result.stdout.split("\n") if b.strip()
    ]
    # Should only have 'main' branch, no temporary branches
    assert all(not b.startswith("mteb-results-") for b in branches), (
        f"Temporary branch should be deleted but found: {branches}"
    )


def _setup_experiment_model_results(
    cache_path: Path, **kwargs: Any
) -> tuple[ModelMeta, list[str]]:
    """Generate experiment results by evaluating the baseline random encoder with experiment kwargs."""
    cache = ResultCache(cache_path=cache_path)
    task = MockRetrievalTask()
    model = mteb.get_model("mteb/baseline-random-encoder", **kwargs)
    mteb.evaluate(model, task, cache=cache)

    model_meta = model.mteb_model_meta
    revision = cast("str", model_meta.revision)
    experiment_name = cast("str", model_meta.experiment_name)
    model_dir = (
        cache.cache_path
        / "results"
        / model_meta.model_name_as_path()
        / revision
        / "experiments"
        / experiment_name
    )
    result_files = [
        f.name for f in model_dir.glob("*.json") if f.name != "model_meta.json"
    ]
    return model_meta, result_files


def _committed_files(remote_path: Path) -> str:
    commit_sha = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=remote_path,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    return subprocess.run(
        ["git", "ls-tree", "-r", "--name-only", commit_sha],
        cwd=remote_path,
        check=True,
        capture_output=True,
        text=True,
    ).stdout


def test_submit_results_experiments_only(tmp_path: Path):
    """Submit results for a model that only has experiment results (no base results)."""
    cache_path, remote_path = _setup_fake_remote(tmp_path)
    test_model, result_files = _setup_experiment_model_results(cache_path, a="test")
    revision = cast("str", test_model.revision)
    experiment_name = cast("str", test_model.experiment_name)
    cache = ResultCache(cache_path=cache_path)

    with patch.object(cache, "download_from_remote", return_value=None):
        result = cache.submit_results(models=[test_model], create_pr=False)

    assert result["status"] == "ready_for_submission"
    assert result["result_count"] == len(result_files)

    committed = _committed_files(remote_path)
    for filename in result_files:
        expected = f"{test_model.model_name_as_path()}/{revision}/experiments/{experiment_name}/{filename}"
        assert expected in committed, f"{expected} not found in commit"
