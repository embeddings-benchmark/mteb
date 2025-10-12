"""Test cases for the ResultCache class in the mteb.cache module."""

from pathlib import Path
from typing import cast

import mteb
from mteb.cache import ResultCache
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
    model_revision_1 = cast(str, model_revision_1)
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
    model_revision_1 = cast(str, model_revision_1)
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
