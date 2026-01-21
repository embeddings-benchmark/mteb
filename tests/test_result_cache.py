"""Test cases for the ResultCache class in the mteb.cache module."""

import gzip
from pathlib import Path
from typing import cast
from unittest.mock import Mock, patch

import numpy as np
import pytest
import requests

import mteb
from mteb.cache import ResultCache
from mteb.results import TaskResult
from tests.mock_tasks import MockMultilingualClusteringTask

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
        "baseline/random-encoder-baseline"
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


# Tests for _download_cached_results_from_branch method


class TestDownloadCachedResultsFromBranch:
    """Test the _download_cached_results_from_branch method."""

    @patch("requests.get")
    def test_successful_download(
        self, mock_get, tmp_path, mock_benchmark_json, mock_gzipped_content
    ):
        """Test successful download and decompression, including parent directory creation."""
        cache = ResultCache(cache_path=tmp_path)

        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.headers = {"content-type": "application/gzip"}
        mock_response.content = mock_gzipped_content(mock_benchmark_json)
        mock_get.return_value = mock_response

        # Test basic download
        output_path = tmp_path / "test_cached_results.json"
        result_path = cache._download_cached_results_from_branch(
            output_path=output_path
        )

        assert result_path == output_path
        assert result_path.exists()
        assert result_path.read_text(encoding="utf-8") == mock_benchmark_json
        mock_response.raise_for_status.assert_called_once()

        # Test parent directory creation
        nested_output_path = tmp_path / "nested" / "dir" / "test_cache.json"
        nested_result_path = cache._download_cached_results_from_branch(
            output_path=nested_output_path
        )

        assert nested_result_path.exists()
        assert nested_result_path.read_text(encoding="utf-8") == mock_benchmark_json

    @patch("requests.get")
    def test_file_too_large(self, mock_get, tmp_path):
        """Test that oversized files raise ValueError."""
        cache = ResultCache(cache_path=tmp_path)

        # Create gzipped content that's 51MB (larger than the limit)
        # We create the actual bytes directly rather than compressing,
        # since we're testing the downloaded size check
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.headers = {"content-type": "application/gzip"}
        mock_response.content = b"x" * (51 * 1024 * 1024)  # 51MB of gzipped data
        mock_get.return_value = mock_response

        with pytest.raises(ValueError, match="Downloaded file too large"):
            cache._download_cached_results_from_branch(max_size_mb=50)

    @pytest.mark.parametrize(
        "error_type,exception_class",
        [
            ("timeout", requests.exceptions.Timeout),
            ("connection", requests.exceptions.ConnectionError),
            ("http_404", requests.exceptions.HTTPError),
            ("invalid_gzip", gzip.BadGzipFile),
        ],
    )
    @patch("requests.get")
    def test_error_handling_consolidated(
        self, mock_get, tmp_path, error_type, exception_class
    ):
        """Test various error conditions in a consolidated manner."""
        cache = ResultCache(cache_path=tmp_path)

        if error_type == "timeout":
            mock_get.side_effect = requests.exceptions.Timeout("Request timed out")
        elif error_type == "connection":
            mock_get.side_effect = requests.exceptions.ConnectionError(
                "Connection failed"
            )
        elif error_type == "http_404":
            mock_response = Mock()
            mock_response.status_code = 404
            mock_response.raise_for_status.side_effect = requests.exceptions.HTTPError(
                "404 Not Found"
            )
            mock_get.return_value = mock_response
        elif error_type == "invalid_gzip":
            mock_response = Mock()
            mock_response.status_code = 200
            mock_response.headers = {"content-type": "application/gzip"}
            mock_response.content = b"not gzipped data"
            mock_get.return_value = mock_response

        with pytest.raises(exception_class):
            cache._download_cached_results_from_branch(timeout=30)

    @patch("requests.get")
    def test_default_output_path(
        self, mock_get, tmp_path, mock_benchmark_json, mock_gzipped_content
    ):
        """Test that default output path is mteb/leaderboard/__cached_results.json when none provided."""
        cache = ResultCache(cache_path=tmp_path)

        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.headers = {"content-type": "application/gzip"}
        mock_response.content = mock_gzipped_content(mock_benchmark_json)
        mock_get.return_value = mock_response

        result_path = cache._download_cached_results_from_branch()

        # Default path should be mteb/leaderboard/__cached_results.json
        # Get the mteb package directory
        mteb_package_dir = Path(mteb.__file__).parent
        expected_path = mteb_package_dir / "leaderboard" / "__cached_results.json"
        assert result_path == expected_path
        assert result_path.exists()

    @pytest.mark.parametrize(
        "content_type,should_raise_exception",
        [
            ("application/gzip", False),  # Expected type
            ("text/html", True),  # Unexpected type - should raise exception
            ("", False),  # Empty content-type should not raise exception
        ],
    )
    @patch("requests.get")
    def test_content_type_handling(
        self,
        mock_get,
        tmp_path,
        content_type,
        should_raise_exception,
        mock_benchmark_json,
        mock_gzipped_content,
    ):
        """Test various content-type header scenarios."""
        cache = ResultCache(cache_path=tmp_path)

        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.headers = {"content-type": content_type}
        mock_response.content = mock_gzipped_content(mock_benchmark_json)
        mock_get.return_value = mock_response

        output_path = tmp_path / "test.json"

        if should_raise_exception:
            with pytest.raises(Exception, match="Unexpected content-type"):
                cache._download_cached_results_from_branch(output_path=output_path)
        else:
            result_path = cache._download_cached_results_from_branch(
                output_path=output_path
            )
            assert result_path.exists()
            assert result_path.read_text(encoding="utf-8") == mock_benchmark_json

    @pytest.mark.parametrize(
        "file_size,max_size_mb,should_fail",
        [
            (1024, 1, False),  # Small file - OK
            (1024 * 1024, 1, False),  # At limit - OK
            (51 * 1024 * 1024, 50, True),  # Over limit - Fail
        ],
    )
    @patch("requests.get")
    def test_file_size_validation(
        self,
        mock_get,
        tmp_path,
        file_size,
        max_size_mb,
        should_fail,
        mock_gzipped_content,
    ):
        """Test file size validation with various sizes.

        Note: file_size represents the size of the gzipped (downloaded) content,
        not the uncompressed size, since validation happens on download.
        """
        cache = ResultCache(cache_path=tmp_path)

        # For size validation, we care about the downloaded (gzipped) size
        # So we mock the response.content directly with the desired byte size
        # But we still need valid gzip content for decompression
        if should_fail:
            # For failure cases, just use raw bytes (will fail on decompression but that's after size check)
            mock_response = Mock()
            mock_response.status_code = 200
            mock_response.headers = {"content-type": "application/gzip"}
            mock_response.content = b"x" * file_size
            mock_get.return_value = mock_response

            with pytest.raises(ValueError, match="Downloaded file too large"):
                cache._download_cached_results_from_branch(max_size_mb=max_size_mb)
        else:
            # For success cases, use valid gzipped content
            # Create small content and gzip it
            content = "x" * min(file_size, 1000)  # Keep content small for faster tests
            gzipped = mock_gzipped_content(content)

            mock_response = Mock()
            mock_response.status_code = 200
            mock_response.headers = {"content-type": "application/gzip"}
            mock_response.content = gzipped
            mock_get.return_value = mock_response

            result_path = cache._download_cached_results_from_branch(
                max_size_mb=max_size_mb
            )
            assert result_path.exists()
