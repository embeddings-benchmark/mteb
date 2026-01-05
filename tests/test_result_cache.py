"""Test cases for the ResultCache class in the mteb.cache module."""

import gzip
import json
from pathlib import Path
from typing import cast
from unittest.mock import Mock, patch

import pytest
import requests

import mteb
from mteb.cache import ResultCache
from mteb.results import BenchmarkResults, TaskResult
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


class TestLoadQuickcacheFromRemote:
    """Test the _load_quickcache_from_remote method."""

    def test_local_cache_exists(self, tmp_path):
        """Test that if local cache exists, it loads from there without downloading."""
        cache = ResultCache(cache_path=tmp_path)

        # Create a mock cache file with valid JSON content
        cache_file = tmp_path / "cached_results.json"
        test_data = {
            "results": [
                {
                    "model_name": "test-model",
                    "model_revision": "abc123",
                    "task_results": [],
                }
            ]
        }
        cache_file.write_text(json.dumps(test_data), encoding="utf-8")

        # Mock BenchmarkResults.from_disk to return a known object
        mock_results = Mock(spec=BenchmarkResults)
        with patch.object(
            BenchmarkResults, "from_disk", return_value=mock_results
        ) as mock_from_disk:
            result = cache._load_quickcache_from_remote(cache_path=cache_file)

        # Verify the result is what we expected
        assert result is mock_results

        # Verify from_disk was called with the correct path
        mock_from_disk.assert_called_once_with(cache_file)

    def test_local_cache_corrupt_downloads_fresh(self, tmp_path):
        """Test that if local cache is corrupt, it downloads a fresh copy."""
        cache = ResultCache(cache_path=tmp_path)

        # Create a corrupt cache file
        cache_file = tmp_path / "cached_results.json"
        cache_file.write_text("CORRUPT DATA", encoding="utf-8")

        # Mock the methods we expect to be called
        mock_download_result = tmp_path / "downloaded.json"
        mock_benchmark_results = Mock(spec=BenchmarkResults)

        with (
            patch.object(BenchmarkResults, "from_disk") as mock_from_disk,
            patch.object(
                cache,
                "_download_cached_results_from_branch",
                return_value=mock_download_result,
            ) as mock_download,
        ):
            # First call to from_disk should fail (corrupt file)
            # Second call should succeed (after download)
            mock_from_disk.side_effect = [
                Exception("Invalid JSON"),
                mock_benchmark_results,
            ]

            result = cache._load_quickcache_from_remote(cache_path=cache_file)

        # Verify the result
        assert result is mock_benchmark_results

        # Verify download was called
        mock_download.assert_called_once_with(output_path=cache_file)

        # Verify from_disk was called twice (once for corrupt, once after download)
        assert mock_from_disk.call_count == 2

    @patch("mteb.get_model_metas")
    def test_download_fails_fallback_to_full_repo(self, mock_get_model_metas, tmp_path):
        """Test that if download from cached-data fails, it falls back to full repo clone."""
        cache = ResultCache(cache_path=tmp_path)

        # Set up the cache path
        cache_file = tmp_path / "cached_results.json"

        # Mock model metas
        mock_model1 = Mock()
        mock_model1.name = "model-1"
        mock_model2 = Mock()
        mock_model2.name = "model-2"
        mock_get_model_metas.return_value = [mock_model1, mock_model2]

        # Mock results that will be returned by load_results
        mock_benchmark_results = Mock(spec=BenchmarkResults)

        with (
            patch.object(
                cache, "_download_cached_results_from_branch"
            ) as mock_download,
            patch.object(cache, "download_from_remote") as mock_download_remote,
            patch.object(
                cache, "load_results", return_value=mock_benchmark_results
            ) as mock_load_results,
            patch.object(mock_benchmark_results, "to_disk") as mock_to_disk,
        ):
            # Make download from cached-data fail
            mock_download.side_effect = requests.exceptions.HTTPError("404 Not Found")

            result = cache._load_quickcache_from_remote(cache_path=cache_file)

        # Verify the result
        assert result is mock_benchmark_results

        # Verify the fallback sequence was called
        mock_download.assert_called_once_with(output_path=cache_file)
        mock_download_remote.assert_called_once()
        mock_load_results.assert_called_once_with(
            models=["model-1", "model-2"],
            only_main_score=True,
            require_model_meta=False,
            include_remote=True,
        )
        mock_to_disk.assert_called_once_with(cache_file)

    def test_default_cache_path(self, tmp_path):
        """Test that default cache path is used when none is provided."""
        cache = ResultCache(cache_path=tmp_path)

        # Create a mock results object
        mock_results = Mock(spec=BenchmarkResults)

        # Get the expected default path
        mteb_package_dir = Path(mteb.__file__).parent
        expected_default_path = (
            mteb_package_dir / "leaderboard" / "__cached_results.json"
        )

        # Create the file at the default location
        expected_default_path.parent.mkdir(parents=True, exist_ok=True)
        test_data = {"results": []}
        expected_default_path.write_text(json.dumps(test_data), encoding="utf-8")

        try:
            with patch.object(
                BenchmarkResults, "from_disk", return_value=mock_results
            ) as mock_from_disk:
                result = cache._load_quickcache_from_remote()  # No cache_path provided

            # Verify the result
            assert result is mock_results

            # Verify from_disk was called with the default path
            mock_from_disk.assert_called_once_with(expected_default_path)
        finally:
            # Clean up the test file we created
            if expected_default_path.exists():
                expected_default_path.unlink()

    @patch("mteb.get_model_metas")
    def test_full_workflow_no_cache(
        self, mock_get_model_metas, tmp_path, mock_benchmark_json, mock_gzipped_content
    ):
        """Test the full workflow when no cache exists and download succeeds."""
        cache = ResultCache(cache_path=tmp_path)

        # Set up the cache path
        cache_file = tmp_path / "cached_results.json"

        # Mock model metas (not needed for this path but good to have)
        mock_model1 = Mock()
        mock_model1.name = "model-1"
        mock_get_model_metas.return_value = [mock_model1]

        # Mock the successful download
        mock_results = Mock(spec=BenchmarkResults)

        with (
            patch.object(
                cache, "_download_cached_results_from_branch", return_value=cache_file
            ) as mock_download,
            patch.object(
                BenchmarkResults, "from_disk", return_value=mock_results
            ) as mock_from_disk,
        ):
            # Ensure cache file doesn't exist
            assert not cache_file.exists()

            result = cache._load_quickcache_from_remote(cache_path=cache_file)

        # Verify the result
        assert result is mock_results

        # Verify download was called since cache didn't exist
        mock_download.assert_called_once_with(output_path=cache_file)

        # Verify from_disk was called to load the downloaded file
        mock_from_disk.assert_called_once_with(cache_file)

    @patch("mteb.get_model_metas")
    def test_all_strategies_fail(self, mock_get_model_metas, tmp_path):
        """Test behavior when all loading strategies fail."""
        cache = ResultCache(cache_path=tmp_path)

        # Set up the cache path
        cache_file = tmp_path / "cached_results.json"

        # Mock model metas
        mock_model1 = Mock()
        mock_model1.name = "model-1"
        mock_get_model_metas.return_value = [mock_model1]

        with (
            patch.object(
                cache, "_download_cached_results_from_branch"
            ) as mock_download,
            patch.object(cache, "download_from_remote") as mock_download_remote,
            patch.object(cache, "load_results") as mock_load_results,
        ):
            # Make everything fail
            mock_download.side_effect = Exception("Download failed")
            mock_download_remote.side_effect = Exception("Remote clone failed")

            # The method should raise an exception when everything fails
            with pytest.raises(Exception, match="Remote clone failed"):
                cache._load_quickcache_from_remote(cache_path=cache_file)

        # Verify the sequence of attempts
        mock_download.assert_called_once()
        mock_download_remote.assert_called_once()
        # load_results should not be called if download_from_remote fails
        mock_load_results.assert_not_called()
