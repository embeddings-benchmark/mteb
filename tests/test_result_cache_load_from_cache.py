"""Test cases for the load_from_cache and _rebuild_from_full_repository methods."""

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from mteb.cache import ResultCache
from mteb.results import BenchmarkResults


class TestLoadFromCache:
    """Test the load_from_cache method."""

    def test_load_from_cache_with_rebuild_true(self, tmp_path):
        """Test that rebuild=True forces a full rebuild."""
        cache = ResultCache(cache_path=tmp_path)

        # Create a mock quick cache file that should be ignored
        quick_cache_path = tmp_path / "quick_cache.json"
        quick_cache_path.write_text('{"test": "should be ignored"}')

        # Mock _rebuild_from_full_repository to verify it's called
        with patch.object(cache, "_rebuild_from_full_repository") as mock_rebuild:
            mock_result = MagicMock(spec=BenchmarkResults)
            mock_rebuild.return_value = mock_result

            result = cache.load_from_cache(
                quick_cache_path=quick_cache_path, rebuild=True
            )

            # Should call rebuild directly, ignoring existing cache
            mock_rebuild.assert_called_once_with(quick_cache_path)
            assert result == mock_result

    def test_load_from_cache_existing_file(self, tmp_path):
        """Test loading from existing quick cache file when rebuild=False."""
        cache = ResultCache(cache_path=tmp_path)

        # Create a valid quick cache file
        quick_cache_path = tmp_path / "quick_cache.json"

        # Mock BenchmarkResults.from_disk to return a result
        with patch("mteb.results.BenchmarkResults.from_disk") as mock_from_disk:
            mock_result = MagicMock(spec=BenchmarkResults)
            mock_from_disk.return_value = mock_result

            # Mock other methods to verify they're NOT called
            with (
                patch.object(
                    cache, "_download_cached_results_from_branch"
                ) as mock_download,
                patch.object(cache, "_rebuild_from_full_repository") as mock_rebuild,
            ):
                # Create the file after patching to ensure it exists
                quick_cache_path.write_text("{}")

                result = cache.load_from_cache(
                    quick_cache_path=quick_cache_path, rebuild=False
                )

                # Should load from existing file
                mock_from_disk.assert_called_once_with(quick_cache_path)
                mock_download.assert_not_called()
                mock_rebuild.assert_not_called()
                assert result == mock_result

    def test_load_from_cache_download_success(self, tmp_path):
        """Test downloading from cached-data branch when local cache doesn't exist."""
        cache = ResultCache(cache_path=tmp_path)
        quick_cache_path = tmp_path / "quick_cache.json"

        # Ensure the file doesn't exist
        assert not quick_cache_path.exists()

        with (
            patch.object(
                cache, "_download_cached_results_from_branch"
            ) as mock_download,
            patch("mteb.results.BenchmarkResults.from_disk") as mock_from_disk,
            patch.object(cache, "_rebuild_from_full_repository") as mock_rebuild,
        ):
            mock_download.return_value = quick_cache_path
            mock_result = MagicMock(spec=BenchmarkResults)
            mock_from_disk.return_value = mock_result

            result = cache.load_from_cache(
                quick_cache_path=quick_cache_path, rebuild=False
            )

            # Should download and then load
            mock_download.assert_called_once_with(output_path=quick_cache_path)
            mock_from_disk.assert_called_once_with(quick_cache_path)
            mock_rebuild.assert_not_called()
            assert result == mock_result

    def test_load_from_cache_fallback_to_rebuild(self, tmp_path):
        """Test fallback to full rebuild when both cache and download fail."""
        cache = ResultCache(cache_path=tmp_path)
        quick_cache_path = tmp_path / "quick_cache.json"

        with (
            patch.object(
                cache, "_download_cached_results_from_branch"
            ) as mock_download,
            patch.object(cache, "_rebuild_from_full_repository") as mock_rebuild,
        ):
            # Simulate download failure
            mock_download.side_effect = Exception("Download failed")
            mock_result = MagicMock(spec=BenchmarkResults)
            mock_rebuild.return_value = mock_result

            result = cache.load_from_cache(
                quick_cache_path=quick_cache_path, rebuild=False
            )

            # Should try download, fail, then rebuild
            mock_download.assert_called_once_with(output_path=quick_cache_path)
            mock_rebuild.assert_called_once_with(quick_cache_path)
            assert result == mock_result

    def test_load_from_cache_corrupt_cache_fallback(self, tmp_path):
        """Test fallback when existing cache file is corrupt."""
        cache = ResultCache(cache_path=tmp_path)
        quick_cache_path = tmp_path / "quick_cache.json"

        # Create a corrupt cache file
        quick_cache_path.write_text("corrupted data")

        with (
            patch("mteb.results.BenchmarkResults.from_disk") as mock_from_disk,
            patch.object(
                cache, "_download_cached_results_from_branch"
            ) as mock_download,
            patch.object(cache, "_rebuild_from_full_repository") as mock_rebuild,
        ):
            # First call fails (corrupt), subsequent calls work
            mock_from_disk.side_effect = [
                Exception("Corrupt file"),
                MagicMock(spec=BenchmarkResults),
            ]
            mock_download.return_value = quick_cache_path

            cache.load_from_cache(quick_cache_path=quick_cache_path, rebuild=False)

            # Should try loading, fail, download, then load again
            assert mock_from_disk.call_count == 2
            mock_download.assert_called_once_with(output_path=quick_cache_path)
            mock_rebuild.assert_not_called()

    def test_load_from_cache_default_path(self, tmp_path):
        """Test that default quick_cache_path is used when not specified."""
        cache = ResultCache(cache_path=tmp_path)

        with patch.object(cache, "_rebuild_from_full_repository") as mock_rebuild:
            mock_result = MagicMock(spec=BenchmarkResults)
            mock_rebuild.return_value = mock_result

            # Call without specifying quick_cache_path
            cache.load_from_cache(rebuild=True)

            # Check that the default path was used
            call_args = mock_rebuild.call_args[0]
            assert len(call_args) == 1
            default_path = call_args[0]
            assert isinstance(default_path, Path)
            assert default_path.name == "__cached_results.json"
            assert "leaderboard" in str(default_path)


class TestRebuildFromFullRepository:
    """Test the _rebuild_from_full_repository method."""

    def test_rebuild_from_full_repository(self, tmp_path):
        """Test the full rebuild process."""
        cache = ResultCache(cache_path=tmp_path)
        quick_cache_path = tmp_path / "rebuilt_cache.json"

        # Mock all the dependencies
        with (
            patch.object(cache, "download_from_remote") as mock_download,
            patch.object(cache, "load_results") as mock_load_results,
            patch("mteb.get_model_metas") as mock_get_model_metas,
        ):
            # Setup mocks
            mock_model_meta1 = MagicMock()
            mock_model_meta1.name = "model1"
            mock_model_meta2 = MagicMock()
            mock_model_meta2.name = "model2"
            mock_model_meta3 = MagicMock()
            mock_model_meta3.name = None  # Should be filtered out

            mock_get_model_metas.return_value = [
                mock_model_meta1,
                mock_model_meta2,
                mock_model_meta3,
            ]

            mock_results = MagicMock(spec=BenchmarkResults)
            mock_load_results.return_value = mock_results

            result = cache._rebuild_from_full_repository(quick_cache_path)

            # Verify the sequence of calls
            mock_download.assert_called_once()
            mock_get_model_metas.assert_called_once()
            mock_load_results.assert_called_once_with(
                models=["model1", "model2"],  # model3 filtered out (name=None)
                only_main_score=True,
                require_model_meta=False,
                include_remote=True,
            )
            mock_results.to_disk.assert_called_once_with(quick_cache_path)
            assert result == mock_results

    def test_rebuild_handles_download_failure(self, tmp_path):
        """Test that rebuild propagates download failures."""
        cache = ResultCache(cache_path=tmp_path)
        quick_cache_path = tmp_path / "cache.json"

        with patch.object(cache, "download_from_remote") as mock_download:
            mock_download.side_effect = Exception("Network error")

            with pytest.raises(Exception, match="Network error"):
                cache._rebuild_from_full_repository(quick_cache_path)

    def test_rebuild_handles_load_results_failure(self, tmp_path):
        """Test that rebuild handles load_results failures."""
        cache = ResultCache(cache_path=tmp_path)
        quick_cache_path = tmp_path / "cache.json"

        with (
            patch.object(cache, "download_from_remote"),
            patch.object(cache, "load_results") as mock_load_results,
            patch("mteb.get_model_metas") as mock_get_model_metas,
        ):
            mock_model_meta = MagicMock()
            mock_model_meta.name = "model1"
            mock_get_model_metas.return_value = [mock_model_meta]
            mock_load_results.side_effect = Exception("Load failed")

            with pytest.raises(Exception, match="Load failed"):
                cache._rebuild_from_full_repository(quick_cache_path)

    def test_rebuild_saves_cache_even_if_empty(self, tmp_path):
        """Test that rebuild saves cache file even with empty results."""
        cache = ResultCache(cache_path=tmp_path)
        quick_cache_path = tmp_path / "empty_cache.json"

        with (
            patch.object(cache, "download_from_remote"),
            patch.object(cache, "load_results") as mock_load_results,
            patch("mteb.get_model_metas") as mock_get_model_metas,
        ):
            # No models with valid names
            mock_model_meta = MagicMock()
            mock_model_meta.name = None
            mock_get_model_metas.return_value = [mock_model_meta]

            mock_results = MagicMock(spec=BenchmarkResults)
            mock_load_results.return_value = mock_results

            result = cache._rebuild_from_full_repository(quick_cache_path)

            # Should still save even with empty model list
            mock_load_results.assert_called_once_with(
                models=[],  # Empty list since all models filtered
                only_main_score=True,
                require_model_meta=False,
                include_remote=True,
            )
            mock_results.to_disk.assert_called_once_with(quick_cache_path)
            assert result == mock_results
