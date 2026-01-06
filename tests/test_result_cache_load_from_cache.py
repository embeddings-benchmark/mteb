"""Test cases for the load_from_cache and _rebuild_from_full_repository methods."""

from unittest.mock import MagicMock, patch

import pytest

from mteb.cache import ResultCache
from mteb.results import BenchmarkResults


class TestLoadFromCache:
    """Test the load_from_cache method."""

    def test_rebuild_flag_forces_full_rebuild(self, tmp_path):
        """Test rebuild=True bypasses cache and forces rebuild."""
        cache = ResultCache(cache_path=tmp_path)
        quick_cache_path = tmp_path / "quick_cache.json"
        quick_cache_path.write_text('{"test": "should be ignored"}')

        with patch.object(cache, "_rebuild_from_full_repository") as mock_rebuild:
            mock_result = MagicMock(spec=BenchmarkResults)
            mock_rebuild.return_value = mock_result
            result = cache.load_from_cache(quick_cache_path, rebuild=True)
            mock_rebuild.assert_called_once_with(quick_cache_path)
            assert result == mock_result

    def test_loading_strategies_in_order(self, tmp_path):
        """Test the 3-tier loading strategy: local -> download -> rebuild."""
        cache = ResultCache(cache_path=tmp_path)
        quick_cache_path = tmp_path / "quick_cache.json"
        mock_result = MagicMock(spec=BenchmarkResults)

        # Test 1: Load from existing local cache
        quick_cache_path.write_text("{}")
        with patch("mteb.results.BenchmarkResults.from_disk") as mock_from_disk:
            mock_from_disk.return_value = mock_result
            result = cache.load_from_cache(quick_cache_path, rebuild=False)
            mock_from_disk.assert_called_once_with(quick_cache_path)
            assert result == mock_result

        # Test 2: Download from cached-data branch when local doesn't exist
        quick_cache_path.unlink()  # Remove local file
        with (
            patch.object(cache, "_download_cached_results_from_branch") as mock_dl,
            patch("mteb.results.BenchmarkResults.from_disk") as mock_from_disk,
        ):
            mock_dl.return_value = quick_cache_path
            mock_from_disk.return_value = mock_result
            result = cache.load_from_cache(quick_cache_path, rebuild=False)
            mock_dl.assert_called_once_with(output_path=quick_cache_path)
            assert result == mock_result

        # Test 3: Fallback to full rebuild when download fails
        with (
            patch.object(cache, "_download_cached_results_from_branch") as mock_dl,
            patch.object(cache, "_rebuild_from_full_repository") as mock_rebuild,
        ):
            mock_dl.side_effect = Exception("Download failed")
            mock_rebuild.return_value = mock_result
            result = cache.load_from_cache(quick_cache_path, rebuild=False)
            mock_rebuild.assert_called_once_with(quick_cache_path)
            assert result == mock_result

    def test_corrupt_cache_triggers_fallback(self, tmp_path):
        """Test that corrupt cache files trigger next strategy."""
        cache = ResultCache(cache_path=tmp_path)
        quick_cache_path = tmp_path / "quick_cache.json"
        quick_cache_path.write_text("invalid json {{{")

        with (
            patch("mteb.results.BenchmarkResults.from_disk") as mock_from_disk,
            patch.object(cache, "_download_cached_results_from_branch") as mock_dl,
            patch.object(cache, "_rebuild_from_full_repository") as mock_rebuild,
        ):
            mock_from_disk.side_effect = Exception("Invalid JSON")
            mock_dl.side_effect = Exception("Download failed")
            mock_result = MagicMock(spec=BenchmarkResults)
            mock_rebuild.return_value = mock_result
            result = cache.load_from_cache(quick_cache_path, rebuild=False)
            mock_rebuild.assert_called_once_with(quick_cache_path)
            assert result == mock_result


class TestRebuildFromFullRepository:
    """Test the _rebuild_from_full_repository method."""

    def test_full_rebuild_process(self, tmp_path):
        """Test rebuild downloads repo, loads results, and saves cache."""
        cache = ResultCache(cache_path=tmp_path)
        quick_cache_path = tmp_path / "cache.json"

        with (
            patch.object(cache, "download_from_remote") as mock_download,
            patch.object(cache, "load_results") as mock_load_results,
            patch("mteb.get_model_metas") as mock_get_model_metas,
        ):
            # Mock model metas - None names should be filtered
            meta1 = MagicMock()
            meta1.name = "model1"
            meta2 = MagicMock()
            meta2.name = "model2"
            meta3 = MagicMock()
            meta3.name = None  # Should be filtered
            mock_get_model_metas.return_value = [meta1, meta2, meta3]
            mock_results = MagicMock(spec=BenchmarkResults)
            mock_load_results.return_value = mock_results

            result = cache._rebuild_from_full_repository(quick_cache_path)

            mock_download.assert_called_once()
            mock_load_results.assert_called_once_with(
                models=["model1", "model2"],  # None filtered out
                only_main_score=True,
                require_model_meta=False,
                include_remote=True,
            )
            mock_results.to_disk.assert_called_once_with(quick_cache_path)
            assert result == mock_results

    def test_rebuild_error_propagation(self, tmp_path):
        """Test that errors during rebuild are properly propagated."""
        cache = ResultCache(cache_path=tmp_path)
        quick_cache_path = tmp_path / "cache.json"

        # Test download failure
        with patch.object(cache, "download_from_remote") as mock_download:
            mock_download.side_effect = Exception("Network error")
            with pytest.raises(Exception, match="Network error"):
                cache._rebuild_from_full_repository(quick_cache_path)

        # Test load_results failure
        with (
            patch.object(cache, "download_from_remote"),
            patch.object(cache, "load_results") as mock_load_results,
            patch("mteb.get_model_metas") as mock_get_model_metas,
        ):
            meta = MagicMock()
            meta.name = "model1"
            mock_get_model_metas.return_value = [meta]
            mock_load_results.side_effect = Exception("Load failed")
            with pytest.raises(Exception, match="Load failed"):
                cache._rebuild_from_full_repository(quick_cache_path)
