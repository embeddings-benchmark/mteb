"""Integration tests for leaderboard caching workflow with ResultCache."""

from unittest.mock import Mock, patch

import pytest

from mteb.cache import ResultCache


class TestIntegrationScenarios:
    """Test integration scenarios that use ResultCache method."""

    @patch("mteb.cache.requests.get")
    def test_full_caching_workflow_success(
        self, mock_get, tmp_path, mock_benchmark_json, mock_gzipped_content
    ):
        """Test the complete workflow from download to file write via ResultCache."""
        cache = ResultCache(cache_path=tmp_path)

        # Setup mocks for successful workflow
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.headers = {"content-type": "application/gzip"}
        mock_response.content = mock_gzipped_content(mock_benchmark_json)
        mock_get.return_value = mock_response

        # Test the full workflow through ResultCache
        output_path = tmp_path / "cached_results.json"
        result_path = cache.download_cached_results_from_branch(output_path=output_path)

        # Verify the workflow completed correctly
        assert result_path == output_path
        assert result_path.exists()
        assert result_path.read_text(encoding="utf-8") == mock_benchmark_json
        mock_get.assert_called_once()
        mock_response.raise_for_status.assert_called_once()

    @patch("mteb.cache.requests.get")
    def test_download_failure_handling(self, mock_get, tmp_path):
        """Test that download failures are properly handled in the workflow."""
        cache = ResultCache(cache_path=tmp_path)
        mock_get.side_effect = Exception("Network error")

        with pytest.raises(Exception, match="Network error"):
            cache.download_cached_results_from_branch()


@pytest.fixture
def mock_benchmark_json():
    """Sample valid benchmark JSON for testing."""
    return '{"results": [{"task": "test_task", "score": 0.85, "model": "test_model"}]}'


@pytest.fixture
def mock_gzipped_content():
    """Generate mock gzipped content for testing."""
    import gzip
    import io

    def _generate_gzipped(text_content: str) -> bytes:
        buffer = io.BytesIO()
        with gzip.open(buffer, "wt", encoding="utf-8") as gz_file:
            gz_file.write(text_content)
        return buffer.getvalue()

    return _generate_gzipped
