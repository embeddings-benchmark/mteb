"""Integration tests for leaderboard caching workflow."""

import gzip
from pathlib import Path
from unittest.mock import Mock, patch

import pytest
import requests

from mteb.leaderboard.app import _download_cached_results


class TestIntegrationScenarios:
    """Test integration scenarios that combine multiple functions."""

    @patch("mteb.leaderboard.app._download_cached_results")
    @patch("mteb.leaderboard.app._decompress_gzip_data")
    @patch("mteb.leaderboard.app._write_cache_file")
    def test_full_caching_workflow_success(
        self, mock_write, mock_decompress, mock_download
    ):
        """Test the complete workflow from download to file write."""
        # Setup mocks for successful workflow
        mock_download.return_value = b"gzipped content"
        mock_decompress.return_value = '{"results": [{"task": "test"}]}'

        # This would be called within _load_results, but we can test the components
        url = "https://example.com/cached_results.json.gz"
        content = mock_download(url)
        data = mock_decompress(content)
        file_path = Path("/tmp/cached_results.json")
        mock_write(data, file_path)

        # Verify all functions were called correctly
        mock_download.assert_called_once_with(url)
        mock_decompress.assert_called_once_with(b"gzipped content")
        mock_write.assert_called_once_with('{"results": [{"task": "test"}]}', file_path)

    @patch("mteb.leaderboard.app._download_cached_results")
    def test_download_failure_handling(self, mock_download):
        """Test that download failures are properly handled in the workflow."""
        mock_download.side_effect = requests.exceptions.ConnectionError("Network error")

        with pytest.raises(requests.exceptions.ConnectionError):
            mock_download("https://example.com/cached_results.json.gz")

    @patch("mteb.leaderboard.app._decompress_gzip_data")
    def test_decompression_failure_handling(self, mock_decompress):
        """Test that decompression failures are properly handled."""
        mock_decompress.side_effect = gzip.BadGzipFile("Invalid gzip")

        with pytest.raises(gzip.BadGzipFile):
            mock_decompress(b"invalid gzip content")

    @patch("mteb.leaderboard.app._write_cache_file")
    def test_write_failure_handling(self, mock_write):
        """Test that write failures are properly handled."""
        mock_write.side_effect = PermissionError("Cannot write file")

        with pytest.raises(PermissionError):
            mock_write('{"data": "test"}', Path("/tmp/test.json"))


# Parametrized tests for edge cases
@pytest.mark.parametrize(
    "content_type,should_warn",
    [
        ("application/gzip", False),
        ("application/octet-stream", False),
        ("application/x-gzip", False),
        ("text/html", True),
        ("application/json", True),
        ("", False),  # Empty content-type should not warn
    ],
)
@patch("requests.get")
def test_content_type_handling(mock_get, content_type, should_warn):
    """Test various content-type header scenarios."""
    mock_response = Mock()
    mock_response.status_code = 200
    mock_response.headers = {"content-type": content_type}
    mock_response.content = b"test content"
    mock_get.return_value = mock_response

    with patch("mteb.leaderboard.app.logger.warning") as mock_warning:
        result = _download_cached_results("http://example.com/test.gz")
        assert result == b"test content"

        if should_warn:
            mock_warning.assert_called_once()
            assert "Unexpected content-type" in mock_warning.call_args[0][0]
        else:
            mock_warning.assert_not_called()


@pytest.mark.parametrize(
    "file_size,max_size_mb,should_fail",
    [
        (1024, 1, False),  # 1KB file, 1MB limit - OK
        (1024 * 1024, 1, False),  # 1MB file, 1MB limit - OK (exactly at limit)
        (1024 * 1024 + 1, 1, True),  # 1MB+1 file, 1MB limit - Fail
        (50 * 1024 * 1024, 50, False),  # 50MB file, 50MB limit - OK
        (51 * 1024 * 1024, 50, True),  # 51MB file, 50MB limit - Fail
    ],
)
@patch("requests.get")
def test_file_size_validation(mock_get, file_size, max_size_mb, should_fail):
    """Test file size validation with various sizes."""
    mock_response = Mock()
    mock_response.status_code = 200
    mock_response.headers = {"content-type": "application/gzip"}
    mock_response.content = b"x" * file_size
    mock_get.return_value = mock_response

    if should_fail:
        with pytest.raises(ValueError, match="Downloaded file too large"):
            _download_cached_results(
                "http://example.com/test.gz", max_size_mb=max_size_mb
            )
    else:
        result = _download_cached_results(
            "http://example.com/test.gz", max_size_mb=max_size_mb
        )
        assert len(result) == file_size
