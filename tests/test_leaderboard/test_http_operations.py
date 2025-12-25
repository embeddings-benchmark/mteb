"""Tests for leaderboard HTTP operations."""

from unittest.mock import Mock, patch

import pytest
import requests

from mteb.leaderboard.app import _download_cached_results


class TestDownloadCachedResults:
    """Test HTTP downloading with validation."""

    @patch("requests.get")
    def test_successful_download(self, mock_get):
        """Test successful HTTP download."""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.headers = {"content-type": "application/gzip"}
        mock_response.content = b"test content"
        mock_get.return_value = mock_response

        result = _download_cached_results("http://example.com/test.gz")

        assert result == b"test content"
        mock_get.assert_called_once_with("http://example.com/test.gz", timeout=60)
        mock_response.raise_for_status.assert_called_once()

    @patch("requests.get")
    def test_file_too_large(self, mock_get):
        """Test that oversized files raise ValueError."""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.headers = {"content-type": "application/gzip"}
        mock_response.content = b"x" * (51 * 1024 * 1024)  # 51MB, over limit
        mock_get.return_value = mock_response

        with pytest.raises(ValueError, match="Downloaded file too large"):
            _download_cached_results("http://example.com/test.gz", max_size_mb=50)

    @patch("requests.get")
    def test_unexpected_content_type(self, mock_get):
        """Test that unexpected content-type logs warning but continues."""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.headers = {"content-type": "text/html"}
        mock_response.content = b"test content"
        mock_get.return_value = mock_response

        with patch("mteb.leaderboard.app.logger.warning") as mock_warning:
            result = _download_cached_results("http://example.com/test.gz")

            assert result == b"test content"
            mock_warning.assert_called_once()
            assert "Unexpected content-type" in mock_warning.call_args[0][0]

    @patch("requests.get")
    def test_http_timeout(self, mock_get):
        """Test that HTTP timeout is properly handled."""
        mock_get.side_effect = requests.exceptions.Timeout("Request timed out")

        with pytest.raises(requests.exceptions.Timeout):
            _download_cached_results("http://example.com/test.gz", timeout=30)

    @patch("requests.get")
    def test_connection_error(self, mock_get):
        """Test that connection errors are properly handled."""
        mock_get.side_effect = requests.exceptions.ConnectionError("Connection failed")

        with pytest.raises(requests.exceptions.ConnectionError):
            _download_cached_results("http://example.com/test.gz")

    @patch("requests.get")
    def test_http_error_status(self, mock_get):
        """Test that HTTP error status codes are handled."""
        mock_response = Mock()
        mock_response.status_code = 404
        mock_response.raise_for_status.side_effect = requests.exceptions.HTTPError(
            "404 Not Found"
        )
        mock_get.return_value = mock_response

        with pytest.raises(requests.exceptions.HTTPError):
            _download_cached_results("http://example.com/test.gz")

    @patch("requests.get")
    def test_unexpected_exception(self, mock_get):
        """Test that unexpected exceptions are logged and re-raised."""
        mock_get.side_effect = RuntimeError("Unexpected error")

        with pytest.raises(RuntimeError):
            _download_cached_results("http://example.com/test.gz")
