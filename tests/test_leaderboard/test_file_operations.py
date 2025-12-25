"""Tests for leaderboard file operations (gzip decompression and file writing)."""

import gzip
import io
from pathlib import Path
from unittest.mock import patch

import pytest

from mteb.leaderboard.app import _decompress_gzip_data, _write_cache_file


class TestDecompressGzipData:
    """Test gzip decompression with JSON validation."""

    def test_successful_decompression(self, mock_benchmark_json, mock_gzipped_content):
        """Test successful gzip decompression."""
        gzipped_content = mock_gzipped_content(mock_benchmark_json)
        result = _decompress_gzip_data(gzipped_content)
        assert result == mock_benchmark_json

    def test_invalid_gzip(self):
        """Test that invalid gzip data raises BadGzipFile."""
        invalid_gzip = b"not gzipped data"

        with pytest.raises(gzip.BadGzipFile):
            _decompress_gzip_data(invalid_gzip)

    def test_unicode_decode_error(self):
        """Test that non-UTF-8 content raises UnicodeDecodeError."""
        # Create gzipped content that's not valid UTF-8
        buffer = io.BytesIO()
        with gzip.open(buffer, "wb") as gz_file:
            gz_file.write(b"\x80\x81\x82")  # Invalid UTF-8 sequence
        gzipped_content = buffer.getvalue()

        with pytest.raises(UnicodeDecodeError):
            _decompress_gzip_data(gzipped_content)

    @patch("gzip.open")
    def test_unexpected_exception(self, mock_gzip_open):
        """Test that unexpected exceptions are logged and re-raised."""
        mock_gzip_open.side_effect = RuntimeError("Unexpected error")

        with pytest.raises(RuntimeError):
            _decompress_gzip_data(b"test content")


class TestWriteCacheFile:
    """Test cache file writing with error handling."""

    def test_successful_write(self, tmp_path):
        """Test successful file writing."""
        test_file = tmp_path / "test_cache.json"
        test_data = '{"results": [{"task": "test"}]}'

        _write_cache_file(test_data, test_file)

        assert test_file.exists()
        assert test_file.read_text(encoding="utf-8") == test_data

    def test_create_parent_directory(self, tmp_path):
        """Test that parent directories are created if they don't exist."""
        test_file = tmp_path / "nested" / "dir" / "test_cache.json"
        test_data = '{"results": []}'

        _write_cache_file(test_data, test_file)

        assert test_file.exists()
        assert test_file.read_text(encoding="utf-8") == test_data

    @patch("pathlib.Path.mkdir")
    @patch("pathlib.Path.write_text")
    def test_permission_error(self, mock_write_text, mock_mkdir):
        """Test that permission errors are properly handled."""
        mock_write_text.side_effect = PermissionError("Permission denied")
        test_file = Path("/fake/path/test.json")

        with pytest.raises(PermissionError):
            _write_cache_file("test data", test_file)

    @patch("pathlib.Path.write_text")
    def test_os_error(self, mock_write_text):
        """Test that OS errors are properly handled."""
        mock_write_text.side_effect = OSError("Disk full")
        test_file = Path("/fake/path/test.json")

        with pytest.raises(OSError):
            _write_cache_file("test data", test_file)

    @patch("pathlib.Path.mkdir")
    @patch("pathlib.Path.write_text")
    def test_unexpected_exception(self, mock_write_text, mock_mkdir):
        """Test that unexpected exceptions are logged and re-raised."""
        mock_write_text.side_effect = RuntimeError("Unexpected error")
        test_file = Path("/fake/path/test.json")

        with pytest.raises(RuntimeError):
            _write_cache_file("test data", test_file)

    @patch("pathlib.Path.mkdir")
    def test_directory_creation_error(self, mock_mkdir):
        """Test that directory creation errors are handled."""
        mock_mkdir.side_effect = OSError("Cannot create directory")
        test_file = Path("/fake/nonexistent/test.json")

        with pytest.raises(OSError):
            _write_cache_file("test data", test_file)
